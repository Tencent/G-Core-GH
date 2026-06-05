# copyright (c) 2024 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com
# NOTE: 不要修改这个文件，用 v1 / v3。

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from types import SimpleNamespace
import json
import copy

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from datasets import load_dataset

from megatron_datasets.utils import print_rank_0

from megatron_datasets.packed_indexed_jsonl_pretrain_dataset import PackedPretrainDataset
from megatron_datasets.mega_indexed_jsonl_dataset_v2 import (
    MegaIndexedJsonlDatasetV2,
    get_domain_epoch_and_line,
    update_domain_consumed,
    adjust_domain_id_list_for_dp,
    generate_global_batch_domain_id,
    add_domain_id,
)


def tokenize_text_for_packed(tokenizer, text):
    y = tokenizer(f'{tokenizer.bos_token}{text}{tokenizer.eos_token}', add_special_tokens=False)
    input_ids = y.input_ids
    labels = copy.deepcopy(input_ids)
    input_ids = input_ids[:-1]
    labels = labels[1:]
    attention_mask = y.attention_mask
    return input_ids, labels, attention_mask


class PackedMegaIndexedJsonlDatasetV2(MegaIndexedJsonlDatasetV2):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        path_likes,
        domain_probabilities,
        domain_names,
        global_batch_size,
        train_data_consuming_progresses=None,
        rank=0,
        dp_rank=0,
        dp_size=1,
        access_policy_interleave=False,
        shuffle_buffer_size=1000,
        seed=0,
        train=False,
        retention_rates_per_domains=[],
        unsplit_eval_data=False,
        enable_pareto=[],
        pareto_alphas=[],
        pareto_scales=[],
        pareto_score_scales=[],
        top_domains_to_cut=1,
    ):
        if dp_rank == 0:
            print("Dataset type: PackedMegaIndexedJsonlDatasetV2")
        super().__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            path_likes=path_likes,
            domain_probabilities=domain_probabilities,
            domain_names=domain_names,
            global_batch_size=global_batch_size,
            train_data_consuming_progresses=train_data_consuming_progresses,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            access_policy_interleave=access_policy_interleave,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            train=train,
            retention_rates_per_domains=retention_rates_per_domains,
            unsplit_eval_data=unsplit_eval_data,
            enable_pareto=enable_pareto,
            pareto_alphas=pareto_alphas,
            pareto_scales=pareto_scales,
            pareto_score_scales=pareto_score_scales,
            top_domains_to_cut=top_domains_to_cut,
        )

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True

        bufs = [
            SimpleNamespace(input_ids=[], labels=[], n_packed=0)
            for domain_id, _ in enumerate(self.domain_names)
        ]

        while True:
            domain_id = self.global_batch_domain_id[self.curr_pointer_position]
            ds = self.ds_list[domain_id]

            # 先在 buffer 里面看看是否有足够的数据能弹出
            while len(bufs[domain_id].input_ids) < self.max_seq_len:
                try:
                    idx = next(ds)
                except StopIteration:
                    self.domain_consumed[domain_id].epoch += 1
                    domain_epoch = self.domain_consumed[domain_id].epoch
                    # 该 domain 满了，重开一次
                    self.ds_list[domain_id] = iter(
                        self.create_ds(
                            domain_id, self.path_likes[domain_id], domain_epoch=domain_epoch
                        )
                    )
                    ds = self.ds_list[domain_id]
                    idx = next(ds)

                fname = idx['data_file_name']
                offset = idx['offset']
                length = idx['length']
                assert domain_id == idx['domain_id'].item()
                example = self.read_and_parse_obj_from_jsonl(fname, offset, length)
                # TODO: n_packed 添加到 buf 里
                bufs[domain_id].n_packed += 1
                if example.get('deleted', False):  # 被黑名单的垃圾数据
                    continue
                text = example['content']
                _input_ids, _labels, _ = tokenize_text_for_packed(self.tokenizer, text)
                bufs[domain_id].input_ids += _input_ids
                bufs[domain_id].labels += _labels

            input_ids = bufs[domain_id].input_ids[:self.max_seq_len]
            labels = bufs[domain_id].labels[:self.max_seq_len]
            ret_d = {
                'input_ids':
                    torch.tensor(input_ids, dtype=torch.int64),
                'labels':
                    torch.tensor(labels, dtype=torch.int64),
                'train':
                    self.train,
                'domain_id':
                    torch.tensor(domain_id, dtype=torch.int64),
                'n_packed':
                    torch.tensor(bufs[domain_id].n_packed, dtype=torch.int64),
                'domain_epoch':
                    torch.tensor(self.domain_consumed[domain_id].epoch, dtype=torch.int64),
                'curr_pointer_position':
                    self.curr_pointer_position
            }
            bufs[domain_id].input_ids = bufs[domain_id].input_ids[self.max_seq_len:]
            bufs[domain_id].labels = bufs[domain_id].labels[self.max_seq_len:]
            bufs[domain_id].n_packed = 0
            self.curr_pointer_position = (self.curr_pointer_position +
                                          1) % len(self.global_batch_domain_id)
            # TODO: 如果 curr_pointer_position 达到 global_batch_domain_id 的尾巴
            # 是否要对 global_batch_domain_id 重新生成，或者 shuffle？
            # 其实主要是为了 shuffle global_batch_domain_id 的id?
            # if self.curr_pointer_position == 0:
            #     # shuffle global_batch_domain_id
            yield ret_d

        assert False, 'never reachable'


def build_train_valid_test_datasets(args, tokenizer, rank=0, dp_rank=0, dp_size=1):
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    retention_rates_per_domains = args.px_retention_rates_per_domain
    domain_names = args.px_train_data_domain_names
    enable_pareto = args.px_train_apply_pareto
    pareto_alpha = args.px_train_pareto_alpha
    pareto_scale = args.px_train_pareto_scale
    pareto_score_scale = args.train_pareto_score_scale
    assert args.num_workers <= 1

    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {args.train_data_consuming_progresses}'
    )
    train_ds = PackedMegaIndexedJsonlDatasetV2(
        tokenizer,
        args.seq_length,
        train_path_likes,
        domain_probabilities,
        domain_names,
        args.global_batch_size,
        train_data_consuming_progresses=args.train_data_consuming_progresses,
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        access_policy_interleave=False,  # todo: 测试 nr 对这个flag 的去处是否有影响
        shuffle_buffer_size=args.px_shuffle_buffer_size,
        seed=args.seed,
        train=True,
        retention_rates_per_domains=retention_rates_per_domains,
        unsplit_eval_data=False,
        enable_pareto=enable_pareto,
        pareto_alphas=pareto_alpha,
        pareto_scales=pareto_scale,
        pareto_score_scales=pareto_score_scale,
        top_domains_to_cut=args.px_top_domains_to_cut,
    )
    # eval data 不必再单独设计，复用之前的
    eval_ds = None
    if eval_path_likes is not None:
        eval_samples = args.eval_iters * args.global_batch_size // dp_size
        eval_ds = PackedPretrainDataset(
            tokenizer,
            args.seq_length,
            args.max_position_embeddings,
            eval_path_likes,
            None,
            args.px_eval_data_domain_names,
            train_data_consuming_progresses=None,
            train=False,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            access_policy_interleave=args.px_indexed_jsonl_dataset_access_policy_interleave,
            shuffle_buffer_size=0,
            eval_samples=eval_samples,
            seed=0
        )
    test_ds = None

    return train_ds, eval_ds, test_ds
