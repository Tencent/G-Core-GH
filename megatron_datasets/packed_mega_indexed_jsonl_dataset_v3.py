# copyright (c) 2024 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from types import SimpleNamespace
import json
import copy
import os

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset
from datasets import load_dataset

from megatron_datasets.utils import print_rank_0
from megatron_datasets.indexed_jsonl_pretrain_dataset import PretrainDataset
from megatron_datasets.mega_indexed_jsonl_dataset_v3 import (
    MegaIndexedJsonlDatasetV3, get_consumed_by_this_worker, get_consumed_in_this_domain
)


def tokenize_text_for_packed(tokenizer, text):
    y = tokenizer(f'{tokenizer.bos_token}{text}{tokenizer.eos_token}', add_special_tokens=False)
    input_ids = y.input_ids
    labels = copy.deepcopy(input_ids)
    input_ids = input_ids[:-1]
    labels = labels[1:]
    attention_mask = y.attention_mask
    return input_ids, labels, attention_mask


class PackedMegaIndexedJsonlDatasetV3(MegaIndexedJsonlDatasetV3):
    # 中间数据结构变化较大，无法兼容 indexed jsonl dataset v2。

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
        num_workers=1,
        access_policy_interleave=False,
        shuffle_buffer_size=1000,
        seed=0,
        train=False,
        retention_rates_per_domains=None,
        unsplit_eval_data=False,
        enable_pareto=[],
        pareto_alphas=[],
        pareto_scales=[],
        pareto_score_scales=[],
        top_domains_to_cut=1,
    ):
        if dp_rank == 0:
            print("Dataset type: PackedMegaIndexedJsonlDatasetV3")
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
            num_workers=num_workers,
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
            domain_id = self.global_batch_domain_id[self.domain_cand_off]
            ds = self.ds_list[domain_id]

            # 先在 buffer 里面看看是否有足够的数据能弹出
            while len(bufs[domain_id].input_ids) < self.max_seq_len:
                try:
                    idx = next(ds)
                except StopIteration:
                    # 该 domain 空了，重开一次
                    self.ds_list[domain_id] = iter(
                        self.create_dataset(domain_id, self.path_likes[domain_id], new_epoch=True)
                    )
                    ds = self.ds_list[domain_id]
                    idx = next(ds)

                fname = idx['data_file_name']
                offset = idx['offset']
                length = idx['length']
                assert idx['domain_id'].item() == domain_id
                worker_id = idx['worker_id']

                example = self.read_and_parse_obj_from_jsonl(fname, offset, length)
                bufs[domain_id].n_packed += 1
                if example.get('deleted', False):  # 被黑名单的垃圾数据
                    continue
                text = example['content']
                doc_id = example['docid']
                _input_ids, _labels, _ = tokenize_text_for_packed(self.tokenizer, text)
                bufs[domain_id].input_ids += _input_ids
                bufs[domain_id].labels += _labels

            input_ids = bufs[domain_id].input_ids[:self.max_seq_len]
            labels = bufs[domain_id].labels[:self.max_seq_len]

            domain_epoch = get_consumed_by_this_worker(
                get_consumed_in_this_domain(self.consumed_by_this_rank, domain_id), worker_id
            ).epoch
            ret_d = {
                'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'train': self.train,
                'domain_id': torch.tensor(domain_id, dtype=torch.int64),
                'worker_id': torch.tensor(worker_id, dtype=torch.int64),
                'domain_epoch': torch.tensor(domain_epoch, dtype=torch.int64),
                'domain_line': torch.tensor(bufs[domain_id].n_packed, dtype=torch.int64),
                'domain_cand_off': self.domain_cand_off
            }
            bufs[domain_id].input_ids = bufs[domain_id].input_ids[self.max_seq_len:]
            bufs[domain_id].labels = bufs[domain_id].labels[self.max_seq_len:]
            bufs[domain_id].n_packed = 0
            self.domain_cand_off = (self.domain_cand_off + 1) % len(self.global_batch_domain_id)
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
    train_ds = PackedMegaIndexedJsonlDatasetV3(
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
        access_policy_interleave=False,
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
    eval_ds = None
    assert eval_path_likes is None
    test_ds = None

    return train_ds, eval_ds, test_ds
