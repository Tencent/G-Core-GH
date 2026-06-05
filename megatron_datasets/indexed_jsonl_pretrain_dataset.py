# coding=utf-8
#
# Copyright (c) 2024 Tencent Inc. All Rights Reserved.
# Author: xiaotaoliu@tencent.com, nrwu@tencent.com

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Sequence
import copy
import itertools

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset as TorchIterableDataset
import datasets
import torch
import transformers

from megatron_datasets.utils import print_rank_0, print_datetime
from megatron_datasets.mega_indexed_jsonl_dataset import MegaIndexedJsonlDataset
from megatron_datasets.mega_indexed_jsonl_dataset import get_epoch_and_line, update_epoch_and_line


def tokenize_text(tokenizer, text):
    y = tokenizer(f'{tokenizer.bos_token}{text}{tokenizer.eos_token}', add_special_tokens=False)
    input_ids = y.input_ids
    attention_mask = y.attention_mask
    return input_ids, attention_mask


@dataclass
class PretrainDataset(TorchIterableDataset):
    # 已测试：
    # 1. 每个 eval epoch 都是相同的数据。
    # 2. 多次热启动，每个 rank，观察到数据顺序完全一致。
    # 3. 热启动后，skip 的数据就是 consumed 数据。

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        path_likes,
        domain_probabilities,
        domain_names,
        train_data_consuming_progresses=None,
        train=False,
        rank=0,
        dp_rank=0,
        dp_size=1,
        access_policy_interleave=False,
        shuffle_buffer_size=1000,
        eval_samples=None,
        seed=0,
        retention_rates_per_domains=[],
        enable_pareto=[],
        pareto_alphas=[],
        pareto_scales=[],
        pareto_score_scales=[],
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.train = train
        self.path_likes = path_likes
        self.domain_probabilities = domain_probabilities
        self.retention_rates_per_domains = retention_rates_per_domains
        if train:
            assert len(self.retention_rates_per_domains) == len(domain_probabilities)

        self.domain_names = domain_names
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.access_policy_interleave = access_policy_interleave
        self.shuffle_buffer_size = shuffle_buffer_size
        self.eval_samples = eval_samples
        self.seed = seed
        self.in_iter = False

        self.enable_pareto = enable_pareto
        self.pareto_alphas = pareto_alphas
        self.pareto_scales = pareto_scales
        self.pareto_score_scales = pareto_score_scales

        if self.train:
            assert self.eval_samples is None
        else:
            assert train_data_consuming_progresses is None
            assert self.eval_samples is not None and self.eval_samples > 0
            assert self.shuffle_buffer_size == 0

        self.train_data_consuming_progresses = train_data_consuming_progresses
        self.start_epoch, line = get_epoch_and_line(self.train_data_consuming_progresses, rank)
        self.underlying = MegaIndexedJsonlDataset(
            self.path_likes,
            self.domain_probabilities,
            self.domain_names,
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
            epoch=self.start_epoch,
            consumed=line,
            access_policy_interleave=access_policy_interleave,
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed,
            train=self.train,
            retention_rates_per_domains=self.retention_rates_per_domains,
            enable_pareto=self.enable_pareto,
            pareto_alphas=self.pareto_alphas,
            pareto_scales=self.pareto_scales,
            pareto_score_scales=self.pareto_score_scales,
        )

    def iter_in_epoch(self, epoch, bufs):
        if torch.distributed.get_rank() == 0:
            print(f'PretrainDataset.iter_in_epoch epoch {epoch} train {self.train}')

        # 每个epoch开始的时候重置 n_packed，避免累计多 epoch 的 n_packed
        for domain_id, _ in enumerate(self.domain_names):
            bufs[domain_id].n_packed = 0

        for example in self.underlying:
            domain_id = example['domain_id']
            bufs[domain_id].n_packed += 1
            if example.get('deleted', False):  # 被黑名单的垃圾数据
                continue

            # TODO
            # 1. 要不要 pad 啊...，保持上下文完整性？
            # 2. position id? 分段 attention？看下 pai 的做法？
            # 3. return 0s to perf
            text = example['content']  # BoG 一直都是这个 key
            _input_ids, _ = tokenize_text(self.tokenizer, text)
            bufs[domain_id].input_ids += _input_ids

            while len(bufs[domain_id].input_ids) >= self.max_seq_len + 1:
                input_ids = bufs[domain_id].input_ids[:self.max_seq_len + 1]
                labels = copy.deepcopy(input_ids)
                ret_d = {
                    'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                    'labels': torch.tensor(labels, dtype=torch.int64),
                    'train': self.train,
                    'epoch': epoch,
                    'line': bufs[domain_id].n_packed,
                }
                yield ret_d
                # 参考 nv 的实现，捡回来一个 token
                bufs[domain_id].input_ids = bufs[domain_id].input_ids[self.max_seq_len:]
                bufs[domain_id].n_packed = 0

                if not self.train:
                    self.eval_yielded += 1
                    if self.eval_yielded >= self.eval_samples:  # 保证 each eval epoch 都看到相同的数据
                        self.eval_yielded = 0
                        return

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True
        self.eval_yielded = 0
        bufs = [
            SimpleNamespace(input_ids=[], n_packed=0)
            for domain_id, _ in enumerate(self.domain_names)
        ]
        for epoch in itertools.count(start=self.start_epoch):
            yield from self.iter_in_epoch(epoch, bufs)

            self.underlying = MegaIndexedJsonlDataset(
                self.path_likes,
                self.domain_probabilities,
                self.domain_names,
                dp_rank=self.dp_rank,
                dp_size=self.dp_size,
                epoch=epoch + 1,
                consumed=0,
                shuffle_buffer_size=self.shuffle_buffer_size,
                seed=self.seed,
                train=self.train,
                retention_rates_per_domains=self.retention_rates_per_domains,
                enable_pareto=self.enable_pareto,
                pareto_alphas=self.pareto_alphas,
                pareto_scales=self.pareto_scales,
                pareto_score_scales=self.pareto_score_scales,
            )
        assert False, 'never reachable'


def build_train_valid_test_datasets(args, tokenizer, rank=0, dp_rank=0, dp_size=1):
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    retention_rates_per_domains = args.px_retention_rates_per_domain
    enable_pareto = args.px_train_apply_pareto
    pareto_alpha = args.px_train_pareto_alpha
    pareto_scale = args.px_train_pareto_scale
    pareto_score_scale = args.train_pareto_score_scale
    domain_names = args.px_train_data_domain_names
    assert args.num_workers <= 1

    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {args.train_data_consuming_progresses}'
    )
    train_ds = PretrainDataset(
        tokenizer,
        args.seq_length,
        train_path_likes,
        domain_probabilities,
        domain_names,
        train_data_consuming_progresses=args.train_data_consuming_progresses,
        train=True,
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        access_policy_interleave=args.px_indexed_jsonl_dataset_access_policy_interleave,
        shuffle_buffer_size=args.px_shuffle_buffer_size,
        seed=args.seed,
        retention_rates_per_domains=retention_rates_per_domains,
        enable_pareto=enable_pareto,
        pareto_alphas=pareto_alpha,
        pareto_scales=pareto_scale,
        pareto_score_scales=pareto_score_scale,
    )
    eval_ds = None
    if eval_path_likes is not None:
        eval_samples = args.eval_iters * args.global_batch_size // dp_size
        eval_ds = PretrainDataset(
            tokenizer,
            args.seq_length,
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
