# coding=utf-8
#
# Copyright (c) 2024 Tencent Inc. All Rights Reserved.
# Author: xiaotaoliu@tencent.com, nrwu@tencent.com, erikfu@tencent.com

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


def tokenize_text(tokenizer, max_len, prompt, target):
    text = prompt + target
    assert tokenizer.pad_token is not None
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    text_tokenized = tokenizer(f"{text}", add_special_tokens=False)

    text_input_ids = text_tokenized.input_ids
    text_attention_mask = text_tokenized.attention_mask
    labels = copy.deepcopy(text_input_ids)
    if not len(prompt_input_ids) < len(labels):
        print(f"WARN empty target prompt {prompt} target {target}")
        return None, None, None
    assert labels[:len(prompt_input_ids)] == prompt_input_ids

    if len(text_input_ids) < max_len + 1:
        text_input_ids += [tokenizer.pad_token_id] * (max_len + 1 - len(text_input_ids))
        text_attention_mask += [0] * (max_len + 1 - len(text_attention_mask))

    labels[:len(prompt_input_ids)] = [-100] * len(prompt_input_ids)
    if len(labels) < max_len + 1:
        labels += [-100] * (max_len + 1 - len(labels))

    if len(text_input_ids) > max_len + 1:
        text_input_ids = text_input_ids[-(max_len + 1):]
        text_attention_mask = text_attention_mask[-(max_len + 1):]
        labels = labels[-(max_len + 1):]

    return text_input_ids, labels, text_attention_mask


@dataclass
class SftDataset(TorchIterableDataset):
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
        shuffle_buffer_size=1000,
        seed=0,
        eos_token=None,
        prompt_format=None,
        eval_samples=None,
        target_key='target'
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.train = train
        self.path_likes = path_likes
        self.domain_probabilities = domain_probabilities
        self.domain_names = domain_names
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.eval_samples = eval_samples
        self.seed = seed
        self.target_key = target_key

        # line 表示下层的数据库跳过多少条 jsonl 样本。
        # 注意：
        # 1. 这里的 line 不是 megatron 的 consumed_samples，而是原始 jsonl 数据集消费了多少 json obj。对于
        # pretrain，存在多条 json 拼接成一条 4096 满长度的 seq 的情况。
        # 2. 在不同的 dp rank 上，line 可能是不同的，因为 tokenize 后的 seq len 不同，有的 worker 多消费几个
        # sample，有的少消费几个。

        if self.train:
            assert self.eval_samples is None
        else:
            assert train_data_consuming_progresses is None
            assert self.eval_samples is not None and self.eval_samples

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
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed
        )
        if eos_token is None:
            self.eos_token = self.tokenizer.eos_token
        else:
            self.eos_token = eos_token
        if prompt_format is None:
            self.prompt_format = "###{input}\n### Response:\n"
        else:
            self.prompt_format = prompt_format

    def __iter__(self):
        self.eval_yielded = 0
        for epoch in itertools.count(start=self.start_epoch):
            print(f'SftDataset.__iter__ rank {torch.distributed.get_rank()} epoch {epoch}')
            for example in self.underlying:
                '''
                NOTE：数据格式是：
                ```json
                {
                    'input': 'why is the sky blue?'
                    'target': 'because of science.',
                }
                ```
                最早 luckytyang 的格式不是这样的，但对齐 erikfu 实际测试过的 code and case。
                '''
                prompt = self.prompt_format.format_map(example)
                target = f"{example[self.target_key]}{self.eos_token}"
                input_ids, labels, attention_mask = tokenize_text(
                    self.tokenizer, self.max_seq_len, prompt, target
                )

                if input_ids is None:
                    continue
                data = {
                    'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                    'labels': torch.tensor(labels, dtype=torch.int64),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),
                    'train': self.train,
                    'epoch': epoch,
                    'line': 1,
                }
                yield data
                if not self.train:
                    self.eval_yielded += 1
                    if self.eval_yielded >= self.eval_samples:  # 保证 each eval epoch 都看到相同的数据
                        self.eval_yielded = 0
                        return

            self.underlying = MegaIndexedJsonlDataset(
                self.path_likes,
                self.domain_probabilities,
                self.domain_names,
                dp_rank=self.dp_rank,
                dp_size=self.dp_size,
                epoch=epoch + 1,
                consumed=0,
                shuffle_buffer_size=self.shuffle_buffer_size,
                seed=self.seed
            )
        assert False, 'never reachable'


def build_train_valid_test_datasets(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    prompt_format=None,
    eos_token=None,
    target_key='target'
):
    max_seq_len = args.seq_length
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    domain_names = args.px_train_data_domain_names
    shuffle_buffer_size = args.px_shuffle_buffer_size
    seed = args.seed
    assert args.num_workers <= 1
    assert all([dr == 1.0 for dr in args.px_retention_rates_per_domain])
    assert len(domain_names) == 1  # SFT 没有多个 domain 的说法
    assert len(domain_probabilities) == 1
    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {args.train_data_consuming_progresses}'
    )
    train_ds = SftDataset(
        tokenizer,
        max_seq_len,
        train_path_likes,
        domain_probabilities,
        domain_names,
        train=True,
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        train_data_consuming_progresses=args.train_data_consuming_progresses,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        prompt_format=prompt_format,
        eos_token=eos_token,
        target_key=target_key
    )
    eval_ds = None
    if eval_path_likes is not None:
        eval_samples = args.eval_iters * args.global_batch_size // dp_size
        eval_ds = SftDataset(
            tokenizer,
            max_seq_len,
            eval_path_likes,
            None,
            domain_names,
            train=False,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            train_data_consuming_progresses=None,
            shuffle_buffer_size=0,
            seed=0,
            prompt_format=prompt_format,
            eos_token=eos_token,
            eval_samples=eval_samples,
            target_key=target_key
        )
    test_ds = None
    return train_ds, eval_ds, test_ds
