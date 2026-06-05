# coding=utf-8
#
# Copyright (c) 2024 Tencent Inc. All Rights Reserved.
# Author: nrwu@tencent.com, erikfu@tencent.com

from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Sequence
import copy
import itertools
import random
import re

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset as TorchIterableDataset
import datasets
import torch
import transformers

try:
    from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer as TokenizerType
except ImportError:
    from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer as TokenizerType

from megatron_datasets.utils import print_rank_0, print_datetime
from megatron_datasets.mega_indexed_jsonl_dataset import MegaIndexedJsonlDataset
from megatron_datasets.mega_indexed_jsonl_dataset import get_epoch_and_line, update_epoch_and_line
from megatron.core.utils import divide


def tokenize_text(tokenizer, prompt_seq_len, prompt, ppo_rollout_debug_pad_to_prompt_seq_len):
    assert tokenizer.pad_token is not None
    prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
    input_ids = prompt_tokenized.input_ids
    attention_mask = prompt_tokenized.attention_mask
    unpadded_lens = len(input_ids)

    # debugging purpose only
    if ppo_rollout_debug_pad_to_prompt_seq_len and len(input_ids) < prompt_seq_len:
        # 都是 pad token id 会不均衡
        # input_ids += [tokenizer.pad_token_id] * (prompt_seq_len - len(input_ids))
        tmpi = copy.copy(input_ids)
        while len(input_ids) < prompt_seq_len:
            input_ids += tmpi
        input_ids = input_ids[-prompt_seq_len:]
        unpadded_lens = len(input_ids)

    if len(input_ids) > prompt_seq_len:
        input_ids = input_ids[-prompt_seq_len:]
        unpadded_lens = prompt_seq_len
    assert unpadded_lens > 0 and unpadded_lens == len(input_ids)
    return input_ids, unpadded_lens


def extract_gt_answer(text):
    match = re.search(r'####\s*(-?\d+(\.\d+)?)', text)
    if match:
        return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    else:
        return None


@dataclass
class PpoActorDataset(TorchIterableDataset):
    def __init__(
        self,
        tokenizer,
        seq_len,
        max_position_embeddings,
        resp_seq_len,
        rollout_micro_batch_size,
        rollout_global_batch_size,
        rollout_max_prompt_len_diff,
        path_likes,
        domain_probabilities,
        domain_names,
        train_data_consuming_progresses=None,
        train=False,
        rank=0,
        dp_rank=0,
        dp_size=1,
        shuffle_buffer_size=1000,
        eval_samples=None,
        ppo_rollout_debug_pad_to_prompt_seq_len=False,
        seed=0,
        eos_token=None,
        prompt_format=None,
        rm_1_prompt_format=None,  # 目前只 hardcoding 了 rms[1] 的特殊规则，不过应该很容易扩展。
    ):
        self.seq_len = seq_len
        self.max_position_embeddings = max_position_embeddings
        self.resp_seq_len = resp_seq_len
        self.rollout_micro_batch_size = rollout_micro_batch_size
        self.rollout_global_batch_size = rollout_global_batch_size
        self.rollout_max_prompt_len_diff = rollout_max_prompt_len_diff
        assert self.seq_len % self.rollout_max_prompt_len_diff == 0
        self.tokenizer = tokenizer
        self.train = train
        self.path_likes = path_likes
        self.domain_probabilities = domain_probabilities
        self.domain_names = domain_names
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.eval_samples = eval_samples
        self.ppo_rollout_debug_pad_to_prompt_seq_len = ppo_rollout_debug_pad_to_prompt_seq_len
        self.seed = seed
        self.in_iter = False

        if eos_token is None:
            self.eos_token = self.tokenizer._tokenizer.eos_token
        else:
            self.eos_token = eos_token
        if prompt_format is None:
            self.prompt_format = "###{instruction}\n### Response:\n"
        else:
            self.prompt_format = prompt_format
        self.rm_1_prompt_format = rm_1_prompt_format

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
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed
        )

    def iter_in_epoch(self, epoch, pool):
        if torch.distributed.get_rank() == 0:
            print(f'PpoActorDataset.iter_in_epoch epoch {epoch} train {self.train}')
        rng = random.Random(self.seed + epoch)
        max_fl = -1

        for ei, example in enumerate(self.underlying):
            if isinstance(self.prompt_format, list):
                if rng.random() < 0.8:
                    prompt = self.prompt_format[0].format_map(example)  # systerm pt
                else:
                    prompt = self.prompt_format[1].format_map(example)
            else:
                # temp setting
                if "question" in example.keys():
                    example_prompt = example["question"]
                    if isinstance(example_prompt, dict):
                        prompt = self.prompt_format.format("{}", problem=example_prompt["problem"])
                    else:
                        prompt = self.prompt_format.format("{}", problem=example_prompt)
                else:
                    prompt = self.prompt_format.format_map(example)
            input_ids, unpadded_lens = tokenize_text(
                self.tokenizer._tokenizer, self.seq_len - self.resp_seq_len, prompt,
                self.ppo_rollout_debug_pad_to_prompt_seq_len
            )

            if 'answer' in example.keys():
                gt_label = extract_gt_answer(example['answer'])
                if gt_label is None:
                    gt_label = 0
            else:
                gt_label = 0

            o = {
                'input_ids': input_ids,
                'unpadded_lens': unpadded_lens,
                'train': self.train,
                'fl': ei,
                'gt_label': gt_label,
            }

            # 如果 rms[1] 有特殊规则的话
            rm_1_prompt = None
            if self.rm_1_prompt_format is not None:
                rm_1_prompt = self.rm_1_prompt_format.format_map(example)
                rm_1_input_ids, rm_1_attention_mask, rm_1_unpadded_lens = tokenize_text(
                    self.tokenizer._tokenizer, self.seq_len - self.resp_seq_len, rm_1_prompt,
                    self.ppo_rollout_debug_pad_to_prompt_seq_len
                )
                o.update(
                    {
                        'rm_1_input_ids': rm_1_input_ids,
                        'rm_1_unpadded_lens': rm_1_unpadded_lens,
                    }
                )
            pool.append(o)

            # TODO(@nrwu): drop this
            assert len(
                pool
            ) <= self.seq_len // self.rollout_max_prompt_len_diff * self.rollout_micro_batch_size
            if len(pool) >= self.rollout_micro_batch_size:
                pool.sort(key=lambda o: o['unpadded_lens'], reverse=True)

                diffs = [
                    pool[i]['unpadded_lens'] -
                    pool[i + self.rollout_micro_batch_size - 1]['unpadded_lens']
                    for i in range(len(pool) - self.rollout_micro_batch_size + 1)
                ]
                for i in range(len(diffs)):
                    if diffs[i] <= self.rollout_max_prompt_len_diff:
                        for j in range(self.rollout_micro_batch_size):
                            o = pool.pop(i)
                            assert len(o['input_ids']) == o['unpadded_lens']
                            o['epoch'] = epoch
                            fl = o['fl']
                            del o['fl']
                            if fl > max_fl:
                                o['line'] = fl - max_fl
                                max_fl = fl
                            else:
                                assert fl != max_fl
                                o['line'] = 0
                            yield o

                        if not self.train:
                            self.eval_yielded += 1
                            if self.eval_yielded >= self.eval_samples:  # 保证 each eval epoch 都看到相同的数据
                                self.eval_yielded = 0
                                return
                        break

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True
        self.eval_yielded = 0
        pool = []
        for epoch in itertools.count(start=self.start_epoch):
            yield from self.iter_in_epoch(epoch, pool)
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
            for x in pool:
                x['fl'] = -2
        assert False, 'never reachable'


def build_train_valid_test_datasets(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    prompt_format=None,
    rm_1_prompt_format=None,
    eos_token=None
):
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    domain_names = args.px_train_data_domain_names
    assert args.num_workers <= 1
    assert all([dr == 1.0 for dr in args.px_retention_rates_per_domain])
    assert len(domain_names) == 1  # PpoActor 没有多个 domain 的说法
    assert len(domain_probabilities) == 1

    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {args.train_data_consuming_progresses}'
    )
    train_ds = PpoActorDataset(
        tokenizer,
        args.seq_length,
        args.max_position_embeddings,
        args.ppo_resp_seq_len,
        args.ppo_rollout_micro_batch_size,
        args.ppo_rollout_global_batch_size,
        args.ppo_rollout_max_prompt_len_diff,
        train_path_likes,
        domain_probabilities,
        domain_names,
        train_data_consuming_progresses=args.train_data_consuming_progresses,
        train=True,
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        shuffle_buffer_size=args.px_shuffle_buffer_size,
        ppo_rollout_debug_pad_to_prompt_seq_len=args.ppo_rollout_debug_pad_to_prompt_seq_len,
        seed=args.seed,
        prompt_format=prompt_format,
        rm_1_prompt_format=rm_1_prompt_format,
        eos_token=eos_token
    )
    eval_ds = None
    if eval_path_likes is not None:
        eval_samples = args.eval_iters * args.global_batch_size // dp_size
        eval_ds = PpoActorDataset(
            tokenizer,
            args.seq_length,
            args.max_position_embeddings,
            args.ppo_resp_seq_len,
            args.ppo_rollout_micro_batch_size,
            args.ppo_rollout_global_batch_size,
            args.ppo_rollout_max_prompt_len_diff,
            eval_path_likes,
            None,
            domain_names,
            train_data_consuming_progresses=None,
            train=False,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            shuffle_buffer_size=0,
            eval_samples=eval_samples,
            ppo_rollout_debug_pad_to_prompt_seq_len=args.ppo_rollout_debug_pad_to_prompt_seq_len,
            seed=0,
            prompt_format=prompt_format,
            rm_1_prompt_format=rm_1_prompt_format,
            eos_token=eos_token
        )
    test_ds = None
    return train_ds, eval_ds, test_ds


@dataclass
class DataCollator(object):
    tokenizer: TokenizerType
    seq_len: int  # == args.seq_length
    resp_seq_len: int
    gen_left_pad: bool

    def __call__(self, batch):
        pad_token_id = self.tokenizer._tokenizer.pad_token_id

        batch_lens = [item['unpadded_lens'] for item in batch]
        input_ids_lens = [len(item['input_ids']) for item in batch]
        batch_max_len = max(batch_lens)
        lpad_lens = []
        num_lpads = []
        for item in batch:
            input_ids = item['input_ids']
            assert isinstance(input_ids, list)
            to_pad = batch_max_len - len(input_ids)
            assert batch_lens == input_ids_lens and to_pad >= 0, f'wtf batch_lens {batch_lens} input_ids_lens {input_ids_lens}'

            if self.gen_left_pad:
                lpad_lens.append(batch_max_len)
                num_lpads.append(to_pad)
                item['input_ids'] = [pad_token_id] * to_pad + input_ids
            else:
                lpad_lens.append(len(input_ids))
                num_lpads.append(0)
                item['input_ids'] += [pad_token_id] * to_pad
            assert len(input_ids) <= self.seq_len

        input_ids = torch.as_tensor([item['input_ids'] for item in batch], dtype=torch.int64)
        unpadded_lens = torch.as_tensor(
            [item['unpadded_lens'] for item in batch], dtype=torch.int64
        )
        train = torch.as_tensor([item['train'] for item in batch], dtype=torch.bool)
        epoch = torch.as_tensor([item['epoch'] for item in batch], dtype=torch.int64)
        line = torch.as_tensor([item['line'] for item in batch], dtype=torch.int64)
        num_lpads = torch.as_tensor(num_lpads, dtype=torch.int64)
        lpad_lens = torch.as_tensor(lpad_lens, dtype=torch.int64)
        # gsm 数据集 gt 都是 int, 看是否要将 gt 设置成 float, 如果设置 float 的话， get_batch
        # 里 broadcast 的时候则需要用 torch.float 做广播
        gt_label = torch.as_tensor([item['gt_label'] for item in batch], dtype=torch.int64)
        ret = dict(
            input_ids=input_ids,
            unpadded_lens=unpadded_lens,
            num_lpads=num_lpads,
            lpad_lens=lpad_lens,
            train=train,
            epoch=epoch,
            line=line,
            gt_label=gt_label,
        )

        # rms[1] 特殊 prompt
        tweak_rm_prompt = 'rm_1_input_ids' in batch[0]
        if tweak_rm_prompt:
            rm_1_unpadded_lens = [item['rm_1_unpadded_lens'] for item in batch]
            assert rm_1_unpadded_lens == [len(item['rm_1_input_ids']) for item in batch]
            batch_max_len = max(rm_1_unpadded_lens)
            for item in batch:
                # rm must be right padding
                assert not self.gen_left_pad
                to_pad = batch_max_len - len(item['rm_1_input_ids'])
                item['rm_1_input_ids'] += [pad_token_id] * to_pad

            rm_1_input_ids = torch.as_tensor(
                [item['rm_1_input_ids'] for item in batch], dtype=torch.int64
            )
            rm_1_unpadded_lens = torch.as_tensor(
                [item['rm_1_unpadded_lens'] for item in batch], dtype=torch.int64
            )
            ret.update(
                {
                    'rm_1_input_ids': rm_1_input_ids,
                    'rm_1_unpadded_lens': rm_1_unpadded_lens,
                }
            )
        return ret
