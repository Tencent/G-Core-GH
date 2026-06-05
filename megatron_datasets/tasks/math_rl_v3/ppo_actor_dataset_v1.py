# Copyright (c) 2024 Tencent Inc. All Rights Reserved.
# Author: nrwu@tencent.com, erikfu@tencent.com, yeazhao@tencent.com
# coding=utf-8

import copy
from dataclasses import dataclass
import itertools
import random
import re
from typing_extensions import Optional, List, Dict

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

try:
    from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer as TokenizerType
except ImportError:
    from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer as TokenizerType
from megatron_datasets.utils import print_rank_0, print_datetime
from megatron_datasets.mega_indexed_jsonl_dataset import (
    MegaIndexedJsonlDataset, get_epoch_and_line, update_epoch_and_line
)

from gpatch.core.aligner_helper import random_pad_list


def tokenize_text(tokenizer, prompt_seq_len, prompt):
    assert tokenizer.pad_token is not None
    prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
    input_ids = prompt_tokenized.input_ids
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
        path_likes,
        domain_probabilities,
        domain_names,
        train_data_consuming_progresses=None,
        train=False,
        rank=0,
        dp_rank=0,
        dp_size=1,
        shuffle_strategy="interleave",
        shuffle_buffer_size=1000,
        eval_samples=None,
        seed=0,
        eos_token=None,
        prompt_format=None,
        apply_chat_template=False,
        system_prompt=None,
    ):
        self.seq_len = seq_len
        self.max_position_embeddings = max_position_embeddings
        self.resp_seq_len = resp_seq_len
        self.rollout_micro_batch_size = rollout_micro_batch_size
        self.rollout_global_batch_size = rollout_global_batch_size
        self.tokenizer = tokenizer
        self.train = train
        self.path_likes = path_likes
        self.domain_probabilities = domain_probabilities
        self.domain_names = domain_names
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.shuffle_strategy = shuffle_strategy
        self.shuffle_buffer_size = shuffle_buffer_size
        self.eval_samples = eval_samples
        self.seed = seed
        self.in_iter = False

        if eos_token is None:
            self.eos_token = self.tokenizer._tokenizer.eos_token
        else:
            self.eos_token = eos_token
        self.apply_chat_template = apply_chat_template
        self.system_prompt = system_prompt
        if self.apply_chat_template:
            self.prompt_format = None
        elif prompt_format is None:
            assert system_prompt is None
            self.prompt_format = "###{instruction}\n### Response:\n"
        else:
            assert system_prompt is None
            self.prompt_format = prompt_format

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
            shuffle_strategy=self.shuffle_strategy,
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed,
        )

    def _apply_chat_template(self, example):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": example})
        chat_text = self.tokenizer._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return chat_text, messages

    def iter_in_epoch(self, epoch):
        if torch.distributed.get_rank() == 0:
            print(f'PpoActorDataset.iter_in_epoch epoch {epoch} train {self.train}')
        rng = random.Random(self.seed + epoch)

        messages: Optional[List[Dict[str, str]]] = None

        for example in self.underlying:
            if isinstance(self.prompt_format, list):
                if rng.random() < 0.8:
                    prompt = self.prompt_format[0].format_map(example)  # systerm pt
                else:
                    prompt = self.prompt_format[1].format_map(example)
            else:
                if self.apply_chat_template:
                    assert "question" in example.keys(), f'example keys {example.keys()}'
                    if isinstance(example["question"], dict):
                        example_prompt = example["question"]["problem"]
                    else:
                        example_prompt = example["question"]
                    prompt, messages = self._apply_chat_template(example_prompt)
                else:
                    if "question" in example.keys():
                        example_prompt = example["question"]
                        if isinstance(example_prompt, dict):
                            prompt = self.prompt_format.format(
                                "{}", problem=example_prompt["problem"]
                            )
                        else:
                            prompt = self.prompt_format.format("{}", problem=example_prompt)
                    else:
                        prompt = self.prompt_format.format_map(example)

            input_ids, unpadded_lens = tokenize_text(
                self.tokenizer._tokenizer,
                self.seq_len - self.resp_seq_len,
                prompt,
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
                'gt_label': gt_label,
                'messages': messages,
                'epoch': epoch,
                'line': 1,
            }
            yield o

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True
        for epoch in itertools.count(start=self.start_epoch):
            yield from self.iter_in_epoch(epoch)
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
    args, tokenizer, rank=0, dp_rank=0, dp_size=1, prompt_format=None, eos_token=None
):
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    domain_names = args.px_train_data_domain_names
    apply_chat_template = args.px_apply_chat_template
    system_prompt = args.px_system_prompt
    assert args.num_workers <= 1
    assert all([dr == 1.0 for dr in args.px_retention_rates_per_domain])
    # assert len(domain_names) == 1  # PpoActor 没有多个 domain 的说法
    # assert len(domain_probabilities) == 1

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
        train_path_likes,
        domain_probabilities,
        domain_names,
        train_data_consuming_progresses=args.train_data_consuming_progresses,
        train=True,
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        shuffle_strategy=args.px_shuffle_strategy,
        shuffle_buffer_size=args.px_shuffle_buffer_size,
        seed=args.seed,
        eos_token=eos_token,
        prompt_format=prompt_format,
        apply_chat_template=apply_chat_template,
        system_prompt=system_prompt,
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
            eval_path_likes,
            None,
            domain_names,
            train_data_consuming_progresses=None,
            train=False,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            shuffle_strategy="no_shuffle",
            shuffle_buffer_size=0,
            eval_samples=eval_samples,
            seed=0,
            eos_token=eos_token,
            prompt_format=prompt_format,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
        )
    test_ds = None
    return train_ds, eval_ds, test_ds


@dataclass
class DataCollator(object):
    tokenizer: TokenizerType
    seq_len: int  # == args.seq_length
    resp_seq_len: int
    gen_left_pad: bool
    random_pad: bool = False

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

            assert not (
                self.random_pad and self.gen_left_pad
            ), "random_pad and gen_left_pad are mutually exclusive"
            if self.gen_left_pad:
                lpad_lens.append(batch_max_len)
                num_lpads.append(to_pad)
                item['input_ids'] = [pad_token_id] * to_pad + input_ids
            elif self.random_pad:
                lpad_lens.append(len(input_ids))
                num_lpads.append(0)
                item['input_ids'] = random_pad_list(item['input_ids'], to_pad)
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
        messages: List[Optional[List[Dict[str, str]]]] = [item['messages'] for item in batch]
        ret = dict(
            input_ids=input_ids,
            unpadded_lens=unpadded_lens,
            num_lpads=num_lpads,
            lpad_lens=lpad_lens,
            train=train,
            epoch=epoch,
            line=line,
            gt_label=gt_label,
            messages=messages,
        )
        return ret
