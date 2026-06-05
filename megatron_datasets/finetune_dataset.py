# coding=utf-8

# Copyright (c) 2024 Tencent Inc. All Rights Reserved.
# Author: xiaotaoliu@tencent.com
"""Iterable style finetune dataset."""

import copy
from datetime import datetime
from typing import Dict, Sequence
from dataclasses import dataclass
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

import torch
from torch.utils.data import IterableDataset
import transformers

from megatron_datasets.utils import print_rank_0, print_datetime, IGNORE_INDEX


def build_train_valid_test_datasets(tokenizer, args, mpu, train_data_path, eval_data_path):
    return make_train_eval_dataset(tokenizer, args, mpu, train_data_path, eval_data_path)


def _tokenize_fn(
    sequence_length, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    padding_method = "longest"

    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=padding_method,
            max_length=sequence_length + 1,
            truncation=True,
        ) for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def tokenize_text(tokenizer, px_pad_to_max_len, seq_length, sources, targets):
    examples = [s + t for s, t in zip([sources], [targets])]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(seq_length, strings, tokenizer) for strings in (examples, [sources])
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    if px_pad_to_max_len:
        input_ids.append(torch.zeros([seq_length + 1], dtype=input_ids[0].dtype))
        labels.append(torch.zeros([seq_length + 1], dtype=labels[0].dtype))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

    if px_pad_to_max_len:
        input_ids = input_ids[0]
        labels = labels[0]
    return input_ids, labels


@dataclass
class LlamaSftIterableDataset(IterableDataset):
    def __init__(
        self,
        args,
        tokenizer,
        files,
        micro_batch_size: int,
        data_parallel_size: int,
        data_parallel_rank: int,
        seed: int,
        px_data_file_format: str,
        is_train: bool,
        split: str = "train"
    ):

        self.args = args
        self.tokenizer = tokenizer
        self.files = files
        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = data_parallel_size
        self.data_parallel_rank = data_parallel_rank
        self.seed = seed
        self.split = split
        self.dataset_list = []
        self.epoch = 0
        self.is_train = is_train
        self.px_data_file_format = px_data_file_format
        assert self.px_data_file_format == "jsonl"
        print_rank_0(f"Data path {self.files} {data_parallel_size} {data_parallel_rank}")

    def __iter__(self):
        if self.is_train:
            px_total_samples_of_dataset = self.args.px_total_samples_of_dataset
            assert px_total_samples_of_dataset > 0, f"To use llama sft-iterable-dataset, the total amount of data must be provided"
        else:
            px_total_samples_of_dataset = self.args.px_finetune_eval_num
            assert px_total_samples_of_dataset > 0, f"To use llama sft-eval-iterable-dataset, the total amount of data must be provided"

        self.total_micro_batch_per_rank = px_total_samples_of_dataset // (
            self.data_parallel_size * self.micro_batch_size
        )
        self.total_samples_per_rank = self.total_micro_batch_per_rank * self.micro_batch_size
        current_consumed_samples = 0
        if self.is_train and self.args.consumed_train_samples is not None and self.args.consumed_train_samples > 0:
            print_datetime(
                f"[before skip samples] {self.args.consumed_train_samples} samples will be skipped"
            )
            consumed_samples_per_dp_rank = self.args.consumed_train_samples // self.data_parallel_size
            print_rank_0(
                f"[restart model] consumed_samples_per_dp_rank {consumed_samples_per_dp_rank}"
            )
            self.epoch = consumed_samples_per_dp_rank // self.total_samples_per_rank
            current_consumed_samples = consumed_samples_per_dp_rank % self.total_samples_per_rank

        while True:
            cnt = 0
            ds = None
            if self.px_data_file_format == "jsonl":
                ds = load_dataset("json", data_files=self.files, split=self.split, streaming=True)
            elif self.px_data_file_format == "pkl":
                sample_rate = None
                if self.args.px_retention_rates_per_domain is not None:
                    sample_rate = self.args.px_retention_rates_per_domain[0]
                ds = load_dataset(
                    "json",
                    data_files=self.files,
                    split=self.split,
                    streaming=True,
                    sample_rate=sample_rate
                )
            else:
                raise NotImplementedError(
                    f"data version {self.px_data_file_format} is not implemented"
                )

            if self.args.px_shuffle_data and self.is_train:
                ds = ds.shuffle(
                    buffer_size=self.args.px_shuffle_buffer_size, seed=self.seed + self.epoch
                )

            distributed_dataset = split_dataset_by_node(
                ds, rank=self.data_parallel_rank, world_size=self.data_parallel_size
            )
            dataset_iter = iter(distributed_dataset)

            if self.is_train and current_consumed_samples > 0:
                print_datetime(
                    f"[skip consumed samples] epoch {self.epoch} remaining skip samples {current_consumed_samples}"
                )
                count = current_consumed_samples
                while count > 0:
                    _ = next(dataset_iter)
                    cnt += 1
                    if count % 10000 == 0:
                        print_datetime(
                            f"[skip consumed samples] last {count} samples will be skipped"
                        )
                    count -= 1
                current_consumed_samples = 0
                consumed_samples_per_dp_rank = 0
                print_datetime(
                    f"[after skip samples] args.consumed_train_samples is set to {current_consumed_samples}"
                )

            for example in dataset_iter:
                if cnt >= self.total_samples_per_rank:
                    break
                cnt += 1
                if self.args.llama_prompt_pattern == "sft_wo_prompt":
                    prompt_template = "{input}"
                    source = prompt_template.format(input=example['input'])
                    target = f"{example['target']}{self.tokenizer.eos_token}"

                    input_ids, labels = tokenize_text(
                        self.tokenizer, self.args.px_pad_to_max_len, self.args.seq_length, source,
                        target
                    )

                    yield dict(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                    )

                else:
                    raise NotImplementedError(
                        'Prompt-pattern {} for iterable_dataset is not implemented.'.format(
                            self.args.llama_prompt_pattern
                        )
                    )
            if self.is_train:
                self.epoch += 1


def make_train_eval_dataset(tokenizer, args, mpu, train_data_path, eval_data_path):
    train_dataset, eval_dataset = None, None

    print_rank_0(f"load samples for finetune model {args.llama_use_iterable_dataset}")

    assert args.llama_use_iterable_dataset
    assert 1 == len(train_data_path), f"the nums of sft data path must be 1"
    if eval_data_path is not None:
        assert 1 == len(eval_data_path), f"the nums of sft eval data path must be 1"

    train_dataset = LlamaSftIterableDataset(
        args,
        tokenizer,
        train_data_path,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=mpu.get_data_parallel_world_size(),
        data_parallel_rank=mpu.get_data_parallel_rank(),
        seed=args.seed,
        px_data_file_format=args.px_data_file_format,
        is_train=True,
        split="train"
    )

    if eval_data_path is not None:
        eval_dataset = LlamaSftIterableDataset(
            args,
            tokenizer,
            eval_data_path,
            micro_batch_size=args.micro_batch_size,
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_parallel_rank=mpu.get_data_parallel_rank(),
            seed=args.seed,
            px_data_file_format=args.px_data_file_format,
            is_train=False,
            split="train"
        )

    return train_dataset, eval_dataset, None
