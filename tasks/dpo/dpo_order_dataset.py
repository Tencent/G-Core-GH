# coding=utf-8
#
# Copyright (c) 2024 Tencent Inc. All Rights Reserved.

from dataclasses import dataclass
import itertools
import json
import os

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

import megatron.core.parallel_state as mpu


def tokenize_text(tokenizer, seq_len, prompt, answer):
    assert tokenizer.pad_token is not None
    prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
    answer_tokenized = tokenizer(answer, add_special_tokens=False)

    input_ids = prompt_tokenized.input_ids + answer_tokenized.input_ids
    attention_mask = prompt_tokenized.attention_mask + answer_tokenized.attention_mask
    labels = [-100] * len(prompt_tokenized.input_ids) + answer_tokenized.input_ids
    unpadded_lens = min(len(input_ids), seq_len)
    prompt_lens = len(prompt_tokenized.input_ids)

    if len(input_ids) < seq_len + 1:
        input_ids += [tokenizer.pad_token_id] * (seq_len + 1 - len(input_ids))
        attention_mask += [0] * (seq_len + 1 - len(attention_mask))
        labels += [-100] * (seq_len + 1 - len(labels))

    input_ids = input_ids[:-1]
    attention_mask = attention_mask[:-1]
    labels = labels[1:]

    if len(input_ids) > seq_len:
        input_ids = input_ids[-seq_len:]
        attention_mask = attention_mask[-seq_len:]
        labels = labels[-seq_len:]
        unpadded_lens = seq_len
    assert unpadded_lens > 0

    return input_ids, attention_mask, labels, unpadded_lens, prompt_lens


@dataclass
class JsonlOrderDataset(TorchIterableDataset):
    def __init__(self, data_path, jsonl_files, domain_name) -> None:
        super().__init__()
        self.data_path = data_path
        self.jsonl_files = jsonl_files
        self.domain_name = domain_name

    def __iter__(self):
        for jsonl_file in self.jsonl_files:
            with open(os.path.join(self.data_path, jsonl_file), "r") as f:
                line = f.readline()
                while line:
                    yield json.loads(line), self.domain_name, jsonl_file
                    line = f.readline()


@dataclass
class DpoOrderDataset(TorchIterableDataset):
    def __init__(
        self,
        tokenizer,
        seq_len,
        path_likes,
        domain_names,
        prompt_format=None,
        eos_token=None,
        golden_loss=False,
    ):
        assert len(path_likes) > 0
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.path_likes = path_likes
        self.jsonl_files_list = []
        self.in_iter = False
        self.domain_names = domain_names
        assert not golden_loss
        if eos_token is None:
            self.eos_token = self.tokenizer._tokenizer.eos_token
        else:
            self.eos_token = eos_token
        if prompt_format is None:
            self.prompt_format = "###{input}\n### Response:\n"
        else:
            self.prompt_format = prompt_format

        for data_path in self.path_likes:
            jsonl_files = [file for file in os.listdir(data_path) if file.endswith('.jsonl')]
            sorted(jsonl_files)
            self.jsonl_files_list.append(jsonl_files)
        assert len(self.jsonl_files_list) == len(self.path_likes)
        assert len(self.path_likes) == len(
            self.domain_names
        ), f"error:{self.path_likes} {self.domain_names}"

        self.underlying = None

    def iter_in_epoch(self):
        for example, domain_name, jsonl_file in self.underlying:
            real_input = example['input'] if example.get('input') else example['instruction']
            prompt = self.prompt_format.format_map({"input": real_input, "instruction": real_input})
            chosen = f"{example['chosen']}{self.eos_token}"
            input_ids, attention_mask, labels, unpadded_lens, prompt_lens = tokenize_text(
                self.tokenizer._tokenizer, self.seq_len, prompt, chosen
            )
            example_json_str = json.dumps(example)
            yield {
                'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),
                'unpadded_lens': torch.tensor(unpadded_lens, dtype=torch.int64),
                'is_chosen': True,
                'is_rejected': False,
                'is_golden': False,
                'line': 0,
                'src_json': example_json_str,
                'domain_name': domain_name,
                'jsonl_file': jsonl_file,
            }

            rejected = f"{example['rejected']}{self.eos_token}"
            input_ids, attention_mask, labels, unpadded_lens, prompt_lens = tokenize_text(
                self.tokenizer._tokenizer, self.seq_len, prompt, rejected
            )
            yield {
                'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),
                'unpadded_lens': torch.tensor(unpadded_lens, dtype=torch.int64),
                'is_chosen': False,
                'is_rejected': True,
                'is_golden': False,
                'line': 1,
                'src_json': example_json_str,
                'domain_name': domain_name,
                'jsonl_file': jsonl_file,
            }

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True
        self.eval_yielded = 0
        for i in range(len(self.path_likes)):
            print(
                f"Reading the order jsonl file: path={self.path_likes[i]}, jsonl={self.jsonl_files_list[i]} domain_name={self.domain_names[i]}"
            )
            self.underlying = JsonlOrderDataset(
                self.path_likes[i], self.jsonl_files_list[i], self.domain_names[i]
            )
            yield from self.iter_in_epoch()


def build_train_valid_test_datasets(
    args, tokenizer, prompt_format=None, eos_token=None, dpo_golden_loss=False
):
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path

    assert mpu.get_data_parallel_world_size() == 1, "only support the data parallel is 1"
    assert args.num_workers <= 1
    assert all([dr == 1.0 for dr in args.px_retention_rates_per_domain])

    train_ds = DpoOrderDataset(
        tokenizer,
        args.seq_length,
        train_path_likes,
        args.px_train_data_domain_names,
        prompt_format=prompt_format,
        eos_token=eos_token,
        golden_loss=dpo_golden_loss,
    )
    eval_ds = None
    if eval_path_likes is not None:
        eval_ds = DpoOrderDataset(
            tokenizer,
            args.seq_length,
            eval_path_likes,
            args.px_eval_data_domain_names,
            prompt_format=prompt_format,
            eos_token=eos_token,
            golden_loss=dpo_golden_loss,
        )
    test_ds = None
    return train_ds, eval_ds, test_ds
