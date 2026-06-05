# coding=utf-8

# Copyright (c) 2024 Tencent Inc. All Rights Reserved.
# Author: xiaotaoliu@tencent.com
"""Iterable style pretrain dataset."""

import os
import copy
import json
from typing import List
from dataclasses import dataclass
from datasets import load_dataset
from datasets import interleave_datasets
from datasets.distributed import split_dataset_by_node
from functools import partial
import pickle

import torch
from torch.utils.data import IterableDataset

from megatron_datasets.utils import print_rank_0, print_datetime, IGNORE_INDEX
from megatron_datasets.utils import _get_ltor_masks_and_position_ids


def build_train_valid_test_datasets(tokenizer, args, mpu, train_data_path, eval_data_path):
    """Build train, valid, and test datasets."""
    return make_train_eval_dataset(tokenizer, args, mpu, train_data_path, eval_data_path)


@dataclass
class LlamaPretrainInterleavingIterableDataset(IterableDataset):
    """Dataset for LLaMA pretrain with multi source"""
    def __init__(
        self,
        args,
        tokenizer,
        files,
        micro_batch_size: int,
        data_parallel_size: int,
        data_parallel_rank: int,
        consumed_samples: int,
        probabilities: List[int],
        domain_names: List[str],
        seed: int,
        px_data_file_format: str,
        split: str = "train"
    ):

        self.args = args
        self.tokenizer = tokenizer
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files
        self.probabilities = probabilities
        self.domain_names = domain_names

        assert len(self.probabilities) == len(
            self.files
        ), f"num of probabilities {len(self.probabilities)} must be equal to files {len(self.files)}"

        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = data_parallel_size
        self.data_parallel_rank = data_parallel_rank
        self.seed = seed
        self.consumed_samples = consumed_samples
        self.consumed_samples_per_dp_rank = self.consumed_samples / data_parallel_size
        self.px_data_file_format = px_data_file_format
        self.split = split
        print_datetime(
            f"Init dataset {self.px_data_file_format} consumed_samples {self.consumed_samples} {self.consumed_samples_per_dp_rank} {len(self.files)}"
        )

    def __iter__(self):
        def add_domain_id(domain_id, example):
            example["domain_id"] = torch.tensor(domain_id, dtype=torch.int64)
            return example

        dataset_list = []
        for domain_id, file in enumerate(self.files):
            if self.px_data_file_format == "jsonl":
                ds = load_dataset(file, split=self.split, streaming=True)
                ds = ds.map(partial(add_domain_id, domain_id))
                dataset_list.append(ds)
            elif self.px_data_file_format == "pkl":
                sample_rate = None
                if self.args.px_retention_rates_per_domain is not None:
                    sample_rate = self.args.px_retention_rates_per_domain[domain_id]
                ds = load_dataset(file, split=self.split, streaming=True, sample_rate=sample_rate)
                ds = ds.map(partial(add_domain_id, domain_id))
                dataset_list.append(ds)
            else:
                raise NotImplementedError(
                    f"data version {self.px_data_file_format} is not implemented"
                )

        # 这样 log awk 根本无法看，最好是人看。
        domain_id_map_str = 'LlamaPretrainInterleavingIterableDataset id / domain mapping {\n'
        for domain_id, file in enumerate(self.files):
            domain_id_map_str += f'\tdomain_id {domain_id:3d}\tdomain_name {self.domain_names[domain_id].rjust(16)}\tp {self.probabilities[domain_id]:0.3f}\tdomain_file {file}\n'
        domain_id_map_str += '}'
        print_rank_0(domain_id_map_str)

        multi_stream_dataset = interleave_datasets(
            dataset_list,
            probabilities=self.probabilities,
            seed=self.seed,
            stopping_strategy="all_exhausted"
        )
        if self.args.px_shuffle_data:
            multi_stream_dataset = multi_stream_dataset.shuffle(
                buffer_size=self.args.px_shuffle_buffer_size, seed=self.seed
            )
        distributed_dataset = split_dataset_by_node(
            multi_stream_dataset, rank=self.data_parallel_rank, world_size=self.data_parallel_size
        )
        dataset_iter = iter(distributed_dataset)

        if self.consumed_samples_per_dp_rank > 0:
            count = self.consumed_samples_per_dp_rank
            print_datetime(f"Begin skip consumed samples {count}")
            while count > 0:
                if count % 100000 == 0:
                    print_datetime(f"Skip samples last {count}")
                next(dataset_iter)
                count -= 1
            self.consumed_samples = 0
            self.consumed_samples_per_dp_rank = 0
            print_datetime(f"Finished skipping samples")

        for raw_sample in dataset_iter:
            sample = None
            if self.px_data_file_format == "jsonl":
                sample = json.loads(raw_sample["data"])
                input_ids = sample["input_ids"]
            elif self.px_data_file_format == "pkl":
                abs_dir = raw_sample["abs_dir"]
                sample = json.loads(raw_sample["data"])
                input_ids_path = os.path.join(abs_dir, sample["pkl_file_name"])
                offset = sample["offset"]

                with open(input_ids_path, 'rb') as f:
                    f.seek(offset)
                    input_ids = pickle.load(f)
            else:
                raise NotImplementedError(
                    f"data version {self.px_data_file_format} is not implemented"
                )

            assert len(input_ids) == self.args.seq_length

            input_ids = [torch.tensor(input_ids, dtype=torch.int64)]
            labels = copy.deepcopy(input_ids)

            if self.args.px_pad_to_max_len:
                input_ids.append(torch.zeros([self.args.seq_length + 1], dtype=input_ids[0].dtype))
                labels.append(torch.zeros([self.args.seq_length + 1], dtype=labels[0].dtype))

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )

            if self.args.px_pad_to_max_len:
                input_ids = input_ids[0]
                labels = labels[0]

            # for get_batch_v2 ==> get_batch_on_this_tp_rank
            # input_ids = input_ids[:-1].contiguous()
            # labels = labels[1:].contiguous()

            # attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            #     labels,
            #     IGNORE_INDEX,
            #     self.args.reset_position_ids,
            #     self.args.reset_attention_mask,
            #     self.args.eod_mask_loss
            # )
            # yield dict(tokens=input_ids,
            #            labels=labels,
            #            attention_mask=attention_mask,
            #            loss_mask=loss_mask,
            #            position_ids=position_ids,
            #            domain_id=raw_sample['domain_id'],
            #            sample_id=sample["sample_id"]
            #            if self.px_data_file_format == "pkl" else sample["doc_id"])

            yield dict(
                input_ids=input_ids,
                labels=labels,
                domain_id=raw_sample['domain_id'],
                sample_id=sample["sample_id"]
                if self.px_data_file_format == "pkl" else sample["doc_id"]
            )


@dataclass
class LlamaPretrainEvalDataset(IterableDataset):
    """Dataset for LLaMA pretrain with multi source"""
    def __init__(
        self,
        args,
        tokenizer,
        files,
        micro_batch_size: int,
        data_parallel_size: int,
        data_parallel_rank: int,
        px_data_file_format: str,
        px_eval_samples_per_domain,
        split: str = "train"
    ):

        self.args = args
        self.tokenizer = tokenizer
        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = data_parallel_size
        self.data_parallel_rank = data_parallel_rank
        self.px_data_file_format = px_data_file_format
        self.split = split

        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files

        if isinstance(px_eval_samples_per_domain, list):
            self.px_eval_samples_per_domain = [
                int(eval_samples_num // self.args.global_batch_size) * self.args.global_batch_size
                for eval_samples_num in px_eval_samples_per_domain
            ]
        else:
            self.px_eval_samples_per_domain = [
                int(px_eval_samples_per_domain // self.args.global_batch_size) *
                self.args.global_batch_size
            ]
        assert len(self.px_eval_samples_per_domain) == len(self.files)

        print_datetime(f"Init dataset {self.px_data_file_format} {len(self.files)}")

    def __iter__(self):
        def add_domain_id(domain_id, example):
            example["domain_id"] = torch.tensor(domain_id, dtype=torch.int64)
            return example

        data_iter_list = []
        eval_sample_per_domain = []
        for domain_id, file in enumerate(self.files):
            ds = None
            if self.px_data_file_format in ["jsonl", "pkl"]:
                ds = load_dataset(file, split=self.split, streaming=True)
                ds = ds.map(
                    partial(add_domain_id, domain_id)
                )  # domain id 是给 doremi 的，doremi 没有 eval，所有 domain id 都无用，不做了。
            else:
                raise NotImplementedError(
                    f"data version {self.px_data_file_format} is not implemented"
                )

            distributed_dataset = split_dataset_by_node(
                ds, rank=self.data_parallel_rank, world_size=self.data_parallel_size
            )
            dataset_iter = iter(distributed_dataset)
            data_iter_list.append(dataset_iter)
            total_micro_batch_per_rank = self.px_eval_samples_per_domain[domain_id] // (
                self.data_parallel_size * self.args.micro_batch_size
            )
            eval_sample_per_domain.append(total_micro_batch_per_rank * self.args.micro_batch_size)

        for di, data_iter in enumerate(data_iter_list):
            try:
                cnt = 0
                for raw_sample in data_iter:
                    if cnt >= eval_sample_per_domain[di]:
                        break
                    cnt += 1
                    sample = None
                    if self.px_data_file_format == "jsonl":
                        sample = json.loads(raw_sample["data"])
                        input_ids = sample["input_ids"]
                    elif self.px_data_file_format == "pkl":
                        abs_dir = raw_sample["abs_dir"]
                        sample = json.loads(raw_sample["data"])
                        input_ids_path = os.path.join(abs_dir, sample["pkl_file_name"])
                        offset = sample["offset"]

                        with open(input_ids_path, 'rb') as f:
                            f.seek(offset)
                            input_ids = pickle.load(f)
                    else:
                        raise NotImplementedError(
                            f"data version {self.px_data_file_format} is not implemented"
                        )

                    # assert len(
                    #     input_ids
                    # ) == self.args.seq_length, f'expect {self.args.seq_length} actual {len(input_ids)}'

                    input_ids = [torch.tensor(input_ids, dtype=torch.int64)]
                    labels = copy.deepcopy(input_ids)

                    if self.args.px_pad_to_max_len:
                        input_ids.append(
                            torch.zeros([self.args.seq_length + 1], dtype=input_ids[0].dtype)
                        )
                        labels.append(
                            torch.zeros([self.args.seq_length + 1], dtype=labels[0].dtype)
                        )

                    input_ids = torch.nn.utils.rnn.pad_sequence(
                        input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
                    )
                    labels = torch.nn.utils.rnn.pad_sequence(
                        labels, batch_first=True, padding_value=IGNORE_INDEX
                    )

                    if self.args.px_pad_to_max_len:
                        input_ids = input_ids[0]
                        labels = labels[0]

                    # input_ids = input_ids[:-1].contiguous()
                    # labels = labels[1:].contiguous()

                    # attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                    #     labels,
                    #     IGNORE_INDEX,
                    #     self.args.reset_position_ids,
                    #     self.args.reset_attention_mask,
                    #     self.args.eod_mask_loss
                    # )
                    # yield dict(tokens=input_ids,
                    #            labels=labels,
                    #            attention_mask=attention_mask,
                    #            loss_mask=loss_mask,
                    #            position_ids=position_ids,
                    #            domain_id=raw_sample['domain_id'])
                    yield dict(
                        input_ids=input_ids, labels=labels, domain_id=raw_sample['domain_id']
                    )
            except StopIteration:
                pass


def make_train_eval_dataset(tokenizer, args, mpu, train_data_path, eval_data_path):
    train_dataset, eval_dataset = None, None

    print_rank_0(f"load samples for pretrain model")
    train_dataset = LlamaPretrainInterleavingIterableDataset(
        args,
        tokenizer,
        train_data_path,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=mpu.get_data_parallel_world_size(),
        data_parallel_rank=mpu.get_data_parallel_rank(),
        consumed_samples=args.consumed_train_samples,
        probabilities=args.px_domain_probabilities,
        domain_names=args.px_train_data_domain_names,
        seed=args.seed,
        px_data_file_format=args.px_data_file_format,
        split="train"
    )
    eval_dataset = None
    if eval_data_path is not None:
        if not args.px_do_eval_per_domain:
            eval_dataset = LlamaPretrainEvalDataset(
                args,
                tokenizer,
                eval_data_path,
                micro_batch_size=args.micro_batch_size,
                data_parallel_size=mpu.get_data_parallel_world_size(),
                data_parallel_rank=mpu.get_data_parallel_rank(),
                px_data_file_format=args.px_eval_data_file_format,
                px_eval_samples_per_domain=args.px_eval_samples_per_domain,
                split="train"
            )
        else:
            eval_datasets = []
            if isinstance(eval_data_path, str):
                eval_data_path = [eval_data_path]

            for fi, eval_path in enumerate(eval_data_path):
                eval_dataset = LlamaPretrainEvalDataset(
                    args,
                    tokenizer,
                    eval_path,
                    micro_batch_size=args.micro_batch_size,
                    data_parallel_size=mpu.get_data_parallel_world_size(),
                    data_parallel_rank=mpu.get_data_parallel_rank(),
                    px_data_file_format=args.px_eval_data_file_format,
                    px_eval_samples_per_domain=args.px_eval_samples_per_domain[fi],
                    split="train"
                )
                eval_datasets.append(eval_dataset)
            eval_dataset = eval_datasets

    return train_dataset, eval_dataset, None
