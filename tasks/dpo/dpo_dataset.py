# coding=utf-8
#
# Copyright (c) 2024 Tencent Inc. All Rights Reserved.
# Author: nrwu@tencent.com, xiaotaoliu@tencent.com

from dataclasses import dataclass
import copy
import itertools
import json
import random

from torch.utils.data import IterableDataset as TorchIterableDataset
import torch
import transformers

from megatron_datasets.utils import print_rank_0
from megatron_datasets.mega_indexed_jsonl_dataset import MegaIndexedJsonlDataset
from megatron_datasets.mega_indexed_jsonl_dataset import get_epoch_and_line, update_epoch_and_line
from megatron_datasets.utils import random_pad_list


def tokenize_text(
    tokenizer,
    seq_len,
    prompt,
    answer,
    train_with_dynamic_len=False,
    assert_too_long=False,
    moe_pad_with_random_token=False
):
    assert tokenizer.pad_token is not None
    prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
    answer_tokenized = tokenizer(answer, add_special_tokens=False)

    prompt_len = len(prompt_tokenized.input_ids)
    answer_len = len(answer_tokenized.input_ids)
    total_len = prompt_len + answer_len

    input_ids = prompt_tokenized.input_ids + answer_tokenized.input_ids
    attention_mask = prompt_tokenized.attention_mask + answer_tokenized.attention_mask
    labels = [-100] * len(prompt_tokenized.input_ids) + answer_tokenized.input_ids
    unpadded_lens = len(input_ids)
    prompt_lens = len(prompt_tokenized.input_ids)

    # 检查序列是否过长，并区分是prompt超长还是整体超长
    if len(input_ids) > seq_len:
        if prompt_len > seq_len:
            error_message = f"Prompt长度 {prompt_len} 超过最大长度 {seq_len} \n prompt: {prompt}"
        else:
            error_message = f"序列总长度 {total_len} 超过最大长度 {seq_len}, 其中prompt长度: {prompt_len}, answer长度: {answer_len} \n prompt: {prompt} \n answer: {answer}"

        if assert_too_long:
            assert False, error_message
        else:
            return None, None, None, None, None

    if not train_with_dynamic_len:
        if len(input_ids) < seq_len + 1:
            if moe_pad_with_random_token:
                len_to_pad = seq_len + 1 - len(input_ids)
                input_ids = random_pad_list(input_ids, len_to_pad)
            else:
                input_ids += [tokenizer.pad_token_id] * (seq_len + 1 - len(input_ids))
            attention_mask += [0] * (seq_len + 1 - len(attention_mask))
            labels += [-100] * (seq_len + 1 - len(labels))
    else:
        # 因为在下面提前进行错位了，为避免错位后丢失有效token，提前补一个pad_token
        if len(input_ids) < seq_len + 1:
            input_ids += [tokenizer.pad_token_id]
            attention_mask += [0]
            labels += [-100]

    input_ids = input_ids[:-1]
    attention_mask = attention_mask[:-1]
    labels = labels[1:]

    if len(input_ids) > seq_len:
        input_ids = input_ids[-seq_len:]
        attention_mask = attention_mask[-seq_len:]
        labels = labels[-seq_len:]
        unpadded_lens = seq_len
    assert unpadded_lens > 0
    if not train_with_dynamic_len:
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        unpadded_lens = torch.tensor(unpadded_lens, dtype=torch.int64)
        prompt_lens = torch.tensor(prompt_lens, dtype=torch.int64)

    return input_ids, attention_mask, labels, unpadded_lens, prompt_lens


@dataclass
class DpoDataset(TorchIterableDataset):
    def __init__(
        self,
        tokenizer,
        seq_len,
        max_position_embeddings,
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
        seed=0,
        reward_models_cnts=0,
        margin_keys=[],
        prompt_format=None,
        eos_token=None,
        only_using_policy=False,
        retention_rates_per_domains=[],
        unsplit_eval_data=False,
        train_with_dynamic_len=False,
        assert_too_long=False,
        moe_pad_with_random_token=False,
    ):
        self.seq_len = seq_len
        self.max_position_embeddings = max_position_embeddings
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
        self.in_iter = False
        self.reward_models_cnts = reward_models_cnts  # no use now
        self.margin_keys = margin_keys
        self.retention_rates_per_domains = retention_rates_per_domains
        self.unsplit_eval_data = unsplit_eval_data
        self.train_with_dynamic_len = train_with_dynamic_len
        self.assert_too_long = assert_too_long
        self.moe_pad_with_random_token = moe_pad_with_random_token
        if self.unsplit_eval_data:
            assert self.train is False, f"only support unsplit eval data"
        if eos_token is None:
            self.eos_token = self.tokenizer._tokenizer.eos_token
        else:
            self.eos_token = eos_token
        if prompt_format is None:
            self.prompt_format = "###{instruction}\n### Response:\n"
        else:
            self.prompt_format = prompt_format
        self.only_using_policy = only_using_policy

        if self.train:
            assert self.eval_samples is None
        else:
            assert train_data_consuming_progresses is None
            assert self.eval_samples is not None and self.eval_samples > 0
            assert self.shuffle_buffer_size == 0
        print(
            f"DpoDataset init. rank {torch.distributed.get_rank()} eval samples {self.eval_samples}"
            f" moe_pad_with_random_token {self.moe_pad_with_random_token}"
        )

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
            seed=self.seed,
            train=self.train,
            retention_rates_per_domains=self.retention_rates_per_domains,
            unsplit_eval_data=self.unsplit_eval_data
        )

    def _get_margin(self, example):
        if len(self.margin_keys) == 0:
            return 0, 0, 0
        margin_chosen = []
        margin_reject = []
        w0 = [int(1000 * example["dpo-weight"]["w0"])]
        for k in self.margin_keys:
            # convert to int, beta = 0.1
            margin_chosen.append(
                int(example["dpo-weight"][k] * example['modpo-chosen-margin'][k] * 1000.0 * 0.1)
            )
            margin_reject.append(
                int(example["dpo-weight"][k] * example['modpo-rejected-margin'][k] * 1000.0 * 0.1)
            )

        return w0, margin_chosen, margin_reject

    def _get_ref_logps(self, example):
        if not self.only_using_policy:
            return 0, 0
        return [int(1000 * example["ref-model-logps"]["chosen"])], [
            int(1000 * example["ref-model-logps"]["rejected"])
        ]

    def iter_in_epoch(self, epoch):
        if torch.distributed.get_rank() == 0:
            print(f'DpoDataset.iter_in_epoch epoch {epoch} train {self.train}')

        rng = random.Random(self.seed + epoch)

        for example in self.underlying:
            w0, margin_chosen, margin_reject = self._get_margin(example)
            ref_logps_chosen, ref_logps_reject = self._get_ref_logps(example)
            real_input = example['input'] if example.get('input') else example['instruction']

            if isinstance(self.prompt_format, list):
                if rng.random() < 0.6:
                    prompt = self.prompt_format[0].format_map(
                        {
                            "input": real_input,
                            "instruction": real_input
                        }
                    )  # systerm pt
                else:
                    prompt = self.prompt_format[1].format_map(
                        {
                            "input": real_input,
                            "instruction": real_input
                        }
                    )
            else:

                prompt = self.prompt_format.format_map(
                    {
                        "input": real_input,
                        "instruction": real_input
                    }
                )
            chosen = f"{example['chosen']}{self.eos_token}"

            input_ids, attention_mask, labels, unpadded_lens, prompt_lens = tokenize_text(
                self.tokenizer._tokenizer,
                self.seq_len,
                prompt,
                chosen,
                self.train_with_dynamic_len,
                self.assert_too_long,
                moe_pad_with_random_token=self.moe_pad_with_random_token
            )
            if input_ids is None:  # 序列过长，跳过这条数据
                continue
            chosen_data = {
                'input_ids':
                    input_ids,
                'labels':
                    labels,
                'attention_mask':
                    attention_mask,
                'unpadded_lens':
                    unpadded_lens,
                'prompt_lens':
                    prompt_lens,
                'is_chosen':
                    True,
                'is_rejected':
                    False,
                'train':
                    self.train,
                'epoch':
                    epoch,
                'line':
                    0,
                'margin':
                    0 if margin_chosen == 0 else torch.tensor(margin_chosen, dtype=torch.int64),
                'w0':
                    0 if w0 == 0 else torch.tensor(w0, dtype=torch.int64),
                'ref_logps':
                    0
                    if ref_logps_chosen == 0 else torch.tensor(ref_logps_chosen, dtype=torch.int64),
            }

            rejected = f"{example['rejected']}{self.eos_token}"
            input_ids, attention_mask, labels, unpadded_lens, prompt_lens = tokenize_text(
                self.tokenizer._tokenizer,
                self.seq_len,
                prompt,
                rejected,
                self.train_with_dynamic_len,
                self.assert_too_long,
                moe_pad_with_random_token=self.moe_pad_with_random_token
            )
            if input_ids is None:  # 序列过长，跳过这条数据
                continue
            rejected_data = {
                'input_ids':
                    input_ids,
                'labels':
                    labels,
                'attention_mask':
                    attention_mask,
                'unpadded_lens':
                    unpadded_lens,
                'prompt_lens':
                    prompt_lens,
                'is_chosen':
                    False,
                'is_rejected':
                    True,
                'train':
                    self.train,
                'epoch':
                    epoch,
                'line':
                    1,
                'margin':
                    0 if margin_reject == 0 else torch.tensor(margin_reject, dtype=torch.int64),
                'w0':
                    0 if w0 == 0 else torch.tensor(w0, dtype=torch.int64),
                'ref_logps':
                    0
                    if ref_logps_reject == 0 else torch.tensor(ref_logps_reject, dtype=torch.int64),
            }

            yield chosen_data
            yield rejected_data

            if not self.train:
                self.eval_yielded += 1
                if self.eval_yielded >= self.eval_samples:  # 保证 each eval epoch 都看到相同的数据
                    self.eval_yielded = 0
                    return

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True
        self.eval_yielded = 0
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
                seed=self.seed,
                train=self.train,
                retention_rates_per_domains=self.retention_rates_per_domains,
                unsplit_eval_data=self.unsplit_eval_data
            )
        assert False, 'never reachable'


@dataclass
class DpoDataCollator(object):
    tokenizer: transformers.AutoTokenizer
    seq_len: int
    train_with_dynamic_len: int
    pad_to_multiple_of: int
    moe_pad_with_random_token: bool = False

    def __call__(self, batch):
        pad_token_id = self.tokenizer._tokenizer.pad_token_id

        input_ids_lens = [len(item['input_ids']) for item in batch]
        batch_max_len = max(input_ids_lens)
        if self.train_with_dynamic_len:
            pad_to_multiple_of = self.pad_to_multiple_of
            batch_max_len = (
                (batch_max_len + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of
            batch_max_len = min(batch_max_len, self.seq_len)

            for item in batch:
                input_ids = item['input_ids']
                assert isinstance(input_ids, list)
                to_pad = batch_max_len - len(input_ids)
                assert to_pad >= 0
                if self.moe_pad_with_random_token:
                    item['input_ids'] = random_pad_list(item['input_ids'], to_pad)
                else:
                    item['input_ids'] += [pad_token_id] * to_pad
                item['labels'] += [-100] * to_pad
                item['attention_mask'] += [0] * to_pad
                assert len(input_ids) <= self.seq_len

        input_ids = torch.as_tensor([item['input_ids'] for item in batch], dtype=torch.int64)
        labels = torch.as_tensor([item['labels'] for item in batch], dtype=torch.int64)
        attention_mask = torch.as_tensor(
            [item['attention_mask'] for item in batch], dtype=torch.int64
        )
        unpadded_lens = torch.as_tensor(
            [item['unpadded_lens'] for item in batch], dtype=torch.int64
        )
        prompt_lens = torch.as_tensor([item['prompt_lens'] for item in batch], dtype=torch.int64)
        is_chosen = torch.as_tensor([item['is_chosen'] for item in batch], dtype=torch.bool)
        is_rejected = torch.as_tensor([item['is_rejected'] for item in batch], dtype=torch.bool)
        train = torch.as_tensor([item['train'] for item in batch], dtype=torch.bool)
        epoch = torch.as_tensor([item['epoch'] for item in batch], dtype=torch.int64)
        line = torch.as_tensor([item['line'] for item in batch], dtype=torch.int64)
        margin = torch.as_tensor([item['margin'] for item in batch], dtype=torch.int64)
        w0 = torch.as_tensor([item['w0'] for item in batch], dtype=torch.int64)
        ref_logps = torch.as_tensor([item['ref_logps'] for item in batch], dtype=torch.int64)

        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            unpadded_lens=unpadded_lens,
            prompt_lens=prompt_lens,
            is_chosen=is_chosen,
            is_rejected=is_rejected,
            train=train,
            epoch=epoch,
            line=line,
            margin=margin,
            w0=w0,
            ref_logps=ref_logps,
        )
        return ret


def build_train_valid_test_datasets(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    prompt_format=None,
    eos_token=None,
    only_using_policy=False,
):
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    domain_names = args.px_train_data_domain_names
    retention_rates_per_domains = args.px_retention_rates_per_domain
    train_with_dynamic_len = args.px_inputs_pad_to_longest
    moe_pad_with_random_token = args.moe_pad_with_random_token
    assert args.num_workers <= 1
    assert all([dr == 1.0 for dr in args.px_retention_rates_per_domain])
    # assert len(domain_names) == 1  # Dpo 没有多个 domain 的说法
    # assert len(domain_probabilities) == 1

    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {args.train_data_consuming_progresses}'
    )
    train_ds = DpoDataset(
        tokenizer,
        args.seq_length,
        args.max_position_embeddings,
        train_path_likes,
        domain_probabilities,
        domain_names,
        train_data_consuming_progresses=args.train_data_consuming_progresses,
        train=True,
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        shuffle_buffer_size=args.px_shuffle_buffer_size,
        seed=args.seed,
        reward_models_cnts=args.dpo_reward_models_cnt,
        margin_keys=args.dpo_margin_keys,
        prompt_format=prompt_format,
        eos_token=eos_token,
        only_using_policy=only_using_policy,
        retention_rates_per_domains=retention_rates_per_domains,
        train_with_dynamic_len=train_with_dynamic_len,
        assert_too_long=args.assert_too_long,
        moe_pad_with_random_token=moe_pad_with_random_token,
    )
    eval_ds = None
    if eval_path_likes is not None:
        eval_samples = args.eval_iters * args.global_batch_size // dp_size
        eval_ds = DpoDataset(
            tokenizer,
            args.seq_length,
            args.max_position_embeddings,
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
            seed=0,
            reward_models_cnts=args.dpo_reward_models_cnt,
            margin_keys=args.dpo_margin_keys,
            prompt_format=prompt_format,
            eos_token=eos_token,
            only_using_policy=only_using_policy,
            assert_too_long=args.assert_too_long,
            moe_pad_with_random_token=moe_pad_with_random_token,
        )
    test_ds = None
    return train_ds, eval_ds, test_ds
