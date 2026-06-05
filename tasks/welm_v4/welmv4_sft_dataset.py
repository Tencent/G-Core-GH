# coding=utf-8
#
# Copyright (c) 2026 Tencent Inc. All Rights Reserved.
# Author: xiaotaoliu@tencent.com, nrwu@tencent.com

from dataclasses import dataclass
import copy
import random
import itertools

from torch.utils.data import IterableDataset as TorchIterableDataset
import datasets
import torch
import transformers

try:
    from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer as TokenizerType
except ImportError:
    from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer as TokenizerType

from gpatch.core.device_type import is_wxacc1
from megatron_datasets.utils import print_rank_0, print_datetime
from megatron_datasets.mega_indexed_jsonl_dataset import MegaIndexedJsonlDataset
from megatron_datasets.mega_indexed_jsonl_dataset import get_epoch_and_line, update_epoch_and_line
from megatron_datasets.utils import random_pad_list


def tokenize_text(
    tokenizer,
    max_len,
    prompt,
    target,
    train_with_dynamic_len=False,
    moe_pad_with_random_token=False
):
    text = prompt + target
    assert tokenizer._tokenizer.pad_token is not None

    prompt_input_ids = tokenizer._tokenizer(prompt, add_special_tokens=False).input_ids
    text_tokenized = tokenizer._tokenizer(f"{text}", add_special_tokens=False)

    text_input_ids = text_tokenized.input_ids
    text_attention_mask = text_tokenized.attention_mask
    labels = copy.deepcopy(text_input_ids)
    if not len(prompt_input_ids) < len(labels):
        print(f"WARN empty target prompt {prompt} target {target}")
        return None, None, None

    assert labels[:len(prompt_input_ids)] == prompt_input_ids, \
        f"labels {labels[:len(prompt_input_ids)]} prompt_input_ids {prompt_input_ids}" \
        f" prompt {prompt[-10:]=} target {target[:20]=}"

    labels[:len(prompt_input_ids)] = [-100] * len(prompt_input_ids)
    if not train_with_dynamic_len:
        if len(text_input_ids) < max_len + 1:
            if moe_pad_with_random_token:
                len_to_pad = max_len + 1 - len(text_input_ids)
                text_input_ids = random_pad_list(text_input_ids, len_to_pad)
            else:
                text_input_ids += [tokenizer._tokenizer.pad_token_id
                                  ] * (max_len + 1 - len(text_input_ids))
            text_attention_mask += [0] * (max_len + 1 - len(text_attention_mask))

        if len(labels) < max_len + 1:
            labels += [-100] * (max_len + 1 - len(labels))
    else:
        # 因为在下面提前进行错位了，为避免错位后丢失有效token，提前补一个pad_token
        if len(text_input_ids) < max_len + 1:
            text_input_ids += [tokenizer._tokenizer.pad_token_id]
            text_attention_mask += [0]
        if len(labels) < max_len + 1:
            labels += [-100]

    text_input_ids = text_input_ids[:-1]
    text_attention_mask = text_attention_mask[:-1]
    labels = labels[1:]

    if len(text_input_ids) > max_len:
        text_input_ids = text_input_ids[-max_len:]
        text_attention_mask = text_attention_mask[-max_len:]
        labels = labels[-max_len:]

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
        train_with_dynamic_len=False,
        moe_pad_with_random_token=False,
        smart_padding_buffer_size=1000,
        global_batch_size=None,
        apply_chat_template=False,
        system_prompt=None,
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
        self.train_with_dynamic_len = train_with_dynamic_len
        self.smart_padding_buffer_size = smart_padding_buffer_size
        self.global_batch_size = global_batch_size
        self.random_for_unordered = random.Random(seed)
        self.unordered_examples_count = 0
        self.moe_pad_with_random_token = moe_pad_with_random_token

        print_rank_0(
            f"SftDataset build with moe_pad_with_random_token {self.moe_pad_with_random_token} "
            f"unorded_buffer_size {self.smart_padding_buffer_size} global_batch_size {self.global_batch_size}"
        )

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
            self.eos_token = self.tokenizer._tokenizer.eos_token
        else:
            self.eos_token = eos_token

        self.apply_chat_template = apply_chat_template
        self.system_prompt = system_prompt
        if self.apply_chat_template:
            self.prompt_format = None
        elif prompt_format is None:
            assert system_prompt is None
            self.prompt_format = "###{input}\n### Response:\n"
        else:
            assert system_prompt is None
            self.prompt_format = prompt_format

    def handle_unordered_examples(self, unordered_examples: list, epoch):
        def split_batch(it: list, batch_size):
            it = iter(it)
            batch_list = []
            for batch in iter(lambda: list(itertools.islice(it, batch_size)), []):
                batch_list.append(batch)
            return batch_list

        if len(unordered_examples) > 0:
            # 根据 input ids 做排序
            ordered_examples = sorted(unordered_examples, key=lambda x: len(x[0]))
            ordered_batchs = split_batch(ordered_examples, self.global_batch_size // self.dp_size)
            self.random_for_unordered.seed(self.unordered_examples_count)
            self.random_for_unordered.shuffle(ordered_batchs)
            self.unordered_examples_count += len(unordered_examples)
            for batch in ordered_batchs:
                # batch_lens_list = [len(e[0]) for e in batch]
                # print(f'rank {torch.distributed.get_rank()} unordered batch {len(batch)} max_len {max(batch_lens_list)} {batch_lens_list=}')
                for example in batch:
                    input_ids, labels, attn_mask = example
                    assert len(input_ids) == len(labels), f"{len(input_ids)} != {len(labels)}"
                    assert len(input_ids) == len(attn_mask), f"{len(input_ids)} != {len(attn_mask)}"

                    data = {
                        'input_ids': input_ids,
                        'labels': labels,
                        'attention_mask': attn_mask,
                        'train': self.train,
                        'epoch': epoch,
                        'line': 1,
                    }
                    yield data
            unordered_examples.clear()

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
        return chat_text

    def __iter__(self):
        unordered_examples = []
        self.eval_yielded = 0
        for epoch in itertools.count(start=self.start_epoch):
            print(f'SftDataset.__iter__ rank {torch.distributed.get_rank()} epoch {epoch}')
            for example in self.underlying:
                if self.apply_chat_template:
                    if "problem" in example.keys():
                        prompt = self._apply_chat_template(example["problem"])
                        target = f"{example['solution']}{self.eos_token}".lstrip().lstrip('\n')
                    else:
                        prompt = self._apply_chat_template(example["question"])
                        target = f"{example['target']}{self.eos_token}".lstrip().lstrip('\n')
                else:
                    if "problem" in example.keys():
                        prompt = self.prompt_format.format("{}", problem=example["problem"])

                        target = f"{example['solution']}{self.eos_token}".lstrip().lstrip('\n')
                    else:
                        prompt = self.prompt_format.format_map(example)
                        target = f"{example['target']}{self.eos_token}".lstrip().lstrip('\n')

                input_ids, labels, attention_mask = tokenize_text(
                    self.tokenizer,
                    self.max_seq_len,
                    prompt,
                    target,
                    train_with_dynamic_len=self.train_with_dynamic_len,
                    moe_pad_with_random_token=self.moe_pad_with_random_token
                )

                if input_ids is None:
                    continue

                if self.train_with_dynamic_len:
                    assert self.smart_padding_buffer_size > 0, "smart_padding_buffer_size must be > 0"
                    unordered_examples.append((input_ids, labels, attention_mask))
                    if len(unordered_examples) >= self.smart_padding_buffer_size:
                        yield from self.handle_unordered_examples(unordered_examples, epoch)
                else:
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


@dataclass
class SftDataCollator(object):
    tokenizer: TokenizerType
    seq_len: int
    train_with_dynamic_len: int
    pad_to_multiple_of: int
    moe_pad_with_random_token: bool = False
    smart_pad_by_truncated: bool = False

    def __call__(self, batch):
        pad_token_id = self.tokenizer._tokenizer.pad_token_id

        input_ids_lens = [len(item['input_ids']) for item in batch]
        batch_max_len = max(input_ids_lens)
        if self.smart_pad_by_truncated:
            # 如果 enable 这个，那么先 用 random token/pad token pad 满，后面 padding iterator 做截断
            batch_max_len = self.seq_len

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
        train = torch.as_tensor([item['train'] for item in batch], dtype=torch.bool)
        epoch = torch.as_tensor([item['epoch'] for item in batch], dtype=torch.int64)
        line = torch.as_tensor([item['line'] for item in batch], dtype=torch.int64)
        seq_lengths = torch.tensor(input_ids_lens, dtype=torch.int64)

        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            train=train,
            epoch=epoch,
            line=line,
            seq_length=seq_lengths,
        )
        return ret


def build_train_valid_test_datasets(
    args, tokenizer, rank=0, dp_rank=0, dp_size=1, prompt_format=None, eos_token=None
):
    max_seq_len = args.seq_length
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    domain_names = args.px_train_data_domain_names
    shuffle_buffer_size = args.px_shuffle_buffer_size
    seed = args.seed
    train_with_dynamic_len = args.px_inputs_pad_to_longest
    moe_pad_with_random_token = args.moe_pad_with_random_token
    smart_padding_buffer_size = args.px_smart_padding_buffer_size
    global_batch_size = args.global_batch_size
    apply_chat_template = args.px_apply_chat_template
    system_prompt = args.px_system_prompt
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
        train_with_dynamic_len=train_with_dynamic_len,
        moe_pad_with_random_token=moe_pad_with_random_token,
        smart_padding_buffer_size=smart_padding_buffer_size,
        global_batch_size=global_batch_size,
        apply_chat_template=apply_chat_template,
        system_prompt=system_prompt,
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
            moe_pad_with_random_token=moe_pad_with_random_token,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
        )
    test_ds = None
    return train_ds, eval_ds, test_ds
