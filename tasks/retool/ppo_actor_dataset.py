from dataclasses import dataclass
import itertools
import json
import random
import re
from typing_extensions import Optional, List, Dict

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from megatron.training import get_args
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron_datasets.utils import print_rank_0
from gpatch.training.v3.ppo_actor import iter_to_ppo_epoch_step

from gdataset import GDatasetV4


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
        domain_probabilities,
        domain_names,
        train=False,
        metadata_file=None,
        dp_rank=0,
        dp_size=1,
        shuffling_buffer_size=1000,
        seed=0,
        eos_token=None,
        prompt_format=None,
        apply_chat_template=False,
        system_prompt=None,
        disable_shuffle_data_files=False,
        answer_format=None,
        tools=None,
    ):
        #TODO: 将 apply_chat_template 设置为默认行为，当前不影响以往业务使用，慢慢推
        self.seq_len = seq_len
        self.max_position_embeddings = max_position_embeddings
        self.resp_seq_len = resp_seq_len
        self.rollout_micro_batch_size = rollout_micro_batch_size
        self.rollout_global_batch_size = rollout_global_batch_size
        self.tokenizer = tokenizer
        self.train = train
        # TODO(parkeychen): interleave datasets
        self.domain_probabilities = domain_probabilities
        self.domain_names = domain_names
        self.metadata_file = metadata_file
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.shuffling_buffer_size = shuffling_buffer_size
        self.seed = seed
        self.in_iter = False
        self.disable_shuffle_data_files = disable_shuffle_data_files

        if eos_token is None:
            self.eos_token = self.tokenizer._tokenizer.eos_token
        else:
            self.eos_token = eos_token
        self.apply_chat_template = apply_chat_template
        self.system_prompt = system_prompt
        self.answer_format = answer_format
        self.tools = tools
        if self.apply_chat_template:
            self.prompt_format = None
        elif prompt_format is None:
            assert system_prompt is None
            self.prompt_format = "###{instruction}\n### Response:\n"
        else:
            assert system_prompt is None
            self.prompt_format = prompt_format

        if not self.train:
            assert self.shuffling_buffer_size == 0
        args = get_args()
        self.start_epoch, ppo_step = iter_to_ppo_epoch_step(args.iteration)
        consumed = ppo_step * rollout_global_batch_size
        self.make_underlying(consumed=consumed, epoch=self.start_epoch)

    def make_underlying(self, consumed, epoch):
        self.underlying = GDatasetV4(
            metadata_file=self.metadata_file,
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
            gbs=self.rollout_global_batch_size,
            shuffling_buffer_size=self.shuffling_buffer_size,
            seed=self.seed,
            consumed=consumed,
            disable_shuffle_data_files=self.disable_shuffle_data_files,
        )
        self.underlying.set_epoch(epoch)

    def _apply_chat_template(self, example):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        # Append answer_format to user message if provided (VERL-style)
        user_content = example
        if self.answer_format is not None:
            user_content = example + self.answer_format
        messages.append({"role": "user", "content": user_content})
        # Pass tools parameter if provided (for Qwen tool-use template)
        chat_text = self.tokenizer._tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
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
                    elif isinstance(example["question"], list) and len(example["question"]) > 0:
                        if isinstance(example["question"][0],
                                      dict) and "content" in example["question"][0]:
                            example_prompt = example["question"][0]["content"]
                        else:
                            example_prompt = str(example["question"])
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

            if 'target' in example.keys():
                # gt_label = extract_gt_answer(example['answer'])
                gt_label = example['target']
                if gt_label is None:
                    gt_label = ''
            else:
                gt_label = ''

            o = {
                'input_ids': input_ids,
                'unpadded_lens': unpadded_lens,
                'train': self.train,
                'gt_label': gt_label,
                'messages': messages,
            }

            yield o

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True
        for epoch in itertools.count(start=self.start_epoch):
            yield from self.iter_in_epoch(epoch)
            self.make_underlying(consumed=0, epoch=epoch + 1)
        assert False, 'never reachable'


def build_train_valid_test_datasets(
    args, tokenizer, dp_rank=0, dp_size=1, prompt_format=None, eos_token=None
):
    train_metadata_file = args.gdatasetv4_train_metadata_file
    eval_metadata_file = args.gdatasetv4_eval_metadata_file
    domain_probabilities = args.px_domain_probabilities
    domain_names = args.px_train_data_domain_names
    apply_chat_template = args.px_apply_chat_template
    system_prompt = args.px_system_prompt
    # ReTool/VERL-style: answer_format and tools for Qwen tool-use template
    answer_format = getattr(args, 'px_answer_format', None)
    tools_json = getattr(args, 'px_tools_json', None)
    tools = None
    if tools_json is not None:
        try:
            tools = json.loads(tools_json)
        except json.JSONDecodeError as e:
            print_rank_0(f"Warning: Failed to parse px_tools_json: {e}")

    assert args.num_workers <= 1
    assert all([dr == 1.0 for dr in args.px_retention_rates_per_domain])
    assert len(domain_names) == 1  # PpoActor 没有多个 domain 的说法
    assert len(domain_probabilities) == 1

    train_ds = PpoActorDataset(
        tokenizer=tokenizer,
        seq_len=args.seq_length,
        max_position_embeddings=args.max_position_embeddings,
        resp_seq_len=args.ppo_resp_seq_len,
        rollout_micro_batch_size=args.ppo_rollout_micro_batch_size,
        rollout_global_batch_size=args.ppo_rollout_global_batch_size,
        domain_probabilities=domain_probabilities,
        domain_names=domain_names,
        train=True,
        metadata_file=train_metadata_file,
        dp_rank=dp_rank,
        dp_size=dp_size,
        shuffling_buffer_size=args.px_shuffle_buffer_size,
        seed=args.seed,
        eos_token=eos_token,
        prompt_format=prompt_format,
        apply_chat_template=apply_chat_template,
        system_prompt=system_prompt,
        disable_shuffle_data_files=args.disable_shuffle_data_files,
        answer_format=answer_format,
        tools=tools,
    )
    eval_ds = None
    if eval_metadata_file is not None:
        eval_ds = PpoActorDataset(
            tokenizer=tokenizer,
            seq_len=args.seq_length,
            max_position_embeddings=args.max_position_embeddings,
            resp_seq_len=args.ppo_resp_seq_len,
            rollout_micro_batch_size=args.ppo_eval_rollout_micro_batch_size,
            rollout_global_batch_size=args.ppo_eval_rollout_global_batch_size,
            domain_probabilities=None,
            domain_names=domain_names,
            train=False,
            metadata_file=eval_metadata_file,
            dp_rank=dp_rank,
            dp_size=dp_size,
            shuffling_buffer_size=0,
            seed=0,
            eos_token=eos_token,
            prompt_format=prompt_format,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
            disable_shuffle_data_files=args.disable_shuffle_data_files,
            answer_format=answer_format,
            tools=tools,
        )
    test_ds = None
    return train_ds, eval_ds, test_ds


@dataclass
class DataCollator(object):
    tokenizer: MegatronTokenizer
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
        num_lpads = torch.as_tensor(num_lpads, dtype=torch.int64)
        lpad_lens = torch.as_tensor(lpad_lens, dtype=torch.int64)
        # gsm 数据集 gt 都是 int, 看是否要将 gt 设置成 float, 如果设置 float 的话， get_batch
        # 里 broadcast 的时候则需要用 torch.float 做广播
        # gt_label = torch.as_tensor([item['gt_label'] for item in batch], dtype=torch.int64)
        gt_label = [item['gt_label'] for item in batch]
        messages: List[Optional[List[Dict[str, str]]]] = [item['messages'] for item in batch]
        ret = dict(
            input_ids=input_ids,
            unpadded_lens=unpadded_lens,
            num_lpads=num_lpads,
            lpad_lens=lpad_lens,
            train=train,
            gt_label=gt_label,
            messages=messages,
        )

        return ret
