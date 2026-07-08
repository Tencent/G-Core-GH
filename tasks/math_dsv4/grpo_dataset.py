import glob
import os
import re
from typing import Any, Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler
from gpatch_v4.models.deepseek_v4.chat_template import DSV4_CHAT_TEMPLATE


def tokenize_text(tokenizer, prompt_seq_len, prompt):
    assert tokenizer.pad_token is not None
    prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
    input_ids = prompt_tokenized.input_ids
    prompt_len = len(input_ids)

    if len(input_ids) > prompt_seq_len:
        input_ids = input_ids[-prompt_seq_len:]
        prompt_len = prompt_seq_len
    assert prompt_len > 0 and prompt_len == len(input_ids)
    return input_ids, prompt_len


def extract_gt_answer(text):
    match = re.search(r'####\s*(-?\d+(\.\d+)?)', text)
    if match:
        return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    else:
        return None


class SimpleDataset(Dataset):
    def __init__(
        self, config: RlConfig,
        tokenizer: AutoTokenizer,
        json_pattern="*.jsonl",
        split="train"
    ):
        self.config = config
        self.tokenizer = tokenizer
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = DSV4_CHAT_TEMPLATE
        self.train = True
        self.system_prompt = config.data.system_prompt
        self.seq_len = config.training.seq_length
        resp_seq_len = -1
        for infer_engine in config.sampler.infer_engine_configs:
            resp_seq_len = max(resp_seq_len, infer_engine.generate_max_tokens)
        self.resp_seq_len = resp_seq_len

        if split == "train":
            json_files = {
                "train": glob.glob(os.path.join(config.data.data_pathes[0], json_pattern))
            }
        elif split == "test":
            json_files = {
                "test": glob.glob(os.path.join(config.data.eval_data_pathes[0], json_pattern))
            }

        self.dataset = load_dataset('json', data_files=json_files, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        assert "question" in example.keys()
        assert "answer" in example.keys()
        q_str = example['question']
        a_str = example['answer']
        answer = extract_gt_answer(a_str)
        if answer is None:
            answer = 0.0
        prompt, messages = self._apply_chat_template(q_str)
        input_ids, prompt_len = tokenize_text(
            self.tokenizer,
            self.seq_len - self.resp_seq_len,
            prompt,
        )

        return {
            'input_ids': input_ids,
            'prompt_len': prompt_len,
            'gt_label': answer,
        }

    def get_features(self) -> list:
        return self.dataset.column_names

    def _apply_chat_template(self, example):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": example})
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.training.enable_thinking,
        )
        return chat_text, messages


def collate_fn(examples):
    prompt_token_ids = []
    prompt_lens = []
    gt_label = []
    for example in examples:
        prompt_len = example['prompt_len']
        input_ids = example['input_ids'][:prompt_len]

        prompt_token_ids.append({'prompt_token_ids': input_ids})
        prompt_lens.append(torch.tensor(prompt_len, dtype=torch.long))
        gt_label.append(torch.tensor(example['gt_label'], dtype=torch.float32))

    return {
        "prompt_token_ids": prompt_token_ids,
        "prompt_lens": prompt_lens,
        "gt_label": gt_label,
    }


def get_dataset_and_dataloader(
    config: RlConfig = None, tokenizer=None, dp_rank=0, dp_size=1, meta_info=None
):
    train_dataset = SimpleDataset(config=config, tokenizer=tokenizer)

    train_sampler = ResumableDistributedSampler(
        train_dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed,
        drop_last=True,
    )
    if meta_info is not None and "resume_step" in meta_info:
        resume_step = meta_info["resume_step"]
        rollout_gas = config.training.rollout_gbs // (dp_size * config.training.rollout_mbs)
        consumed_batches = resume_step * rollout_gas
        train_sampler.set_start_index(consumed_batches, config.training.rollout_mbs)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.rollout_mbs,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
        multiprocessing_context=config.data.multiprocessing_method,
    )

    if config.training.eval_interval > 0:
        eval_dataset = SimpleDataset(config=config, tokenizer=tokenizer, split="test")

        eval_sampler = DistributedSampler(
            eval_dataset,
            rank=dp_rank,
            num_replicas=dp_size,
            shuffle=False,
            seed=config.data.sampler_seed
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            collate_fn=collate_fn,
            pin_memory=config.data.dataloader_pin_memory,
            batch_size=config.training.rollout_mbs,
            num_workers=config.data.dataloader_num_workers,
            drop_last=True,
            multiprocessing_context=config.data.multiprocessing_method,
        )
    else:
        eval_dataset = None
        eval_dataloader = None

    return {
        'train_dataset': train_dataset,
        'train_sampler': train_sampler,
        'train_dataloader': train_dataloader,
        'eval_dataset': eval_dataset,
        'eval_dataloader': eval_dataloader,
    }


def get_batched_data(batched_data=None):
    assert batched_data is not None
    # 看起来不太有必要保留这个
    return batched_data
