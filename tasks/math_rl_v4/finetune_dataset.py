import glob
import os
import re
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler


def tokenize_text(tokenizer, seq_length, prompt, full_text):
    assert tokenizer.pad_token is not None
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    real_seq_length = len(input_ids)

    return input_ids, labels, real_seq_length, prompt_len


class SimpleDataset(Dataset):
    def __init__(
        self,
        config: FinetuneConfig,
        tokenizer: AutoTokenizer,
        json_pattern="*.jsonl",
        split="train"
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.data_dir = config.data.data_pathes[0]
        self.train = True
        self.system_prompt = config.data.system_prompt
        self.seq_len = config.training.seq_length

        json_files = glob.glob(os.path.join(self.data_dir, json_pattern))
        self.dataset = load_dataset('json', data_files=json_files, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        if "problem" in example.keys():
            assert "problem" in example.keys()
            assert "solution" in example.keys()
            q_str = example['problem']
            a_str = example['solution']
        elif "question" in example.keys():
            assert "question" in example.keys()
            assert "target" in example.keys()
            q_str = example['question']
            a_str = example['target']
        else:
            raise ValueError(f"Unknown keys in example: {example.keys()}")

        prompt, full_text, messages = self._apply_chat_template(q_str, a_str)
        input_ids, labels, seq_length, prompt_len = tokenize_text(
            self.tokenizer,
            self.seq_len,
            prompt,
            full_text,
        )
        return {
            'tokens': input_ids,
            'labels': labels,
            'seq_length': seq_length,
            "prompt_len": prompt_len,
            'offpd_loss_alpha': example.get("offpd_loss_alpha", 0.0)
        }

    def get_features(self) -> list:
        return self.dataset.column_names

    def _apply_chat_template(self, question, answer):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.training.enable_thinking,
        )
        messages.append({"role": "assistant", "content": answer})

        if self.config.data.custom_add_eos:
            eos_token = self.config.data.tokenizer_eos_token
            assert eos_token is not None

            full_text = prompt_text + answer + eos_token
        else:
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            full_text = full_text.rstrip('\n')
        return prompt_text, full_text, messages


def collate_fn(examples):
    input_ids_list = []
    labels_list = []
    seq_length_list = []
    prompt_len_lst = []
    offpd_loss_alpha_list = []
    for example in examples:
        input_ids = example['tokens']
        labels = example['labels']
        seq_length = example['seq_length']
        prompt_len = example['prompt_len']
        offpd_loss_alpha = example.get('offpd_loss_alpha', 0.0)

        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        seq_length_list.append(torch.tensor(seq_length, dtype=torch.long))
        prompt_len_lst.append(torch.tensor(prompt_len, dtype=torch.long))
        offpd_loss_alpha_list.append(torch.tensor(offpd_loss_alpha, dtype=torch.float32))

    return {
        'tokens': input_ids_list,
        'sequence_lengths': seq_length_list,
        'prompt_lengths': prompt_len_lst,
        'offpd_loss_alpha': offpd_loss_alpha_list,
        'labels': labels_list,
    }


def get_dataset_and_dataloader(
    config: FinetuneConfig = None, tokenizer=None, dp_rank=0, dp_size=1, meta_info=None
):
    dataset = SimpleDataset(config=config, tokenizer=tokenizer)
    sampler = ResumableDistributedSampler(
        dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed,
        drop_last=True,
    )
    if meta_info is not None and "resume_step" in meta_info:
        resume_step = meta_info["resume_step"]
        gas = config.training.train_gbs // (dp_size * config.training.train_mbs)
        consumed_batches = resume_step * gas
        sampler.set_start_index(consumed_batches, config.training.train_mbs)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.train_mbs,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
        multiprocessing_context=config.data.multiprocessing_method,
    )
    return {
        'train_dataset': dataset,
        'train_sampler': sampler,
        'train_dataloader': dataloader,
    }


def get_batched_data(batched_data=None):
    assert batched_data is not None
    # 看起来不太有必要保留这个
    return batched_data
