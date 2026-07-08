"""DSV4 DPO dataset for gpatch_v4.

Data format (JSONL)::

    {
      "conversations": [
        {"role": "system", "content": "..."},   // optional
        {"role": "user", "content": "..."}
      ],
      "chosen": "preferred response text",
      "rejected": "dispreferred response text"
    }

Each ``__getitem__`` returns a ``(chosen_dict, rejected_dict)`` tuple;
the collator reorders to ``[all_chosen ..., all_rejected ...]``.
"""

import copy
import glob
import os
from typing import Any, Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from gpatch_v4.configs.config import DpoConfig
from gpatch_v4.models.deepseek_v4.chat_template import DSV4_CHAT_TEMPLATE


def tokenize_func(tokenizer, prompt_text, full_text):
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return full_ids, labels


class DSV4DpoDataset(Dataset):
    def __init__(self, config: DpoConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = DSV4_CHAT_TEMPLATE
        self.seq_len = config.training.seq_length
        self.system_prompt = getattr(config.data, "system_prompt", None)

        data_path = config.data.data_pathes[0]
        if os.path.isdir(data_path):
            json_files = glob.glob(os.path.join(data_path, "*.jsonl"))
            if not json_files:
                json_files = glob.glob(os.path.join(data_path, "*.json"))
            self.dataset = load_dataset("json", data_files=json_files, split="train")
        else:
            self.dataset = load_dataset(data_path, split="train")
        self.dataset = self.dataset.shuffle(seed=42)

    def __len__(self) -> int:
        return len(self.dataset)

    def _build_prompt_and_full(self, conversations, answer):
        messages = list(conversations)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        messages.append({"role": "assistant", "content": answer})
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        full_text = full_text.rstrip("\n")
        return prompt_text, full_text

    def __getitem__(self, idx: int) -> tuple:
        example = self.dataset[idx]

        conversations = []
        if self.system_prompt:
            conversations.append({"role": "system", "content": self.system_prompt})
        conversations.append({"role": "user", "content": example["instruction"]})
        conversations2 = copy.deepcopy(conversations)

        chosen_text = example["chosen"]
        rejected_text = example["rejected"]

        prompt_text, chosen_full = self._build_prompt_and_full(conversations, chosen_text)
        _, rejected_full = self._build_prompt_and_full(conversations, rejected_text)

        chosen_ids, chosen_labels = tokenize_func(
            self.tokenizer, prompt_text, chosen_full
        )
        rejected_ids, rejected_labels = tokenize_func(
            self.tokenizer, prompt_text, rejected_full
        )

        chosen_dict = {"tokens": chosen_ids, "labels": chosen_labels}
        rejected_dict = {"tokens": rejected_ids, "labels": rejected_labels}
        return (chosen_dict, rejected_dict)


def dpo_collate_fn(examples):
    chosen_list = []
    rejected_list = []
    for chosen, rejected in examples:
        chosen_list.append(chosen)
        rejected_list.append(rejected)
    all_samples = chosen_list + rejected_list

    tokens_list = [torch.tensor(s["tokens"], dtype=torch.long) for s in all_samples]
    labels_list = [torch.tensor(s["labels"], dtype=torch.long) for s in all_samples]

    return {
        "tokens": tokens_list,
        "labels": labels_list,
    }


def get_dataset_and_dataloader(
    config: DpoConfig = None, tokenizer=None, dp_rank=0, dp_size=1, meta_info=None
):
    dataset = DSV4DpoDataset(config=config, tokenizer=tokenizer)
    sampler = DistributedSampler(
        dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=dpo_collate_fn,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.train_mbs,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
        multiprocessing_context=config.data.multiprocessing_method,
    )
    return {
        "train_dataset": dataset,
        "train_sampler": sampler,
        "train_dataloader": dataloader,
    }
