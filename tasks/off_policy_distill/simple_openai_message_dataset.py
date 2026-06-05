import glob
import os
import re
import bisect
from typing import Optional, Dict, Any

import torch
import torch.distributed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer

from gpatch_v4.configs.config import OffPolicyDistillConfig


def tokenize_text(tokenizer, prompt_seq_len, prompt):
    assert tokenizer.pad_token is not None
    prompt_tokenized = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    text_input_ids = prompt_tokenized.input_ids

    pattern = re.compile(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', re.DOTALL)
    content_starts = []
    content_ends = []
    assistant_contents = []
    for match in pattern.finditer(prompt):
        content_starts.append(match.start(1))
        content_ends.append(match.end(1) - 1)
        assistant_contents.append(match.group(1).strip())
    offsets_mapping = prompt_tokenized["offset_mapping"]
    labels = [-100] * len(text_input_ids)
    token_ends = [offset[1] - 1 for offset in offsets_mapping]
    first_prompt_pos = None
    for i, (cs, ce, content) in enumerate(zip(content_starts, content_ends, assistant_contents)):
        s = bisect.bisect(token_ends, cs)
        e = bisect.bisect(token_ends, ce) + 1
        labels[s:e] = text_input_ids[s:e]
        if i == 0:
            first_prompt_pos = s - 1
    real_seq_length = len(text_input_ids)
    prompt_len = first_prompt_pos
    return text_input_ids, labels, real_seq_length, prompt_len


class SimpleOpenaiMessageDataset(Dataset):
    def __init__(
        self,
        config: OffPolicyDistillConfig,
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
        assert "messages" in example.keys()
        assert "tools" in example.keys()
        conversations = example['messages']
        tools = example['tools']

        prompt = self.conversation_apply_chat_template(
            conversation=conversations,
            tools=tools,
        )

        input_ids, labels, seq_length, prompt_len = tokenize_text(
            self.tokenizer,
            self.seq_len,
            prompt,
        )

        return {
            'input_ids': input_ids,
            'labels': labels,
            'seq_length': seq_length,
            'prompt_len': prompt_len,
        }

    def get_features(self) -> list:
        return self.dataset.column_names

    def conversation_apply_chat_template(self, conversation=None, tools=None):
        assert conversation is not None, "conversation is None"
        assert tools is not None, "tools is None"

        formatted_text = self.tokenizer.apply_chat_template(
            conversation=conversation,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        return formatted_text


def collate_fn(examples):
    input_ids_list = []
    labels_list = []
    seq_length_list = []
    prompt_len_lst = []
    for example in examples:
        input_ids = example['tokens']
        labels = example['labels']
        seq_length = example['seq_length']
        prompt_len = example['prompt_len']

        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        seq_length_list.append(torch.tensor(seq_length, dtype=torch.long))
        prompt_len_lst.append(torch.tensor(prompt_len, dtype=torch.long))

    return {
        'tokens': input_ids_list,
        'sequence_lengths': seq_length_list,
        'prompt_lengths': prompt_len_lst,
        'labels': labels_list,
    }


def get_dataset_and_dataloader(
    config: OffPolicyDistillConfig = None, tokenizer=None, dp_rank=0, dp_size=1
):
    dataset = SimpleOpenaiMessageDataset(config=config, tokenizer=tokenizer)
    sampler = DistributedSampler(
        dataset, rank=dp_rank, num_replicas=dp_size, shuffle=True, seed=config.data.sampler_seed
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.train_mbs,
        num_workers=config.data.dataloader_num_workers,
        prefetch_factor=config.data.dataloader_prefetch_factor,
        drop_last=True,
    )
    return {
        'train_dataset': dataset,
        'train_sampler': sampler,
        'train_dataloader': dataloader,
    }
