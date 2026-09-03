"""Dataset & DataLoader for DAPO-Math-17K.

Supports both jsonl and parquet formats (auto-detected by file extension).

DAPO-Math-17k field format (per record):
    {"question": "...", "target": "12"}

Field semantics:
    - ``question``: the math problem (user prompt).
    - ``target``: ground-truth answer string (e.g. ``"12"``, ``"\\frac{1}{2}"``).
      Passed as ``gt_label`` for rule-based reward evaluation via
      ``bt_reward_dapo.py::dapo_math_rule_reward``.
"""
import glob
import os
from typing import Any, Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler


def _load_dataset_auto(data_path: str, split: str):
    """Load a jsonl/parquet file or a directory containing either format.

    Parameters
    ----------
    data_path : str
        A ``.jsonl`` / ``.parquet`` file, or a directory containing either
        ``*.jsonl`` or ``*.parquet`` files.
    split : str
        Dataset split name (used as the HF datasets split label).

    Returns
    -------
    datasets.Dataset

    Raises
    ------
    AssertionError
        If no supported files are found in ``data_dir``.
    """
    if os.path.isfile(data_path):
        if data_path.endswith(".jsonl"):
            jsonl_files, parquet_files = [data_path], []
        elif data_path.endswith(".parquet"):
            jsonl_files, parquet_files = [], [data_path]
        else:
            raise AssertionError(f"unsupported dataset file format: {data_path}")
    else:
        jsonl_files = glob.glob(os.path.join(data_path, "*.jsonl"))
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))

    if jsonl_files:
        return load_dataset("json", data_files={split: jsonl_files}, split=split)
    elif parquet_files:
        return load_dataset("parquet", data_files={split: parquet_files}, split=split)
    else:
        raise AssertionError(
            f"no *.jsonl or *.parquet files found at {data_path}. "
            f"Download DAPO-Math-17K and place files there."
        )


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


class DapoMathDataset(torch.utils.data.Dataset):
    """Dataset for DAPO-Math-17K (jsonl or parquet).

    Parameters
    ----------
    config : RlConfig
    tokenizer : AutoTokenizer
    split : str
        ``"train"`` uses ``config.data.data_pathes[0]``;
        ``"test"`` uses ``config.data.eval_data_pathes[0]``.
    """
    def __init__(
        self,
        config: RlConfig,
        tokenizer: AutoTokenizer,
        split: str = "train",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.system_prompt = config.data.system_prompt
        self.seq_len = config.training.seq_length
        resp_seq_len = -1
        for infer_engine in config.sampler.infer_engine_configs:
            resp_seq_len = max(resp_seq_len, infer_engine.generate_max_tokens)
        self.resp_seq_len = resp_seq_len

        if split == "train":
            data_dir = config.data.data_pathes[0]
        elif split == "test":
            data_dir = config.data.eval_data_pathes[0]
        else:
            raise ValueError(f"unknown split: {split}")

        self.dataset = _load_dataset_auto(data_dir, split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        if "prompt" in example and "label" in example:
            question = example["prompt"]
            target = str(example["label"])
        elif "problem" in example and "answer" in example:
            # HuggingFaceH4/aime_2024 stores a plain problem string instead
            # of a chat message list. Convert it to the DAPO chat format.
            question = [{"role": "user", "content": example["problem"]}]
            target = str(example["answer"])
        else:
            raise AssertionError(
                "expected either DAPO fields ('prompt', 'label') or "
                "AIME fields ('problem', 'answer') at "
                f"idx={idx}, got {sorted(example)}"
            )

        prompt, _ = self._apply_chat_template(question)
        input_ids, prompt_len = tokenize_text(
            self.tokenizer,
            self.seq_len - self.resp_seq_len,
            prompt,
        )

        # gt_label: pass as float if it's a pure number; otherwise keep string.
        # bt_reward.py handles both paths (float comparison for integers,
        # json-string path for complex answers).
        try:
            gt_label = float(target)
        except (ValueError, TypeError):
            gt_label = target

        return {
            "input_ids": input_ids,
            "prompt_len": prompt_len,
            "gt_label": gt_label,
        }

    def get_features(self) -> list:
        return self.dataset.column_names

    def _apply_chat_template(self, question: list[dict[str, str]]):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(question)
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
        prompt_len = example["prompt_len"]
        input_ids = example["input_ids"][:prompt_len]

        prompt_token_ids.append({"prompt_token_ids": input_ids})
        prompt_lens.append(torch.tensor(prompt_len, dtype=torch.long))

        label = example["gt_label"]
        if isinstance(label, float):
            gt_label.append(torch.tensor(label, dtype=torch.float32))
        else:
            # String labels are kept as-is; collation stores them in a list.
            gt_label.append(label)

    return {
        "prompt_token_ids": prompt_token_ids,
        "prompt_lens": prompt_lens,
        "gt_label": gt_label,
    }


def get_dataset_and_dataloader(
    config: RlConfig = None,
    tokenizer=None,
    dp_rank: int = 0,
    dp_size: int = 1,
    meta_info=None,
):
    train_dataset = DapoMathDataset(config=config, tokenizer=tokenizer, split="train")

    train_sampler = ResumableDistributedSampler(
        train_dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=config.data.shuffle,
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
        eval_dataset = DapoMathDataset(config=config, tokenizer=tokenizer, split="test")
        eval_sampler = DistributedSampler(
            eval_dataset,
            rank=dp_rank,
            num_replicas=dp_size,
            shuffle=False,
            seed=config.data.sampler_seed,
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
        "train_dataset": train_dataset,
        "train_sampler": train_sampler,
        "train_dataloader": train_dataloader,
        "eval_dataset": eval_dataset,
        "eval_dataloader": eval_dataloader,
    }


def get_batched_data(batched_data=None):
    assert batched_data is not None
    return batched_data
