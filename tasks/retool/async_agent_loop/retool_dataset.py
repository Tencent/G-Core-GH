"""Dataset for RetoolEnv async rollout.

Returns raw ``question`` and ``target`` strings so that
``EnvAgentLoopActor`` can pass them as ``task_data`` to ``RetoolEnv.reset()``.

YAML config example::

    data:
      py_path: "tasks/retool/agentic_rl/retool_dataset.py"
      fn_name: "get_dataset_and_dataloader"
      data_pathes:
        - "tasks/retool/agentic_rl/fixtures/dapo_mini/metadata.json"
"""
from __future__ import annotations

from typing import Any, Dict, List

from torch.utils.data import DataLoader, Dataset

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler

from tasks.retool.async_agent_loop.dataset_utils import load_examples_from_metadata


class RetoolDataset(Dataset):
    """Each sample is a ``(question, target)`` pair loaded from metadata jsonl."""
    def __init__(self, config: RlConfig, metadata_file: str):
        self.examples = load_examples_from_metadata(metadata_file)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        question, target = self.examples[idx]
        return {"question": question, "target": target}


def _collate_fn(samples: List[Dict[str, str]]) -> Dict[str, List[str]]:
    return {
        "question": [s["question"] for s in samples],
        "target": [s["target"] for s in samples],
    }


def get_dataset_and_dataloader(
    config: RlConfig,
    tokenizer=None,
    dp_rank: int = 0,
    dp_size: int = 1,
    meta_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata_file = config.data.data_pathes[0]
    train_dataset = RetoolDataset(config=config, metadata_file=metadata_file)

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
        collate_fn=_collate_fn,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.rollout_mbs,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
    )

    return {
        "train_dataset": train_dataset,
        "train_sampler": train_sampler,
        "train_dataloader": train_dataloader,
    }
