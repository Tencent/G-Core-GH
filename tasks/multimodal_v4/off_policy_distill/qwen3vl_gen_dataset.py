"""
这个是离线生成样本的 dataset，生成的 dataset 会被传送到
tasks/multimodal_v4/off_policy_distill/offline_generate_examples.py:offline_generate_func
被使用
"""

from typing import Any, Sequence
from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset

from gpatch_v4.configs.config import RlConfig


def collate_func(instances: Sequence[dict]) -> dict[str, Any]:
    res = defaultdict(list)
    for instance in instances:
        res["images"].append([np.array(img) for img in instance["images"]])
        res["answers"].append(instance["answer"])
        res["problems"].append(instance["problem"])

    return res


def get_dataset_and_dataloader(config: RlConfig = None, tokenizer=None, dp_rank=0, dp_size=1):
    assert config.training.train_mbs == 1, "mbs must be 1"
    dataset = load_dataset(config.data.data_pathes[0])
    dataset_key = "train"
    if config.task is not None and "dataset_key" in config.task:
        dataset_key = config.task["dataset_key"]
    train_dataset = dataset[dataset_key].shuffle(seed=42)

    sampler = DistributedSampler(
        train_dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed
    )

    dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=collate_func,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.train_mbs,
        num_workers=config.data.dataloader_num_workers,
        prefetch_factor=config.data.dataloader_prefetch_factor,
        drop_last=True,
    )
    return {
        'train_dataset': train_dataset,
        'train_sampler': sampler,
        'train_dataloader': dataloader,
    }


def verify_dataloader_func(train_dataset, train_sampler, train_dataloader):
    print(f"{len(train_dataset)=}")
    data = train_dataset[0]
    print(f"{data.keys()=}")
    print(f"{data=}")

    it = iter(train_dataloader)
    data_dl = next(it)
    print(f"{data_dl.keys()=}")
    print(f"{data_dl=}")
