import os
import json
import time
import random
import traceback

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# 仅仅是 demo，实际的数据集建议用专用管线或者 energon。
class SimpleTextPromptDataset(Dataset):
    def __init__(self, dataset_file, dp_rank):
        self.file_path = dataset_file
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        self.rng = random.Random(108 + dp_rank)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if not self.file_path.endswith('.jsonl'):
            return {"prompt": self.prompts[idx]}
        else:
            # 这个是給 xh 写的特殊 case，先这样写着，假装看不见就好了。
            jobj = json.loads(self.prompts[idx])

            if "caption" in jobj:
                caption_obj = jobj.get("caption")
                target_keys = [
                    "qwen72b_caption_short", "qwen72b_caption_medium", "qwen72b_caption_long"
                ]
            else:
                caption_obj = jobj
                target_keys = [
                    "Qwen2.5VL_72B_CN_Short", "Qwen2.5VL_72B_CN_Medium", "Qwen2.5VL_72B_CN_Long"
                ]

            ks = filter(lambda k: k in caption_obj and caption_obj[k] != None, target_keys)
            filter_ks = list(ks)
            assert len(filter_ks) > 0, f"No caption found in {self.file_path} {jobj=}"
            k = self.rng.choice(filter_ks)

            return {"prompt": caption_obj[k]}


def collate_fn(examples):
    prompts = [example["prompt"] for example in examples]
    return {"prompt": prompts}


def get_dataset_and_dataloader(config=None, dp_rank=0, dp_size=1):
    dataset = SimpleTextPromptDataset(config.data.data_pathes[0], dp_rank)  # 假装只有一个数据文件
    sampler = DistributedSampler(
        dataset, rank=dp_rank, num_replicas=dp_size, shuffle=True, seed=config.data.sampler_seed
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        batch_size=config.training.rollout_mbs,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
    )
    return {
        'train_dataset': dataset,
        'train_sampler': sampler,
        'train_dataloader': dataloader,
    }
