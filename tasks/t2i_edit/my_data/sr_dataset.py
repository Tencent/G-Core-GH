import os
import io
import json
import time
import random
import traceback
import tarfile
import asyncio
import itertools

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from gpatch_v4.utils.common_utils import log, logging_rank0
from gpatch_v4.core.parallel_state import cpu_barrier
from tasks.t2i_edit.my_data.helper import write_tars_if_havnt, read_data_k_lst


class SrDataset(Dataset):
    def __init__(self, img_k_lst, dp_rank, kv_store_client, target_height, target_width):
        self.img_k_lst = img_k_lst
        self.kv_store_client = kv_store_client
        self.rng = random.Random(108 + dp_rank)
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self):
        return len(self.img_k_lst)

    def _resize_and_random_crop(self, img):
        """Resize the short side to match target, then random crop to (target_width, target_height)."""
        w, h = img.size
        th, tw = self.target_height, self.target_width

        # resize so that the image covers the target area (scale to fit the larger ratio)
        scale = max(tw / w, th / h)
        new_w = round(w * scale)
        new_h = round(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # random crop to exact target size
        crop_x = self.rng.randint(0, new_w - tw)
        crop_y = self.rng.randint(0, new_h - th)
        img = img.crop((crop_x, crop_y, crop_x + tw, crop_y + th))
        return img

    def __getitem__(self, idx):
        img_k = self.img_k_lst[idx]  # f'/data/{tar_p}/{tar_mem.name}'
        fcont = self.kv_store_client.get_http(img_k)
        img = Image.open(io.BytesIO(fcont)).convert("RGB")

        # resize + random crop to unified target size
        img = self._resize_and_random_crop(img)

        # SR: target_image is the high-res crop; downsample 16x then upsample back to make a blurry cond
        w, h = img.size
        small_img = img.resize((w // 16, h // 16), Image.BICUBIC)
        cond_img = small_img.resize((w, h), Image.BICUBIC)

        return {
            "target_image": img,
            "cond_images": [cond_img],
            'prompt': 'SR',
        }


def collate_fn(examples):
    target_images = [example["target_image"] for example in examples]
    cond_images = [example["cond_images"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    return {
        "target_image": target_images,
        "cond_images": cond_images,
        "prompt": prompts,
    }


async def get_dataset_and_dataloader(
    *,
    config: 'gpatch_v4.configs.utils.MappingProtocol',
    dp_rank: int,
    dp_size: int,
    kv_store_client: 'gpatch_v4.client.KvStoreClient',
):
    await write_tars_if_havnt(config.data.data_pathes, dp_rank, dp_size, kv_store_client)

    t0 = time.time()
    img_k_lst = await read_data_k_lst(config.data.data_pathes, dp_rank, dp_size, kv_store_client)
    logging_rank0(f'get_dataset_and_dataloader read_data_k_lst elapsed {time.time() - t0}')

    dataset = SrDataset(
        img_k_lst,
        dp_rank,
        kv_store_client,
        target_height=config.training.height,
        target_width=config.training.width,
    )
    sampler = DistributedSampler(
        dataset, rank=dp_rank, num_replicas=dp_size, shuffle=True, seed=config.data.sampler_seed
    )
    assert config.data.dataloader_num_workers == 1
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        batch_size=config.training.train_mbs,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
    )

    return {
        'train_dataset': dataset,
        'train_sampler': sampler,
        'train_dataloader': dataloader,
    }
