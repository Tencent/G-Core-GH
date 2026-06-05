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
from tqdm import tqdm

from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.utils import logging_rank0


async def write_tars_into_db(tar_dir, dp_rank, dp_size, kv_store_client):
    IMG_POSTFIXES = ['jpg', 'jpeg', 'JPG', 'png', 'PNG']

    # list tar balls
    tar_pathes = []
    for tar_p in os.listdir(tar_dir):
        if tar_p.endswith('.tar'):
            tar_pathes.append(os.path.join(tar_dir, tar_p))
    tar_pathes = sorted(tar_pathes)
    tar_pathes = tar_pathes[dp_rank::dp_size]
    cpu_barrier()

    # list images
    img_k_lst = []
    for tar_p in tar_pathes:
        with tarfile.open(tar_p, 'r:') as tar_f:
            for tar_mem in tqdm(tar_f.getmembers()):
                if tar_mem.name.split('.')[-1] in IMG_POSTFIXES:
                    f = tar_f.extractfile(tar_mem)
                    f_bytes = f.read()
                    img_k = f'/data/{tar_p}/{tar_mem.name}'
                    img_k_lst.append(img_k)
                    await kv_store_client.set_co(img_k, f_bytes)

    # 几百万条能 hold 住，不行再改。
    l = [None for _ in range(dp_size)]
    assert dp_size == torch.distributed.get_world_size(), f'FIXME later, dp-size != g-size'
    torch.distributed.all_gather_object(l, img_k_lst)
    g_img_k_list = list(itertools.chain(*l))
    if dp_rank == 0:
        buf = io.BytesIO()
        torch.save(g_img_k_list, buf)
        b = buf.getvalue()
        await kv_store_client.set_co(f'/meta/keys/{tar_dir}', b)
        # Write guard key to indicate this tar_dir has been fully written
        await kv_store_client.set_co(f'/meta/guard/{tar_dir}', b'1')
    cpu_barrier()


async def write_tars_if_havnt(tar_dirs, dp_rank, dp_size, kv_store_client):
    # list tar balls
    for dir_i, tar_dir in enumerate(tar_dirs):
        # Check guard key to skip already-written tar dirs
        guard = await kv_store_client.get_co(f'/meta/guard/{tar_dir}')
        if guard is not None:
            logging_rank0(f'[write_tars_if_havnt] Skipping {tar_dir}, already written to kv store.')
        else:
            await write_tars_into_db(tar_dir, dp_rank, dp_size, kv_store_client)
        cpu_barrier()


async def read_data_k_lst(tar_dirs, dp_rank, dp_size, kv_store_client):
    img_k_lst = []
    for dir_i, tar_dir in enumerate(tar_dirs):
        _img_k_lst = await kv_store_client.get_co(f'/meta/keys/{tar_dir}')
        _img_k_lst = torch.load(io.BytesIO(_img_k_lst), weights_only=False)
        img_k_lst += _img_k_lst
    cpu_barrier()
    return img_k_lst
