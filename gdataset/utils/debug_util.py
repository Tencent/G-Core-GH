import hashlib

import torch
from megatron.core import mpu


def md5_sum(t):
    tmp = t.detach()
    if torch.is_floating_point(tmp):
        tmp = tmp.float()
    tmp = tmp.cpu().numpy().to_bytes()
    tmp = hashlib.md5(tmp).hexdigest()
    return tmp


def md5_sum_across_tp(t):
    md5 = md5_sum(t)
    group = mpu.get_tensor_model_parallel_group()
    all_md5 = [None] * torch.distributed.get_world_size(group=group)
    torch.distributed.all_gather_object(all_md5, md5, group=group)
    return all_md5
