import os

import torch


def save_data(data, save_dir, file_name, only_save_rank0=False):
    os.makedirs(save_dir, exist_ok=True)
    if only_save_rank0:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            torch.save(data, os.path.join(save_dir, file_name))
    else:
        torch.save(data, os.path.join(save_dir, file_name))
