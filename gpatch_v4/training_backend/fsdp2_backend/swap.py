import gc
from typing import Any

import torch

from gpatch_v4.utils.common_utils import clear_memory


# offload and onload normal model, fsdp2 model and optimizer
@torch.no_grad()
def offload_model(model, non_blocking=False, do_clear_memory=True):
    model.to('cpu', non_blocking=non_blocking)
    if do_clear_memory:
        clear_memory()


@torch.no_grad()
def onload_model(model, non_blocking=False, do_clear_memory=True):
    model.to(torch.cuda.current_device(), non_blocking=non_blocking)
    if do_clear_memory:
        clear_memory()


@torch.no_grad()
def offload_optimizer(optimizer):
    if optimizer is None:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)
    clear_memory()


@torch.no_grad()
def onload_optimizer(optimizer):
    if optimizer is None:
        return
    device_id = torch.cuda.current_device()
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
    clear_memory()
