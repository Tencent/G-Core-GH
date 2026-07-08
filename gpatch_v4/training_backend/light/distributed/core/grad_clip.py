"""Gradient clipping helpers for FSDP2/EP training."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor, Shard


def _dtensor_shard_groups(grad: DTensor) -> tuple[dist.ProcessGroup, ...]:
    """Return process-groups for mesh dimensions where ``grad`` is sharded."""
    groups: list[dist.ProcessGroup] = []
    for mesh_dim, placement in enumerate(grad.placements):
        if isinstance(placement, Shard):
            groups.append(grad.device_mesh.get_group(mesh_dim))
    return tuple(groups)


def get_grad_norm(
    parameters: Iterable[nn.Parameter] | nn.Module,
    norm_type: float = 2.0,
    *,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Compute global grad norm from local grads.

    Supports both replicated grads (plain ``torch.Tensor``) and sharded grads
    (``DTensor``). For ``DTensor`` grads, reduction is performed only across mesh
    dimensions where placement is ``Shard``.

    The computation is split into two stages:
    1) local accumulation by shard-group signature;
    2) one global reduction per group signature.
    """
    if isinstance(parameters, nn.Module):
        params = list(parameters.parameters())
    else:
        params = list(parameters)

    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0, dtype=torch.float32)

    norm_type = float(norm_type)
    if norm_type <= 0.0:
        raise ValueError(f"norm_type must be > 0 or inf, got {norm_type}")

    ref_grad = grads[0]
    device = ref_grad.to_local().device if isinstance(ref_grad, DTensor) else ref_grad.device

    if norm_type == float("inf"):
        replicated_max = torch.zeros((), dtype=torch.float32, device=device)
        sharded_max_by_group: dict[tuple[int, ...], tuple[torch.Tensor, tuple[dist.ProcessGroup,
                                                                              ...]]] = {}

        for grad in grads:
            if isinstance(grad, DTensor):
                groups = _dtensor_shard_groups(grad)
                local_max = grad.to_local().detach().float().abs().max()
                if not groups:
                    replicated_max = torch.maximum(replicated_max, local_max)
                    continue
                key = tuple(id(g) for g in groups)
                bucket = sharded_max_by_group.get(key)
                if bucket is None:
                    sharded_max_by_group[key] = (local_max, groups)
                else:
                    sharded_max_by_group[key] = (torch.maximum(bucket[0], local_max), groups)
            else:
                replicated_max = torch.maximum(
                    replicated_max,
                    grad.detach().float().abs().max().to(device)
                )

        total_norm = replicated_max
        for group_max, groups in sharded_max_by_group.values():
            for group in groups:
                dist.all_reduce(group_max, op=dist.ReduceOp.MAX, group=group)
            total_norm = torch.maximum(total_norm, group_max)
    else:
        replicated_total = torch.zeros((), dtype=torch.float32, device=device)
        sharded_sum_by_group: dict[tuple[int, ...], tuple[torch.Tensor, tuple[dist.ProcessGroup,
                                                                              ...]]] = {}

        for grad in grads:
            if isinstance(grad, DTensor):
                groups = _dtensor_shard_groups(grad)
                local_sum = grad.to_local().detach().float().abs().pow(norm_type).sum()
                if not groups:
                    replicated_total += local_sum
                    continue
                key = tuple(id(g) for g in groups)
                bucket = sharded_sum_by_group.get(key)
                if bucket is None:
                    sharded_sum_by_group[key] = (local_sum, groups)
                else:
                    sharded_sum_by_group[key] = (bucket[0] + local_sum, groups)
            else:
                replicated_total += grad.detach().float().abs().pow(norm_type).sum().to(device)

        total = replicated_total
        for group_sum, groups in sharded_sum_by_group.values():
            for group in groups:
                dist.all_reduce(group_sum, op=dist.ReduceOp.SUM, group=group)
            total += group_sum
        total_norm = total.pow(1.0 / norm_type)

    if error_if_nonfinite and not torch.isfinite(total_norm):
        raise RuntimeError(f"Non-finite grad norm detected: {total_norm.item()}")
    return total_norm


def clip_grad(
    parameters: Iterable[nn.Parameter] | nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
    *,
    error_if_nonfinite: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip gradients by global norm and return the pre-clip global norm."""
    if isinstance(parameters, nn.Module):
        params = list(parameters.parameters())
    else:
        params = list(parameters)

    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0, dtype=torch.float32)

    max_norm = float(max_norm)
    if max_norm < 0.0:
        raise ValueError(f"max_norm must be >= 0, got {max_norm}")

    total_norm = get_grad_norm(
        params,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
    )
    clip_coef = max_norm / (total_norm + float(eps))
    if clip_coef >= 1.0:
        return total_norm

    for grad in grads:
        if isinstance(grad, DTensor):
            local = grad.to_local()
            local.mul_(clip_coef.to(device=local.device, dtype=local.dtype))
        else:
            grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))
    return total_norm
