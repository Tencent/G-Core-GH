# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.

"""Global prev-entropy quantile cuts: postprocess hook + GRPO loss (print cuts).

Register via::

    ppo:
      feature_store_enable: true
      use_legacy_loss: true
      post_compute_logprobs: prev_entropy_quantile
      post_compute_logprobs_py_path: tasks/math_rl_v4/pre_entropy_quantile.py
      post_compute_logprobs_py_name: prev_entropy_quantile_postprocess
      loss_func: prev_entropy_quantile_grpo
      loss_func_py_path: tasks/math_rl_v4/pre_entropy_quantile.py
      loss_func_py_name: prev_entropy_quantile_grpo_loss_func

Optional ``task.prev_entropy_quantile_num_bins`` (default 10).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from gpatch_v4.core.ppo_feature_store import (
    get_ppo_feature_store,
    is_ppo_feature_store_enabled,
)
from gpatch_v4.training_backend.loss_factory import PolicyLossInput, grpo_loss_func
from gpatch_v4.utils.common_utils import logging_rank0
from gpatch_v4.utils.ppo_utils import create_response_mask

PREV_ENTROPY_QUANTILE_CUTS_KEY = "prev_entropy_quantile_cuts"
_DEFAULT_NUM_BINS = 10


def _task_get(task: Any, key: str, default: Any) -> Any:
    if task is None:
        return default
    if isinstance(task, dict):
        return task[key] if key in task else default
    if hasattr(task, key):
        return getattr(task, key)
    return default


def get_prev_entropy_quantile_num_bins(config: Any) -> int:
    task = getattr(config, "task", None)
    num_bins = int(_task_get(task, "prev_entropy_quantile_num_bins", _DEFAULT_NUM_BINS))
    assert num_bins >= 2, f"prev_entropy_quantile_num_bins must be >= 2, got {num_bins}"
    return num_bins


def collect_valid_prev_entropies(
    rollout_batches: List[Dict[str, Any]],
) -> torch.Tensor:
    """Concatenate response-token ``prev_per_token_entropies`` across local batches.

    Returns
    -------
    torch.Tensor
        1-D ``float32`` CPU tensor of valid response entropies.
    """
    pieces: List[torch.Tensor] = []
    for rb in rollout_batches:
        entropies = rb["prev_per_token_entropies"]
        prompt_lengths = rb["prompt_lengths"]
        sequence_lengths = rb["sequence_lengths"]
        masks = create_response_mask(
            values=entropies,
            prompt_lengths=prompt_lengths,
            sequence_lengths=sequence_lengths,
            dtype=torch.float32,
        )
        for ent, mask in zip(entropies, masks, strict=True):
            valid = ent.detach().float().reshape(-1)[mask.reshape(-1).bool()]
            if valid.numel() > 0:
                pieces.append(valid.cpu())
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(pieces, dim=0)


def all_gather_1d_tensor(
    local: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """All-gather a 1-D tensor across ``group`` (pad to max local length)."""
    assert local.dim() == 1, f"expected 1-D tensor, got {tuple(local.shape)}"
    if (not dist.is_available()) or (not dist.is_initialized()):
        return local
    world = dist.get_world_size(group=group)
    if world == 1:
        return local

    device = local.device
    local_n = torch.tensor([local.numel()], device=device, dtype=torch.long)
    size_list = [torch.zeros_like(local_n) for _ in range(world)]
    dist.all_gather(size_list, local_n, group=group)
    max_n = max(int(s.item()) for s in size_list)
    if max_n == 0:
        return local.new_empty(0)

    padded = local.new_zeros(max_n)
    if local.numel() > 0:
        padded[:local.numel()].copy_(local)
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded, group=group)
    parts = [g[:int(s.item())] for g, s in zip(gathered, size_list)]
    return torch.cat(parts, dim=0)


def compute_quantile_cuts(entropy_1d: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Quantile cuts: ``linspace(0,1,num_bins+1)[1:-1]``.

    Parameters
    ----------
    entropy_1d : torch.Tensor
        1-D float entropies (all valid tokens).
    num_bins : int
        ``>= 2``; returned cuts have shape ``[num_bins - 1]``.
    """
    assert num_bins >= 2, f"num_bins must be >= 2, got {num_bins}"
    assert entropy_1d.numel() > 0, "compute_quantile_cuts requires at least one valid entropy"
    entropy_1d = entropy_1d.detach().to(torch.float32).reshape(-1)
    qs = torch.linspace(
        0.0, 1.0, num_bins + 1, device=entropy_1d.device, dtype=torch.float32
    )[1:-1]
    return torch.quantile(entropy_1d, qs).contiguous()


def prev_entropy_quantile_postprocess(
    config: Any,
    rollout_batches: List[Dict[str, Any]],
) -> None:
    """Gather response prev-entropies across DP, write quantile cuts to step-local store."""
    assert is_ppo_feature_store_enabled(), (
        "prev_entropy_quantile_postprocess requires ppo.feature_store_enable=True"
    )
    assert rollout_batches, "rollout_batches must be non-empty"
    assert "prev_per_token_entropies" in rollout_batches[0], (
        "prev_entropy_quantile_postprocess requires prev_per_token_entropies on rollout batches"
    )

    num_bins = get_prev_entropy_quantile_num_bins(config)
    local = collect_valid_prev_entropies(rollout_batches)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    local = local.to(device=device, dtype=torch.float32)

    group = None
    if dist.is_available() and dist.is_initialized():
        from megatron.core import mpu
        group = mpu.get_data_parallel_group()
    all_entropy = all_gather_1d_tensor(local, group=group)
    assert all_entropy.numel() > 0, "global valid prev_per_token_entropy is empty"

    cuts = compute_quantile_cuts(all_entropy, num_bins).detach().cpu().contiguous()
    get_ppo_feature_store().set_step_local(PREV_ENTROPY_QUANTILE_CUTS_KEY, cuts)


def prev_entropy_quantile_grpo_loss_func(config, loss_input: PolicyLossInput):
    """Plain GRPO; fetch step-local quantile cuts and print them."""
    cuts = get_ppo_feature_store().get_step_local(PREV_ENTROPY_QUANTILE_CUTS_KEY)
    num_bins = get_prev_entropy_quantile_num_bins(config)
    assert cuts.numel() == num_bins - 1, (
        f"cuts shape mismatch: got {cuts.numel()}, expect {num_bins - 1}"
    )
    logging_rank0(f"prev_entropy_quantile_cuts={cuts.tolist()}")
    return grpo_loss_func(config, loss_input)
