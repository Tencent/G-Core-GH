# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""DeepSeek-V4 loss-free router correction bias update (auxiliary-loss-free load balancing).

Each :class:`DeepseekV4TopKRouter` holds an ``e_score_correction_bias`` buffer
that is added to the routing score before top-k selection (paper:
https://arxiv.org/abs/2408.15664). During training we track how many tokens
each expert receives and update the bias at the end of the training step
so underloaded experts become more likely and overloaded ones less likely.

Counting is done with a *scoped* forward hook (:func:`router_load_tracking_ctx`)
that wraps only the forward call, so recompute during ``backward``
never re-triggers it, and every token is counted exactly once.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager, nullcontext
from typing import Iterator, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter

_ACCUM_ATTR = "local_tokens_per_expert"
ComputePhase = Literal["outside", "forward", "recompute"]
_is_train_forward: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "is_train_forward", default=False
)


def _iter_topk_routers(model: nn.Module) -> Iterator[nn.Module]:
    """Yield each ``DeepseekV4TopKRouter`` in *model*.

    Skips ``DeepseekV4HashRouter`` — its routing is deterministic and has no
    ``e_score_correction_bias``.
    """

    for module in model.modules():
        if isinstance(module, DeepseekV4TopKRouter):
            yield module


def freeze_router_weights(model: nn.Module) -> None:
    """Set ``requires_grad=False`` on every TopKRouter ``weight``.

    Must run before optimizer construction so the frozen weights are excluded
    from the optimizer's parameter list.
    """
    for router in _iter_topk_routers(model):
        router.weight.requires_grad_(False)


def init_router_correction_bias_accumulators(model: nn.Module) -> None:
    """Register a non-persistent per-router token-count accumulator.

    Shape ``[num_experts]``, fp32.
    """
    for router in _iter_topk_routers(model):
        if hasattr(router, _ACCUM_ATTR):
            router.get_buffer(_ACCUM_ATTR).zero_()
            continue
        bias = router.e_score_correction_bias
        router.register_buffer(
            _ACCUM_ATTR,
            torch.zeros_like(bias, dtype=torch.float32),
            persistent=False,
        )


def reset_router_correction_bias_accum(model: nn.Module) -> None:
    """Zero every router's token-count accumulator (call at each training step start)."""
    for router in _iter_topk_routers(model):
        router.get_buffer(_ACCUM_ATTR).zero_()


def register_router_correction_bias_accum_tracking_hook(model: nn.Module):
    """Register a forward hook to track router correction bias accum.

    Register a forward hook on each TopKRouter, only applied in training forward pass.
    The hook reads the router's top-k ``indices`` (``out[2]``) and adds
    per-expert counts into the ``local_tokens_per_expert`` buffer.
    Recomputation will not trigger the hook again.
    """
    def _make_hook():
        def hook(router: nn.Module, inp, out):
            if not (router.training and _is_train_forward.get()):
                return
            indices = out[2].detach().clone()
            flat = indices.reshape(-1)
            num_experts = router.e_score_correction_bias.shape[0]
            counts = torch.bincount(flat, minlength=num_experts).to(torch.float32)
            router.get_buffer(_ACCUM_ATTR).add_(counts)

        return hook

    for router in _iter_topk_routers(model):
        router.register_forward_hook(_make_hook())


@contextmanager
def train_forward_context():
    """Set the _is_train_forward contextvar on enter and reset it on exit"""
    token = _is_train_forward.set(True)
    try:
        yield
    finally:
        _is_train_forward.reset(token)


def checkpoint_context_fn():
    """Used as torch.utils.checkpoint.checkpoint context_fn
    Returns two context managers for forward pass and recompute pass respectively.
    """
    return train_forward_context(), nullcontext()


@torch.no_grad()
def update_router_correction_bias(
    model: nn.Module,
    update_speed: float,
    use_abs_update: bool = True
) -> Tuple[Optional[float], Optional[float]]:
    """Apply the loss-free load-balancing update to every TopKRouter correction bias.
    See https://arxiv.org/abs/2408.15664 §3 for more details.
    """
    routers = [r for r in _iter_topk_routers(model) if r.training]
    if not routers:
        return None, None

    counts = torch.stack([r.get_buffer(_ACCUM_ATTR) for r in routers], dim=0)  # [L, E]

    # NOTE(parkeychen): With FSDP2 + EP + CP, counts should be all_reduced in the global group
    # For possible future work:
    #   if using PP, all_reduce should be done in the corresponding PP rank (PP stage of that router)
    #   if using TP, each token may be counted multiple times in different TP ranks
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

    avg = counts.mean(dim=-1)  # [L]
    assert (avg == avg[0]
           ).all(), (f"avg load should be the same for all routers, S * K / E, but got {avg}")

    for i, router in enumerate(routers):
        if use_abs_update:
            router.e_score_correction_bias.add_(torch.sign(avg[i] - counts[i]) * update_speed)
        else:
            eps = 1.0
            router.e_score_correction_bias.add_(
                torch.log((avg[i] + eps) / (counts[i] + eps)) * update_speed
            )

    # maximal violation, see https://arxiv.org/abs/2408.15664 §4.1 for more details.
    maxvio = (counts.max(dim=-1).values - avg) / avg.clamp_min(1e-12)  # [L]
    return maxvio.max(), maxvio.mean()
