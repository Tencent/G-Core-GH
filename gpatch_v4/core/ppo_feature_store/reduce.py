"""Per-feature scalar reduce registry for ``PpoFeatureStore``."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

FeatureReduceFn = Callable[
    [list[float], Optional[list[float]], Any],
    Optional[float],
]
_FEATURE_REDUCE_REGISTRY: dict[str, FeatureReduceFn] = {}


def register_feature_reduce(
    name: str,
    fn: Optional[FeatureReduceFn] = None,
    *,
    override: bool = False,
):
    """Register a scalar reduce used by ``configure(..., reduce=name)``.

    Signature::

        (local_values, local_weights, group) -> float | None
    """
    def _register(func: FeatureReduceFn) -> FeatureReduceFn:
        if name in _FEATURE_REDUCE_REGISTRY and not override:
            raise ValueError(
                f"feature reduce {name!r} already registered; pass override=True to replace"
            )
        _FEATURE_REDUCE_REGISTRY[name] = func
        return func

    if fn is not None:
        return _register(fn)
    return _register


def get_feature_reduce(name: str) -> FeatureReduceFn:
    if name not in _FEATURE_REDUCE_REGISTRY:
        available = ", ".join(sorted(_FEATURE_REDUCE_REGISTRY))
        raise KeyError(f"Unknown feature reduce {name!r}. Available: [{available}]")
    return _FEATURE_REDUCE_REGISTRY[name]


def default_dp_group():
    if not (dist.is_available() and dist.is_initialized()):
        return None
    try:
        from megatron.core import mpu
        return mpu.get_data_parallel_group(with_context_parallel=True)
    except Exception:
        return None


def _device_for_stats() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@register_feature_reduce("mean")
def _reduce_mean(
    local_values: list[float],
    local_weights: Optional[list[float]],
    group,
) -> Optional[float]:
    if local_weights is None:
        raise ValueError("reduce='mean' requires local_weights (counts)")
    local_sum = float(sum(local_values))
    local_count = float(sum(local_weights))
    stats = torch.tensor([local_sum, local_count], dtype=torch.float64, device=_device_for_stats())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
    total_sum = float(stats[0].item())
    total_count = float(stats[1].item())
    if total_count <= 0:
        return None
    return total_sum / total_count


@register_feature_reduce("sum")
def _reduce_sum(
    local_values: list[float],
    local_weights: Optional[list[float]],
    group,
) -> Optional[float]:
    del local_weights
    has_dist = (
        dist.is_available() and dist.is_initialized() and group is not None and
        dist.get_world_size(group=group) > 1
    )
    if not local_values and not has_dist:
        return None
    local_sum = float(sum(local_values)) if local_values else 0.0
    stats = torch.tensor([local_sum], dtype=torch.float64, device=_device_for_stats())
    if has_dist:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
    return float(stats[0].item())


@register_feature_reduce("max")
def _reduce_max(
    local_values: list[float],
    local_weights: Optional[list[float]],
    group,
) -> Optional[float]:
    del local_weights
    local_max = max(local_values) if local_values else float("-inf")
    stats = torch.tensor([local_max], dtype=torch.float64, device=_device_for_stats())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=group)
    value = float(stats[0].item())
    if not math.isfinite(value):
        return None
    return value


@register_feature_reduce("min")
def _reduce_min(
    local_values: list[float],
    local_weights: Optional[list[float]],
    group,
) -> Optional[float]:
    del local_weights
    local_min = min(local_values) if local_values else float("inf")
    stats = torch.tensor([local_min], dtype=torch.float64, device=_device_for_stats())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.MIN, group=group)
    value = float(stats[0].item())
    if not math.isfinite(value):
        return None
    return value
