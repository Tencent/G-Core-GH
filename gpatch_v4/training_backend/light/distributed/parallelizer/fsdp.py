"""FSDP2 model wrapping with optional EP (used by ``ModelParallelizer.apply_fsdp``)."""

from __future__ import annotations

import contextlib
import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed._tensor import Shard
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPCommContext,
    FSDPParamGroup,
)
from torch.distributed.fsdp._fully_shard._fsdp_state import (
    FSDPState,
    _get_module_fsdp_state,
)

from ..core.parallel_state import ParallelState, get_parallel_state
from ..core.utils import is_activation_checkpoint_inner_layer
from .ep import ExpertsParallelPlan
from .fsdp_patch import (
    apply_fsdp2_post_forward_patch,
    apply_fsdp2_reduce_scatter_ring_buffer_patch,
    enable_rs_ring,
    fully_shard,
)


def _collect_decoder_layers(
    model: nn.Module,
    auto_wrap_policy: Callable[[nn.Module], bool],
) -> list[nn.Module]:
    """Collect transformer decoder layers to wrap (``FSDP(CheckpointWrapper(layer))``)."""
    layers: list[nn.Module] = []
    for name, module in model.named_modules():
        if module is model:
            continue
        if is_activation_checkpoint_inner_layer(name):
            continue
        if auto_wrap_policy(module) and not isinstance(module, FSDPModule):
            if torch.distributed.get_rank() == 0:
                print(f"auto wrap {name} {type(module)}", flush=True)
            layers.append(module)

    return layers


# dtype 别名 → torch.dtype。param_dtype 与 reduce_dtype 共用这张表。
# 梯度规约默认 fp32（数值更稳）；设环境变量 FSDP2_REDUCE_DTYPE=bfloat16 可复现旧版
# Accelerate FSDP1 的 bf16 规约行为，便于做两版收敛对齐实验。
_DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _resolve_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Invalid dtype {name!r}; expected one of {sorted(_DTYPE_ALIASES)}")
    return _DTYPE_ALIASES[key]


def _fsdp_mixed_precision_policy(param_dtype_name: str) -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(
        param_dtype=_resolve_dtype(param_dtype_name),
        reduce_dtype=_resolve_dtype(os.environ.get("FSDP2_REDUCE_DTYPE", "float32")),
        cast_forward_inputs=True,
    )


def wrap_model_fsdp2(
    model: nn.Module,
    *,
    auto_wrap_policy: Callable[[nn.Module], bool],
    experts_parallel_plan: Optional[ExpertsParallelPlan] = None,
    reshard_after_forward: bool = True,
    mixed_precision: str = "bf16",
    state: Optional[ParallelState] = None,
    reduce_scatter_ring_buffer: int = 0,
) -> nn.Module:
    # Backport PyTorch PR #164009: avoid `post_forward` re-running reshard /
    # post-forward bookkeeping when AC re-invokes the FSDP `_pre_forward` hook
    # during the backward pass. Idempotent across calls.
    apply_fsdp2_post_forward_patch()

    # Optional FSDP2 reduce-scatter input ring buffer (torch >= 2.10). The
    # ``enable_rs_ring()`` context makes the patched ``fully_shard`` auto-mark
    # every module wrapped inside it as ring-eligible; the root (wrapped outside
    # the context) stays on the upstream serialized path. See
    # fsdp_patch.apply_fsdp2_reduce_scatter_ring_buffer_patch.
    use_rs_ring = bool(reduce_scatter_ring_buffer and reduce_scatter_ring_buffer > 1)
    if use_rs_ring:
        apply_fsdp2_reduce_scatter_ring_buffer_patch(ring_size=reduce_scatter_ring_buffer)
    rs_ring_ctx = enable_rs_ring() if use_rs_ring else contextlib.nullcontext()

    state = state or get_parallel_state()

    if experts_parallel_plan is not None and state.ep_device_mesh is not None:
        experts_parallel_plan.apply(
            model, state.ep_device_mesh["ep"], state.ep_device_mesh["ep_fsdp"]
        )

    mixed_precision_policy = _fsdp_mixed_precision_policy(mixed_precision)
    fsdp_kwargs = {
        "mesh": state.fsdp_mesh,
        "reshard_after_forward": reshard_after_forward,
        "mp_policy": mixed_precision_policy,
    }
    ep_fsdp_kwargs = {
        "mesh": state.ep_fsdp_mesh,
        "reshard_after_forward": reshard_after_forward,
        "mp_policy": mixed_precision_policy,
        "shard_placement_fn": lambda _p: Shard(1),
    }

    ep_wrapped: set[str] = set()
    # EP modules + decoder layers are wrapped inside enable_rs_ring() so the
    # patched fully_shard marks them ring-eligible when use_rs_ring is on.
    with rs_ring_ctx:
        if experts_parallel_plan is not None and state.ep_device_mesh is not None:
            for fqn, ep_mod in experts_parallel_plan.find_ep_modules(model).items():
                if fqn in ep_wrapped or isinstance(ep_mod, FSDPModule):
                    continue
                fully_shard(ep_mod, **ep_fsdp_kwargs)
                if hasattr(ep_mod, "set_gradient_divide_factor"):
                    # EP FSDP mesh is ep_fsdp-only; scale to global microbatch count.
                    # CP ranks share a microbatch (seq shards), so exclude cp_size here.
                    ep_mod.set_gradient_divide_factor(state.ep_gradient_divide_factor)
                ep_wrapped.add(fqn)

        for layer in _collect_decoder_layers(model, auto_wrap_policy):
            if torch.distributed.get_rank() == 0:
                print(f"_collect_decoder_layers wrap layer {type(layer)}", flush=True)
            fully_shard(layer, **fsdp_kwargs)

    # Root is wrapped outside the context: it keeps the upstream serialized RS path.
    root_kwargs = {k: v for k, v in fsdp_kwargs.items() if k != "reshard_after_forward"}
    if not isinstance(model, FSDPModule):
        fully_shard(model, **root_kwargs)
    return model


class _SharedFSDPCommContext(FSDPCommContext):
    """Idempotent FSDPCommContext: lazy_init is a no-op after the first call.

    When two FSDP2 root modules share a single _comm_ctx, FSDP2's
    ``_init_shared_state`` calls ``comm_ctx.lazy_init(device)`` for each root.
    Without the guard this would re-create all streams and break sharing.
    """
    def lazy_init(self, device: torch.device) -> None:
        if hasattr(self, "all_gather_stream"):
            return  # already initialized, skip
        super().lazy_init(device)


def share_fsdp2_comm_ctx(
    model_a: nn.Module,
    model_b: nn.Module,
    device: torch.device,
) -> None:
    """Share one FSDPCommContext (and its NPU streams) between two FSDP2 roots.

    Both models will use the same ``all_gather_stream``,
    ``all_gather_copy_in_stream``, ``reduce_scatter_stream``, and
    ``all_reduce_stream``.  Because the two roots run **sequentially**
    (text_encoder in no_grad, transformer3d in the main train step),
    sharing ``all_gather_state`` / ``post_forward_order`` is safe: each
    root's ``_post_forward`` clears them before the other model runs.

    Must be called **after** ``fully_shard()`` but **before** the first
    forward pass of either model (i.e. before ``lazy_init`` fires).
    """
    state_a = _get_module_fsdp_state(model_a)
    state_b = _get_module_fsdp_state(model_b)
    if state_a is None or state_b is None:
        raise RuntimeError("Both models must be wrapped with fully_shard before sharing comm_ctx")

    shared_ctx = _SharedFSDPCommContext()
    shared_ctx.lazy_init(device)  # pre-initialize once; subsequent calls are no-ops

    # Replace both root states' _comm_ctx.  Child states are not yet linked
    # (lazy_init hasn't run), but _init_shared_state will propagate
    # shared_ctx to all descendants when the first forward fires.
    state_a._comm_ctx = shared_ctx
    state_b._comm_ctx = shared_ctx
    print(
        "[NPU] Shared FSDPCommContext (streams) between %s and %s" % (
            type(model_a).__name__,
            type(model_b).__name__,
        )
    )
