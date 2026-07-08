"""Dynamically bind EP mixins onto HuggingFace Experts modules.

Instead of registering a new ``@use_experts_implementation`` forward, we swap
``module.__class__`` to ``(DispatchMixin, OriginalExpertsClass)``.  Weights and
buffers stay on the same instance; only ``forward`` / dispatch logic changes.
"""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn

from ..core.utils import iter_experts_modules, validate_experts_forward_signature
from .parallel_experts_alltoall import AllToAllEPExpertsMixin
from .parallel_experts_deepep import (
    DEEPEP_AVAILABLE,
    DEEPEP_IMPORT_ERROR,
    DeepEPEPExpertsMixin,
)
from .parallel_experts_mixin import EPExpertsMixin
from .parellel_experts_alltoall_npu import AllToAllEPExpertsMixinNPU

# ``ParallelConfig.ep_dispatch`` -> mixin that implements ``_ep_forward``.
EP_DISPATCH_MIXINS: dict[str, Type[EPExpertsMixin]] = {
    "alltoall": AllToAllEPExpertsMixin,
    "alltoall_npu": AllToAllEPExpertsMixinNPU,
}
if DEEPEP_AVAILABLE:
    EP_DISPATCH_MIXINS["deepep"] = DeepEPEPExpertsMixin

EP_DISPATCH_MODES = tuple(EP_DISPATCH_MIXINS)

# Records which dispatch mode is bound; avoids re-creating the dynamic class.
_BOUND_ATTR = "_fsdp_ep_cp_ep_bound"
# One dynamic class per (dispatch, original HF Experts class) pair.
_EP_BOUND_CLASS_CACHE: dict[tuple[str, type], type] = {}


def _get_ep_bound_class(dispatch: str, base_cls: type) -> type:
    """Build ``class EP{dispatch}_{Name}(Mixin, HFExperts): ...`` with caching."""
    key = (dispatch, base_cls)
    cached = _EP_BOUND_CLASS_CACHE.get(key)
    if cached is not None:
        return cached

    mixin_cls = EP_DISPATCH_MIXINS[dispatch]
    # Mixin before base so dispatch ``forward`` / ``_ep_forward`` take priority.
    cached = type(
        f"EP{dispatch}_{base_cls.__name__}",
        (mixin_cls, base_cls),
        {"_fsdp_ep_cp_dispatch": dispatch},
    )
    _EP_BOUND_CLASS_CACHE[key] = cached
    return cached


def _unwrap_experts_class(cls: type) -> type:
    """Strip prior EP bindings so rebinding uses the original HF class."""
    if getattr(cls, "_fsdp_ep_cp_dispatch", "base") == "base":
        return cls
    for base in cls.__bases__:
        if issubclass(base, EPExpertsMixin):
            continue
        if issubclass(base, nn.Module):
            return _unwrap_experts_class(base)
    return cls


def _ensure_experts_runtime_attrs(module: nn.Module, base_cls: type) -> None:
    # QwenImageDeepseekV3GeluExperts is a non-gated GELU expert container:
    # up_proj has shape (E, intermediate, hidden), while gated DeepSeek experts
    # expose gate_up_proj and _apply_gate for SwiGLU.  The EP mixins default to
    # gated mode, so mark this custom GELU container explicitly.  This mirrors
    # the dedicated GELU EP test setup and avoids looking up a non-existent
    # _apply_gate during all-to-all dispatch.
    if base_cls.__name__ == "QwenImageDeepseekV3GeluExperts":
        module.has_gate = False
        module.is_transposed = False
        module.has_bias = False
        module.is_concatenated = True


def bind_ep_experts(module: nn.Module, dispatch: str = "alltoall") -> nn.Module:
    """Bind an EP dispatch mixin onto a single Experts module via ``__class__`` swap."""
    if dispatch not in EP_DISPATCH_MIXINS:
        if dispatch == "deepep":
            raise ImportError(DEEPEP_IMPORT_ERROR)
        raise ValueError(f"Unknown ep_dispatch={dispatch!r}, choose from {list(EP_DISPATCH_MODES)}")

    validate_experts_forward_signature(module)

    base_cls = _unwrap_experts_class(module.__class__)
    _ensure_experts_runtime_attrs(module, base_cls)
    if getattr(module, _BOUND_ATTR, None) == dispatch:
        return module

    module.__class__ = _get_ep_bound_class(dispatch, base_cls)
    setattr(module, _BOUND_ATTR, dispatch)
    _ensure_experts_runtime_attrs(module, base_cls)

    # DeepEP needs a one-time dispatcher built on the EP process group.
    if dispatch == "deepep" and hasattr(module, "init_ep_dispatcher"):
        module.init_ep_dispatcher()
    return module


def bind_ep_experts_model(
    model: nn.Module,
    dispatch: str = "alltoall",
    *,
    experts_cls: str,
) -> nn.Module:
    """Bind EP dispatch onto every Experts submodule in ``model``."""
    cls_name = experts_cls.strip()
    bound = 0
    for experts in iter_experts_modules(model, cls_name):
        if torch.distributed.get_rank() == 0:
            print(f"bind_ep_experts_model: {type(experts)}", flush=True)
        bind_ep_experts(experts, dispatch=dispatch)
        bound += 1
    if bound == 0:
        raise ValueError(f"No module with class name {cls_name!r} found in model.")
    return model
