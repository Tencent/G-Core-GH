"""Shared helpers."""

from __future__ import annotations

import fnmatch
import inspect
from typing import Iterator, Tuple

import torch.nn as nn

_CHECKPOINT_WRAPPED_MODULE = "_checkpoint_wrapped_module"
_EXPERTS_FORWARD_PARAMS = ("hidden_states", "top_k_index", "top_k_weights")


def check_fqn_match(pattern: str, fqn: str) -> bool:
    parts = pattern.split(".")
    fqn_parts = fqn.split(".")
    if len(parts) != len(fqn_parts):
        return False
    return all(fnmatch.fnmatch(f, p) for f, p in zip(fqn_parts, parts))


def get_module_from_path(model: nn.Module, path: str) -> nn.Module:
    mod = model
    for name in path.split("."):
        mod = getattr(mod, name)
    return mod


def set_module_from_path(model: nn.Module, path: str, value: nn.Parameter) -> None:
    parts = path.split(".")
    parent = model
    for name in parts[:-1]:
        parent = getattr(parent, name)
    setattr(parent, parts[-1], value)


def get_module_children_bottom_up(model: nn.Module) -> list[nn.Module]:
    modules = [model]
    for child in model.modules():
        if child is not model:
            modules.append(child)
    return modules


def module_matches_cls_name(module: nn.Module, cls_name: str, match_base_cls: bool = False) -> bool:
    """Match a module by exact class name (``type(module).__name__``)."""
    if match_base_cls:
        return any(cls.__name__ == cls_name for cls in module.__class__.__mro__)
    return module.__class__.__name__ == cls_name


def module_matches_layer_cls(module: nn.Module, cls_name: str) -> bool:
    """Match decoder layer by class name, including AC ``CheckpointWrapper`` shells.

    PyTorch registers the wrapped layer as ``_checkpoint_wrapped_module`` (see
    ``nn.Module.__setattr__``). FSDP should ``fully_shard`` the outer wrapper;
    skip the inner module via ``is_activation_checkpoint_inner_layer``.
    """
    if module_matches_cls_name(module, cls_name):
        return True
    wrapped = getattr(module, "_checkpoint_wrapped_module", None)
    return isinstance(wrapped, nn.Module) and module_matches_cls_name(wrapped, cls_name)


def is_activation_checkpoint_inner_layer(module_name: str) -> bool:
    """True when ``module_name`` is the inner layer registered by ``checkpoint_wrapper``."""
    return module_name.endswith(f".{_CHECKPOINT_WRAPPED_MODULE}")


def module_matches_experts_cls(
    module: nn.Module, cls_name: str, match_base_cls: bool = False
) -> bool:
    """Match experts module by exact class name (same as ``module_matches_cls_name``)."""
    return module_matches_cls_name(module, cls_name, match_base_cls)


def _unwrap_forward(func):
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def validate_experts_forward_signature(module: nn.Module) -> None:
    """Require HF-style experts ``forward(hidden_states, top_k_index, top_k_weights)``."""
    cls_name = module.__class__.__name__
    forward = _unwrap_forward(module.__class__.forward)
    if forward is nn.Module.forward:
        raise ValueError(
            f"Experts module {cls_name!r} must define forward("
            f"{', '.join(_EXPERTS_FORWARD_PARAMS)})."
        )

    params = [
        p.name
        for p in inspect.signature(forward).parameters.values() if p.name != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    missing = [name for name in _EXPERTS_FORWARD_PARAMS if name not in params]
    if missing:
        raise ValueError(
            f"Experts module {cls_name!r} forward must accept "
            f"{_EXPERTS_FORWARD_PARAMS}, missing {missing} (got {params})."
        )


def iter_experts_modules(model: nn.Module, experts_cls: str) -> Iterator[nn.Module]:
    """Yield Experts submodules matched by exact class name."""
    cls_name = experts_cls.strip()
    if not cls_name:
        raise ValueError("experts_cls is required and must be a non-empty class name string.")
    for module in model.modules():
        if module_matches_experts_cls(module, cls_name):
            validate_experts_forward_signature(module)
            yield module


def sort_fqn_by_submodule_first(fqns: list[str]) -> list[str]:
    return sorted(fqns, key=lambda x: (-x.count("."), x))
