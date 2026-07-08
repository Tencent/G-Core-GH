"""Shared utilities."""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

from .device import get_device_module


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility (replaces ``accelerate.utils.set_seed``)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    get_device_module().manual_seed_all(seed)


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False) -> Iterator[None]:
    """Initialize modules with parameters (and optionally buffers) on ``device``.

    Adapted from ``accelerate.big_modeling.init_on_device``.
    """
    if include_buffers:
        with device:
            yield
        return

    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            _is_hf_initialized = kwargs.pop("_is_hf_initialized", None)
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            if _is_hf_initialized is not None:
                module._parameters[name]._is_hf_initialized = _is_hf_initialized

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


@contextmanager
def init_empty_weights(include_buffers: bool = False) -> Iterator[None]:
    """Build a model on the meta device without allocating real weight memory.

    Adapted from ``accelerate.big_modeling.init_empty_weights``.
    """
    with init_on_device(torch.device("meta"), include_buffers=include_buffers):
        yield


def _module_is_expert(module: nn.Module) -> bool:
    """True if any direct parameter is an EP-sharded ``DTensor`` (on an ``ep_fsdp`` mesh).

    After ``parallelize_model`` expert params live on the ``ep_fsdp`` mesh (dim
    names ``ep_fsdp`` or ``ep_fsdp_replicate``/``ep_fsdp_shard``), while dense
    params live on the dense FSDP mesh (``dp_shard`` / ``dp_replicate`` / ``cp``).
    The mesh dim name is therefore enough to tell experts apart, without needing
    the parallel state here.
    """
    from torch.distributed.tensor import DTensor

    for param in module.parameters(recurse=False):
        if isinstance(param, DTensor):
            names = param.device_mesh.mesh_dim_names or ()
            if any(name.startswith("ep_fsdp") for name in names):
                return True
    return False


def _split_owner_pname(root: nn.Module, fname: str) -> tuple:
    """Return (owner_module, param_name) for a dotted parameter path."""
    if "." in fname:
        *parts, pname = fname.split(".")
        owner: nn.Module = root
        for part in parts:
            owner = getattr(owner, part)
        return owner, pname
    return root, fname


def _reinit_subtree_params(module: nn.Module, *, seed: int, device: torch.device) -> None:
    """Re-initialize an entire module subtree in a shard-correct way.

    Preserves original ``nn.Parameter`` object identity (hooks, grad metadata, etc.).

    Algorithm:
    1. For every DTensor param in the subtree, save the original Parameter object
       and swap it out with a temporary full-shape CPU tensor so that
       ``reset_parameters()`` sees global shapes (needed for correct fan_in/fan_out).
    2. Seed the RNG and call ``module.reset_parameters()``.
    3. Shard each initialized full tensor back to the DTensor's mesh/placements,
       then copy the local shard **into the original DTensor param's local storage**
       via ``to_local().copy_()``.
    4. Restore the original Parameter object pointer in ``_parameters`` — the object
       identity is unchanged so any attached hooks or metadata survive intact.
    """
    from torch.distributed.tensor import DTensor, distribute_tensor

    # Phase 1: swap DTensor params for temp full-shape CPU tensors.
    # saved: fname -> (original_param, mesh, placements)
    saved: dict[str, tuple] = {}
    for fname, param in list(module.named_parameters(recurse=True)):
        if not isinstance(param, DTensor):
            continue
        saved[fname] = (param, param.device_mesh, param.placements)
        owner, pname = _split_owner_pname(module, fname)
        owner._parameters[pname] = nn.Parameter(
            torch.empty(param.shape, dtype=param.dtype),  # full-shape, CPU
            requires_grad=param.requires_grad,
        )

    # Phase 2: seed + initialize (operates on full-shape CPU tensors).
    set_seed(seed)
    module.reset_parameters()

    # Phase 3 + 4: copy initialized values into original DTensor local storage,
    # then restore the original Parameter objects.
    for fname, (orig_param, mesh, placements) in saved.items():
        owner, pname = _split_owner_pname(module, fname)
        full_initialized = owner._parameters[pname].data  # full CPU, just initialized
        local_shard = distribute_tensor(full_initialized, mesh, placements).to_local()
        with torch.no_grad():
            orig_param.to_local().copy_(local_shard.to(device))
        owner._parameters[pname] = orig_param  # restore original Parameter object


def _reset_tree(
    module: nn.Module,
    *,
    seed: int,
    device: torch.device,
    ep_rank: int,
    index: list,
) -> None:
    """DFS traversal for :func:`reset_meta_parameters`.

    * If ``module`` defines ``reset_parameters``: call :func:`_reinit_subtree_params`
      (which un-shards the subtree, calls ``reset_parameters``, re-shards) and
      **stop recursing** — ``reset_parameters`` is responsible for the whole subtree.
    * Otherwise: recurse into children, then apply a safe fallback for any direct
      params that remain uninitialised (e.g. ``RMSNorm.weight`` from diffusers).

    Safe fallback heuristic (no ``reset_parameters``):
      * 2-D+ params → ``kaiming_uniform``
      * 1-D params named ``*bias*`` / ``*beta*`` → zeros
      * 1-D weight/scale/gamma → ones
    """
    from torch.distributed.tensor import DTensor, distribute_tensor

    reset_fn = getattr(module, "reset_parameters", None)
    if callable(reset_fn):
        idx = index[0]
        index[0] += 1
        module_seed = (seed + idx) & 0x7FFF_FFFF
        if _module_is_expert(module):
            module_seed = (module_seed + (ep_rank + 1) * 1_000_003) & 0x7FFF_FFFF
        _reinit_subtree_params(module, seed=module_seed, device=device)
        return  # reset_parameters handles the entire subtree; do not recurse.

    for child in module.children():
        _reset_tree(child, seed=seed, device=device, ep_rank=ep_rank, index=index)

    # Safe fallback: initialise any direct params of this node that were not
    # covered by a child's reset_parameters (e.g. RMSNorm.weight from diffusers).
    # The original Parameter object is never replaced — only its underlying data is
    # written in-place so hooks and grad metadata are preserved.
    for pname, param in module.named_parameters(recurse=False):
        if param is None:
            continue
        if isinstance(param, DTensor):
            # Create a full-shape CPU tensor, apply init with correct global shape,
            # then shard and copy into the DTensor's local storage in-place.
            full = torch.empty(param.shape, dtype=param.dtype)
            if full.dim() >= 2:
                nn.init.kaiming_uniform_(full, a=math.sqrt(5))
            elif any(k in pname for k in ("bias", "beta")):
                nn.init.zeros_(full)
            else:
                nn.init.ones_(full)
            local_shard = distribute_tensor(full, param.device_mesh, param.placements).to_local()
            with torch.no_grad():
                param.to_local().copy_(local_shard.to(device))
        else:
            # Regular tensor: init directly in-place on param.data.
            data = param.data
            if data.dim() >= 2:
                nn.init.kaiming_uniform_(data, a=math.sqrt(5))
            elif any(k in pname for k in ("bias", "beta")):
                nn.init.zeros_(data)
            else:
                nn.init.ones_(data)


def reset_meta_parameters(
    model: nn.Module,
    *,
    device: torch.device,
    seed: int = 0,
    parallel_state: Optional[object] = None,
    buffer_device: Optional[torch.device] = None,
) -> nn.Module:
    """Materialize an already-parallelized meta model and re-init its weights.

    From-scratch counterpart to Accelerate's checkpoint-broadcast path, for the
    *post-* :func:`parallelize_model` state where params are ``DTensor`` (dense on
    the FSDP mesh, experts on the ``ep_fsdp`` mesh). Allocates real storage via
    ``to_empty`` then walks the module tree with :func:`_reset_tree`:

    * Modules that define ``reset_parameters()`` are treated as initialisation
      roots — the function is called once for the entire subtree and the traversal
      stops there (DFS early-stop).  Implement ``reset_parameters`` on a few large
      parent modules (e.g. a transformer block) rather than on every leaf.
    * Modules without ``reset_parameters`` are recursed into; any direct params that
      remain after children are handled receive a safe default init (kaiming / ones /
      zeros based on name heuristic).

    EP / DP correctness:

    * **DP (dense params):** seeded from ``seed`` + DFS visit index — identical on
      every rank so shards agree.
    * **EP (expert params):** seed additionally offset by ``ep_rank`` so different
      expert-parallel groups initialize distinct experts.

    Args:
        model: Parallelized model whose params are ``DTensor`` (possibly on meta).
        device: Target device for parameters / buffers.
        seed: Base RNG seed; combined with each module's DFS visit index.
        parallel_state: Provides ``ep_rank``; auto-fetched when model has expert params.
        buffer_device: Optional device for buffers (defaults to ``device``).

    Returns:
        The same ``model`` instance, materialized and initialized in-place.
    """
    buffer_device = buffer_device or device

    # Persist buffer values that were computed on CPU during __init__ (e.g. RoPE freqs
    # registered with persistent=False).  model.to_empty() allocates empty storage for
    # ALL tensors including non-meta buffers, so we must save and restore them.
    saved_buffers = {n: b.detach().clone() for n, b in model.named_buffers() if not b.is_meta}

    model.to_empty(device=device)

    # Restore saved buffer values; move to buffer_device in the same pass.
    for n, b in model.named_buffers():
        if n in saved_buffers:
            b.data.copy_(saved_buffers[n].to(buffer_device))
        elif b.device != buffer_device:
            b.data = b.data.to(buffer_device)
    del saved_buffers

    has_experts = any(_module_is_expert(m) for m in model.modules())
    ep_rank = 0
    if has_experts:
        if parallel_state is None:
            from ..distributed.core.parallel_state import get_parallel_state
            parallel_state = get_parallel_state()
        ep_rank = parallel_state.ep_rank

    _reset_tree(model, seed=seed, device=device, ep_rank=ep_rank, index=[0])
    set_seed(seed)
    return model


def sync_weights(from_model, to_model):
    for from_p, to_p in zip(from_model.parameters(), to_model.parameters()):
        to_p.data.copy_(from_p.data)

    named_buffers = {
        name.replace("._checkpoint_wrapped_module", ""): p
        for name, p in from_model.named_buffers()
    }

    for name, to_p in to_model.named_buffers():
        name = name.replace("._checkpoint_wrapped_module", "")
        try:
            to_p.data.copy_(named_buffers[name].data)
        except Exception as e:
            print(f"Failed to sync buffer {name} {named_buffers.keys()}: {e}", flush=True)
            raise e


@contextmanager
def to_meta_context(model):
    """
    Context manager to move model to meta device and restore original data.
    """
    parameters = [p.detach() for p in list(model.parameters())]
    named_buffers = {name: p.detach() for name, p in model.named_buffers()}
    try:
        model.to_empty(device="meta")
        yield
    finally:
        for (p, meta_p) in zip(parameters, model.parameters()):
            meta_p._data = p.data
        for (name, meta_p) in model.named_buffers():
            meta_p._data = named_buffers[name].data

        del parameters
        del named_buffers

        def apply(t):
            data = t._data
            del t._data
            return data

        model._apply(apply)
