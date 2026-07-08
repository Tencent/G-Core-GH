"""Expert parallelism sharding plan (used by ``ModelParallelizer.apply_fsdp``)."""

from __future__ import annotations

import contextlib
from typing import Dict, Set

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh

from ..core.utils import (
    check_fqn_match,
    module_matches_experts_cls,
    set_module_from_path,
    validate_experts_forward_signature,
)


def _fqn_to_wildcard_pattern(fqn: str) -> str:
    return ".".join("*" if part.isdigit() else part for part in fqn.split("."))


def _experts_module_prefixes(ep_plan: Dict[str, Shard]) -> set[str]:
    """Derive experts-module FQN prefixes (one per matched experts submodule)."""
    return {".".join(pattern.split(".")[:-1]) for pattern in ep_plan}


def build_experts_parallel_plan(model: nn.Module, experts_cls: str) -> "ExpertsParallelPlan":
    """Build EP shard plan from experts module class name and its parameters."""
    cls_name = experts_cls.strip()
    if not cls_name:
        raise ValueError("experts_cls is required and must be a non-empty class name string.")

    ep_plan: Dict[str, Shard] = {}
    matched = False
    for mod_name, module in model.named_modules():
        if not module_matches_experts_cls(module, cls_name, match_base_cls=True):
            continue
        validate_experts_forward_signature(module)
        matched = True
        mod_pattern = _fqn_to_wildcard_pattern(mod_name)
        for param_name, _ in module.named_parameters(recurse=True):
            ep_plan[f"{mod_pattern}.{param_name}"] = Shard(0)

    if not matched:
        raise ValueError(f"No module with class name {cls_name!r} found in model.")
    if not ep_plan:
        raise ValueError(f"Experts module {cls_name!r} has no parameters to shard.")
    return ExpertsParallelPlan(ep_plan)


class ExpertsParallelPlan:
    """Maps FQN patterns to EP shard placements before FSDP2 wrapping."""
    def __init__(self, ep_plan: Dict[str, Shard]):
        self.ep_plan = ep_plan
        self.ep_fsdp_no_shard_prefix = _experts_module_prefixes(ep_plan)

    def apply(self, model: nn.Module, ep_mesh: DeviceMesh, ep_fsdp_mesh: DeviceMesh) -> None:
        ep_size = ep_mesh.size()
        replicate = [Replicate() for _ in range(ep_mesh.ndim)]
        ep_fqns: set[str] = set()
        for fqn, param in list(model.named_parameters()):
            for pattern, shard in self.ep_plan.items():
                if not check_fqn_match(pattern, fqn):
                    continue
                assert param.size(
                    shard.dim
                ) % ep_size == 0, f"{fqn} dim {shard.dim} not divisible by ep_size"
                placement = replicate[:-1] + [shard]
                dtensor = DTensor.from_local(param.data, ep_mesh, replicate)
                local = torch.nn.Parameter(
                    dtensor.redistribute(ep_mesh, placement).to_local(),
                    requires_grad=param.requires_grad,
                )
                set_module_from_path(model, fqn, local)
                # strip _checkpoint_wrapped_module prefix
                fqn = fqn.replace("_checkpoint_wrapped_module.", "")
                ep_fqns.add(fqn)
                break

        # Store EP FQNs on the model *before* fully_shard() is called.
        # fully_shard() replaces every nn.Parameter with a new DTensor-backed
        # nn.Parameter, which drops any custom attributes (e.g. spec_info)
        # on the originals.  Plain module attributes survive FSDP wrapping, so
        # the checkpoint code can reliably read model._ep_fqns after training.
        model._ep_fqns: set[str] = ep_fqns

    def find_ep_modules(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Map experts-module FQNs to modules (for ``ep_fsdp`` FSDP wrap)."""
        modules: Dict[str, nn.Module] = {}
        for fqn, mod in model.named_modules():
            for prefix in self.ep_fsdp_no_shard_prefix:
                if check_fqn_match(prefix, fqn):
                    modules[fqn] = mod
                    break
        return modules


# ─────────────────────────────────────────────────────────────────────────────
# EP dimension (un)wrap helpers
# ─────────────────────────────────────────────────────────────────────────────
#
# After ``ExpertsParallelPlan.apply()`` shards experts on dim 0 across the EP
# mesh and ``fully_shard()`` then shards dim 1 across ``ep_fsdp_mesh``, an expert
# parameter's in-memory DTensor is::
#
#     DTensor([Shard(1)], ep_fsdp_mesh)            # non-HSDP
#     DTensor([Replicate(), Shard(1)], ep_fsdp_mesh)  # EP HSDP
#
# i.e. the EP shard dimension is *baked into* the local expert count and not
# represented in the mesh.  Code that needs to see the full EP shard boundary
# (EP-aware DCP save, or a streaming HF load that slices the *global* expert
# tensor) must temporarily prepend ``Shard(0)`` on the 2-/3-D ``ep_device_mesh``.
# These helpers are shared with ``..checkpoint.stateful``.


def _build_ep_fqn_set(model: nn.Module) -> Set[str]:
    """Return the set of FQNs for EP-sharded parameters.

    ``ExpertsParallelPlan.apply()`` stores ``model._ep_fqns`` (a plain
    ``set[str]``) on the model *before* ``fully_shard()`` is called, which is the
    only reliable source afterwards (FSDP wrapping drops custom param attrs).
    Returns an empty set when EP is not enabled (``_ep_fqns`` absent).
    """
    return getattr(model, "_ep_fqns", set())


def _assert_even_sharding(tensor: DTensor) -> None:
    """Fail-fast guard against uneven / padded shards before ``from_local``.

    ``DTensor.from_local`` infers the global shape as ``local_size * mesh_size``
    on each ``Shard`` dim without validation; padded/indivisible dims would
    silently corrupt the re-declared shape. This is a local, collective-free
    check: ``local * mesh`` must equal the true global size on every ``Shard`` dim.
    """
    mesh = tensor.device_mesh
    local_shape = tensor._local_tensor.shape
    for axis, p in enumerate(tensor.placements):
        if not isinstance(p, Shard):
            continue
        d = p.dim
        m = mesh.size(axis)
        expected = local_shape[d] * m
        if tensor.shape[d] != expected:
            raise RuntimeError(
                f"_restore_ep_dim: uneven sharding on tensor dim {d} "
                f"(mesh axis {axis}, size {m}): global={tensor.shape[d]} != "
                f"local({local_shape[d]}) * mesh({m})={expected}. "
                "EP checkpoint requires evenly-divisible expert/FSDP dims; "
                "padded/uneven shards would corrupt the checkpoint."
            )


def _restore_ep_dim(tensor: torch.Tensor, ep_device_mesh) -> torch.Tensor:
    """Prepend the EP shard dimension (``Shard(0)``) onto an FSDP-only tensor.

        Non-HSDP  [Shard(1)]              → [Shard(0), Shard(1)]              (2-D ep_device_mesh)
        HSDP      [Replicate(), Shard(1)] → [Shard(0), Replicate(), Shard(1)] (3-D ep_device_mesh)

    ``tensor._local_tensor`` is unchanged; only the declared placement widens so
    the EP shard boundary becomes visible. A plain ``torch.Tensor`` (EP without
    FSDP) is wrapped as ``[Shard(0)]`` on the 1-D ``"ep"`` sub-mesh.
    """
    if isinstance(tensor, DTensor):
        _assert_even_sharding(tensor)
        new_placements = [Shard(0)] + list(tensor.placements)
        if ep_device_mesh.ndim != len(new_placements):
            raise RuntimeError(
                f"ep_device_mesh.ndim={ep_device_mesh.ndim} does not match "
                f"expected placement length {len(new_placements)} "
                f"(FSDP placements={list(tensor.placements)}).  "
                "Check that ep_device_mesh matches the mesh used for FSDP wrapping."
            )
        return DTensor.from_local(tensor._local_tensor, ep_device_mesh, new_placements)
    elif torch.is_tensor(tensor):
        return DTensor.from_local(tensor, ep_device_mesh["ep"], [Shard(0)])
    raise RuntimeError(f"_restore_ep_dim: unexpected tensor type {type(tensor)!r}")


def _drop_ep_dim(tensor: torch.Tensor, ep_fsdp_mesh) -> torch.Tensor:
    """Remove the leading EP shard dimension, re-applying the rest on ``ep_fsdp_mesh``.

        Non-HSDP  [Shard(0), Shard(1)]              → [Shard(1)]              on ep_fsdp_mesh_1d
        HSDP      [Shard(0), Replicate(), Shard(1)] → [Replicate(), Shard(1)] on ep_fsdp_mesh_2d

    A single-placement (EP-only, no FSDP) tensor is unwrapped to a local tensor.
    """
    if not isinstance(tensor, DTensor):
        return tensor
    n = len(tensor.placements)
    if n >= 2:
        fsdp_placements = list(tensor.placements[1:])
        return DTensor.from_local(tensor._local_tensor, ep_fsdp_mesh, fsdp_placements)
    elif n == 1:
        return tensor.to_local()
    raise RuntimeError(f"_drop_ep_dim: unexpected placements {tensor.placements}")


@contextlib.contextmanager
def ep_full_placement_hooks(model: nn.Module):
    """Temporarily make EP experts present their *full* EP placement to state-dict I/O.

    Everything (the EP FQN set and the ``ep_device_mesh`` / ``ep_fsdp_mesh``) is
    derived from ``model`` + the registered parallel state, exactly like
    :class:`EPDimManager`. When EP is disabled (no expert params or no EP mesh) the
    context is a no-op, so callers can wrap unconditionally.

    With Expert Parallelism an expert param's in-memory ``DTensor`` lives on the
    ``ep_fsdp`` mesh with the EP shard dimension baked into its local expert count
    (dim0 == ``n_experts // ep_size``); the mesh does not carry the EP axis. While
    this context is active:

      * ``model.state_dict()`` post-hook widens each expert to
        ``[Shard(0), *fsdp_placements]`` on the 2-/3-D ``ep_device_mesh``
        (:func:`_restore_ep_dim`), so the template reports the param's *true* full
        global shape — usable as a redistribute reference for a streaming load.
      * ``model.load_state_dict()`` pre-hook narrows each incoming expert back to the
        real ``ep_fsdp`` placement (:func:`_drop_ep_dim`) so the in-place
        ``DTensor.copy_`` matches.

    Dense (non-expert) params are untouched by both hooks. Handles are removed on
    exit so the model's normal state-dict behaviour is restored. This is the hook
    counterpart of :class:`EPDimManager` for callers that want to drive the standard
    ``nn.Module`` state-dict machinery (e.g. an EP-aware broadcast-from-rank0 load)
    rather than transform state dicts by hand.
    """
    ep_fqns = _build_ep_fqn_set(model)
    ep_device_mesh = ep_fsdp_mesh = None
    if ep_fqns:
        try:
            from ..core.parallel_state import get_parallel_state

            ps = get_parallel_state()
        except (RuntimeError, ImportError):
            ps = None
        if ps is not None and ps.ep_device_mesh is not None and ps.config.ep_enabled:
            ep_device_mesh = ps.ep_device_mesh
            ep_fsdp_mesh = ps.ep_fsdp_mesh

    if ep_device_mesh is None:  # EP disabled — hooks would be no-ops.
        yield
        return

    norm_ep_fqns = {f.replace("_checkpoint_wrapped_module.", "") for f in ep_fqns}

    def _is_ep(key: str) -> bool:
        return key.replace("_checkpoint_wrapped_module.", "") in norm_ep_fqns

    def _state_dict_post_hook(module, state_dict, prefix, local_metadata):
        for key in list(state_dict.keys()):
            if _is_ep(key) and isinstance(state_dict[key], DTensor):
                state_dict[key] = _restore_ep_dim(state_dict[key], ep_device_mesh)

    def _load_state_dict_pre_hook(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for key in list(state_dict.keys()):
            if _is_ep(key) and isinstance(state_dict[key], DTensor):
                state_dict[key] = _drop_ep_dim(state_dict[key], ep_fsdp_mesh)

    sd_handle = model.register_state_dict_post_hook(_state_dict_post_hook)
    ld_handle = model.register_load_state_dict_pre_hook(_load_state_dict_pre_hook)
    try:
        yield
    finally:
        sd_handle.remove()
        ld_handle.remove()


class EPDimManager:
    """Per-key EP dimension (un)wrap helper.

    Both methods are non-inplace: they return a new DTensor (or the original
    tensor unchanged when EP is disabled / key not found), so callers can pass
    the result directly to ``distribute_tensor`` without touching the model params.

    Design note
    -----------
    The earlier ``restore_ep_context`` approach used ``param.data = new_dtensor``
    to patch the placement in-place.  That only updates the *C++ TensorImpl*
    (shape / stride / storage); ``DTensor.__slots__`` (``_spec``, ``_local_tensor``)
    are Python-level attributes that ``param.data =`` never reaches.  FSDP2 reads
    ``param._spec`` directly, so the placement change was silently a no-op.  The
    non-inplace approach used here sidesteps the problem entirely.

    No-op (``enabled=False``) when EP is disabled or the parallel state is absent.
    """
    def __init__(self, model: nn.Module) -> None:
        self._ep_fqns: Set[str] = _build_ep_fqn_set(model)
        self._ep_device_mesh = None
        self._ep_fsdp_mesh = None
        self._enabled = False

        if not self._ep_fqns:
            return
        try:
            from ..core.parallel_state import get_parallel_state
            ps = get_parallel_state()
        except (RuntimeError, ImportError):
            return
        if ps.ep_device_mesh is None or not ps.config.ep_enabled:
            return
        self._ep_device_mesh = ps.ep_device_mesh
        self._ep_fsdp_mesh = ps.ep_fsdp_mesh
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _normalize_key(self, key: str) -> str:
        return key.replace("_checkpoint_wrapped_module.", "")

    def may_add_ep_dim(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Return *tensor* with the EP shard dimension prepended if *key* is an EP parameter.

        Returns the original tensor unchanged when EP is disabled or *key* does not
        correspond to an EP-sharded parameter.
        """
        if not self._enabled or self._normalize_key(key) not in self._ep_fqns:
            return tensor

        return _restore_ep_dim(tensor, self._ep_device_mesh)

    def may_drop_ep_dim(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Return *tensor* with the leading EP shard dimension removed if *key* is an EP parameter.

        Returns the original tensor unchanged when EP is disabled or *key* does not
        correspond to an EP-sharded parameter.
        """
        if not self._enabled or self._normalize_key(key) not in self._ep_fqns:
            return tensor
        if not isinstance(tensor, DTensor):
            return tensor
        return _drop_ep_dim(tensor, self._ep_fsdp_mesh)
