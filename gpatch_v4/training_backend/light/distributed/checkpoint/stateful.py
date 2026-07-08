"""DCP ``Stateful`` wrappers for FSDP2 model / optimizer / LR scheduler.

EP (Expert Parallelism) awareness
----------------------------------
When EP is enabled, ``ExpertsParallelPlan.apply()`` shards expert parameters
along the expert dimension (dim 0) across EP ranks and attaches a
``param.spec_info`` (SpecInfo) to each sharded parameter.  After FSDP2
wrapping with ``ep_fsdp_mesh``, the in-memory placement becomes::

    DTensor([Shard(1)], ep_fsdp_mesh_1d)   # FSDP shard along dim 1

DCP needs to see the **full 2-D placement** ``[Shard(0), Shard(1)]`` on the
2-D ``ep_device_mesh`` to correctly record the EP shard boundary and enable
resharding across different EP topologies.  We therefore:

  * **On save**: call :func:`_restore_ep_dim` to convert
    ``[Shard(1)] on ep_fsdp_mesh`` →
    ``[Shard(0), Shard(1)] on ep_device_mesh``.
  * **On load**: call :func:`_drop_ep_dim` to convert back.

EP parameters are identified via ``model._ep_fqns`` (a ``set[str]`` stored
by ``ExpertsParallelPlan.apply()`` before ``fully_shard()`` runs).

Optimizer state follows the same transformation: ``exp_avg`` / ``exp_avg_sq``
are sharded DTensors with the same FSDP placement, so they receive the same
restore / drop treatment as model weights.

EP + HSDP
---------
For EP HSDP, ``ep_device_mesh`` is 3-D ``("ep", "ep_fsdp_replicate",
"ep_fsdp_shard")`` and ``ep_fsdp_mesh`` is the 2-D sub-mesh
``("ep_fsdp_replicate", "ep_fsdp_shard")``.  After FSDP2 HSDP wrapping,
expert parameter placements are ``[Replicate(), Shard(1)]`` on the 2-D
``ep_fsdp_mesh``.

Both cases are handled uniformly by **dynamically prepending** ``Shard(0)``
to the tensor's *current* placement list rather than hard-coding
``[Shard(0), Shard(1)]``::

    Non-HSDP  [Shard(1)]              → [Shard(0), Shard(1)]
    HSDP      [Replicate(), Shard(1)] → [Shard(0), Replicate(), Shard(1)]

``_drop_ep_dim`` performs the reverse by stripping the leading ``Shard(0)``
from the loaded placement list.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Set

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

# EP-dimension (un)wrap helpers live with the EP sharding plan; re-exported here
# for backward compatibility with existing imports of this module.
from ..parallelizer.ep import (  # noqa: F401
    _assert_even_sharding,
    _build_ep_fqn_set,
    _drop_ep_dim,
    _restore_ep_dim,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# EP dimension helpers (state-dict level)
# ─────────────────────────────────────────────────────────────────────────────


def _apply_ep_dim_model(
    state_dict: dict,
    ep_fqns: Set[str],
    action: str,
    ep_device_mesh,
    ep_fsdp_mesh,
) -> dict:
    """Apply restore/drop EP dimension to expert parameters in a **model** state dict.

    The model state dict is a flat ``{fqn: tensor}`` mapping as returned by
    ``get_state_dict(model, optimizers=[])``.  We look up each EP FQN directly.
    """
    assert action in ("restore", "drop")
    for fqn in ep_fqns:
        if fqn not in state_dict:
            continue
        # print(f"_apply_ep_dim_model restore or drop {fqn}", flush=True)
        t = state_dict[fqn]
        if not (torch.is_tensor(t) or isinstance(t, DTensor)):
            continue
        if isinstance(t, torch.Tensor) and t.ndim == 0:
            continue  # scalar — skip
        if action == "restore":
            state_dict[fqn] = _restore_ep_dim(t, ep_device_mesh)
        else:
            state_dict[fqn] = _drop_ep_dim(t, ep_fsdp_mesh)
    return state_dict


def _apply_ep_dim_optim(
    optim_sd: dict,
    ep_fqns: Set[str],
    action: str,
    ep_device_mesh,
    ep_fsdp_mesh,
) -> dict:
    """Apply restore/drop EP dimension to expert parameters in an **optimizer** state dict.

    The optimizer state dict returned by ``get_state_dict(model, optimizer)``
    has the structure::

        {
            "state": {
                "<fqn>": {"step": Tensor, "exp_avg": DTensor, "exp_avg_sq": DTensor},
                ...
            },
            "param_groups": [...],
        }

    We match by FQN (exact key in ``"state"``) and transform every non-scalar
    state tensor for EP parameters.
    """
    assert action in ("restore", "drop")
    state = optim_sd.get("state", {})
    for param_key, param_state in state.items():
        if not isinstance(param_key, str) or param_key not in ep_fqns:
            continue

        # 浅 copy param_state, ovoid inplace modification
        param_state = copy.copy(param_state)
        # print(f"_apply_ep_dim_optim restore or drop {param_key}", flush=True)
        for state_name, sv in list(param_state.items()):
            if not (torch.is_tensor(sv) or isinstance(sv, DTensor)):
                continue
            if isinstance(sv, torch.Tensor) and sv.ndim == 0:
                continue  # scalar step — skip
            if action == "restore":
                param_state[state_name] = _restore_ep_dim(sv, ep_device_mesh)
            else:
                param_state[state_name] = _drop_ep_dim(sv, ep_fsdp_mesh)
        state[param_key] = param_state
    return optim_sd


# ─────────────────────────────────────────────────────────────────────────────
# Stateful wrappers
# ─────────────────────────────────────────────────────────────────────────────


class _EPAwareMixin:
    """Shared EP-awareness init logic for ModelState and OptimizerState."""
    def _init_ep_state(self, model: nn.Module) -> None:
        self._ep_fqns: Set[str] = _build_ep_fqn_set(model)
        self._ep_aware: bool = False
        self._ep_device_mesh = None
        self._ep_fsdp_mesh = None

        if not self._ep_fqns:
            return

        try:
            from ..core.parallel_state import get_parallel_state
            ps = get_parallel_state()
        except (RuntimeError, ImportError):
            return

        if ps.ep_device_mesh is None or not ps.config.ep_enabled:
            return

        self._ep_aware = True
        self._ep_device_mesh = ps.ep_device_mesh
        self._ep_fsdp_mesh = ps.ep_fsdp_mesh

        logger.debug(
            "EP-aware checkpoint enabled: %d EP params detected, "
            "ep_device_mesh=%s, ep_fsdp_mesh=%s",
            len(self._ep_fqns),
            self._ep_device_mesh,
            self._ep_fsdp_mesh,
        )


class ModelState(_EPAwareMixin, Stateful):
    """FSDP2 model state wrapper with optional EP-dimension restore/drop.

    On ``state_dict()`` the EP expert parameters are re-declared with their
    full ``[Shard(0), Shard(1)]`` placement on ``ep_device_mesh`` so DCP
    records the EP shard boundary.  On ``load_state_dict()`` the EP dimension
    is dropped before handing the tensors back to FSDP2's ``set_state_dict``.
    """
    def __init__(self, model: nn.Module, state_dict_preprocess=None) -> None:
        self.model = model
        # Optional ``{fqn: tensor} -> {fqn: tensor}`` hook applied to the *template*
        # state dict before DCP matches keys. Popping a key here means DCP will not
        # load it (the module keeps its freshly-(re)initialised value) — used for
        # runtime-regenerated buffers/params (e.g. sincos pos-embeds) so a
        # cross-resolution resume does not hit a shape mismatch. When set, the load
        # is non-strict (mirrors the legacy wgov3 ``FSDPCheckpoint``).
        self.state_dict_preprocess = state_dict_preprocess
        # Populated by ``load_state_dict``: the ``set_state_dict`` result
        # (``_IncompatibleKeys(missing_keys, unexpected_keys)``) so callers of
        # ``load_model`` can inspect which model params were not filled from the
        # checkpoint (e.g. the intentionally-popped runtime-only pos-embeds).
        self.load_result = None
        self._init_ep_state(model)

    def state_dict(self) -> dict:
        model_sd, _ = get_state_dict(self.model, optimizers=[])
        if self._ep_aware:
            model_sd = _apply_ep_dim_model(
                model_sd,
                self._ep_fqns,
                "restore",
                self._ep_device_mesh,
                self._ep_fsdp_mesh,
            )
        if self.state_dict_preprocess is not None:
            model_sd = self.state_dict_preprocess(model_sd)
        return {"model": model_sd}

    def load_state_dict(self, state_dict: dict) -> None:
        model_sd = state_dict["model"]
        if self._ep_aware:
            model_sd = _apply_ep_dim_model(
                model_sd,
                self._ep_fqns,
                "drop",
                self._ep_device_mesh,
                self._ep_fsdp_mesh,
            )
        self.load_result = set_state_dict(
            self.model,
            optimizers=[],
            model_state_dict=model_sd,
            optim_state_dict=None,
            options=StateDictOptions(strict=self.state_dict_preprocess is None),
        )


class OptimizerState(_EPAwareMixin, Stateful):
    """FSDP2 optimizer state wrapper with optional EP-dimension restore/drop.

    Expert Adam states (``exp_avg``, ``exp_avg_sq``) carry the same FSDP2
    shard placement as the model weights and receive identical EP-dim treatment.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state_dict_preprocess=None
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        # Optional hook on the optimizer state dict (``{"state": {fqn: {...}},
        # "param_groups": [...]}``) applied to the template before DCP matches keys.
        # Dropping a param's ``state`` entry here means DCP will not load it (the
        # optimizer keeps its freshly-initialised zero state) — used for
        # runtime-regenerated params so a cross-resolution resume does not hit a
        # shape mismatch. When set, the load is non-strict.
        self.state_dict_preprocess = state_dict_preprocess
        self._init_ep_state(model)

    def state_dict(self) -> dict:
        _, optim_sd = get_state_dict(self.model, optimizers=self.optimizer)
        if self._ep_aware:
            optim_sd = _apply_ep_dim_optim(
                optim_sd,
                self._ep_fqns,
                "restore",
                self._ep_device_mesh,
                self._ep_fsdp_mesh,
            )
        if self.state_dict_preprocess is not None:
            optim_sd = self.state_dict_preprocess(optim_sd)
        return {"optim": optim_sd}

    def load_state_dict(self, state_dict: dict) -> None:
        optim_sd = state_dict["optim"]
        if self._ep_aware:
            optim_sd = _apply_ep_dim_optim(
                optim_sd,
                self._ep_fqns,
                "drop",
                self._ep_device_mesh,
                self._ep_fsdp_mesh,
            )
        set_state_dict(
            self.model,
            optimizers=self.optimizer,
            model_state_dict=None,
            optim_state_dict=optim_sd,
            options=StateDictOptions(strict=self.state_dict_preprocess is None),
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state."""
    def __init__(self, lr_scheduler) -> None:
        self.lr_scheduler = lr_scheduler

    def state_dict(self) -> dict:
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
