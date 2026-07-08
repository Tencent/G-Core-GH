# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Router replay utilities for DeepSeek-V4 MoE.

Mirrors :mod:`gpatch_v4.models.qwen3_5_moe.router_replay` (intentionally — the
test scripts and replay mechanics stay symmetric across MoE forks). Two
complementary mechanisms:

1. **Observation** — :func:`capture_routing_decisions` uses PyTorch forward
   hooks to record per-layer routing indices from *any* model (upstream
   transformers or our fork) without modifying source code.

2. **Substitution** — :class:`RouterReplay` pins the top-k computation inside
   our fork's :class:`DeepseekV4TopKRouter.forward` to pre-recorded indices
   via :func:`enable_router_replay`.

Each ``DeepseekV4TopKRouter`` holds at most one :class:`RouterReplay`. Replay
is *enabled* by attaching an instance and populating its ``target_topk_idx``;
fully *disabled* by detaching the instance. For scoped use prefer
:func:`router_replay_ctx`, which handles both setup and teardown.

Note on DSV4 vs Qwen3.5-MoE: DSV4 has two router classes — :class:`DeepseekV4TopKRouter`
(learned top-k routing with ``e_score_correction_bias``) and
:class:`DeepseekV4HashRouter` (deterministic ``tid2eid[input_ids]`` lookup,
used by the first ``default_num_hash_layers`` layers). HashRouter is fully
input-determined and doesn't need replay — baseline and EP necessarily make
the same routing decisions for hash_moe layers. We only replay TopKRouter.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from typing import Generator


class RouterReplay:
    """Per-layer replay state, attached to a ``DeepseekV4TopKRouter`` module.

    Replay is active iff ``target_topk_idx is not None``. There is **no**
    global registry — instances live on their owning module and are discovered
    via ``model.modules()`` by :func:`enable_router_replay` /
    :func:`disable_router_replay` / :func:`router_replay_ctx`. State stays
    fully scoped to a model, so loading multiple models in the same process
    is safe.
    """
    def __init__(self) -> None:
        self.target_topk_idx: Optional[torch.Tensor] = None

    def set_target_indices(self, idx: torch.Tensor) -> None:
        """Set target indices; enables replay for this layer."""
        self.target_topk_idx = idx

    def get_replay_topk(
        self,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(values, indices)`` by gathering *scores* at the saved target.

        Caller MUST ensure ``self.target_topk_idx`` is set before calling;
        the no-target case is handled at the call site (see
        :meth:`DeepseekV4TopKRouter.forward`).
        """
        assert self.target_topk_idx is not None, (
            "target_topk_idx must be set before get_replay_topk; "
            "caller is responsible for the no-replay path"
        )
        indices = self.target_topk_idx.to(scores.device)
        return scores.gather(1, indices), indices


# ======================================================================
# layer-index helpers
# ======================================================================


def get_topk_layer_indices(config) -> list[int]:
    """Return decoder-layer indices that use ``DeepseekV4TopKRouter``.

    Hash-MoE layers (deterministic routing) are excluded — they don't
    need replay.  The returned indices index into the ``num_layers``
    dimension of the ``routed_experts`` tensor produced by the sampler.

    Parameters
    ----------
    config
        A ``DeepseekV4Config`` (or any object with ``mlp_layer_types``).
    """
    mlp_layer_types = config.mlp_layer_types
    return [i for i, t in enumerate(mlp_layer_types) if t != "hash_moe"]


def extract_topk_layers(
    routed_experts: torch.Tensor,
    topk_layer_indices: list[int],
) -> list[torch.Tensor]:
    """Slice ``routed_experts`` to only the TopKRouter layers.

    Parameters
    ----------
    routed_experts : Tensor
        Shape ``(seq_len, num_all_layers, topk)`` — from sampler.
    topk_layer_indices : list[int]
        Output of :func:`get_topk_layer_indices`.

    Returns
    -------
    list[Tensor]
        One ``(seq_len, topk)`` tensor per TopKRouter layer, in order.
    """
    return [routed_experts[:, i, :] for i in topk_layer_indices]


# ======================================================================
# enable / disable (our fork only)
# ======================================================================


def _iter_routers(model: nn.Module):
    """Yield each ``DeepseekV4TopKRouter`` in *model* (layer order).

    Skips ``DeepseekV4HashRouter`` — it's fully deterministic and doesn't need replay.
    """
    from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter

    for module in model.modules():
        if isinstance(module, DeepseekV4TopKRouter):
            yield module


def enable_router_replay(model: nn.Module) -> list[RouterReplay]:
    """Attach a :class:`RouterReplay` to every ``DeepseekV4TopKRouter`` in *model*.

    Uses ``isinstance`` for strict matching — only works on **our fork**.
    For upstream transformers, use :func:`capture_routing_decisions` instead.

    Returns the list of :class:`RouterReplay` instances in layer order
    (same order as ``model.modules()``). Idempotent: re-calling reuses
    existing instances.
    """
    instances: list[RouterReplay] = []
    for module in _iter_routers(model):
        if module.router_replay is None:
            module.router_replay = RouterReplay()
        instances.append(module.router_replay)
    assert instances, (
        "No DeepseekV4TopKRouter found in model; "
        "use capture_routing_decisions for upstream transformers models"
    )
    return instances


def disable_router_replay(model: nn.Module) -> None:
    """Set ``router_replay = None`` on every ``DeepseekV4TopKRouter`` in *model*."""
    for module in _iter_routers(model):
        module.router_replay = None


@contextmanager
def router_replay_ctx(
    model: nn.Module,
    per_layer_indices: list[torch.Tensor],
) -> Generator[None, None, None]:
    """Scope a router-replay session: attach + set on enter, detach on exit.

    Equivalent to::

        routers = enable_router_replay(model)
        for r, idx in zip(routers, per_layer_indices):
            r.set_target_indices(idx)
        try:
            ...
        finally:
            disable_router_replay(model)
    """
    routers = enable_router_replay(model)
    if len(routers) != len(per_layer_indices):
        raise ValueError(
            f"Expected {len(routers)} index tensors (one per layer), "
            f"got {len(per_layer_indices)}"
        )
    for r, idx in zip(routers, per_layer_indices):
        r.set_target_indices(idx)
    try:
        yield
    finally:
        disable_router_replay(model)


# ======================================================================
# capture_routing_decisions (works on ANY model)
# ======================================================================


@contextmanager
def capture_routing_decisions(
    model: nn.Module,
    *,
    router_class_names: tuple[str, ...] = ("DeepseekV4TopKRouter", ),
) -> Generator[list[Optional[torch.Tensor]], None, None]:
    """Record per-layer routing indices via forward hooks.

    Works on **any** model (our fork or upstream transformers) without
    modifying source code. Matches gate modules by ``type(module).__name__``.

    Only ``DeepseekV4TopKRouter`` is captured by default —
    :class:`DeepseekV4HashRouter` decisions are deterministic from
    ``input_ids`` and don't need replay. Pass a wider tuple if you want to
    capture HashRouter too (e.g. for debugging).

    Usage::

        with capture_routing_decisions(model) as recorded:
            outputs = model(input_ids=input_ids)
        # recorded[i] is a (seq_len, top_k) int64 tensor for the i-th
        # TopKRouter layer (in module-iteration order).

    The hooks are automatically removed when the context exits.
    """
    recorded: list[Optional[torch.Tensor]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for module in model.modules():
        if type(module).__name__ in router_class_names:
            slot = len(recorded)
            recorded.append(None)

            def _make_hook(i: int):
                def hook(mod, inp, out):
                    # out = (router_logits, router_scores, router_indices)
                    recorded[i] = out[2].detach().clone()

                return hook

            handles.append(module.register_forward_hook(_make_hook(slot)))

    try:
        yield recorded
    finally:
        for h in handles:
            h.remove()
