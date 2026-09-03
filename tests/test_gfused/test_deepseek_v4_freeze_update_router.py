# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""CPU unit tests for DeepSeek-V4-Flash freeze / loss-free-update of the MoE router.

Exercises the real ``DeepseekV4TopKRouter`` + the real
``gpatch_v4/models/deepseek_v4/freeze_update_router.py`` helpers (the code the
``freeze_router_weight`` / ``freeze_router_correction_bias`` /
``router_correction_bias_update_speed`` training switches drive), covering:

1. ``freeze_router_weight`` switch — ``freeze_router_weights`` makes the
   router ``weight`` non-trainable so an optimizer step leaves it untouched;
   without it the weight updates normally.
2. ``freeze_router_correction_bias`` switch — when the update runs
   (``freeze_router_correction_bias=False``) ``e_score_correction_bias`` moves by the
   exact loss-free rule ``bias += sign(mean - load) * speed`` (numerically
   checked); when it is not run (frozen) the buffer never moves under a normal
   forward / backward / optimizer step.

Token counting is driven by a permanently-registered forward hook
(:func:`register_router_correction_bias_accum_tracking_hook`) that only
increments the per-expert accumulator while the _is_train_forward (a
``contextvars.ContextVar`` set by :func:`train_forward_context`) is ``True``.
Recompute during gradient checkpointing is therefore never double-counted;
any forward outside the training forward (eval, logprob, etc.) is guard by router.training

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_freeze_update_router.py
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from gpatch_v4.models.deepseek_v4.freeze_update_router import (
    _ACCUM_ATTR,
    train_forward_context,
    checkpoint_context_fn,
    freeze_router_weights,
    init_router_correction_bias_accumulators,
    register_router_correction_bias_accum_tracking_hook,
    reset_router_correction_bias_accum,
    update_router_correction_bias,
)
from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter

HIDDEN = 16
NUM_EXPERTS = 8
TOP_K = 2


@pytest.fixture(scope="module", autouse=True)
def _single_rank_pg():
    """world_size=1 gloo group: makes update_router_correction_bias's all_reduce a no-op
    and lets the real FSDP2 DCP save/load run single-process."""
    created = not dist.is_initialized()
    if created:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


def _router_config(
    num_experts: int = NUM_EXPERTS,
    top_k: int = TOP_K,
    hidden: int = HIDDEN,
    scoring_func: str = "sigmoid",
    routed_scaling_factor: float = 2.5,
) -> SimpleNamespace:
    """Minimal duck-typed config for ``DeepseekV4TopKRouter.__init__``."""
    return SimpleNamespace(
        num_experts_per_tok=top_k,
        num_local_experts=num_experts,
        hidden_size=hidden,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        moe_router_force_load_balancing=False,
    )


class _RouterModel(nn.Module):
    """A container of real TopK routers (mimics the backbone + MTP routers).

    ``scale`` stands in for the rest of a real model's trainable params so an
    optimizer built with ``filter(requires_grad)`` (as in ``setup_optimizer``)
    is never empty even when all router weights are frozen.
    """

    def __init__(self, num_routers: int = 2, **cfg):
        super().__init__()
        config = _router_config(**cfg)
        self.gates = nn.ModuleList(
            [DeepseekV4TopKRouter(config) for _ in range(num_routers)]
        )
        # weight is created via torch.empty -> initialize it deterministically.
        for gate in self.gates:
            nn.init.normal_(gate.weight, mean=0.0, std=0.02)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, hidden: torch.Tensor):
        return [gate(hidden) for gate in self.gates]

    def router_loss(self, hidden: torch.Tensor) -> torch.Tensor:
        """Differentiable scalar depending on every router weight + ``scale``."""
        outs = self(hidden)
        return sum(o[1].float().sum() for o in outs) * self.scale


def _counts(gate: nn.Module) -> torch.Tensor:
    return gate.get_buffer(_ACCUM_ATTR)


def _trainable_params(model: nn.Module):
    """Mirror fsdp2_backend/optimizer.py::setup_optimizer parameter filter."""
    return list(filter(lambda p: p.requires_grad, model.parameters()))


def _reference_updated_bias(bias: torch.Tensor, counts: torch.Tensor, speed: float):
    """Loss-free rule reproduced independently for numerical comparison."""
    avg = counts.sum() / counts.numel()
    return bias + torch.sign(avg - counts) * speed


def _enable_bias_tracking(model: nn.Module) -> None:
    """Set up the new counting path: accumulator buffers + the permanent
    forward hook. Counting itself only fires inside ``train_forward_context``.
    """
    init_router_correction_bias_accumulators(model)
    register_router_correction_bias_accum_tracking_hook(model)


# ---------------------------------------------------------------------------
# 1. freeze_router_weight switch
# ---------------------------------------------------------------------------


def test_freeze_switch_on_blocks_router_weight_update():
    """freeze ON -> weight.requires_grad False, excluded from optimizer, unchanged."""
    torch.manual_seed(0)
    model = _RouterModel(num_routers=2)

    freeze_router_weights(model)

    for gate in model.gates:
        assert gate.weight.requires_grad is False
    # The frozen router weights must not appear in the optimizer's param list.
    trainable = _trainable_params(model)
    assert all(gate.weight is not p for gate in model.gates for p in trainable)
    assert any(model.scale is p for p in trainable)

    before = [gate.weight.detach().clone() for gate in model.gates]
    optimizer = torch.optim.AdamW(trainable, lr=0.1)

    hidden = torch.randn(4, HIDDEN)
    loss = model.router_loss(hidden)
    loss.backward()
    optimizer.step()

    for gate, snap in zip(model.gates, before):
        assert gate.weight.grad is None, "frozen weight must not receive grad"
        assert torch.equal(gate.weight, snap), "frozen weight must not move"


def test_freeze_switch_off_allows_router_weight_update():
    """freeze OFF (default) -> weight trains and moves after an optimizer step."""
    torch.manual_seed(0)
    model = _RouterModel(num_routers=2)

    for gate in model.gates:
        assert gate.weight.requires_grad is True

    before = [gate.weight.detach().clone() for gate in model.gates]
    optimizer = torch.optim.AdamW(_trainable_params(model), lr=0.1)

    hidden = torch.randn(4, HIDDEN)
    loss = model.router_loss(hidden)
    loss.backward()
    for gate in model.gates:
        assert gate.weight.grad is not None
        assert gate.weight.grad.abs().sum() > 0
    optimizer.step()

    for gate, snap in zip(model.gates, before):
        assert not torch.equal(gate.weight, snap), "unfrozen weight should move"


# ---------------------------------------------------------------------------
# 2. freeze_router_correction_bias switch + numerical correctness of the update
# ---------------------------------------------------------------------------


def test_bias_tracking_counts_routed_tokens():
    """The forward hook (under train_forward_context) counts the assigned experts.

    The new tracking hook has no ``valid_mask`` / pad argument — it bincounts the
    router's top-k ``indices`` over every token routed during train forward.
    Verify the accumulator equals that bincount and totals
    ``n_tokens * top_k``.
    """
    torch.manual_seed(1)
    model = _RouterModel(num_routers=2)
    model.train()
    _enable_bias_tracking(model)

    hidden = torch.randn(6, HIDDEN)

    with train_forward_context():
        outs = model(hidden)

    for gate, out in zip(model.gates, outs):
        indices = out[2]  # [n_tokens, top_k], deterministic given fixed weights
        expected = torch.bincount(
            indices.reshape(-1), minlength=gate.num_experts
        )[: gate.num_experts].float()
        assert torch.equal(_counts(gate), expected)
        # every token contributes exactly top_k assignments.
        assert _counts(gate).sum().item() == hidden.shape[0] * gate.top_k


def test_bias_update_matches_reference_formula():
    """bias += sign(mean - load) * speed, checked numerically per router."""
    torch.manual_seed(2)
    model = _RouterModel(num_routers=3)
    model.train()
    init_router_correction_bias_accumulators(model)

    speed = 1e-3
    # Distinct non-trivial start bias + known counts per router.
    known_counts = [
        torch.tensor([10.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0]),
        torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),     # balanced -> no move
        torch.tensor([0.0, 0.0, 16.0, 0.0, 0.0, 16.0, 0.0, 0.0]),
    ]
    known_counts_t = torch.stack(known_counts, dim=0)
    start_bias = []
    for gate, counts in zip(model.gates, known_counts):
        gate.e_score_correction_bias.copy_(torch.randn(gate.num_experts) * 0.01)
        start_bias.append(gate.e_score_correction_bias.detach().clone())
        _counts(gate).copy_(counts)

    maxvio_max, maxvio_mean = update_router_correction_bias(model, speed)
    expected_maxvio = (known_counts_t.max(dim=-1).values - known_counts_t.mean(dim=-1)) / known_counts_t.mean(dim=-1).clamp_min(1e-12)
    assert torch.equal(maxvio_max, expected_maxvio.max())
    assert torch.equal(maxvio_mean, expected_maxvio.mean())

    for gate, counts, bias0 in zip(model.gates, known_counts, start_bias):
        expected = _reference_updated_bias(bias0, counts, speed)
        assert torch.allclose(gate.e_score_correction_bias, expected, atol=1e-7, rtol=0)

    # balanced router (all counts equal) has sign(mean - load) == 0 -> unchanged.
    assert torch.equal(model.gates[1].e_score_correction_bias, start_bias[1])


def test_bias_update_sign_direction():
    """Over-loaded experts get a smaller bias, under-loaded a larger bias."""
    torch.manual_seed(3)
    model = _RouterModel(num_routers=1)
    model.train()
    init_router_correction_bias_accumulators(model)

    speed = 0.5
    counts = torch.tensor([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # expert 0 hot
    _counts(model.gates[0]).copy_(counts)
    before = model.gates[0].e_score_correction_bias.detach().clone()

    update_router_correction_bias(model, speed)
    after = model.gates[0].e_score_correction_bias

    assert after[0] < before[0], "over-loaded expert bias must decrease"
    assert torch.all(after[1:] > before[1:]), "under-loaded expert bias must increase"


def test_bias_update_end_to_end_from_real_routing():
    """reset -> phase-scoped forward count -> update; result matches formula on observed counts."""
    torch.manual_seed(4)
    model = _RouterModel(num_routers=2)
    model.train()
    _enable_bias_tracking(model)
    for gate in model.gates:
        gate.e_score_correction_bias.copy_(torch.randn(gate.num_experts) * 0.01)

    speed = 2e-3
    hidden = torch.randn(32, HIDDEN)

    reset_router_correction_bias_accum(model)
    start_bias = [g.e_score_correction_bias.detach().clone() for g in model.gates]
    with train_forward_context():
        model(hidden)
    observed = [_counts(g).detach().clone() for g in model.gates]
    observed_t = torch.stack(observed, dim=0)

    maxvio_max, maxvio_mean = update_router_correction_bias(model, speed)
    expected_maxvio = (
        observed_t.max(dim=-1).values - observed_t.mean(dim=-1, dtype=torch.float64)
    ) / observed_t.mean(dim=-1, dtype=torch.float64).clamp_min(1e-12)
    assert torch.equal(maxvio_max, expected_maxvio.max())
    assert torch.equal(maxvio_mean, expected_maxvio.mean())

    for gate, counts, bias0 in zip(model.gates, observed, start_bias):
        # every token contributes exactly top_k assignments (no pad here).
        assert counts.sum().item() == hidden.shape[0] * gate.top_k
        expected = _reference_updated_bias(bias0, counts, speed)
        assert torch.allclose(gate.e_score_correction_bias, expected, atol=1e-7, rtol=0)


def test_bias_frozen_when_update_not_called():
    """freeze_router_correction_bias semantics: without the update, a normal train step
    never moves the buffer (forward + backward + optimizer.step touch only params)."""
    torch.manual_seed(5)
    model = _RouterModel(num_routers=2)
    model.train()
    _enable_bias_tracking(model)
    for gate in model.gates:
        gate.e_score_correction_bias.copy_(torch.randn(gate.num_experts) * 0.1)
    frozen = [g.e_score_correction_bias.detach().clone() for g in model.gates]

    optimizer = torch.optim.AdamW(_trainable_params(model), lr=0.1)
    hidden = torch.randn(8, HIDDEN)

    # Simulate a step with tracking on but the update deliberately skipped.
    reset_router_correction_bias_accum(model)
    with train_forward_context():
        loss = model.router_loss(hidden)
    loss.backward()
    optimizer.step()

    for gate, snap in zip(model.gates, frozen):
        assert torch.equal(gate.e_score_correction_bias, snap), (
            "e_score_correction_bias must stay frozen when the update is not run"
        )


def test_bias_update_skipped_in_eval():
    """eval routers are guarded out of the update (bias unchanged)."""
    torch.manual_seed(6)
    model = _RouterModel(num_routers=2)
    init_router_correction_bias_accumulators(model)
    model.eval()
    for gate in model.gates:
        _counts(gate).copy_(torch.tensor([9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    before = [g.e_score_correction_bias.detach().clone() for g in model.gates]

    update_router_correction_bias(model, 0.1)

    for gate, snap in zip(model.gates, before):
        assert torch.equal(gate.e_score_correction_bias, snap)


def test_counting_only_happens_inside_forward_phase():
    """Counting fires only within ``train_forward_context``.

    The tracking hook is registered permanently, but it only increments the
    accumulator while the _is_train_forward is ``True``. Forwards run in eval mode and
    forwards run in the recompute phase (gradient-checkpointing recompute)
    must NOT touch ``local_tokens_per_expert``. This is what keeps those forwards
    from polluting the per-step token counts and guards against double-counting
    on recompute.
    """
    torch.manual_seed(9)
    model = _RouterModel(num_routers=2)
    model.train()
    _enable_bias_tracking(model)
    fwd_ctx, recompute_ctx = checkpoint_context_fn()

    hidden = torch.randn(8, HIDDEN)

    # 1. forward outside train_forward_context -> hook is a no-op -> no counting.
    model(hidden)
    for gate in model.gates:
        assert _counts(gate).sum().item() == 0.0, "forward outside train_forward_context must not count"

    # 2. forward INSIDE train_forward_context -> counts exactly this forward.
    with fwd_ctx:
        model(hidden)
    counted = [_counts(gate).detach().clone() for gate in model.gates]
    for gate, snap in zip(model.gates, counted):
        assert snap.sum().item() == hidden.shape[0] * gate.top_k

    # 3. forward in the "recompute" phase -> hook is a no-op -> counts frozen
    #    (this is the gradient-checkpointing double-count guard).
    with recompute_ctx:
        model(hidden)
    for gate, snap in zip(model.gates, counted):
        assert torch.equal(_counts(gate), snap), (
            "recompute-phase forward must not count (double-count guard)"
        )

    # 4. more forwards outside train_forward_context -> counts still frozen.
    for _ in range(3):
        model(hidden)
    for gate, snap in zip(model.gates, counted):
        assert torch.equal(_counts(gate), snap), (
            "forward outside train_forward_context must not count"
        )
