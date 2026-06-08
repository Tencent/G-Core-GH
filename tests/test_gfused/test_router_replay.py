# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""CPU unit tests for deepseek_v4 router_replay module."""

import pytest
import torch
import torch.nn as nn

from gpatch_v4.models.deepseek_v4.router_replay import (
    RouterReplay,
    capture_routing_decisions,
)


# ------------------------------------------------------------------
# RouterReplay core
# ------------------------------------------------------------------


def test_replay_gather_and_grad():
    """Replay returns scores.gather(1, target) and gradients flow back."""
    rr = RouterReplay()
    target = torch.tensor([[2, 0], [1, 3]], dtype=torch.long)
    rr.set_target_indices(target)

    scores = torch.randn(2, 5, requires_grad=True)

    values, indices = rr.get_replay_topk(scores)
    assert torch.equal(indices, target)
    expected_values = scores.gather(1, target)
    assert torch.equal(values, expected_values)

    loss = values.sum()
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0


def test_replay_reuses_same_target_across_calls():
    """Multiple get_replay_topk calls re-read the same target (grad-ckpt safe)."""
    rr = RouterReplay()
    target = torch.tensor([[0, 1]], dtype=torch.long)
    rr.set_target_indices(target)
    scores = torch.randn(1, 5)

    _, got_a = rr.get_replay_topk(scores)
    _, got_b = rr.get_replay_topk(scores)
    assert torch.equal(got_a, target)
    assert torch.equal(got_b, target)


def test_get_replay_topk_without_target_asserts():
    """Calling without setting target raises AssertionError."""
    rr = RouterReplay()
    scores = torch.randn(2, 6)

    with pytest.raises(AssertionError, match="caller is responsible"):
        rr.get_replay_topk(scores)


# ------------------------------------------------------------------
# capture_routing_decisions
# ------------------------------------------------------------------


class _FakeTopKRouter(nn.Module):
    """Minimal mock matching DeepseekV4TopKRouter.forward signature."""

    def __init__(self, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.randn(num_experts, 8))

    def forward(self, x):
        logits = x @ self.weight.T
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, self.top_k, dim=-1)
        return logits, values, indices


# Override __name__ so capture_routing_decisions can match via type(inst).__name__
_FakeTopKRouter.__name__ = "DeepseekV4TopKRouter"


class _TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate0 = _FakeTopKRouter()
        self.gate1 = _FakeTopKRouter()

    def forward(self, x):
        _, _, idx0 = self.gate0(x)
        _, _, idx1 = self.gate1(x)
        return idx0, idx1


def test_capture_routing_decisions():
    model = _TwoLayerModel()
    x = torch.randn(3, 8)

    with capture_routing_decisions(model) as recorded:
        idx0, idx1 = model(x)

    assert len(recorded) == 2
    assert torch.equal(recorded[0], idx0)
    assert torch.equal(recorded[1], idx1)


def test_capture_hooks_removed_after_exit():
    model = _TwoLayerModel()
    x = torch.randn(3, 8)

    with capture_routing_decisions(model) as recorded:
        model(x)

    first_capture = [t.clone() for t in recorded]

    with torch.no_grad():
        model.gate0.weight.fill_(0.0)

    model(x)
    assert torch.equal(recorded[0], first_capture[0]), (
        "Hooks should be removed; recorded should NOT update after context exit"
    )
