# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""CPU unit tests for deepseek_v4 router_replay module."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from gpatch_v4.models.deepseek_v4.router_replay import (
    RouterReplay,
    capture_routing_decisions,
    extract_topk_layers,
    get_topk_layer_indices,
    router_replay_ctx,
)

# _iter_routers 用 isinstance(module, DeepseekV4TopKRouter) 匹配,
# 但 CPU 测试用 _FakeTopKRouter. 这里 patch 成按 router_replay 属性匹配.
_PATCH_ITER = patch(
    "gpatch_v4.models.deepseek_v4.router_replay._iter_routers",
    side_effect=lambda model: (
        m for m in model.modules() if hasattr(m, "router_replay")
        and not isinstance(m, nn.Sequential) and m is not model
    ),
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
        self.router_replay: "RouterReplay | None" = None

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        logits = flat @ self.weight.T
        scores = torch.softmax(logits, dim=-1)
        rr = self.router_replay
        if rr is not None and rr.target_topk_idx is not None:
            values, indices = rr.get_replay_topk(scores)
        else:
            values, indices = torch.topk(scores, self.top_k, dim=-1)
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


# ------------------------------------------------------------------
# get_topk_layer_indices / extract_topk_layers
# ------------------------------------------------------------------


def test_get_topk_layer_indices_basic():
    # 模拟 8 层: 前 2 层 hash, 其余 topk
    cfg = SimpleNamespace(
        mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe", "moe", "moe", "moe", "moe"]
    )
    indices = get_topk_layer_indices(cfg)
    assert indices == [2, 3, 4, 5, 6, 7]


def test_get_topk_layer_indices_all_topk():
    cfg = SimpleNamespace(mlp_layer_types=["moe", "moe", "moe"])
    indices = get_topk_layer_indices(cfg)
    assert indices == [0, 1, 2]


def test_get_topk_layer_indices_all_hash():
    cfg = SimpleNamespace(mlp_layer_types=["hash_moe", "hash_moe"])
    indices = get_topk_layer_indices(cfg)
    assert indices == []


def test_get_topk_layer_indices_interleaved():
    # hash, topk, hash, topk
    cfg = SimpleNamespace(
        mlp_layer_types=["hash_moe", "moe", "hash_moe", "moe"]
    )
    indices = get_topk_layer_indices(cfg)
    assert indices == [1, 3]


def test_extract_topk_layers():
    num_layers, topk = 6, 2
    seq_len = 10
    re = torch.arange(seq_len * num_layers * topk).reshape(seq_len, num_layers, topk)
    topk_indices = [1, 3, 5]
    result = extract_topk_layers(re, topk_indices)
    assert len(result) == 3
    for i, layer_i in enumerate(topk_indices):
        assert torch.equal(result[i], re[:, layer_i, :])


# ------------------------------------------------------------------
# router_replay_ctx 完整性
# ------------------------------------------------------------------


@_PATCH_ITER
def test_router_replay_ctx_pins_routing(_):
    """router_replay_ctx 应当固定 TopKRouter 的路由结果到给定 indices。"""
    model = _TwoLayerModel()
    x = torch.randn(3, 8)

    # 1. 正常 forward → 记录 baseline routing
    idx0_baseline, idx1_baseline = model(x)

    # 2. 构造一个"完全不同"的 replay target
    num_experts = model.gate0.weight.shape[0]
    fake_target_0 = (idx0_baseline + 1) % num_experts
    fake_target_1 = (idx1_baseline + 1) % num_experts

    # 3. replay → 路由应被 pin 到 fake_target
    with router_replay_ctx(model, [fake_target_0, fake_target_1]):
        _, _, replayed_0 = model.gate0(x)
        _, _, replayed_1 = model.gate1(x)
    assert torch.equal(replayed_0, fake_target_0)
    assert torch.equal(replayed_1, fake_target_1)

    # 4. 退出 ctx 后路由恢复正常
    idx0_after, idx1_after = model(x)
    assert torch.equal(idx0_after, idx0_baseline)
    assert torch.equal(idx1_after, idx1_baseline)


@_PATCH_ITER
def test_router_replay_ctx_survives_double_forward(_):
    """模拟 gradient checkpointing: forward 两次，replay indices 不变。"""
    model = _TwoLayerModel()
    x = torch.randn(4, 8)
    num_experts = model.gate0.weight.shape[0]

    target_0 = torch.randint(0, num_experts, (4, 2))
    target_1 = torch.randint(0, num_experts, (4, 2))

    with router_replay_ctx(model, [target_0, target_1]):
        # 第一次 forward
        _, _, idx0_first = model.gate0(x)
        _, _, idx1_first = model.gate1(x)
        # 第二次 forward（模拟 recompute）
        _, _, idx0_second = model.gate0(x)
        _, _, idx1_second = model.gate1(x)

    assert torch.equal(idx0_first, target_0)
    assert torch.equal(idx0_second, target_0)
    assert torch.equal(idx1_first, target_1)
    assert torch.equal(idx1_second, target_1)


@_PATCH_ITER
def test_router_replay_ctx_must_cover_backward_recompute(_):
    """FSDP2 GRPO 场景：ctx 在 backward 前退出则 recompute 拿不到 expert_idx。"""
    from torch.utils.checkpoint import checkpoint

    model = _TwoLayerModel()
    x = torch.randn(4, 8, requires_grad=True)
    num_experts = model.gate0.weight.shape[0]
    target_0 = torch.randint(0, num_experts, (4, 2))
    target_1 = torch.randint(0, num_experts, (4, 2))

    def _run_with_hook(backward_inside_ctx: bool) -> list[torch.Tensor]:
        captured: list[torch.Tensor] = []

        def _gate0_hook(mod, inp, out):
            captured.append(out[2].detach().clone())

        handle = model.gate0.register_forward_hook(_gate0_hook)
        try:
            with router_replay_ctx(model, [target_0, target_1]):
                weights0 = checkpoint(model.gate0, x, use_reentrant=False)[1]
                weights1 = checkpoint(model.gate1, x, use_reentrant=False)[1]
                loss = weights0.sum() + weights1.sum()
                if backward_inside_ctx:
                    loss.backward()
                else:
                    pass
            if not backward_inside_ctx:
                loss.backward()
        finally:
            handle.remove()
        return captured

    outside = _run_with_hook(backward_inside_ctx=False)
    assert len(outside) == 2
    assert torch.equal(outside[0], target_0)
    assert not torch.equal(outside[1], target_0), (
        "recompute must not fall back to natural topk when replay indices are pinned"
    )

    model.zero_grad(set_to_none=True)
    x.grad = None
    inside = _run_with_hook(backward_inside_ctx=True)
    assert len(inside) == 2
    assert torch.equal(inside[0], target_0)
    assert torch.equal(inside[1], target_0)


@_PATCH_ITER
def test_router_replay_ctx_grad_flows(_):
    """replay 路径下 scores 仍然收到梯度。"""
    model = _TwoLayerModel()
    x = torch.randn(2, 8, requires_grad=True)
    target_0 = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    target_1 = torch.tensor([[1, 0], [3, 2]], dtype=torch.long)

    with router_replay_ctx(model, [target_0, target_1]):
        logits0, weights0, _ = model.gate0(x)
        logits1, weights1, _ = model.gate1(x)
    loss = weights0.sum() + weights1.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


@_PATCH_ITER
def test_router_replay_ctx_mismatched_layers_raises(_):
    """per_layer_indices 数量与 router 数量不匹配时 raise。"""
    model = _TwoLayerModel()
    with pytest.raises(ValueError, match="Expected 2"):
        with router_replay_ctx(model, [torch.zeros(1, 2, dtype=torch.long)]):
            pass


# ------------------------------------------------------------------
# _prepare_replay_indices (FSDP2 RouterReplayMixin)
# ------------------------------------------------------------------

@contextmanager
def _mock_mpu(cp_size=1, cp_rank=0, dp_rank=0):
    """Mock megatron.core.mpu 并行度函数。"""
    with patch("gpatch_v4.training_backend.fsdp2_backend.mixin.mpu") as mock:
        mock.get_context_parallel_world_size.return_value = cp_size
        mock.get_context_parallel_rank.return_value = cp_rank
        mock.get_data_parallel_rank.return_value = dp_rank
        yield mock


def _make_mixin(mlp_layer_types):
    """构造一个最小的 RouterReplayMixin 实例。"""
    from gpatch_v4.training_backend.fsdp2_backend.mixin import RouterReplayMixin

    class _Stub(RouterReplayMixin):
        pass

    stub = _Stub()
    stub.hf_config = SimpleNamespace(mlp_layer_types=mlp_layer_types)
    stub.config = SimpleNamespace(
        training=SimpleNamespace(moe_router_replay=True)
    )
    return stub


def _replay_index_map(seq_length: int, seq_batch: int, dp_rank: int = 0) -> torch.Tensor:
    full_index = torch.arange(seq_length)
    return (full_index + (full_index // seq_batch) * 6 * dp_rank) % seq_batch


def _manual_prepare_replay_indices(
    batches,
    seq_length: int,
    topk_layer_indices: list[int],
    dp_rank: int = 0,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> list[torch.Tensor]:
    """Reference implementation mirroring mixin._prepare_replay_indices."""
    per_layer_accum: list[list[torch.Tensor]] = [[] for _ in topk_layer_indices]
    for batch in batches:
        re = batch["routed_experts"]
        seq_batch = re.shape[0] - 1
        index = _replay_index_map(seq_length, seq_batch, dp_rank)
        if cp_size > 1:
            s_local = seq_length // cp_size
            start = cp_rank * s_local
            index = index[start:start + s_local]
        padded = re[index]
        for out_i, layer_i in enumerate(topk_layer_indices):
            per_layer_accum[out_i].append(padded[:, layer_i, :])
    return [
        torch.cat(chunks, dim=0).contiguous().long()
        for chunks in per_layer_accum
    ]


def _assert_replay_prepared_matches_captured(
    mixin,
    model,
    batches,
    seq_length: int,
    x: torch.Tensor,
    *,
    cp_size: int = 1,
    cp_rank: int = 0,
    dp_rank: int = 0,
) -> None:
    """R3 contract: prepared indices == capture_routing_decisions during replay."""
    with _mock_mpu(cp_size=cp_size, cp_rank=cp_rank, dp_rank=dp_rank):
        prepared = mixin._prepare_replay_indices(batches, seq_length)
        assert prepared is not None
        with mixin._maybe_router_replay(model, batches, seq_length):
            with capture_routing_decisions(model) as recorded:
                model(x)
    assert len(prepared) == len(recorded)
    for layer_i, (prep, cap) in enumerate(zip(prepared, recorded)):
        assert cap is not None, f"layer {layer_i} routing not captured"
        assert torch.equal(prep, cap.long()), (
            f"layer {layer_i} prepared != captured\n"
            f"prepared={prep}\ncaptured={cap}"
        )


def test_prepare_replay_indices_no_cp():
    """CP=1 时 indices 形状 = (B * seq_length, topk)。"""
    # 4 层: 1 hash + 3 topk
    mixin = _make_mixin(["hash_moe", "moe", "moe", "moe"])
    topk = 2
    sample_seq = 10
    seq_length = 16
    B = 3

    re = torch.randint(0, 64, (sample_seq, 4, topk))
    batches = [{"routed_experts": re.clone()} for _ in range(B)]

    with _mock_mpu(cp_size=1, cp_rank=0, dp_rank=0):
        result = mixin._prepare_replay_indices(batches, seq_length)

    assert result is not None
    assert len(result) == 3  # 3 个 TopK 层
    for t in result:
        assert t.shape == (B * seq_length, topk)
        assert t.dtype == torch.long


def test_prepare_replay_indices_with_cp():
    """CP=2 时 indices 形状 = (B * seq_length // 2, topk)。"""
    mixin = _make_mixin(["moe", "moe"])
    topk = 2
    sample_seq = 10
    seq_length = 16
    B = 2
    cp_size = 2

    re = torch.randint(0, 64, (sample_seq, 2, topk))
    batches = [{"routed_experts": re.clone()} for _ in range(B)]

    # rank 0
    with _mock_mpu(cp_size=cp_size, cp_rank=0, dp_rank=0):
        result_r0 = mixin._prepare_replay_indices(batches, seq_length)
    # rank 1
    with _mock_mpu(cp_size=cp_size, cp_rank=1, dp_rank=0):
        result_r1 = mixin._prepare_replay_indices(batches, seq_length)

    s_local = seq_length // cp_size
    for t in result_r0:
        assert t.shape == (B * s_local, topk)
    for t in result_r1:
        assert t.shape == (B * s_local, topk)


def test_prepare_replay_indices_cp_contiguous_split():
    """验证 CP 切分后的 indices 严格对应 contiguous split 的 token 位置。"""
    mixin = _make_mixin(["moe"])
    topk = 1
    sample_seq = 8
    seq_length = 8
    cp_size = 2

    # routed_experts 每个位置值不同，方便验证
    re = torch.arange(sample_seq).reshape(sample_seq, 1, 1).expand(-1, 1, topk)
    batches = [{"routed_experts": re}]

    with _mock_mpu(cp_size=cp_size, cp_rank=0, dp_rank=0):
        r0 = mixin._prepare_replay_indices(batches, seq_length)
    with _mock_mpu(cp_size=cp_size, cp_rank=1, dp_rank=0):
        r1 = mixin._prepare_replay_indices(batches, seq_length)

    # rank 0 应该拿 positions [0,1,2,3], rank 1 拿 [4,5,6,7]
    # 但实际值经过 (idx + ...) % (sample_seq - 1) 的 padding 公式
    # 由于 dp_rank=0, 公式退化为 idx % (sample_seq - 1)
    # positions [0,1,2,3] → indices [0,1,2,3]
    # positions [4,5,6,7] → indices [4,5,6,0]  (7 % 7 = 0)
    expected_r0 = re[torch.tensor([0, 1, 2, 3]) % (sample_seq - 1), 0, :]
    expected_r1 = re[torch.tensor([4, 5, 6, 0]), 0, :]

    assert torch.equal(r0[0], expected_r0.long())
    assert torch.equal(r1[0], expected_r1.long())


def test_prepare_replay_indices_batch_order():
    """验证 micro-batch 内多个 sample 的拼接顺序 = batch-major（匹配 HF hidden_states.reshape(-1, D)）。"""
    mixin = _make_mixin(["moe"])
    topk = 1
    seq_length = 4

    # sample 0: routing 全是 10
    re0 = torch.full((seq_length, 1, topk), 10)
    # sample 1: routing 全是 20
    re1 = torch.full((seq_length, 1, topk), 20)
    batches = [{"routed_experts": re0}, {"routed_experts": re1}]

    with _mock_mpu(cp_size=1, cp_rank=0, dp_rank=0):
        result = mixin._prepare_replay_indices(batches, seq_length)

    # 期望顺序: [sample0 的 seq_length 个, sample1 的 seq_length 个]
    # 由于 padding 公式 idx % (sample_seq - 1)，sample_seq=4:
    #   idx [0,1,2,3] → [0,1,2,0]
    # re0[0]=10, re0[1]=10, re0[2]=10, re0[0]=10 → 全 10
    # re1[0]=20, re1[1]=20, re1[2]=20, re1[0]=20 → 全 20
    vals = result[0].squeeze(-1)
    assert (vals[:seq_length] == 10).all()
    assert (vals[seq_length:] == 20).all()


def test_prepare_replay_indices_returns_none_when_missing():
    """batch 没有 routed_experts 时返回 None。"""
    mixin = _make_mixin(["moe"])
    batches = [{"tokens": torch.zeros(10)}]

    with _mock_mpu():
        result = mixin._prepare_replay_indices(batches, 10)
    assert result is None


def test_prepare_replay_indices_returns_none_when_none():
    """routed_experts 为 None 时返回 None。"""
    mixin = _make_mixin(["moe"])
    batches = [{"routed_experts": None}]

    with _mock_mpu():
        result = mixin._prepare_replay_indices(batches, 10)
    assert result is None


def test_prepare_replay_indices_padding_wraps():
    """seq_length > sample_seq 时 padding 位置用 wrap-around 映射。"""
    mixin = _make_mixin(["moe"])
    topk = 1
    sample_seq = 4
    seq_length = 8

    re = torch.arange(sample_seq).reshape(sample_seq, 1, topk)
    batches = [{"routed_experts": re}]

    with _mock_mpu(cp_size=1, cp_rank=0, dp_rank=0):
        result = mixin._prepare_replay_indices(batches, seq_length)

    # padding 公式: idx % (sample_seq - 1) = idx % 3
    # [0,1,2,0,1,2,0,1]
    expected_indices = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    expected = re[expected_indices, 0, :]
    assert torch.equal(result[0], expected.long())


def test_prepare_replay_indices_matches_manual_pipeline():
    """_prepare_replay_indices 与手动 re[index]+TopK 层抽取结果一致。"""
    mixin = _make_mixin(["hash_moe", "moe", "moe"])
    topk = 2
    sample_seq = 6
    seq_length = 8
    B = 2
    num_layers = 3

    re = torch.randint(0, 64, (sample_seq, num_layers, topk))
    batches = [{"routed_experts": re.clone()} for _ in range(B)]
    topk_layer_indices = get_topk_layer_indices(mixin.hf_config)

    with _mock_mpu(cp_size=1, cp_rank=0, dp_rank=0):
        prepared = mixin._prepare_replay_indices(batches, seq_length)

    expected = _manual_prepare_replay_indices(
        batches, seq_length, topk_layer_indices,
    )
    assert len(prepared) == len(expected) == len(topk_layer_indices)
    for layer_i, (prep, exp) in enumerate(zip(prepared, expected)):
        assert torch.equal(prep, exp), f"layer {layer_i} manual pipeline mismatch"


def test_prepare_replay_indices_manual_pipeline_with_cp():
    """CP 切分后 manual pipeline 仍与 mixin 一致。"""
    mixin = _make_mixin(["moe", "moe"])
    topk = 2
    sample_seq = 8
    seq_length = 8
    cp_size = 2
    re = torch.randint(0, 64, (sample_seq, 2, topk))
    batches = [{"routed_experts": re}]
    topk_layer_indices = get_topk_layer_indices(mixin.hf_config)

    for cp_rank in range(cp_size):
        with _mock_mpu(cp_size=cp_size, cp_rank=cp_rank, dp_rank=0):
            prepared = mixin._prepare_replay_indices(batches, seq_length)
        expected = _manual_prepare_replay_indices(
            batches, seq_length, topk_layer_indices,
            cp_size=cp_size, cp_rank=cp_rank,
        )
        for layer_i, (prep, exp) in enumerate(zip(prepared, expected)):
            assert torch.equal(prep, exp), (
                f"cp_rank={cp_rank} layer {layer_i} manual pipeline mismatch"
            )


# ------------------------------------------------------------------
# _maybe_router_replay 端到端 + R3 正确性
# ------------------------------------------------------------------


def test_maybe_router_replay_disabled():
    """moe_router_replay=False 时不做 replay。"""
    mixin = _make_mixin(["moe", "moe"])
    mixin.config.training.moe_router_replay = False

    model = _TwoLayerModel()
    batches = [{"routed_experts": torch.randint(0, 4, (5, 2, 2))}]

    with _mock_mpu():
        with mixin._maybe_router_replay(model, batches, 5):
            assert model.gate0.router_replay is None
            assert model.gate1.router_replay is None


@_PATCH_ITER
def test_maybe_router_replay_end_to_end(_):
    """moe_router_replay=True 时 forward 使用 replay indices。"""
    mixin = _make_mixin(["moe", "moe"])
    model = _TwoLayerModel()
    topk = 2
    seq_length = 4
    # B=2 micro-batch, seq_length=4 → replay indices are (B * seq_length, topk)
    x = torch.randn(2, seq_length, 8)

    # 构造 routed_experts: (sample_seq, num_layers, topk)
    # 固定 routing = 全 0
    re = torch.zeros(seq_length, 2, topk, dtype=torch.int32)
    batches = [
        {"routed_experts": re.clone()},
        {"routed_experts": re.clone()},
    ]

    with _mock_mpu(cp_size=1, cp_rank=0, dp_rank=0):
        with mixin._maybe_router_replay(model, batches, seq_length):
            _, _, idx0 = model.gate0(x)
            _, _, idx1 = model.gate1(x)

    # replay indices 应全为 0
    assert (idx0 == 0).all()
    assert (idx1 == 0).all()


@_PATCH_ITER
def test_maybe_router_replay_prepared_matches_captured(_):
    """R3 核心 contract: _prepare_replay_indices == forward capture 的 routing。"""
    mixin = _make_mixin(["moe", "moe"])
    model = _TwoLayerModel()
    topk = 2
    seq_length = 4
    x = torch.randn(2, seq_length, 8)

    re = torch.randint(0, 4, (seq_length, 2, topk), dtype=torch.int32)
    batches = [{"routed_experts": re.clone()}, {"routed_experts": re.clone()}]

    _assert_replay_prepared_matches_captured(
        mixin, model, batches, seq_length, x,
    )


@_PATCH_ITER
def test_maybe_router_replay_cp_prepared_matches_captured(_):
    """CP rank 上 prepared 与 capture 仍逐 token 对齐。"""
    mixin = _make_mixin(["moe", "moe"])
    model = _TwoLayerModel()
    topk = 2
    seq_length = 8
    cp_size = 2
    s_local = seq_length // cp_size
    x = torch.randn(1, s_local, 8)

    re = torch.randint(0, 4, (seq_length, 2, topk), dtype=torch.int32)
    batches = [{"routed_experts": re}]

    for cp_rank in range(cp_size):
        _assert_replay_prepared_matches_captured(
            mixin, model, batches, seq_length, x,
            cp_size=cp_size, cp_rank=cp_rank,
        )


@_PATCH_ITER
def test_maybe_router_replay_overrides_natural_topk(_):
    """Replay 强制使用 routed_experts，而非 scores 的自然 topk。"""
    mixin = _make_mixin(["moe", "moe"])
    model = _TwoLayerModel()
    topk = 2
    seq_length = 4
    x = torch.randn(1, seq_length, 8)

    with torch.no_grad():
        _, _, natural_idx0 = model.gate0(x)

    forced_value = 3
    re = torch.full((seq_length, 2, topk), forced_value, dtype=torch.int32)
    batches = [{"routed_experts": re}]

    with _mock_mpu():
        with mixin._maybe_router_replay(model, batches, seq_length):
            _, _, replay_idx0 = model.gate0(x)

    assert (replay_idx0 == forced_value).all()
    assert not (natural_idx0 == forced_value).all(), (
        "natural topk should differ from forced replay target in this fixture"
    )


@_PATCH_ITER
def test_maybe_router_replay_hash_moe_layers_skipped(_):
    """hash_moe 层不出现在 prepared/capture 里，只 replay TopKRouter 层。"""
    mixin = _make_mixin(["hash_moe", "moe", "moe"])
    model = _TwoLayerModel()
    topk = 2
    seq_length = 4
    x = torch.randn(1, seq_length, 8)

    # routed_experts 含 3 层 (1 hash + 2 topk)，model 只有 2 个 TopKRouter。
    # expert id 必须在 [0, num_experts) 内，否则 gather 会 OOB。
    re = torch.stack([
        torch.full((seq_length, topk), 0),
        torch.full((seq_length, topk), 1),
        torch.full((seq_length, topk), 2),
    ], dim=1)
    batches = [{"routed_experts": re}]

    with _mock_mpu():
        prepared = mixin._prepare_replay_indices(batches, seq_length)

    assert prepared is not None
    assert len(prepared) == 2
    topk_layer_indices = get_topk_layer_indices(mixin.hf_config)
    assert topk_layer_indices == [1, 2]

    _assert_replay_prepared_matches_captured(
        mixin, model, batches, seq_length, x,
    )
    assert (prepared[0] == 1).all()
    assert (prepared[1] == 2).all()
    assert not (prepared[0] == 0).any(), "hash_moe layer 0 should be skipped"
