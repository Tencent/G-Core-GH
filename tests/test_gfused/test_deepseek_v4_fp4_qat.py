from types import SimpleNamespace

import pytest
import torch
from torch import nn


if not torch.cuda.is_available():
    pytest.skip("FP4 QAT requires CUDA", allow_module_level=True)

from gpatch_v4.kernel.quantize.qat import fp4_qat_linear, fp4_simulate_qat
from gpatch_v4.kernel.quantize.eager_quant_kernels import (
    dequant_fp4_e2m1_fp8_scale_e8m0_packed,
)
from gpatch_v4.models.deepseek_v4 import modeling_deepseek_v4
from gpatch_v4.generation_backend.sglang_model_specific.sglang_weight_update_dsv4 import (
    quantize_fp4_qat_expert,
)


# clamp(x / scale) 恰好落在这些值上时，硬件 FP4 cast 走 round-to-nearest-even，
# 而导出 kernel 的 _cast_fp4_bucketize_right_false 强制取较低档。
_FP4_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0)

_EXPERT_KEY = "layers.3.ffn.experts.0.w1.weight"


def _sgl_export_effective_weight(weight: torch.Tensor) -> torch.Tensor:
    """sampler 加载导出权重后，它的 GEMM 实际乘的矩阵。

    走生产入口而不是直接调 kernel，否则导出侧的 dtype 视图选择测不到。
    """
    (_, fp8_weight), (_, fp8_scale) = quantize_fp4_qat_expert(_EXPERT_KEY, weight)
    return dequant_fp4_e2m1_fp8_scale_e8m0_packed(fp8_weight, fp8_scale).float()


def test_fp4_simulate_qat_round_trip_and_ste():
    values = torch.tensor(
        [
            -5.5, -4.75, -3.75, -3.25, -2.75, -2.25, -1.85, -1.65,
            -1.35, -1.15, -0.9, -0.6, -0.4, -0.2, -0.1, 0.0,
            0.1, 0.2, 0.4, 0.6, 0.9, 1.15, 1.35, 1.65,
            1.85, 2.25, 2.75, 3.25, 3.75, 4.75, 5.25, 6.0,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected_values = torch.tensor(
        [
            -6.0, -4.0, -4.0, -3.0, -3.0, -2.0, -2.0, -1.5,
            -1.5, -1.0, -1.0, -0.5, -0.5, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5,
            2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = torch.stack(
        [
            torch.cat([values, values]),
            torch.cat([values * 2, values * 2]),
        ],
    ).unsqueeze(1).expand(-1, 3, -1).clone()
    expected = torch.stack(
        [
            torch.cat([expected_values, expected_values]),
            torch.cat([expected_values * 2, expected_values * 2]),
        ],
    ).unsqueeze(1).expand(-1, 3, -1)
    x.requires_grad_()
    grad_out = torch.randn_like(x)

    y = fp4_simulate_qat(x)
    y.backward(grad_out)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()
    assert torch.equal(y, expected)
    assert torch.equal(x.grad, grad_out)


def test_fp4_qat_linear():
    module = nn.Linear(64, 32, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = fp4_qat_linear(module, x)
    output.float().square().mean().backward()

    assert output.shape == (4, 32)
    assert torch.isfinite(output).all()
    assert module.weight.grad is not None
    assert torch.isfinite(module.weight.grad).all()


def test_fp4_qat_forward_matches_sglang_export():
    """训练 forward 的 fake-quant 权重必须等于 sampler 的有效权重。"""
    torch.manual_seed(29)
    # 取 BF16 网格上的值，排除训练侧 BF16 / 导出侧 FP32 的 amax 差异
    weight = torch.randn((256, 512), device="cuda", dtype=torch.bfloat16).float()

    trainer = fp4_simulate_qat(weight.bfloat16()).float()
    sampler = _sgl_export_effective_weight(weight)

    mismatch = trainer != sampler
    assert not mismatch.any(), (
        f"{mismatch.sum().item()} of {weight.numel()} weights differ between the "
        f"FP4 QAT forward and the sglang export; max abs diff "
        f"{(trainer - sampler).abs().max().item():.6f}"
    )


def test_fp4_qat_forward_matches_sglang_export_at_midpoints():
    """精确 midpoint 上两侧必须选同一个 FP4 码。"""
    row = torch.tensor(_FP4_MIDPOINTS, device="cuda", dtype=torch.float32).repeat(16)
    # 每个 1x32 组都含 6.0，因此组 scale 恒为 1，归一化值就是 _FP4_MIDPOINTS 本身
    weight = row.expand(128, 128).clone()
    weight[1::2] *= -1.0

    trainer = fp4_simulate_qat(weight.bfloat16()).float()
    sampler = _sgl_export_effective_weight(weight)

    mismatch = trainer != sampler
    assert not mismatch.any(), (
        f"FP4 QAT forward and sglang export disagree at "
        f"{sorted(weight[mismatch].abs().unique().tolist())}: "
        f"trainer={sorted(trainer[mismatch].abs().unique().tolist())} vs "
        f"sampler={sorted(sampler[mismatch].abs().unique().tolist())}"
    )


def test_fp4_qat_forward_matches_sglang_export_after_optimizer_step():
    """一次 Adam 量级更新后两侧仍须一致，即同时跨过 group scale 的翻档点。

    网格对齐的权重里约 41% 的 1x32 组 amax 恰为 6*scale，正压在
    scale=2^ceil(log2(amax/6)) 的边界上，amax 涨一点 scale 就必须翻倍。导出若量化
    FP32 master、而 forward 量化 BF16 权重，1e-6 的更新（低于 1e-2 处的 BF16 分辨率
    约 4e-5）就只会让导出侧翻档，两侧落到相差 2 倍的网格上。
    """
    torch.manual_seed(31)
    # 先落到 FP4 网格，复现刚加载 DSV4-Flash ckpt 的状态
    weight = fp4_simulate_qat(
        torch.randn((256, 512), device="cuda", dtype=torch.bfloat16)
    ).float()
    signs = torch.randint(0, 2, weight.shape, device=weight.device, dtype=torch.float32)
    stepped = weight + (signs * 2.0 - 1.0) * 1e-6

    settled = (
        fp4_simulate_qat(weight.bfloat16()).float() != _sgl_export_effective_weight(weight)
    ).sum().item()
    trainer = fp4_simulate_qat(stepped.bfloat16()).float()
    sampler = _sgl_export_effective_weight(stepped)

    mismatch = trainer != sampler
    assert not mismatch.any(), (
        f"{mismatch.sum().item()} of {weight.numel()} weights diverged after a "
        f"+-1e-6 step ({settled} already differed before it); max abs diff "
        f"{(trainer - sampler).abs().max().item():.6f}"
    )


def test_fp4_qat_precedes_fp8_qat(monkeypatch):
    config = SimpleNamespace(
        num_local_experts=1,
        hidden_size=32,
        intermediate_size=32,
        hidden_act="silu",
        swiglu_limit=7.0,
        fp4_qat=True,
        fp8_qat=True,
        fp8=False,
    )
    experts = modeling_deepseek_v4.DeepseekV4Experts(config).cuda().bfloat16()
    fp4_calls = []

    def record_fp4(tensor):
        fp4_calls.append(tensor)
        return tensor

    def fail_fp8(*args):
        pytest.fail("fp8_qat must not quantize routed MoE weights when fp4_qat is enabled")

    monkeypatch.setattr(modeling_deepseek_v4, "fp4_simulate_qat", record_fp4)
    # fp8_qat weight path uses fp8_simulate_qat_128x128; keep both mocked so a
    # regression to either entry cannot silently pass under fp4_qat.
    monkeypatch.setattr(modeling_deepseek_v4, "fp8_simulate_qat", fail_fp8)
    monkeypatch.setattr(modeling_deepseek_v4, "fp8_simulate_qat_128x128", fail_fp8)

    output = experts.fwd_gmm(
        torch.randn(2, 32, device="cuda", dtype=torch.bfloat16),
        torch.zeros(2, device="cuda", dtype=torch.long),
        torch.ones(2, device="cuda", dtype=torch.bfloat16),
    )

    assert len(fp4_calls) == 2
    assert fp4_calls[0] is experts.gate_up_proj
    assert fp4_calls[1] is experts.down_proj
    assert torch.isfinite(output).all()
