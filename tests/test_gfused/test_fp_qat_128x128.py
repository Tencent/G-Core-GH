# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""TileLang ``fp_qat_128x128`` vs ``fp_quantize`` reference.

Run (GPU)::

    cd /work/wepsdl/gcore-dev
    PYTHONPATH=. pytest -v -s tests/test_gfused/test_fp_qat_128x128.py \\
        tests/test_gpatch_v4/test_sglang_weight_update_dsv4.py
"""

import pytest
import torch
from types import SimpleNamespace

from gpatch_v4.generation_backend.sglang_model_specific.sglang_weight_update_dsv4 import (
    simulate_fp8_wo_a,
)
from gpatch_v4.kernel.quantize.eager_quant_kernels import (
    dequant_fp4_e2m1_fp8_scale_e8m0_packed,
    quant_fp8_e4m3_scale_e8m0,
)
from gpatch_v4.kernel.quantize.fused_quant_kernels import fp_qat_128x128
from gpatch_v4.kernel.quantize.qat import fp8_qat_linear, fp8_simulate_qat_128x128
from gpatch_v4.models.deepseek_v4 import modeling_deepseek_v4
from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4GroupedLinear


def _torch_qdq(weight_bf16: torch.Tensor) -> torch.Tensor:
    q, s = quant_fp8_e4m3_scale_e8m0(weight_bf16.float(), block_size=(128, 128))
    return dequant_fp4_e2m1_fp8_scale_e8m0_packed(q, s)


def _assert_matches_torch(weight: torch.Tensor) -> None:
    expected = _torch_qdq(weight.detach())
    actual = fp_qat_128x128(weight)
    torch.cuda.synchronize()

    assert actual.dtype == torch.bfloat16
    assert actual.shape == weight.shape

    actual_bits = actual.view(torch.int16)
    expected_bits = expected.view(torch.int16)
    mismatches = actual_bits != expected_bits
    assert not mismatches.any(), (
        f"{mismatches.sum().item()} / {weight.numel()} BF16 values differ; "
        f"max_abs_diff={(actual.float() - expected.float()).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("m", "n", "seed"),
    [
        pytest.param(128, 128, 0, id="single-tile"),
        pytest.param(256, 384, 1, id="multi-tile"),
        pytest.param(128, 512, 2, id="wide"),
        pytest.param(512, 128, 3, id="tall"),
        pytest.param(512, 512, 4, id="large-square"),
    ],
)
def test_fp_qat_128x128_matches_torch_reference(m: int, n: int, seed: int) -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(seed)
    weight = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)

    _assert_matches_torch(weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp_qat_128x128_uses_independent_scale_per_tile() -> None:
    pytest.importorskip("tilelang")
    pattern = torch.linspace(
        -1.0,
        1.0,
        128 * 128,
        device="cuda",
        dtype=torch.float32,
    ).reshape(128, 128)
    weight = torch.empty(256, 256, device="cuda", dtype=torch.bfloat16)
    weight[:128, :128] = pattern * 2.0**-8
    weight[:128, 128:] = pattern * 2.0**-2
    weight[128:, :128] = 0
    weight[128:, 128:] = pattern * 2.0**5

    _assert_matches_torch(weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp_qat_128x128_reduces_all_128x32_stripes() -> None:
    pytest.importorskip("tilelang")
    weight = torch.full(
        (128, 512),
        0.125,
        device="cuda",
        dtype=torch.bfloat16,
    )
    peaks = (224.0, -448.0, 896.0, -1792.0)
    for tile, peak in enumerate(peaks):
        stripe = tile
        row = 17 + tile * 29
        col = tile * 128 + stripe * 32 + 31
        weight[row, col] = peak

    _assert_matches_torch(weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp_qat_128x128_matches_scale_exponent_boundaries() -> None:
    pytest.importorskip("tilelang")
    tile_amaxes = (223.0, 224.0, 225.0, 446.0, 448.0, 450.0, 892.0, 896.0, 900.0)
    weight = torch.zeros(
        (128, 128 * len(tile_amaxes)),
        device="cuda",
        dtype=torch.bfloat16,
    )
    for tile, amax in enumerate(tile_amaxes):
        start = tile * 128
        weight[:, start:start + 128] = torch.linspace(
            -amax / 4,
            amax / 4,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight[tile, start + tile] = -amax if tile % 2 else amax

    _assert_matches_torch(weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp_qat_128x128_matches_wide_scale_exponent_range() -> None:
    pytest.importorskip("tilelang")
    scale_exponents = (-60, -30, -15, 0, 15, 30, 60)
    tile_amaxes = tuple(448.0 * 2.0**exponent for exponent in scale_exponents)
    weight = torch.zeros(
        (128, 128 * len(tile_amaxes)),
        device="cuda",
        dtype=torch.bfloat16,
    )
    pattern = torch.linspace(
        -0.75,
        0.75,
        128,
        device="cuda",
        dtype=torch.float32,
    )
    for tile, amax in enumerate(tile_amaxes):
        start = tile * 128
        weight[:, start:start + 128] = pattern * amax
        weight[tile, start + 127 - tile] = -amax if tile % 2 else amax

    _assert_matches_torch(weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp_qat_128x128_matches_e4m3_rounding_boundaries() -> None:
    pytest.importorskip("tilelang")
    midpoint_neighbors = []
    for midpoint_index in range(8):
        midpoint = 1.0625 + midpoint_index * 0.125
        midpoint_neighbors.extend(
            (midpoint - 0.0078125, midpoint, midpoint + 0.0078125)
        )
    signed_midpoint_neighbors = [
        *(-value for value in reversed(midpoint_neighbors)),
        *midpoint_neighbors,
    ]
    values = torch.tensor(
        [
            -448.0,
            -0.001953125,
            -0.0,
            0.0,
            0.001953125,
            *signed_midpoint_neighbors,
            448.0,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    repeats = (128 * 128 + values.numel() - 1) // values.numel()
    weight = values.repeat(repeats)[:128 * 128].reshape(128, 128)

    _assert_matches_torch(weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp_qat_128x128_matches_noncontiguous_input() -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(5)
    weight = torch.randn(
        256,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(0, 1)
    assert not weight.is_contiguous()

    _assert_matches_torch(weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_simulate_qat_128x128_matches_torch_and_uses_ste() -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(3)
    weight = torch.randn(
        128,
        256,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output_grad = torch.randn_like(weight)
    expected = _torch_qdq(weight.detach())
    actual = fp8_simulate_qat_128x128(weight)
    torch.cuda.synchronize()

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    actual.backward(output_grad)
    assert torch.equal(weight.grad, output_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_linear_fp4_qat_uses_128x128_qdq_weight() -> None:
    """训练侧 o_a_proj（DeepseekV4GroupedLinear）在 fp4_qat 下应用 128×128 QDQ。"""
    pytest.importorskip("tilelang")
    torch.manual_seed(11)
    n_groups = 2
    in_features = 128
    out_features = 256
    module = DeepseekV4GroupedLinear(
        in_features,
        out_features,
        n_groups,
        qat_config=SimpleNamespace(fp4_qat=True),
    ).cuda().to(dtype=torch.bfloat16)
    x = torch.randn(3, 4, n_groups, in_features, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        expected_weight = _torch_qdq(module.weight)
        w = expected_weight.view(n_groups, -1, in_features).transpose(1, 2)
        expected = torch.bmm(
            x.reshape(-1, n_groups, in_features).transpose(0, 1),
            w,
        ).transpose(0, 1).reshape(3, 4, n_groups, -1)

    actual = module(x)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_train_wo_a_qdq_matches_update_simulate_fp8_wo_a() -> None:
    """训练 fake-quant 与 sglang update 的 wo_a 投影必须 bit-exact。"""
    pytest.importorskip("tilelang")
    torch.manual_seed(17)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)

    train_view = fp8_simulate_qat_128x128(weight)
    update_pairs = simulate_fp8_wo_a("layers.0.attn.wo_a.weight", weight)
    torch.cuda.synchronize()

    assert len(update_pairs) == 1
    assert update_pairs[0][0] == "layers.0.attn.wo_a.weight"
    update_view = update_pairs[0][1]
    assert torch.equal(train_view.view(torch.int16), update_view.view(torch.int16))
    assert torch.equal(train_view.view(torch.int16), _torch_qdq(weight).view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp_qat_128x128_zero_tile_scale_is_one() -> None:
    pytest.importorskip("tilelang")
    weight = torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16)
    _assert_matches_torch(weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_qat_linear_uses_128x128_qdq_weight() -> None:
    """Dense ``fp8_qat_linear`` fake-quants weights with 128×128 tiles."""
    pytest.importorskip("tilelang")
    torch.manual_seed(23)
    linear = torch.nn.Linear(128, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        expected = torch.nn.functional.linear(x, _torch_qdq(linear.weight))
    actual = fp8_qat_linear(linear, x, 128)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_qat_moe_calls_128x128_not_1x128(monkeypatch) -> None:
    """纯 ``fp8_qat`` 时 routed MoE 走 ``fp8_simulate_qat_128x128``，不走 1×N。"""
    pytest.importorskip("tilelang")
    config = SimpleNamespace(
        num_local_experts=2,
        hidden_size=128,
        intermediate_size=128,
        hidden_act="silu",
        swiglu_limit=7.0,
        fp4_qat=False,
        fp8_qat=True,
        fp8=False,
    )
    experts = modeling_deepseek_v4.DeepseekV4Experts(config).cuda().bfloat16()
    torch.nn.init.normal_(experts.gate_up_proj, std=0.02)
    torch.nn.init.normal_(experts.down_proj, std=0.02)
    calls_128: list[torch.Tensor] = []

    def record_128(tensor: torch.Tensor) -> torch.Tensor:
        calls_128.append(tensor)
        return tensor

    def fail_1xn(*_args, **_kwargs):
        pytest.fail("fp8_qat MoE must not use 1×N fp8_simulate_qat")

    monkeypatch.setattr(modeling_deepseek_v4, "fp8_simulate_qat_128x128", record_128)
    monkeypatch.setattr(modeling_deepseek_v4, "fp8_simulate_qat", fail_1xn)

    output = experts.fwd_gmm(
        torch.randn(3, 128, device="cuda", dtype=torch.bfloat16),
        torch.tensor([0, 1, 0], device="cuda", dtype=torch.long),
        torch.ones(3, device="cuda", dtype=torch.bfloat16),
    )

    assert len(calls_128) == 2
    assert calls_128[0] is experts.gate_up_proj
    assert calls_128[1] is experts.down_proj
    assert tuple(calls_128[0].shape) == (2, 256, 128)
    assert tuple(calls_128[1].shape) == (2, 128, 128)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_simulate_qat_128x128_3d_moe_matches_per_expert() -> None:
    """3D ``[E,N,K]`` view(-1,K) 与逐 expert 2D QDQ 在 N%128==0 时 bit-exact。"""
    pytest.importorskip("tilelang")
    torch.manual_seed(29)
    # Flash-like alignment: E=2, 2I=256, H=128 → gate_up [2,256,128]
    gate_up = torch.randn(2, 256, 128, device="cuda", dtype=torch.bfloat16)
    stacked = fp8_simulate_qat_128x128(gate_up)
    per_expert = torch.stack(
        [fp8_simulate_qat_128x128(gate_up[i]) for i in range(gate_up.shape[0])],
        dim=0,
    )
    torch.cuda.synchronize()
    assert torch.equal(stacked.view(torch.int16), per_expert.view(torch.int16))
    assert torch.equal(
        stacked.view(torch.int16),
        _torch_qdq(gate_up.view(-1, gate_up.shape[-1])).view_as(gate_up).view(torch.int16),
    )
