# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.

"""Correctness and performance tests for TileLang weight quantization.

Run on a CUDA machine with::

    PYTHONPATH=. pytest -v -s \
        tests/test_gpatch_v4/test_quantize_kernels.py
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
import torch

pytest.importorskip("tilelang")

from gpatch_v4.kernel.quantize.eager_quant_kernels import (  # noqa: E402
    dequant_fp4_e2m1_fp8_scale_e8m0_packed,
    quant_fp4_e2m1_scale_e8m0_packed,
)
from gpatch_v4.kernel.quantize.fused_quant_kernels import (  # noqa: E402
    fp4_qat_then_to_fp8,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required by the TileLang quantization kernel",
)

_FP4_GROUP_SIZE = 32

# Adam's first update is ``lr * m_hat / sqrt(v_hat)`` == ``lr * sign(g)``, so a
# 1e-6 learning rate moves every element by exactly 1e-6.
_ADAM_STEP = 1e-6

_FP4_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _torch_fp4_qat_then_to_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfused PyTorch baseline without the optional underflow check."""
    packed, fp4_scale = quant_fp4_e2m1_scale_e8m0_packed(weight)

    out_dim, packed_in_dim = packed.shape
    in_dim = packed_in_dim * 2
    fp8_block_size = 128
    fp4_block_size = 32

    codes = packed.contiguous().view(torch.uint8)
    low = (codes & 0x0F).long()
    high = ((codes >> 4) & 0x0F).long()
    lut = torch.tensor(_FP4_TABLE, dtype=torch.float32, device=weight.device)
    values = torch.stack([lut[low], lut[high]], dim=-1).flatten(2)

    blocks_out = out_dim // fp8_block_size
    blocks_in = in_dim // fp8_block_size
    values = values.view(
        blocks_out,
        fp8_block_size,
        blocks_in,
        fp8_block_size,
    ).transpose(1, 2)
    local_scales = (
        fp4_scale.float()
        .view(blocks_out, fp8_block_size, blocks_in, -1)
        .transpose(1, 2)
        .flatten(2)
    )
    outer_scale = (local_scales.amax(dim=-1, keepdim=True) / 64.0).clamp(
        min=2.0**-126
    )
    offset = local_scales / outer_scale
    offset = offset.unflatten(
        -1,
        (fp8_block_size, -1),
    ).repeat_interleave(fp4_block_size, dim=-1)
    fp8_values = (values * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return (
        fp8_values.to(torch.float8_e4m3fn).contiguous(),
        outer_scale.squeeze(-1).to(torch.float8_e8m0fnu).contiguous(),
    )


def _dequant_sgl_fp8(
    fp8_weight: torch.Tensor,
    fp8_scale: torch.Tensor,
) -> torch.Tensor:
    return dequant_fp4_e2m1_fp8_scale_e8m0_packed(
        fp8_weight,
        fp8_scale,
    ).float()


def _snap_to_fp4_grid(weight: torch.Tensor) -> torch.Tensor:
    """Project onto the E2M1 1x32 grid DSV4-Flash experts are stored on."""
    packed, scale = quant_fp4_e2m1_scale_e8m0_packed(weight)
    return dequant_fp4_e2m1_fp8_scale_e8m0_packed(packed, scale).float()


def _fp4_group_scales(weight: torch.Tensor) -> torch.Tensor:
    return quant_fp4_e2m1_scale_e8m0_packed(weight)[1].float()


def _sgl_effective_weight(weight: torch.Tensor) -> torch.Tensor:
    """The weight sglang's GEMM sees after loading an exported expert."""
    return _dequant_sgl_fp8(*fp4_qat_then_to_fp8(weight))


def _assert_byte_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str,
) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    actual_bytes = actual.contiguous().view(torch.uint8)
    expected_bytes = expected.contiguous().view(torch.uint8)
    mismatch = actual_bytes != expected_bytes
    if mismatch.any():
        indices = mismatch.nonzero()[:8]
        index_tuples = tuple(indices[:, dim] for dim in range(indices.shape[1]))
        actual_samples = actual[index_tuples].float().cpu().tolist()
        expected_samples = expected[index_tuples].float().cpu().tolist()
        raise AssertionError(
            f"{name} differs byte-for-byte at {mismatch.sum().item()} elements; "
            f"first indices={indices.cpu().tolist()}, "
            f"actual={actual_samples}, expected={expected_samples}"
        )


@pytest.mark.parametrize(
    ("m", "n"),
    [
        pytest.param(128, 128, id="single-fp8-tile"),
        pytest.param(256, 512, id="multiple-fp8-tiles"),
        pytest.param(384, 640, id="rectangular-multiple-tiles"),
        pytest.param(2048, 4096, id="rectangular-multiple-tiles2"),
    ],
)
def test_fp4_qat_then_to_fp8_matches_torch(m: int, n: int) -> None:
    """TileLang output must match the existing PyTorch conversion exactly."""
    torch.manual_seed(7)
    weight = torch.randn((m, n), device="cuda", dtype=torch.float32)

    # Exercise zero blocks and different but representable local scale ranges.
    weight[:16, :32] = 0
    weight[:, 32:64] *= 2.0**-4
    weight[:, 64:96] *= 2.0**4

    expected_weight, expected_scale = _torch_fp4_qat_then_to_fp8(weight)
    actual_weight, actual_scale = fp4_qat_then_to_fp8(weight)
    torch.cuda.synchronize()

    _assert_byte_equal(actual_scale, expected_scale, "FP8 scale")
    _assert_byte_equal(actual_weight, expected_weight, "FP8 weight")
    actual_dequant = _dequant_sgl_fp8(actual_weight, actual_scale)
    expected_dequant = _dequant_sgl_fp8(expected_weight, expected_scale)
    torch.testing.assert_close(
        actual_dequant,
        expected_dequant,
        rtol=0,
        atol=0,
    )

    actual_error = (actual_dequant - weight).abs()
    expected_error = (expected_dequant - weight).abs()
    print(
        f"\nPrecision shape=({m}, {n}): "
        f"TileLang MAE={actual_error.mean().item():.8f}, "
        f"max={actual_error.max().item():.8f}; "
        f"PyTorch MAE={expected_error.mean().item():.8f}, "
        f"max={expected_error.max().item():.8f}"
    )


def test_fp4_scale_boundary_occupancy_under_optimizer_step() -> None:
    """Report how much of a grid-aligned expert sits on the group-scale cliff.

    ``scale = 2**ceil(log2(amax / 6))`` cannot represent ``amax = 6 * scale + eps``
    without clipping, so the scale necessarily doubles the moment a group's top
    element grows. DSV4-Flash ships routed experts already on the E2M1 1x32
    grid, which puts every group whose top code is 6 exactly on that cliff: an
    Adam step of ``1e-6`` re-rounds all 32 of the group's weights onto a grid
    twice as coarse. The cliff is inherent to power-of-two scales and is only
    harmless while the trainer forward crosses it at the same instant, which is
    what ``test_deepseek_v4_fp4_qat.py`` pins down.
    """
    torch.manual_seed(23)
    m, n = 2048, 4096
    weight = _snap_to_fp4_grid(
        torch.randn((m, n), device="cuda", dtype=torch.float32)
    )

    scales = _fp4_group_scales(weight)
    groups = weight.view(m, n // _FP4_GROUP_SIZE, _FP4_GROUP_SIZE)
    top_code = groups.abs().amax(dim=-1) / scales

    signs = torch.randint(0, 2, weight.shape, device=weight.device, dtype=torch.float32)
    step = (signs * 2.0 - 1.0) * _ADAM_STEP
    perturbed = weight + step

    flipped = _fp4_group_scales(perturbed) != scales
    before = _sgl_effective_weight(weight)
    after = _sgl_effective_weight(perturbed)
    input_shift = (step.norm() / weight.norm()).item()
    export_shift = ((after - before).norm() / before.norm()).item()

    print(
        f"\nFP4 scale boundary shape=({m}, {n}): "
        f"groups={scales.numel()}, "
        f"top-code-6={(top_code == 6.0).float().mean().item():.2%}, "
        f"scale-flipped={flipped.float().mean().item():.2%}, "
        f"input shift={input_shift:.3e}, export shift={export_shift:.3e}, "
        f"newly zeroed={((before != 0) & (after == 0)).sum().item()}"
    )
    assert torch.isfinite(after).all()
    off_cliff = flipped & (top_code != 6.0)
    assert not off_cliff.any(), (
        f"{off_cliff.sum().item()} groups changed scale without sitting at top "
        f"code 6; the group-scale rule is not the only source of instability"
    )


def _bench_cuda_ms(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def test_fp4_qat_then_to_fp8_performance_vs_torch() -> None:
    """Report end-to-end TileLang latency and speedup over the PyTorch path."""
    m = int(os.environ.get("DSV4_QUANT_BENCH_M", "2048"))
    n = int(os.environ.get("DSV4_QUANT_BENCH_N", "4096"))
    warmup = int(os.environ.get("DSV4_QUANT_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("DSV4_QUANT_BENCH_ITERS", "20"))
    min_speedup = float(os.environ.get("DSV4_QUANT_MIN_SPEEDUP", "0"))

    assert m % 128 == 0 and n % 128 == 0
    assert warmup > 0 and iters > 0

    torch.manual_seed(11)
    weight = torch.randn((m, n), device="cuda", dtype=torch.float32)

    # Compile TileLang and validate the benchmark input before timing.
    expected_weight, expected_scale = _torch_fp4_qat_then_to_fp8(weight)
    actual_weight, actual_scale = fp4_qat_then_to_fp8(weight)
    torch.cuda.synchronize()
    _assert_byte_equal(actual_scale, expected_scale, "benchmark FP8 scale")

    # Hardware FP4 casts use round-to-nearest-even, while the eager reference
    # uses torch.bucketize and selects the lower value at every exact midpoint.
    # Compare the deployed dequantized values and quantization error without
    # requiring those rare midpoint cases to have identical FP8 bytes.
    actual_dequant = _dequant_sgl_fp8(actual_weight, actual_scale)
    expected_dequant = _dequant_sgl_fp8(expected_weight, expected_scale)
    dequant_abs_diff = (actual_dequant - expected_dequant).abs()
    fp8_byte_mismatches = (
        actual_weight.contiguous().view(torch.uint8)
        != expected_weight.contiguous().view(torch.uint8)
    ).sum()
    actual_abs_error = (actual_dequant - weight).abs()
    expected_abs_error = (expected_dequant - weight).abs()
    actual_mae = actual_abs_error.mean()
    expected_mae = expected_abs_error.mean()

    assert torch.isfinite(actual_dequant).all()
    assert actual_mae <= expected_mae + 1e-6, (
        f"TileLang dequant MAE {actual_mae.item():.8f} exceeds eager MAE "
        f"{expected_mae.item():.8f}"
    )
    print(
        "\nCorrectness on benchmark input: "
        f"FP8 byte mismatches={fp8_byte_mismatches.item()}, "
        f"dequant max diff={dequant_abs_diff.max().item():.8f}, "
        f"dequant mean diff={dequant_abs_diff.mean().item():.8f}, "
        f"TileLang MAE={actual_mae.item():.8f}, "
        f"eager MAE={expected_mae.item():.8f}"
    )

    tilelang_ms = _bench_cuda_ms(
        lambda: fp4_qat_then_to_fp8(weight),
        warmup=warmup,
        iters=iters,
    )
    torch_ms = _bench_cuda_ms(
        lambda: _torch_fp4_qat_then_to_fp8(weight),
        warmup=warmup,
        iters=iters,
    )
    speedup = torch_ms / tilelang_ms

    print(
        "\nFP4-QAT -> SGL FP8 benchmark "
        f"shape=({m}, {n}), warmup={warmup}, iters={iters}"
    )
    print(f"  TileLang fused: {tilelang_ms:.3f} ms")
    print(f"  PyTorch eager:  {torch_ms:.3f} ms")
    print(f"  Speedup:        {speedup:.2f}x")

    if min_speedup > 0:
        assert speedup >= min_speedup, (
            f"TileLang speedup {speedup:.2f}x is below required "
            f"{min_speedup:.2f}x"
        )
