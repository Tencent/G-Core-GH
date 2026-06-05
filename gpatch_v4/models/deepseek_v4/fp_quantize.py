# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com

"""FP4 / FP8 block-wise quantizers — inverse of HF ``Fp8Dequantize``.

Used by :func:`gpatch_v4.models.deepseek_v4.parallelize._save_checkpoint_hp`
to write a checkpoint in DSV4-Flash native format (``.weight`` packed in
``int8`` / ``float8_e4m3fn`` + ``.scale`` in ``float8_e8m0fnu``).

These functions are the *exact inverse* of the load-side dequantization in
``transformers.integrations.finegrained_fp8.Fp8Dequantize._dequantize_one``
(v5.8.1, see ``_FP4_E2M1_LUT`` and the FP4 ``_unpack_fp4`` / FP8
``quantized.to(float32) * scale`` paths). Round-tripping a tensor through
``quantize_*`` then HF dequant lands inside the FP4/FP8 quantization noise
floor; tested in ``test_fp_quantize.py``.

Public API
----------
* :func:`quantize_fp4_e2m1_packed` — FP4 e2m1 packed + E8M0 scale, any
  ``(block_m, block_n)``. DSV4-Flash MoE experts use ``(1, 32)`` (matches
  OCP MXFP4 v1.0 / NVIDIA Blackwell MXFP4 TensorCore layout).
* :func:`quantize_fp8_e4m3_e8m0` — FP8 e4m3 + E8M0 scale, any
  ``(block_m, block_n)``. DSV4-Flash dense linears (attention projections,
  shared-experts, lm_head, embed_tokens, …) use ``(128, 128)``, matching
  HF's ``transformers.integrations.finegrained_fp8``. **Not** OCP MXFP8 —
  MXFP8 mandates a 1D 32-element block; this is DeepSeek's own 2D 128×128
  tile format.

Non-quantized weights (1D norms, biases, ``hc_*``, sinks, ``tid2eid``,
``ffn.gate.weight``) bypass these helpers entirely; the caller writes them
verbatim in their original dtype.

Packing convention (FP4): each int8 byte holds two e2m1 nibbles — low nibble
(``byte & 0xF``) is the even column index, high nibble (``(byte >> 4) & 0xF``)
is the odd column index. Matches HF's ``_unpack_fp4``.
"""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import torch

__all__ = [
    "quant_fp4_e2m1_scale_e8m0_packed",
    "quant_fp8_e4m3_scale_e8m0",
]


# E2M1 (FP4) value table. Mirrors ``Fp8Dequantize._FP4_E2M1_LUT`` so the
# inverse code points map back exactly. Layout: positive values at idx 0..7,
# sign-flipped values at idx 8..15.
_FP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_FP4_POS_LUT_T = torch.tensor(_FP4_E2M1_LUT[:8], dtype=torch.float32)
_FP4_MAX = 6.0


def quant_fp4_e2m1_scale_e8m0_packed(
    value: torch.Tensor,
    block_size: tuple[int, int] = (1, 32),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to FP4 (e2m1) packed bytes + E8M0 per-block scales.

    Inverse of HF ``Fp8Dequantize._dequantize_one`` (FP4 / int8-packed branch).

    Parameters
    ----------
    value : torch.Tensor
        Shape ``(..., M, N)``. ``M % block_m == 0``, ``N % block_n == 0``, ``N`` even.
    block_size : tuple[int, int]
        ``(block_m, block_n)``. DSV4-Flash MoE experts use ``(1, 32)``.

    Returns
    -------
    packed : torch.Tensor
        dtype ``int8``, shape ``(..., M, N//2)``. Low nibble = even column,
        high nibble = odd column. Matches HF ``_unpack_fp4``.
    scale : torch.Tensor
        dtype ``float8_e8m0fnu``, shape ``(..., M//block_m, N//block_n)``.
    """
    M, N = value.shape[-2:]
    bm, bn = block_size
    assert M % bm == 0 and N % bn == 0, (
        f"value shape ({M}, {N}) not divisible by block ({bm}, {bn})"
    )
    assert N % 2 == 0, (
        f"FP4 packing requires N (={N}) to be even (two nibbles per byte)"
    )
    # NaN guard: a single NaN in `value` poisons its block's max_abs → scale,
    # producing NaN code points in the packed output. Cheaper to fail loud here
    # than to debug a NaN-laced checkpoint downstream.
    assert not torch.isnan(value).any(), "quant_fp4_e2m1_scale_e8m0_packed: input contains NaN"
    leading = value.shape[:-2]

    # 1) reshape into per-block tiles for max-abs computation.
    blk = value.float().reshape(*leading, M // bm, bm, N // bn, bn)
    max_abs = blk.abs().amax(dim=(-3, -1))   # (..., M/bm, N/bn)

    # 2) E8M0 scale = 2^k where k = ceil(log2(max_abs / 6.0)). We round UP so
    # the largest |x| in the block fits inside ``±_FP4_MAX`` after dividing by
    # the scale (no clipping needed; the LUT then nearest-rounds it). For
    # all-zero blocks pin scale = 1.0 — the divide is a no-op, codes all 0.
    safe_max = max_abs.clamp(min=torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(safe_max / _FP4_MAX))
    scale_fp32 = torch.pow(2.0, exponent)
    scale_fp32 = torch.where(max_abs > 0, scale_fp32, torch.ones_like(scale_fp32))

    # 3) divide & nearest-round to LUT index.
    scaled = blk / scale_fp32.unsqueeze(-1).unsqueeze(-3)  # broadcast over (bm, bn)
    sign_bit = torch.signbit(scaled)
    abs_scaled = scaled.abs()

    # bucketize uses midpoints between consecutive positive LUT entries to pick
    # the nearest LUT value (the LUT is monotone increasing on idx 0..7).
    pos_lut = _FP4_POS_LUT_T.to(value.device)
    midpoints = (pos_lut[1:] + pos_lut[:-1]) / 2.0          # 7 thresholds → 8 buckets
    idx = torch.bucketize(abs_scaled.contiguous(), midpoints)  # int64, in [0, 7]
    code = (idx + sign_bit.to(torch.int64) * 8).to(torch.uint8)  # in [0, 15]

    # 4) reshape back to ``(..., M, N)`` then pack pairs along last dim.
    code = code.reshape(*leading, M, N)
    pair = code.reshape(*leading, M, N // 2, 2)
    packed = (pair[..., 0] | (pair[..., 1] << 4)).view(torch.int8).contiguous()

    scale = scale_fp32.to(torch.float8_e8m0fnu)
    return packed, scale


def quant_fp8_e4m3_scale_e8m0(
    value: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to ``float8_e4m3fn`` + E8M0 per-block scales.

    Inverse of HF ``Fp8Dequantize._dequantize_one`` (e4m3 + E8M0 branch).
    DSV4-Flash dense linears use ``(128, 128)`` blocks.

    Parameters
    ----------
    value : torch.Tensor
        Shape ``(..., M, N)``. ``M % block_m == 0``, ``N % block_n == 0``.
    block_size : tuple[int, int]

    Returns
    -------
    quant : torch.Tensor
        dtype ``float8_e4m3fn``, shape ``(..., M, N)``.
    scale : torch.Tensor
        dtype ``float8_e8m0fnu``, shape ``(..., M//block_m, N//block_n)``.
    """
    M, N = value.shape[-2:]
    bm, bn = block_size
    assert M % bm == 0 and N % bn == 0, (
        f"value shape ({M}, {N}) not divisible by block ({bm}, {bn})"
    )
    # NaN guard: see `quant_fp4_e2m1_scale_e8m0_packed` — same poison chain
    # (NaN max_abs → torch.where flips scale back to 1.0 → quant = NaN e4m3).
    assert not torch.isnan(value).any(), "quant_fp8_e4m3_scale_e8m0: input contains NaN"
    leading = value.shape[:-2]
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0

    blk = value.float().reshape(*leading, M // bm, bm, N // bn, bn)
    max_abs = blk.abs().amax(dim=(-3, -1))                # (..., M/bm, N/bn)

    safe_max = max_abs.clamp(min=torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(safe_max / fp8_max))
    scale_fp32 = torch.pow(2.0, exponent)
    scale_fp32 = torch.where(max_abs > 0, scale_fp32, torch.ones_like(scale_fp32))

    scaled = blk / scale_fp32.unsqueeze(-1).unsqueeze(-3)
    # Clamp belt-and-suspenders against fp32→fp8 rounding-up edge cases
    # that overshoot ``fp8_max`` by < 1 ulp; ceil-rounded scale guarantees
    # ``|scaled| <= fp8_max`` mathematically, but the cast can still tip.
    quant = scaled.clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
    quant = quant.reshape(*leading, M, N).contiguous()

    scale = scale_fp32.to(torch.float8_e8m0fnu)
    return quant, scale


# ---------------------------------------------------------------------------
# Dequant — verbatim mirror of HF ``Fp8Dequantize._dequantize_one`` (v5.8.1).
# Do NOT edit the math; re-sync from HF if the upstream implementation changes.
# ---------------------------------------------------------------------------


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Two ``e2m1`` FP4 values per byte → float32 tensor twice as wide on the last dim."""
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=packed.device)
    u8 = packed.contiguous().view(torch.uint8)
    low = (u8 & 0xF).long()
    high = ((u8 >> 4) & 0xF).long()
    unpacked = torch.stack([lut[low], lut[high]], dim=-1)
    return unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])


def dequant_fp4_e2m1_fp8_scale_e8m0_packed(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Mirror of HF ``Fp8Dequantize._dequantize_one`` (FP4 + FP8 branches).

    Inlined so tests don't need to construct the full HF quantizer pipeline.
    Logic copied verbatim from
    ``transformers/integrations/finegrained_fp8.py::_dequantize_one`` (v5.8.1).
    """
    # FP4 path: int8 / float4_e2m1fn_x2 stores two nibbles per byte. Unpack to fp32
    # first so the rest of the routine sees a normal (rows, cols) float matrix.
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
        quantized_fp32 = _unpack_fp4(quantized)
    else:
        quantized_fp32 = quantized.to(torch.float32)
    rows, cols = quantized_fp32.shape[-2:]
    # Derive block size from the scale grid so the same dequant handles
    # both MXFP4 (1×32) and FP8 (128×128) blocks.
    scale_rows, scale_cols = scales.shape[-2:]
    if rows % scale_rows or cols % scale_cols:
        raise ValueError(
            f"Weight shape ({rows}, {cols}) not divisible by scale grid ({scale_rows}, {scale_cols})."
        )
    block_m = rows // scale_rows
    block_n = cols // scale_cols
    # E8M0 has no CUDA mul kernel; promote both sides to fp32.
    # Emit in scales.dtype if it's a real float >= 2 bytes, otherwise bf16.
    out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
    original_shape = quantized_fp32.shape
    q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    s = scales.to(torch.float32).reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
    return (q * s).to(out_dtype).reshape(original_shape)