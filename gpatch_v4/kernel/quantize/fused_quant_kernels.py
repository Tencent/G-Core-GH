# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""TileLang kernels used by DeepSeek-V4 weight conversion."""

import tilelang
import tilelang.language as T
import torch

__all__ = ["fp4_qat_then_to_fp8", "fp_qat_128x128"]

FP4 = "float4_e2m1fn"
FP8 = "float8_e4m3"
E8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

_FP4_GROUP_SIZE = 32
_FP8_BLOCK_SIZE = 128
_FP4_MAX = 6.0
_FP8_MAX = 448.0
_FP8_REBASE_OFFSET = 64.0
_MIN_E8M0_SCALE = 2.0**-126
_STRIPE_N = 32

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


def _fast_log2_ceil(x):
    """Compute ceil(log2(x)) from the IEEE-754 FP32 representation."""
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast(INT32, exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def _fast_pow2(x):
    """Compute 2**x by constructing an IEEE-754 FP32 value."""
    bits_x = (x + 127) << 23
    return T.reinterpret(FP32, bits_x)


def _round_fp4_scale(amax):
    """Return the E8M0-compatible FP4 scale used by the PyTorch reference."""
    return T.if_then_else(
        amax > 0,
        T.max(_fast_pow2(_fast_log2_ceil(amax / _FP4_MAX)), _MIN_E8M0_SCALE),
        1.0,
    )


def _round_fp8_tile_scale(amax):
    """Match ``fp_quantize.quant_fp8_e4m3_scale_e8m0`` E8M0 tile scale."""
    # Non-zero tiles: 2^ceil(log2(amax/448)); all-zero tiles: 1.0.
    return T.if_then_else(
        amax > 0,
        _fast_pow2(_fast_log2_ceil(T.max(amax, 1.0e-30) / _FP8_MAX)),
        1.0,
    )


def _cast_fp4_bucketize_right_false(x):
    abs_x = T.if_then_else(x < 0, -x, x)

    # 普通值继续使用硬件 FP4 cast
    fp4_value = T.Cast(FP32, T.Cast(FP4, x))

    # 精确 midpoint 强制取较低档
    lower = T.if_then_else(
        abs_x == 0.25,
        0.0,
        T.if_then_else(
            abs_x == 0.75,
            0.5,
            T.if_then_else(
                abs_x == 1.25,
                1.0,
                T.if_then_else(
                    abs_x == 1.75,
                    1.5,
                    T.if_then_else(
                        abs_x == 2.5,
                        2.0,
                        T.if_then_else(
                            abs_x == 3.5,
                            3.0,
                            T.if_then_else(abs_x == 5.0, 4.0, -1.0),
                        ),
                    ),
                ),
            ),
        ),
    )

    corrected = T.if_then_else(x < 0, -lower, lower)
    return T.if_then_else(lower >= 0, corrected, fp4_value)


@tilelang.jit(pass_configs=pass_configs)
def _fp4_qat_then_to_fp8_kernel(
    n: int,
    threads: int = 512,
):
    """Build an FP32 -> FP4-Q/DQ -> SGLang-FP8 conversion kernel.

    One CTA owns one 128x128 SGLang FP8 scale tile. FP4 values are kept
    unpacked in shared memory because packing nibbles is unnecessary when the
    final destination is E4M3.
    """
    m = T.symbolic("m")
    fp4_groups_per_tile = _FP8_BLOCK_SIZE // _FP4_GROUP_SIZE

    @T.prim_func
    def kernel(
        weight: T.Tensor[(m, n), FP32],
        fp8_weight: T.Tensor[(m, n), FP8],
        fp8_scale: T.Tensor[
            (m // _FP8_BLOCK_SIZE, n // _FP8_BLOCK_SIZE),
            E8M0,
        ],
    ):
        with T.Kernel(
            m // _FP8_BLOCK_SIZE,
            n // _FP8_BLOCK_SIZE,
            threads=threads,
        ) as (pid_m, pid_n):
            # Only one 128x32 input stripe is live at a time. q4_shared stores
            # normalized E2M1 values (not dequantized values) for the rebase.
            x_shared = T.alloc_shared(
                (_FP8_BLOCK_SIZE, _FP4_GROUP_SIZE),
                FP32,
            )
            q4_shared = T.alloc_shared(
                (_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE),
                BF16,
            )

            x_local = T.alloc_fragment(
                (_FP8_BLOCK_SIZE, _FP4_GROUP_SIZE),
                FP32,
            )
            q4_local = T.alloc_fragment(
                (_FP8_BLOCK_SIZE, _FP4_GROUP_SIZE),
                BF16,
            )
            q8_local = T.alloc_fragment(
                (_FP8_BLOCK_SIZE, _FP4_GROUP_SIZE),
                FP8,
            )
            amax_local = T.alloc_fragment((_FP8_BLOCK_SIZE, ), FP32)
            scale_local = T.alloc_fragment((_FP8_BLOCK_SIZE, ), FP32)
            # Layout fix: retain all four 1x32 scales in one fragment. The
            # previous shared-memory columns were written inside a serial loop
            # and then copied as a 2D tensor, which LayoutInference rejected.
            scales_local = T.alloc_fragment(
                (_FP8_BLOCK_SIZE, fp4_groups_per_tile),
                FP32,
            )

            # Stage 1: quantize the FP32 master weight directly; each row has
            # one independent E2M1 scale per 32 elements.
            for group in T.serial(fp4_groups_per_tile):
                T.copy(
                    weight[
                        pid_m * _FP8_BLOCK_SIZE,
                        pid_n * _FP8_BLOCK_SIZE + group * _FP4_GROUP_SIZE,
                    ],
                    x_shared,
                )
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)

                for i in T.Parallel(_FP8_BLOCK_SIZE):
                    scale_local[i] = _round_fp4_scale(amax_local[i])
                    scales_local[i, group] = scale_local[i]

                for i, j in T.Parallel(
                    _FP8_BLOCK_SIZE,
                    _FP4_GROUP_SIZE,
                ):
                    q4_local[i, j] = T.Cast(
                        BF16,
                        _cast_fp4_bucketize_right_false(
                            T.clamp(
                                x_local[i, j] / scale_local[i],
                                -_FP4_MAX,
                                _FP4_MAX,
                            ),
                        ),
                    )
                T.copy(
                    q4_local,
                    q4_shared[0, group * _FP4_GROUP_SIZE],
                )
                T.sync_threads()

            # Stage 2: one E8M0 scale is shared by the full 128x128 FP8 tile.
            row_scale_max = T.alloc_fragment((_FP8_BLOCK_SIZE, ), FP32)
            tile_scale_max = T.alloc_fragment((1, ), FP32)
            outer_scale = T.alloc_fragment((1, ), FP32)

            T.reduce_max(scales_local, row_scale_max, dim=1, clear=True)
            T.reduce_max(row_scale_max, tile_scale_max, dim=0, clear=True)
            outer_scale[0] = T.max(
                tile_scale_max[0] / _FP8_REBASE_OFFSET,
                _MIN_E8M0_SCALE,
            )
            fp8_scale[pid_m, pid_n] = T.Cast(E8M0, outer_scale[0])

            # Stage 3: fold each FP4 group's power-of-two scale offset into
            # E4M3, matching examples/nrwu/deepseek_v4_inference/convert.py.
            for group in T.serial(fp4_groups_per_tile):
                T.copy(
                    q4_shared[0, group * _FP4_GROUP_SIZE],
                    q4_local,
                )
                for i, j in T.Parallel(
                    _FP8_BLOCK_SIZE,
                    _FP4_GROUP_SIZE,
                ):
                    q8_local[i, j] = T.Cast(
                        FP8,
                        T.Cast(FP32, q4_local[i, j]) * scales_local[i, group] / outer_scale[0],
                    )
                T.copy(
                    q8_local,
                    fp8_weight[
                        pid_m * _FP8_BLOCK_SIZE,
                        pid_n * _FP8_BLOCK_SIZE + group * _FP4_GROUP_SIZE,
                    ],
                )

    return kernel


def fp4_qat_then_to_fp8(weight: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an FP32 QAT master weight to SGLang's FP8 representation.

    The conversion is equivalent to:

    1. E2M1 FP4 quantization of the FP32 input with one E8M0 scale per
       1x32 group.
    2. Rebase into E4M3 with one E8M0 scale per 128x128 tile.

    Returns ``(fp8_weight, fp8_scale)``.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight, got shape {tuple(weight.shape)}")
    if weight.dtype != torch.float32:
        raise TypeError(f"Expected an FP32 master weight, got {weight.dtype}")
    if not weight.is_cuda:
        raise ValueError("fp4_qat_then_to_fp8 requires a CUDA tensor")

    m, n = weight.shape
    if m % _FP8_BLOCK_SIZE != 0 or n % _FP8_BLOCK_SIZE != 0:
        raise ValueError(
            "FP4-to-SGL FP8 conversion requires 128x128-aligned weights, "
            f"got {(m, n)}"
        )

    weight = weight.contiguous()
    fp8_weight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    fp8_scale = torch.empty(
        (m // _FP8_BLOCK_SIZE, n // _FP8_BLOCK_SIZE),
        dtype=torch.float8_e8m0fnu,
        device=weight.device,
    )

    kernel = _fp4_qat_then_to_fp8_kernel(n)
    kernel(weight, fp8_weight, fp8_scale)
    return fp8_weight, fp8_scale


@tilelang.jit(pass_configs=pass_configs)
def _fp8_qat_128x128_kernel(n: int, threads: int = 512):
    """Fused E4M3 128×128 Q/DQ → BF16 (one CTA per tile).

    Matches ``quant_fp8_e4m3_scale_e8m0`` + dequant on the BF16 view: one
    power-of-two scale per 128×128 block, then cast through ``float8_e4m3``.

    Layout follows ``_fp4_qat_then_to_fp8_kernel``: stream 128×32 stripes from
    global → shared → fragment (no full-tile shared staging).
    """
    m = T.symbolic("m")
    n_stripes = _FP8_BLOCK_SIZE // _STRIPE_N
    fp8_min = -_FP8_MAX
    fp8_max = _FP8_MAX

    @T.prim_func
    def kernel(
        weight: T.Tensor[(m, n), BF16],
        out: T.Tensor[(m, n), BF16],
    ):
        with T.Kernel(
            m // _FP8_BLOCK_SIZE,
            n // _FP8_BLOCK_SIZE,
            threads=threads,
        ) as (pid_m, pid_n):
            x_shared = T.alloc_shared((_FP8_BLOCK_SIZE, _STRIPE_N), BF16)
            y_shared = T.alloc_shared((_FP8_BLOCK_SIZE, _STRIPE_N), BF16)
            x_local = T.alloc_fragment((_FP8_BLOCK_SIZE, _STRIPE_N), BF16)
            y_local = T.alloc_fragment((_FP8_BLOCK_SIZE, _STRIPE_N), BF16)
            stripe_amax = T.alloc_fragment((_FP8_BLOCK_SIZE, ), FP32)
            row_amax = T.alloc_fragment((_FP8_BLOCK_SIZE, ), FP32)
            tile_amax = T.alloc_fragment((1, ), FP32)
            scale = T.alloc_fragment((1, ), FP32)

            for i in T.Parallel(_FP8_BLOCK_SIZE):
                row_amax[i] = 0.0

            # Pass 1: tile absmax via 128×32 stripes (same copy shape as fp4).
            for stripe in T.serial(n_stripes):
                T.copy(
                    weight[
                        pid_m * _FP8_BLOCK_SIZE,
                        pid_n * _FP8_BLOCK_SIZE + stripe * _STRIPE_N,
                    ],
                    x_shared,
                )
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, stripe_amax, dim=1)
                for i in T.Parallel(_FP8_BLOCK_SIZE):
                    row_amax[i] = T.max(row_amax[i], stripe_amax[i])
                T.sync_threads()

            T.reduce_max(row_amax, tile_amax, dim=0, clear=True)
            scale[0] = _round_fp8_tile_scale(tile_amax[0])

            # Pass 2: Q/DQ each stripe and store through shared (act_quant style).
            for stripe in T.serial(n_stripes):
                T.copy(
                    weight[
                        pid_m * _FP8_BLOCK_SIZE,
                        pid_n * _FP8_BLOCK_SIZE + stripe * _STRIPE_N,
                    ],
                    x_shared,
                )
                T.copy(x_shared, x_local)
                for i, j in T.Parallel(_FP8_BLOCK_SIZE, _STRIPE_N):
                    y_local[i, j] = T.Cast(
                        BF16,
                        T.Cast(
                            FP32,
                            T.Cast(
                                FP8,
                                T.clamp(
                                    T.Cast(FP32, x_local[i, j]) / scale[0],
                                    fp8_min,
                                    fp8_max,
                                ),
                            ),
                        ) * scale[0],
                    )
                T.copy(y_local, y_shared)
                T.copy(
                    y_shared,
                    out[
                        pid_m * _FP8_BLOCK_SIZE,
                        pid_n * _FP8_BLOCK_SIZE + stripe * _STRIPE_N,
                    ],
                )
                T.sync_threads()

    return kernel


def fp_qat_128x128(weight: torch.Tensor) -> torch.Tensor:
    """Fused 128×128 FP8 E4M3 Q/DQ on a BF16 weight (CUDA).

    Mathematically equivalent to::

        q, s = quant_fp8_e4m3_scale_e8m0(weight.float(), (128, 128))
        dequant_fp4_e2m1_fp8_scale_e8m0_packed(q, s)  # bf16

    Parameters
    ----------
    weight : torch.Tensor
        CUDA ``bfloat16``, shape ``(M, N)`` with ``M, N`` divisible by 128.

    Returns
    -------
    torch.Tensor
        BF16 tensor, same shape as ``weight``.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight, got shape {tuple(weight.shape)}")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"Expected bfloat16 weight, got {weight.dtype}")
    if not weight.is_cuda:
        raise ValueError("fp_qat_128x128 requires a CUDA tensor")

    m, n = weight.shape
    if m % _FP8_BLOCK_SIZE != 0 or n % _FP8_BLOCK_SIZE != 0:
        raise ValueError(f"fp_qat_128x128 requires 128x128-aligned weights, got {(m, n)}")

    weight = weight.contiguous()
    out = torch.empty_like(weight)
    kernel = _fp8_qat_128x128_kernel(n)
    kernel(weight, out)
    return out
