# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""FP8 and FP4 fake-quantization for QAT (Quantization-Aware Training).

Simulates precision loss from FP8 E4M3 and FP4 E2M1 block-wise
quantization during training. Gradients pass through unchanged via
Straight-Through Estimator (STE).

Insertion points mirror DeepSeek-V4's inference KV quantization:
  - Main attention KV (nope dims only, block=64)
  - Compressor compressed KV (nope dims only, block=64)
  - Indexer query and compressed keys (nope dims only, block=64)

Quantization kernel
-------------------
``act_quant`` is ported verbatim from
``deepseek-ai/DeepSeek-V4-Pro/inference/kernel.py`` (via
``radixark/miles`` ``ops/kernel/act_quant.py``). It is mathematically
equivalent to ``tile_kernels.quant.per_token_cast`` with
``round_sf=True`` — both compute ``2^ceil(log2(amax/448))`` via IEEE 754
bit manipulation. We keep the ``act_quant`` copy because:

  1. It matches the exact kernel used by DeepSeek inference (bit-exact
     parity with upstream).
  2. It supports arbitrary-shape input (``per_token_cast`` requires 2D).
  3. It supports ``inplace=True`` for fused quant+dequant (not used by
     QAT but useful for future extensions).

Dequantization uses ``tile_kernels.quant.per_token_cast_back``.

FP4 QAT uses the official reference's E2M1 1×32 quantize→dequantize
kernel with power-of-two scales.

Weight FP8 QAT (``fp8_qat_linear``, MoE / shared-expert weights, and
``wo_a`` under ``fp4_qat``) uses ``fp8_simulate_qat_128x128`` (fused TileLang
E4M3 + E8M0 128×128 Q/DQ), matching sglang FineGrainedFP8 storage. KV /
activation fake-quant stays on the 1×N ``act_quant`` path
(``fp8_simulate_qat(..., 64)``).

``act_quant`` vs ``per_token_cast`` differences (for reference):
  - tile size: act_quant fixed ``blk_m=32``; per_token_cast adaptive
  - vectorize: act_quant none; per_token_cast has ``annotate_layout``
  - pipeline: act_quant ``T.Pipelined``; per_token_cast predicated load
  - pre-quantized input: act_quant no; per_token_cast yes
  - reduction: act_quant single-pass; per_token_cast two-stage (fp16→fp32)
  - QAT output: bit-identical (absmax + pow2 ceil + clamp + fp8 cast)
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

try:
    from tile_kernels.quant import per_token_cast_back
except Exception as e:
    per_token_cast_back = None
    print(f"Error importing per_token_cast_back from tile_kernels")

from torch import nn

from .eager_quant_kernels import (
    dequant_fp4_e2m1_fp8_scale_e8m0_packed,
    quant_fp8_e4m3_scale_e8m0,
)
from .fused_quant_kernels import _cast_fp4_bucketize_right_false, fp_qat_128x128

__all__ = [
    "fp4_qat_linear",
    "fp4_simulate_qat",
    "fp8_qat_linear",
    "fp8_simulate_qat",
    "fp8_simulate_qat_128x128",
]

_FP8_WEIGHT_BLOCK = (128, 128)

# ---------------------------------------------------------------------------
# act_quant — ported from deepseek-ai/DeepSeek-V4-Pro/inference/kernel.py
# via radixark/miles miles_plugins/models/deepseek_v4/ops/kernel/act_quant.py
# ---------------------------------------------------------------------------

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
E8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"


def fast_log2_ceil(x):
    """Compute ceil(log2(x)) via IEEE 754 bit manipulation. Avoids slow log/ceil intrinsics."""
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    """Compute 2^x for integer x via IEEE 754 bit manipulation."""
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N,
    block_size=128,
    in_dtype=BF16,
    out_dtype=FP8,
    scale_dtype=FP32,
    round_scale=False,
    inplace=False
):
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16."""
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale or inplace else 2
    blk_m = 32
    group_size = block_size
    # Internal computation in FP32; scale_dtype controls output storage format.
    compute_dtype = FP32
    out_dtype = in_dtype if inplace else out_dtype

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m, ), compute_dtype)
            s_local = T.alloc_fragment((blk_m, ), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(
                                compute_dtype,
                                T.Cast(
                                    out_dtype,
                                    T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
                                )
                            ) * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP8 quantization with fp32 scales.

    When ``scale_fmt`` is set (e.g. ``"ue8m0"``), scales are rounded to
    power-of-2 (matching DeepSeek inference). ``inplace=True`` does fused
    quant+dequant back to BF16.
    """
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else torch.empty_like(z, dtype=torch.float8_e4m3fn)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=torch.float32)
    kernel = act_quant_kernel(
        N,
        block_size,
        scale_dtype=FP32,
        round_scale=scale_fmt is not None,
        inplace=inplace,
    )
    kernel(z.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s


# 一定要小心 round up / down 的问题，参考这里的教训：
# https://git.woa.com/wepsdl/gcore-dev/-/merge_requests/1652/diff/gpatch_v4/models/deepseek_v4/qat.py?sourceObjectId=acc32a379df7288b12a4535cf154e8aa1c529f2b&targetObjectId=5aa6b38ccc51bf017a99fe41f98b59c90cc33965
# 这段代码是 AI 参考 @gcore-dev/examples/nrwu/deepseek_v4_inference/kernel.py:128-183 写的，不知道为什么 AI 决定要魔改而改出了 BUG，quant 逻辑 AI 用 tilelang.FP4，midpoint
# quant 逻辑不同，导致 TI mismatch，没注意看到实在是无语... 目前 xiaotao 已经 fix 好了。
# TODO(@xiaotaoliu): 还有点小问题，midpoint 处理根据 tile-kernels 应该固定向下才对。


@tilelang.jit(pass_configs=pass_configs)
def _fp4_quant_kernel(
    N,
    block_size=32,
    in_dtype=BF16,
):
    M = T.symbolic("M")
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    blk_m = 32

    @T.prim_func
    def fp4_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), in_dtype],
        S: T.Tensor[(M, T.ceildiv(N, block_size)), E8M0],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, block_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, block_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, block_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m, ), FP32)
            scale_local = T.alloc_fragment((blk_m, ), FP32)
            y_local = T.alloc_fragment((blk_m, block_size), in_dtype)
            y_shared = T.alloc_shared((blk_m, block_size), in_dtype)

            for _ in T.Pipelined(1, num_stages=2):
                T.copy(X[pid_m * blk_m, pid_n * block_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
                    scale_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
                for i, j in T.Parallel(blk_m, block_size):
                    # 与导出 kernel / 盘上量化器共用 midpoint 规则：硬件 cast 走 RNE，
                    # 会在 0.75/1.75/3.5 上选到与 fp_quantize.py 的 bucketize 相反的一档。
                    y_local[i, j] = T.Cast(
                        in_dtype,
                        _cast_fp4_bucketize_right_false(
                            T.clamp(
                                x_local[i, j] / scale_local[i],
                                -fp4_max,
                                fp4_max,
                            ),
                        ) * scale_local[i],
                    )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(E8M0, scale_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * block_size])

    return fp4_quant_kernel_


# ---------------------------------------------------------------------------
# fp8_simulate — quant (act_quant) + dequant (per_token_cast_back)
# ---------------------------------------------------------------------------


def _fp8_simulate(x: torch.Tensor, block_size: int) -> torch.Tensor:
    x_c = x.contiguous()
    y, scale = act_quant(x_c, block_size, "ue8m0")

    N = x_c.size(-1)
    y_flat = y.view(-1, N)
    scale_flat = scale.reshape(y_flat.size(0), N // block_size).contiguous()

    out_flat = per_token_cast_back(
        (y_flat, scale_flat),
        'bf16' if x.dtype == torch.bfloat16 else 'fp32',
        block_size,
    )
    return out_flat.view_as(x_c).to(x.dtype)


def _fp8_simulate_128x128_torch(x: torch.Tensor) -> torch.Tensor:
    """CPU / reference path: ``fp_quantize`` Q then dequant."""
    q, s = quant_fp8_e4m3_scale_e8m0(x.bfloat16().float(), block_size=_FP8_WEIGHT_BLOCK)
    return dequant_fp4_e2m1_fp8_scale_e8m0_packed(q, s).to(dtype=x.dtype)


def _fp8_simulate_128x128(x: torch.Tensor) -> torch.Tensor:
    """E4M3 + E8M0 Q/DQ on the BF16 view with one scale per 128×128 tile.

    CUDA uses fused TileLang ``fp_qat_128x128``; otherwise falls back to the
    pure-torch ``fp_quantize`` reference. Rank-3+ weights (e.g. MoE
    ``[E, N, K]``) are viewed as ``(-1, K)`` before Q/DQ, matching
    :func:`quantize_fp8`'s layout, then restored.
    """
    x_c = x.contiguous()
    if x_c.ndim < 2:
        raise ValueError(f"fp8_simulate_qat_128x128 requires ndim>=2, got {tuple(x_c.shape)}")
    x_2d = x_c.view(-1, x_c.shape[-1])
    if x_2d.is_cuda:
        out_2d = fp_qat_128x128(x_2d.bfloat16()).to(dtype=x.dtype)
    else:
        # TODO(@xiaotaoliu) 这里有个坑，如果这里意外的是 cpu tensor，在非 debug 的情况会意外走进来...
        out_2d = _fp8_simulate_128x128_torch(x_2d)
    return out_2d.view_as(x_c)


class _FP8SimulateQAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
        return _fp8_simulate(x, block_size)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


fp8_simulate_qat = _FP8SimulateQAT.apply


class _FP8SimulateQAT128x128(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return _fp8_simulate_128x128(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad


fp8_simulate_qat_128x128 = _FP8SimulateQAT128x128.apply


def _fp4_simulate(x: torch.Tensor, block_size: int) -> torch.Tensor:
    assert x.dtype == torch.bfloat16, f"FP4 QAT requires bfloat16 input, got {x.dtype}"
    assert block_size == 32, f"FP4 QAT requires 1x32 blocks, got {block_size}"
    N = x.size(-1)
    assert N % block_size == 0

    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    scale = torch.empty(
        *x_c.size()[:-1],
        N // block_size,
        dtype=torch.float8_e8m0fnu,
        device=x.device,
    )
    kernel = _fp4_quant_kernel(N, block_size)
    kernel(x_c.view(-1, N), out.view(-1, N), scale.view(-1, N // block_size))
    return out


class _FP4SimulateQAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
        return _fp4_simulate(x, block_size)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


fp4_simulate_qat = _FP4SimulateQAT.apply

# ---------------------------------------------------------------------------
# fp8_qat_linear — nn.Linear forward with FP8 QAT fake-quant on the weight
# ---------------------------------------------------------------------------


def fp8_qat_linear(module: nn.Linear, x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """nn.Linear forward with FP8 QAT fake-quant on the weight tensor.

    Equivalent to ``module(x)`` but the weight is passed through
    ``fp8_simulate_qat_128x128`` (E4M3 + E8M0, 128×128 tiles, STE backward)
    before the GEMM.  The bias (if any) is applied unchanged.

    ``block_size`` is kept for call-site compatibility and must be 128.
    """
    assert block_size == 128, (
        f"fp8_qat_linear uses 128×128 weight tiles, got block_size={block_size}"
    )
    weight = fp8_simulate_qat_128x128(module.weight)
    return F.linear(x, weight, module.bias)


def fp4_qat_linear(module: nn.Linear, x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Run ``module`` with E2M1 FP4 fake-quantized weights.

    Parameters
    ----------
    module : nn.Linear
    x : torch.Tensor
        Must be bfloat16 because the FP4 TileLang kernel operates on
        bfloat16 inputs and weights.
    block_size : int, default=32
        Must be 32, the deployed FP4 block geometry.

    Returns
    -------
    torch.Tensor
        ``module`` output using the Q/DQ weight on this forward pass.

    Raises
    ------
    AssertionError
        If the weight is not bfloat16, ``block_size`` is not 32, or the input
        dimension is not divisible by 32.
    """
    weight = fp4_simulate_qat(module.weight, block_size)
    return F.linear(x, weight, module.bias)
