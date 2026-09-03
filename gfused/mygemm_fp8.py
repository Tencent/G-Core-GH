import math

try:
    import tilelang as tl
    import tilelang.language as T
except ImportError:
    from gfused.fake_tilelang_stub import tilelang_stub as tl, language_stub as T

import torch


@tl.jit
def gemm_impl(
    A,
    B,
    block_M=128,
    block_N=128,
    block_K=64,
    dtype=torch.bfloat16,
    out_dtype=torch.bfloat16,
    accum_dtype=torch.float32,
):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((N, K), dtype)
    C = T.empty((M, N), dtype=out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_N, block_K), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


@tl.jit
def gemm_fp8_block_scaling_impl(
    Aq,
    As,
    Bq,
    Bs,
    BLK_M=64,
    BLK_N=128,
    BLK_K=128,
    FP8_BLK_SZ=128,
    dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float8_e8m0fnu,
    out_dtype=torch.bfloat16,
    accum_dtype=torch.float32,
):
    # ref: https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py
    assert max(BLK_N, FP8_BLK_SZ) % min(BLK_N, FP8_BLK_SZ) == 0
    assert BLK_K == FP8_BLK_SZ

    M, N, K = T.const("M, N, K")
    Aq: T.Tensor((M, K), dtype)
    As: T.Tensor((M, K // FP8_BLK_SZ), scale_dtype)
    Bq: T.Tensor((N, K), dtype)
    Bs: T.Tensor((N // FP8_BLK_SZ, K // FP8_BLK_SZ), scale_dtype)
    C = T.empty((M, N), dtype=out_dtype)

    with T.Kernel(T.ceildiv(M, BLK_M), T.ceildiv(N, BLK_N), threads=128) as (bx, by):
        Aq_shared = T.alloc_shared((BLK_M, BLK_K), dtype)
        As_shared = T.alloc_shared((BLK_M, 1), scale_dtype)
        Bq_shared = T.alloc_shared((BLK_N, BLK_K), dtype)
        Bs_shared = T.alloc_shared((T.ceildiv(BLK_N, FP8_BLK_SZ), 1), scale_dtype)
        C_local = T.alloc_fragment((BLK_M, BLK_N), accum_dtype)
        C_local_acc = T.alloc_fragment((BLK_M, BLK_N), accum_dtype)

        T.clear(C_local_acc)
        for k in T.Pipelined(T.ceildiv(K, BLK_K), num_stages=3):
            T.copy(Aq[bx * BLK_M, k * BLK_K], Aq_shared)
            T.copy(As[bx * BLK_M, k], As_shared)
            T.copy(Bq[by * BLK_N, k * BLK_K], Bq_shared)
            T.copy(Bs[by * BLK_N // FP8_BLK_SZ, k], Bs_shared)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.gemm(Aq_shared, Bq_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(BLK_M, BLK_N):
                C_local_acc[i, j] += C_local[i, j] * (
                    T.Cast(accum_dtype, As_shared[i, 0]) *
                    T.Cast(accum_dtype, Bs_shared[j // FP8_BLK_SZ, 0])
                )

        T.copy(C_local_acc, C[bx * BLK_M, by * BLK_N])

    return C
