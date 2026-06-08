"""
A emtpy template for tilelang fused op.

Example usage:

    cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/wepsdl/gcore-dev
    python3 tests/test_gfused/test_myfa_mask.py
"""

import math
import unittest

import einops
import pytest
import tilelang as tl
import tilelang.language as T
import torch
from flash_attn import flash_attn_func


def ref_eager_attn(q, k, v, mask, scaling):
    b, h, sq, d = q.shape
    s = torch.matmul(q, einops.rearrange(k, 'b h s d -> b h d s')) * scaling
    s = s + mask
    p = torch.nn.functional.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return o


@tl.jit(out_idx=[-2, -1])
def myfa_mask_fwd(
    H,
    D,
    scaling=1.0,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    num_stages=3,
    BLOCK_Q=64,
    BLOCK_K=64,
):
    assert D == tl.math.next_power_of_2(D)
    B = T.dynamic('B', torch.int32)
    LQ = T.dynamic('LQ', torch.int32)
    LK = T.dynamic('LK', torch.int32)

    scaling = scaling * 1.44269504  # log2(e)

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, H, LQ, D), dtype),
        K: T.Tensor((B, H, LK, D), dtype),
        V: T.Tensor((B, H, LK, D), dtype),
        O: T.Tensor((B, H, LQ, D), dtype),
        LSE: T.Tensor((B, H, LQ), dtype),
    ):
        with T.Kernel(B, H, T.ceildiv(LQ, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
            Q_i = T.alloc_shared((BLOCK_Q, D), dtype)
            K_i = T.alloc_shared((BLOCK_K, D), dtype)
            V_i = T.alloc_shared((BLOCK_K, D), dtype)
            # M_i = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            SE_i = T.alloc_fragment((BLOCK_Q,), acc_dtype)

            acc_s = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            acc_s_as_dtype = T.alloc_fragment((BLOCK_Q, BLOCK_K), dtype)
            m = T.alloc_fragment((BLOCK_Q,), acc_dtype)
            m_prev = T.alloc_fragment((BLOCK_Q,), acc_dtype)
            alpha = T.alloc_fragment((BLOCK_Q,), acc_dtype)
            tmp_sum = T.alloc_fragment((BLOCK_Q,), acc_dtype)
            acc_o = T.alloc_fragment((BLOCK_Q, D), acc_dtype)

            T.fill(m, float('-inf'))
            T.fill(SE_i, 0)
            T.fill(acc_o, 0)

            q_beg = q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(Q[b_i, h_i, q_beg:q_end, :], Q_i)

            loop_range = T.ceildiv(q_end, BLOCK_K)
            for i in T.Pipelined(loop_range, num_stages=num_stages):
                k_beg = i * BLOCK_K
                k_end = k_beg + BLOCK_K
                T.copy(K[b_i, h_i, k_beg:k_end, :], K_i)

                for j, k in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[j, k] = T.if_then_else(
                        q_beg + j >= k_beg + k,
                        0.,
                        -T.infinity(acc_s.dtype),
                    )
                T.gemm(Q_i, K_i, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # scaling
                for j, k in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[j, k] = acc_s[j, k] * scaling

                # fix m
                T.copy(m, m_prev)
                T.reduce_max(acc_s, m, dim=1, clear=True)
                for j in T.Parallel(BLOCK_Q):
                    m[j] = T.max(m[j], m_prev[j])
                
                # alpha
                for j in T.Parallel(BLOCK_Q):
                    alpha[j] = T.exp2((m_prev[j] - m[j]))

                # exp(s)
                for j, k in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[j, k] = T.exp2(acc_s[j, k] - m[j])
                T.copy(acc_s, acc_s_as_dtype)

                # SE and fix SE
                T.reduce_sum(acc_s, tmp_sum, dim=1, clear=True) # sum exp(s)
                for j in T.Parallel(BLOCK_Q):
                    SE_i[j] = SE_i[j] * alpha[j] + tmp_sum[j]

                # sv
                for j, k in T.Parallel(BLOCK_Q, D):
                    acc_o[j, k] = alpha[j] * acc_o[j, k]
                T.copy(V[b_i, h_i, k_beg:k_end, :], V_i)
                T.gemm(acc_s_as_dtype, V_i, acc_o, policy=T.GemmWarpPolicy.FullRow)
            
            # o = a / l
            for j, k in T.Parallel(BLOCK_Q, D):
                acc_o[j, k] = acc_o[j, k] / SE_i[j]
            T.copy(acc_o, O[b_i, h_i, q_beg:q_end, :])

            # lse
            for j in T.Parallel(BLOCK_Q):
                SE_i[j] = m[j] + T.log(SE_i[j])
            T.copy(SE_i, LSE[b_i, h_i, q_beg:q_end])

    return kernel


@torch.no_grad()
def main():
    b = 2
    h = 4
    s = 32 * 1024
    d = 128
    scaling = 1.0 / math.sqrt(d)

    causal_mask = torch.triu(
        torch.full(
            (b, 1, s, s), float("-inf"), device='cuda', dtype=torch.bfloat16
        ),
        diagonal=1
    )

    rng = torch.Generator(device='cuda').manual_seed(1919810)
    q = torch.randn(
        b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16
    )

    ref_o = ref_eager_attn(q, k, v, causal_mask, scaling)

    myfa_fwd_kernel = myfa_mask_fwd.compile(H=h, D=d, scaling=scaling)
    my_o, my_lse = myfa_fwd_kernel(q, k, v)

    q_bshd = q.transpose(1, 2)
    k_bshd = k.transpose(1, 2)
    v_bshd = v.transpose(1, 2)
    fa2_o = flash_attn_func(
        q_bshd, k_bshd, v_bshd,
        causal=True, softmax_scale=scaling,
    ).transpose(1, 2)

    print('my  vs ref | avg', (my_o - ref_o).abs().mean().item(),  '| max', (my_o - ref_o).abs().max().item())
    print('fa2 vs ref | avg', (fa2_o - ref_o).abs().mean().item(), '| max', (fa2_o - ref_o).abs().max().item())
    print('my  vs fa2 | avg', (my_o - fa2_o).abs().mean().item(),  '| max', (my_o - fa2_o).abs().max().item())

    # warmup
    for _ in range(10):
        ref_eager_attn(q, k, v, causal_mask, scaling)
        myfa_fwd_kernel(q, k, v)
        flash_attn_func(q_bshd, k_bshd, v_bshd, causal=True, softmax_scale=scaling)
    torch.cuda.synchronize()

    n_iter = 8
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    t0.record()
    for _ in range(n_iter):
        ref_eager_attn(q, k, v, causal_mask, scaling)
    t1.record()
    torch.cuda.synchronize()
    eager_ms = t0.elapsed_time(t1) / n_iter

    t0.record()
    for _ in range(n_iter):
        myfa_fwd_kernel(q, k, v)
    t1.record()
    torch.cuda.synchronize()
    my_ms = t0.elapsed_time(t1) / n_iter

    t0.record()
    for _ in range(n_iter):
        flash_attn_func(q_bshd, k_bshd, v_bshd, causal=True, softmax_scale=scaling)
    t1.record()
    torch.cuda.synchronize()
    fa2_ms = t0.elapsed_time(t1) / n_iter

    print(f'eager: {eager_ms:.4f} ms')
    print(f'myfa : {my_ms:.4f} ms')
    print(f'fa2  : {fa2_ms:.4f} ms')


main()