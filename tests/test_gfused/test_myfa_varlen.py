"""
myfa_mask forward / backward correctness tests.

Example usage::

    cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/wepsdl/gcore-dev
    pytest -s tests/test_gfused/test_myfa_mask_varlen.py -v
"""

import math

import einops
import pytest
import tilelang as tl
import tilelang.language as T
import torch
from flash_attn import flash_attn_func


def ref_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, scaling):
    o = torch.zeros_like(q)
    b = len(cu_seqlens_q) - 1
    for i in range(b):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        qi = q[:, qs:qe]  # (h, sq_i, d)
        ki = k[:, ks:ke]  # (h, sk_i, d)
        vi = v[:, ks:ke]  # (h, sk_i, d)
        s = torch.matmul(qi, ki.transpose(-2, -1)) * scaling
        p = torch.softmax(s, dim=-1)
        oi = torch.matmul(p, vi)
        o[:, qs:qe] = oi
    return o


@tl.jit(
    out_idx=[-2, -1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_varlen_fwd(
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

    @T.prim_func
    def kernel(
        Q: T.Tensor((H, LQ, D), dtype),
        K: T.Tensor((H, LK, D), dtype),
        V: T.Tensor((H, LK, D), dtype),
        cu_seqlens_q: T.Tensor((B,), torch.int32),
        cu_seqlens_k: T.Tensor((B,), torch.int32),
        max_seqlen: T.int32,
        O: T.Tensor((H, LQ, D), dtype),
        LSE: T.Tensor((H, LQ), acc_dtype),
    ):
        with T.Kernel(B - 1, H, T.ceildiv(max_seqlen, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
            q = T.alloc_shared((BLOCK_Q, D), dtype)
            k = T.alloc_shared((BLOCK_K, D), dtype)
            v = T.alloc_shared((BLOCK_K, D), dtype)
            se = T.alloc_fragment((BLOCK_Q, ), acc_dtype)

            acc_s = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            acc_s_cast = T.alloc_fragment((BLOCK_Q, BLOCK_K), dtype)
            m = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            m_prev = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            alpha = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            tmp_sum = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            acc_o = T.alloc_fragment((BLOCK_Q, D), acc_dtype)

            T.fill(m, float('-inf'))
            T.fill(se, 0)
            T.fill(acc_o, 0)

            q_beg = cu_seqlens_q[b_i] + q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(Q[h_i, q_beg:q_end, :], q)
            lk = cu_seqlens_k[b_i + 1] - cu_seqlens_k[b_i]

            for k_idx in T.Pipelined(T.ceildiv(lk, BLOCK_K), num_stages=num_stages):
                k_beg = cu_seqlens_k[b_i] + k_idx * BLOCK_K
                k_end = k_beg + BLOCK_K
                T.copy(K[h_i, k_beg:k_end, :], k)

                # s = qk^T
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    # mask
                    acc_s[i, j] = T.if_then_else(
                        k_beg + j < cu_seqlens_k[b_i] + lk,
                        0.,
                        -T.infinity(acc_s.dtype))
                T.gemm(q, k, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # scaling
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[i, j] = acc_s[i, j] * scaling

                # fix m
                T.copy(m, m_prev)
                T.reduce_max(acc_s, m, dim=1, clear=True)
                for i in T.Parallel(BLOCK_Q):
                    m[i] = T.max(m[i], m_prev[i])

                # alpha
                for i in T.Parallel(BLOCK_Q):
                    alpha[i] = T.exp(m_prev[i] - m[i])

                # exp(s)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[i, j] = T.exp(acc_s[i, j] - m[i])
                T.copy(acc_s, acc_s_cast)

                # l(SE) and fix l
                T.reduce_sum(acc_s, tmp_sum, dim=1, clear=True)  # sum exp(s)
                for i in T.Parallel(BLOCK_Q):
                    se[i] = se[i] * alpha[i] + tmp_sum[i]

                # a = pv
                for i, j in T.Parallel(BLOCK_Q, D):
                    acc_o[i, j] = alpha[i] * acc_o[i, j]
                T.copy(V[h_i, k_beg:k_end, :], v)
                T.gemm(acc_s_cast, v, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # o = a / l
            for i, j in T.Parallel(BLOCK_Q, D):
                acc_o[i, j] = acc_o[i, j] / se[i]
                if q_beg + i < cu_seqlens_q[b_i + 1]:
                    O[h_i, q_beg + i, j] = acc_o[i, j]

            # lse
            for i in T.Parallel(BLOCK_Q):
                se[i] = m[i] + T.log(se[i])
                if q_beg + i < cu_seqlens_q[b_i + 1]:
                    LSE[h_i, q_beg + i] = se[i]

    return kernel


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_varlen_bwd_pre(
    H,
    D,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    BLOCK_Q=64,
):
    assert D == tl.math.next_power_of_2(D)
    LQ = T.dynamic('LQ', torch.int32)

    @T.prim_func
    def kernel(
        O: T.Tensor((H, LQ, D), dtype),
        dO: T.Tensor((H, LQ, D), dtype),
        DELTA: T.Tensor((H, LQ), acc_dtype),
    ):
        with T.Kernel(H, T.ceildiv(LQ, BLOCK_Q), threads=threads) as (h_i, q_idx):
            o = T.alloc_fragment((BLOCK_Q, D), dtype)
            do = T.alloc_fragment((BLOCK_Q, D), dtype)
            acc = T.alloc_fragment((BLOCK_Q, D), acc_dtype)
            delta = T.alloc_fragment((BLOCK_Q, ), acc_dtype)

            # delta = sum_d(o * dO)
            q_beg = q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(O[h_i, q_beg:q_end, :], o)
            T.copy(dO[h_i, q_beg:q_end, :], do)
            for i, j in T.Parallel(BLOCK_Q, D):
                acc[i, j] = o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, dim=1, clear=True)
            T.copy(delta, DELTA[h_i, q_beg:q_end])

    return kernel


@tl.jit(pass_configs={
    tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, )
def myfa_varlen_bwd(
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

    @T.prim_func
    def kernel(
        Q: T.Tensor((H, LQ, D), dtype),
        K: T.Tensor((H, LK, D), dtype),
        V: T.Tensor((H, LK, D), dtype),
        cu_seqlens_q: T.Tensor((B,), torch.int32),
        cu_seqlens_k: T.Tensor((B,), torch.int32),
        max_seqlen: T.int32,
        LSE: T.Tensor((H, LQ), acc_dtype),
        dO: T.Tensor((H, LQ, D), dtype),
        DELTA: T.Tensor((H, LQ), acc_dtype),
        dQ: T.Tensor((H, LQ, D), acc_dtype),
        dK: T.Tensor((H, LK, D), dtype),
        dV: T.Tensor((H, LK, D), dtype),
    ):
        with T.Kernel(B - 1, H, T.ceildiv(max_seqlen, BLOCK_K), threads=threads) as (b_i, h_i, k_idx):
            q = T.alloc_shared((BLOCK_Q, D), dtype)
            k = T.alloc_shared((BLOCK_K, D), dtype)
            v = T.alloc_shared((BLOCK_K, D), dtype)
            lse = T.alloc_shared((BLOCK_Q, ), acc_dtype)
            do = T.alloc_shared((BLOCK_Q, D), dtype)
            delta = T.alloc_shared((BLOCK_Q, ), acc_dtype)

            p = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            p_cast = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            ds = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            ds_cast = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            dq = T.alloc_fragment((BLOCK_Q, D), acc_dtype)
            dk = T.alloc_fragment((BLOCK_K, D), acc_dtype)
            dv = T.alloc_fragment((BLOCK_K, D), acc_dtype)

            k_beg = cu_seqlens_k[b_i] + k_idx * BLOCK_K
            k_end = k_beg + BLOCK_K
            T.copy(K[h_i, k_beg:k_end, :], k)
            T.copy(V[h_i, k_beg:k_end, :], v)
            T.clear(dk)
            T.clear(dv)
            lq = cu_seqlens_q[b_i + 1] - cu_seqlens_q[b_i]
            lk = cu_seqlens_k[b_i + 1] - cu_seqlens_k[b_i]

            for q_idx in T.Pipelined(T.ceildiv(lq, BLOCK_Q), num_stages=num_stages):
                q_beg = cu_seqlens_q[b_i] + q_idx * BLOCK_Q
                q_end = q_beg + BLOCK_Q
                T.copy(Q[h_i, q_beg:q_end, :], q)

                # S = QK^T * scaling
                T.clear(p)
                T.gemm(q, k, p, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = p[i, j] * scaling

                # P = softmax(S)
                for i in T.Parallel(BLOCK_Q):
                    lse[i] = T.if_then_else(
                        q_beg + i < cu_seqlens_q[b_i + 1],
                        LSE[h_i, q_beg + i],
                        0.,
                    )
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = T.exp(p[i, j] - lse[i])

                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = T.if_then_else(
                        T.Or(k_beg + j >= cu_seqlens_k[b_i] + lk,
                             q_beg + i >= cu_seqlens_q[b_i + 1]),
                        0.,
                        p[i, j],
                    )

                # dP = dO @ V^T
                T.copy(dO[h_i, q_beg:q_end, :], do)
                T.clear(ds)
                T.gemm(do, v, ds, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # dV = P^T @ dO
                T.copy(p, p_cast)
                T.gemm(p_cast, do, dv, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                # dS = P * (dO - delta), shape [LQ, LK]
                # delta = sum_d(dO * O)
                for i in T.Parallel(BLOCK_Q):
                    delta[i] = T.if_then_else(
                        q_beg + i < cu_seqlens_q[b_i + 1],
                        DELTA[h_i, q_beg + i],
                        0.,
                    )
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    ds_cast[i, j] = p[i, j] * (ds[i, j] - delta[i])
                    ds_cast[i, j] *= scaling

                # dK = (scaling * dS)^T @ Q
                T.gemm(ds_cast, q, dk, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                # dQ = (scaling * dS) @ K^T)
                T.clear(dq)
                T.gemm(ds_cast, k, dq, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BLOCK_Q, D):
                    if q_beg + i < cu_seqlens_q[b_i + 1]:
                        T.atomic_add(dQ[h_i, q_beg + i, j], dq[i, j])

            for i, j in T.Parallel(BLOCK_K, D):
                if k_beg + i < cu_seqlens_k[b_i + 1]:
                    dK[h_i, k_beg + i, j] = dk[i, j]
                    dV[h_i, k_beg + i, j] = dv[i, j]

    return kernel


class MyfaVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask):
        _, h, _, d = q.shape
        scaling = 1.0 / math.sqrt(d)
        fwd_kernel = myfa_varlen_fwd.compile(H=h, D=d, scaling=scaling)
        o, lse = fwd_kernel(q, k, v, mask)
        ctx.save_for_backward(q, k, v, mask, o, lse)
        ctx.scaling = scaling
        return o

    @staticmethod
    def backward(ctx, dO):
        q, k, v, mask, o, lse = ctx.saved_tensors
        _, h, _, d = q.shape
        scaling = ctx.scaling

        bwd_pre_kernel = myfa_varlen_bwd_pre.compile(H=h, D=d)
        delta = bwd_pre_kernel(o, dO)

        bwd_kernel = myfa_varlen_bwd.compile(H=h, D=d, scaling=scaling)
        dQ = torch.zeros_like(q, dtype=torch.float32)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)
        bwd_kernel(q, k, v, mask, lse, dO, delta, dQ, dK, dV)
        dQ = dQ.to(q.dtype)

        return dQ, dK, dV, None


def myfa_mask(q, k, v, mask):
    return MyfaVarlen.apply(q, k, v, mask)


def _make_inputs():
    seqlens = [33, 1023, 3, 128, 2049, 512, 72]
    h, d = 4, 128
    scaling = 1.0 / math.sqrt(d)

    cu_seqlens = [0]
    for sl in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + sl)
    total = cu_seqlens[-1]

    rng = torch.Generator(device='cuda').manual_seed(1919810)
    q = torch.randn(h, total, d, generator=rng, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(h, total, d, generator=rng, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(h, total, d, generator=rng, device="cuda", dtype=torch.bfloat16)

    return q, k, v, cu_seqlens, cu_seqlens, h, d, scaling


def _check(tag, a, b, atol_avg=0.01, atol_max=0.05):
    avg = (a - b).abs().mean().item()
    mx = (a - b).abs().max().item()
    print(f'{tag} | avg {avg:.6e} | max {mx:.6e}')
    assert avg < atol_avg, f'{tag} avg {avg} >= {atol_avg}'
    assert mx < atol_max, f'{tag} max {mx} >= {atol_max}'


@torch.no_grad()
def test_forward():
    q, k, v, cu_seqlens_q, cu_seqlens_k, h, d, scaling = _make_inputs()
    assert cu_seqlens_q == cu_seqlens_k

    # eager backward
    with torch.enable_grad():
        q_ag = q.detach().requires_grad_(True)
        k_ag = k.detach().requires_grad_(True)
        v_ag = v.detach().requires_grad_(True)
        ref_o = ref_attn(q_ag, k_ag, v_ag, cu_seqlens_q, cu_seqlens_k, scaling)
        dO = torch.rand_like(ref_o)
        ref_o.backward(dO)
    ref_dQ, ref_dK, ref_dV = q_ag.grad, k_ag.grad, v_ag.grad

    # myfa fwd
    cu_seqlens_q_tensor = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
    cu_seqlens_k_tensor = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cuda")
    max_seqlen = max(cu_seqlens_q_tensor[1:] - cu_seqlens_q_tensor[:-1]).item()
    myfa_fwd_kernel = myfa_varlen_fwd.compile(H=h, D=d, scaling=scaling)
    my_o, my_lse = myfa_fwd_kernel(q, k, v, cu_seqlens_q_tensor, cu_seqlens_k_tensor, max_seqlen)

    # bwd preprocess (delta)
    myfa_bwd_preprocess_kernel = myfa_varlen_bwd_pre.compile(H=h, D=d)
    my_delta = myfa_bwd_preprocess_kernel(my_o, dO)

    # myfa backward
    myfa_bwd_kernel = myfa_varlen_bwd.compile(H=h, D=d, scaling=scaling)
    my_dQ = torch.zeros_like(q, device='cuda', dtype=torch.float32)
    my_dK = torch.empty_like(k, device='cuda', dtype=torch.bfloat16)
    my_dV = torch.empty_like(v, device='cuda', dtype=torch.bfloat16)
    myfa_bwd_kernel(q, k, v, cu_seqlens_q_tensor, cu_seqlens_k_tensor, max_seqlen,
        my_lse, dO, my_delta, my_dQ, my_dK, my_dV)
    my_dQ = my_dQ.to(torch.bfloat16)

    print(f'{my_dQ.mean().item()=}')

    tag = 'full'
    _check(f'[{tag}] my  vs ref', my_o, ref_o)
    # _check(f'[{tag}] fa2 vs ref', fa2_o, ref_o)
    # _check(f'[{tag}] my  vs fa2', my_o, fa2_o)

    _check(f'[{tag}] my_dQ  vs ref', my_dQ, ref_dQ)
    _check(f'[{tag}] my_dK  vs ref', my_dK, ref_dK)
    _check(f'[{tag}] my_dV  vs ref', my_dV, ref_dV)