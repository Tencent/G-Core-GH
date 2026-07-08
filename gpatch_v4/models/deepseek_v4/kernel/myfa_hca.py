import math

import torch

try:
    import tilelang as tl
    import tilelang.language as T
except ImportError:
    from gfused.fake_tilelang_stub import language_stub as T
    from gfused.fake_tilelang_stub import tilelang_stub as tl


@tl.jit(
    out_idx=[-2, -1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_hca_fwd(
    H,
    D,
    scaling=1.0,
    sliding_window=128,
    compress_rate=128,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    num_stages=3,
    BLOCK_Q=64,
    BLOCK_K=64,
):
    """Fused HCA forward: sliding-window attn + compressed attn + attn_sink.

    Takes Q, KV (single-head sliding window), CK (single-head compressed)
    as separate inputs — avoids the ``cat([kv, compressed_kv])`` in eager code.
    Two-phase online softmax carries state across both KV sources.

    Parameters
    ----------
    Q : ``(B, H, LQ, D)``
    K : ``(B, 1, LK, D)``   — single-head sliding-window K (K=V in DSV4).
    CK : ``(B, 1, LCK, D)`` — single-head compressed K (K=V, MQA).
    AttnSink : ``(H,)`` fp32   — per-head learned sink logit.
    O : ``(B, H, LQ, D)``
    LSE : ``(B, H, LQ)`` fp32
    """
    assert D == tl.math.next_power_of_2(D)
    SWA_BLOCKS = T.ceildiv(sliding_window + BLOCK_Q - 1, BLOCK_K)

    B = T.dynamic('B', torch.int32)
    LQ = T.dynamic('LQ', torch.int32)
    LK = T.dynamic('LK', torch.int32)
    LCK = T.dynamic('LCK', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, H, LQ, D), dtype),
        K: T.Tensor((B, 1, LK, D), dtype),
        CK: T.Tensor((B, 1, LCK, D), dtype),
        AttnSink: T.Tensor((H, ), acc_dtype),
        O: T.Tensor((B, H, LQ, D), dtype),
        LSE: T.Tensor((B, H, LQ), acc_dtype),
    ):
        with T.Kernel(B, H, T.ceildiv(LQ, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
            q = T.alloc_shared((BLOCK_Q, D), dtype)
            k = T.alloc_shared((BLOCK_K, D), dtype)
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

            q_beg = q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(Q[b_i, h_i, q_beg:q_end, :], q)

            # ---- Phase 1: Sliding window KV (single-head, K=V) ----
            swa_end = T.ceildiv(q_end, BLOCK_K)
            n_swa = T.min(SWA_BLOCKS, swa_end)
            swa_start = swa_end - n_swa

            for ki_off in T.Pipelined(n_swa, num_stages=num_stages):
                k_beg = (swa_start + ki_off) * BLOCK_K
                k_end = k_beg + BLOCK_K
                T.copy(K[b_i, 0, k_beg:k_end, :], k)

                # s = qk^T (sliding window causal mask)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    k_pos = k_beg + j
                    q_pos = q_beg + i
                    acc_s[i, j] = T.if_then_else(
                        k_pos < LK and k_pos <= q_pos and q_pos - k_pos < sliding_window,
                        0.,
                        -T.infinity(acc_s.dtype),
                    )
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
                T.gemm(acc_s_cast, k, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # ---- Phase 2: Compressed KV (single-head MQA, K=V) ----
            for ki in T.Pipelined(T.ceildiv(LCK, BLOCK_K), num_stages=num_stages):
                k_beg = ki * BLOCK_K
                k_end = k_beg + BLOCK_K
                T.copy(CK[b_i, 0, k_beg:k_end, :], k)

                # p = qk^T (causal threshold mask)
                # TODO 支持 varlen
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[i, j] = T.if_then_else(
                        k_beg + j < LCK and k_beg + j < (q_beg + i + 1) // compress_rate,
                        0.,
                        -T.infinity(acc_s.dtype),
                    )
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

                # exp(p)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[i, j] = T.exp(acc_s[i, j] - m[i])
                T.copy(acc_s, acc_s_cast)

                # l(SE) and fix l
                T.reduce_sum(acc_s, tmp_sum, dim=1, clear=True)  # sum exp(s)
                for i in T.Parallel(BLOCK_Q):
                    se[i] = se[i] * alpha[i] + tmp_sum[i]

                # a = pv (K=V)
                for i, j in T.Parallel(BLOCK_Q, D):
                    acc_o[i, j] = alpha[i] * acc_o[i, j]
                T.gemm(acc_s_cast, k, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # ---- attn_sink ----
            for i in T.Parallel(BLOCK_Q):
                se[i] += T.exp(AttnSink[h_i] - m[i])

            # o = a / l
            for i, j in T.Parallel(BLOCK_Q, D):
                acc_o[i, j] = acc_o[i, j] / se[i]
            T.copy(acc_o, O[b_i, h_i, q_beg:q_end, :])

            # lse
            for i in T.Parallel(BLOCK_Q):
                se[i] = m[i] + T.log(se[i])
            T.copy(se, LSE[b_i, h_i, q_beg:q_end])

    return kernel


'''
@tl.jit(
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_hca_bwd(
    H,
    D,
    scaling=1.0,
    sliding_window=128,
    compress_rate=128,
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
    LCK = T.dynamic('LCK', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, H, LQ, D), dtype),
        K: T.Tensor((B, H, LK, D), dtype),
        M: T.Tensor((B, 1, LQ, LK), dtype),
        LSE: T.Tensor((B, H, LQ), acc_dtype),
        dO: T.Tensor((B, H, LQ, D), dtype),
        DELTA: T.Tensor((B, H, LQ), acc_dtype),
        dQ: T.Tensor((B, H, LQ, D), acc_dtype),
        dK: T.Tensor((B, H, LK, D), dtype),
    ):
        with T.Kernel(B, H, T.ceildiv(LK, BLOCK_K), threads=threads) as (b_i, h_i, k_idx):
            q = T.alloc_shared((BLOCK_Q, D), dtype)
            k = T.alloc_shared((BLOCK_K, D), dtype)
            mask = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            lse = T.alloc_shared((BLOCK_Q, ), acc_dtype)
            do = T.alloc_shared((BLOCK_Q, D), dtype)
            delta = T.alloc_shared((BLOCK_Q, ), acc_dtype)

            p = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            p_cast = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            ds = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            ds_cast = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            dq = T.alloc_fragment((BLOCK_Q, D), acc_dtype)
            dk = T.alloc_fragment((BLOCK_K, D), acc_dtype)

            k_beg = k_idx * BLOCK_K
            k_end = k_beg + BLOCK_K
            T.copy(K[b_i, h_i, k_beg:k_end, :], k)
            T.clear(dk)

            for q_idx in T.Pipelined(T.ceildiv(LQ, BLOCK_Q), num_stages=num_stages):
                q_beg = q_idx * BLOCK_Q
                q_end = q_beg + BLOCK_Q
                T.copy(Q[b_i, h_i, q_beg:q_end, :], q)
                T.copy(M[b_i, 0, q_beg:q_end, k_beg:k_end], mask)

                # S = QK^T * scaling
                T.clear(p)
                T.gemm(q, k, p, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = p[i, j] * scaling

                # P = softmax(S)
                T.copy(LSE[b_i, h_i, q_beg:q_end], lse)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = T.exp(p[i, j] - lse[i])

                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = T.if_then_else(
                        k_beg + j >= LK or mask[i, j] == float('-inf'),
                        0.,
                        p[i, j],
                    )

                # dP = dO @ V^T
                T.copy(dO[b_i, h_i, q_beg:q_end, :], do)
                T.clear(ds)
                T.gemm(do, v, ds, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # dV = P^T @ dO
                T.copy(p, p_cast)
                T.gemm(p_cast, do, dv, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                # dS = P * (dO - delta), shape [LQ, LK]
                # delta = sum_d(dO * O)
                T.copy(DELTA[b_i, h_i, q_beg:q_end], delta)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    ds_cast[i, j] = p[i, j] * (ds[i, j] - delta[i])
                    ds_cast[i, j] *= scaling

                # dK = (scaling * dS)^T @ Q
                T.gemm(ds_cast, q, dk, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                # dQ = (scaling * dS) @ K^T)
                T.clear(dq)
                T.gemm(ds_cast, k, dq, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BLOCK_Q, D):
                    T.atomic_add(dQ[b_i, h_i, q_idx * BLOCK_Q + i, j], dq[i, j])

            T.copy(dk, dK[b_i, h_i, k_beg:k_end, :])
            T.copy(dv, dV[b_i, h_i, k_beg:k_end, :])

    return kernel
'''


class MyfaHca(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, ckv, attn_sink, sliding_window, compress_rate):
        _, h, _, d = q.shape
        scaling = 1.0 / math.sqrt(d)
        fwd_kernel = myfa_hca_fwd.compile(
            H=h,
            D=d,
            scaling=scaling,
            sliding_window=sliding_window,
            compress_rate=compress_rate,
        )
        o, lse = fwd_kernel(q, kv, ckv, attn_sink)
        return o

    @staticmethod
    def backward(ctx, dO):
        return None, None, None, None, None, None


def myfa_hca(q, kv, ckv, attn_sink, sliding_window=128, compress_rate=128):
    return MyfaHca.apply(q, kv, ckv, attn_sink, sliding_window, compress_rate)
