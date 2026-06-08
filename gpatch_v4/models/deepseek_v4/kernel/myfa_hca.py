import math

import tilelang as tl
import tilelang.language as T
import torch


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

    Takes Q, KV (multi-head sliding window), CKV (single-head compressed)
    as separate inputs — avoids the ``cat([kv, compressed_kv])`` in eager code.
    Two-phase online softmax carries state across both KV sources.

    Parameters
    ----------
    Q : ``(B, H, LQ, D)``
    KV : ``(B, H, LKV, D)``   — multi-head sliding-window KV (K=V in DSV4).
    CKV : ``(B, 1, LCKV, D)`` — single-head compressed KV (K=V, MQA).
    AttnSink : ``(H,)`` fp32   — per-head learned sink logit.
    O : ``(B, H, LQ, D)``
    LSE : ``(B, H, LQ)`` fp32
    """
    assert D == tl.math.next_power_of_2(D)
    SWA_BLOCKS = tl.math.ceildiv(sliding_window + BLOCK_Q - 1, BLOCK_K)

    B = T.dynamic('B', torch.int32)
    LQ = T.dynamic('LQ', torch.int32)
    LKV = T.dynamic('LKV', torch.int32)
    LCKV = T.dynamic('LCKV', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, H, LQ, D), dtype),
        KV: T.Tensor((B, H, LKV, D), dtype),
        CKV: T.Tensor((B, 1, LCKV, D), dtype),
        AttnSink: T.Tensor((H, ), acc_dtype),
        O: T.Tensor((B, H, LQ, D), dtype),
        LSE: T.Tensor((B, H, LQ), acc_dtype),
    ):
        with T.Kernel(B, H, T.ceildiv(LQ, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
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

            q_beg = q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(Q[b_i, h_i, q_beg:q_end, :], q)

            # ---- Phase 1: Sliding window KV (multi-head, K=V) ----
            swa_end = T.ceildiv(q_end, BLOCK_K)
            n_swa = T.min(SWA_BLOCKS, swa_end)
            swa_start = swa_end - n_swa

            for ki_off in T.Pipelined(n_swa, num_stages=num_stages):
                k_beg = (swa_start + ki_off) * BLOCK_K
                k_end = k_beg + BLOCK_K
                T.copy(KV[b_i, h_i, k_beg:k_end, :], k)

                # s = qk^T (sliding window causal mask)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    k_pos = k_beg + j
                    q_pos = q_beg + i
                    acc_s[i, j] = T.if_then_else(
                        k_pos < LKV and k_pos <= q_pos and q_pos - k_pos < sliding_window,
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

                # a = pv (K=V)
                for i, j in T.Parallel(BLOCK_Q, D):
                    acc_o[i, j] = alpha[i] * acc_o[i, j]
                T.copy(KV[b_i, h_i, k_beg:k_end, :], v)
                T.gemm(acc_s_cast, v, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # ---- Phase 2: Compressed KV (single-head MQA, K=V) ----
            for ki in T.Pipelined(T.ceildiv(LCKV, BLOCK_K), num_stages=num_stages):
                k_beg = ki * BLOCK_K
                k_end = k_beg + BLOCK_K
                T.copy(CKV[b_i, 0, k_beg:k_end, :], k)

                # p = qk^T (causal threshold mask)
                # TODO 支持 varlen
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[i, j] = T.if_then_else(
                        k_beg + j < LCKV and k_beg + j < (q_beg + i + 1) // compress_rate,
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
                T.copy(CKV[b_i, 0, k_beg:k_end, :], v)
                T.gemm(acc_s_cast, v, acc_o, policy=T.GemmWarpPolicy.FullRow)

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
