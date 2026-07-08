import math

try:
    import tilelang as tl
    import tilelang.language as T
except ImportError:
    from gfused.fake_tilelang_stub import tilelang_stub as tl, language_stub as T
import torch

# ---------------------------------------------------------------------------
# TileLang kernels  (BSHD layout: Q/K/V/O are [B, S, H, D])
# ---------------------------------------------------------------------------


@tl.jit(
    out_idx=[-2, -1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_sw_sinks_fwd(
    HQ,
    HK,
    D,
    scaling=1.0,
    is_causal=True,
    window_size=None,
    has_sinks=True,
    has_q_offsets=False,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    num_stages=2,
    BLOCK_Q=64,
    BLOCK_K=64,
):
    assert D == tl.math.next_power_of_2(D)
    if window_size is not None:
        assert is_causal, "non-causal + sliding window is not supported"
        assert window_size % BLOCK_K == 0
    groups = HQ // HK
    B = T.dynamic('B', torch.int32)
    LQ = T.dynamic('LQ', torch.int32)
    LK = T.dynamic('LK', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, LQ, HQ, D), dtype),
        K: T.Tensor((B, LK, HK, D), dtype),
        V: T.Tensor((B, LK, HK, D), dtype),
        SINKS: T.Tensor((HQ, ), dtype),
        Q_OFFSETS: T.Tensor((B, ), torch.int32),
        O: T.Tensor((B, LQ, HQ, D), dtype),
        LSE: T.Tensor((B, HQ, LQ), acc_dtype),
    ):
        with T.Kernel(B, HQ, T.ceildiv(LQ, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
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
            if has_sinks:
                sinks = T.alloc_fragment((BLOCK_Q, ), dtype)
                for i in T.Parallel(BLOCK_Q):
                    sinks[i] = SINKS[h_i]
            kv_h_i = h_i // groups

            q_beg = q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(Q[b_i, q_beg:q_end, h_i, :], q)
            if has_q_offsets:
                q_offset = Q_OFFSETS[b_i]
            else:
                q_offset = LK - LQ

            if is_causal:
                if window_size is not None:
                    k_loop_start = T.max(0, (q_offset + q_beg - window_size + 1) // BLOCK_K)
                else:
                    k_loop_start = 0
                k_loop_end = T.min(T.ceildiv(q_offset + q_end, BLOCK_K), T.ceildiv(LK, BLOCK_K))
            else:
                k_loop_start = 0
                k_loop_end = T.ceildiv(LK, BLOCK_K)
            loop_range = k_loop_end - k_loop_start

            for k_local in T.Pipelined(loop_range, num_stages=num_stages):
                k_actual = k_local + k_loop_start
                k_beg = k_actual * BLOCK_K
                k_end = k_beg + BLOCK_K
                T.copy(K[b_i, k_beg:k_end, kv_h_i, :], k)

                # mask
                if is_causal:
                    if window_size is not None:
                        for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                            acc_s[i, j] = T.if_then_else(
                                (k_beg + j < LK) and (q_beg + i < LQ) and
                                (q_beg + i + q_offset >= k_beg + j) and
                                (k_beg + j >= q_beg + i + q_offset - window_size + 1),
                                0.0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                            acc_s[i, j] = T.if_then_else(
                                (k_beg + j < LK) and (q_beg + i < LQ) and
                                (q_beg + i + q_offset >= k_beg + j),
                                0.0,
                                -T.infinity(acc_s.dtype),
                            )
                else:
                    for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                        acc_s[i, j] = T.if_then_else(
                            (k_beg + j < LK) and (q_beg + i < LQ),
                            0.0,
                            -T.infinity(acc_s.dtype),
                        )

                # s = qk^T
                T.gemm(q, k, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # scaling
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[i, j] = acc_s[i, j] * scaling

                # fix m
                T.copy(m, m_prev)
                T.reduce_max(acc_s, m, dim=1, clear=True)
                for i in T.Parallel(BLOCK_Q):
                    m[i] = T.max(m[i], m_prev[i])
                for i in T.Parallel(BLOCK_Q):
                    m[i] = T.if_then_else(
                        m[i] == -T.infinity(acc_dtype),
                        0,
                        m[i],
                    )

                # alpha
                for i in T.Parallel(BLOCK_Q):
                    alpha[i] = T.exp(m_prev[i] - m[i])

                # exp(s)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[i, j] = T.exp(acc_s[i, j] - m[i])
                T.copy(acc_s, acc_s_cast)

                # l(SE) and fix l
                T.reduce_sum(acc_s, tmp_sum, dim=1, clear=True)
                for i in T.Parallel(BLOCK_Q):
                    se[i] = se[i] * alpha[i] + tmp_sum[i]

                # a = pv
                for i, j in T.Parallel(BLOCK_Q, D):
                    acc_o[i, j] = alpha[i] * acc_o[i, j]
                T.copy(V[b_i, k_beg:k_end, kv_h_i, :], v)
                T.gemm(acc_s_cast, v, acc_o, policy=T.GemmWarpPolicy.FullRow)

            if has_sinks:
                for i in T.Parallel(BLOCK_Q):
                    se[i] += T.exp(sinks[i] - m[i])

            # o = a / l
            for i, j in T.Parallel(BLOCK_Q, D):
                acc_o[i, j] = acc_o[i, j] / se[i]
            T.copy(acc_o, O[b_i, q_beg:q_end, h_i, :])

            # lse
            for i in T.Parallel(BLOCK_Q):
                se[i] = m[i] + T.log(se[i])
            T.copy(se, LSE[b_i, h_i, q_beg:q_end])

    return kernel


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_sw_sinks_bwd_pre(
    HQ,
    D,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    BLOCK_Q=64,
):
    assert D == tl.math.next_power_of_2(D)
    B = T.dynamic('B', torch.int32)
    LQ = T.dynamic('LQ', torch.int32)

    @T.prim_func
    def kernel(
        O: T.Tensor((B, LQ, HQ, D), dtype),
        dO: T.Tensor((B, LQ, HQ, D), dtype),
        DELTA: T.Tensor((B, HQ, LQ), acc_dtype),
    ):
        with T.Kernel(B, HQ, T.ceildiv(LQ, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
            o = T.alloc_fragment((BLOCK_Q, D), dtype)
            do = T.alloc_fragment((BLOCK_Q, D), dtype)
            acc = T.alloc_fragment((BLOCK_Q, D), acc_dtype)
            delta = T.alloc_fragment((BLOCK_Q, ), acc_dtype)

            # delta = sum_d(o * dO)
            q_beg = q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(O[b_i, q_beg:q_end, h_i, :], o)
            T.copy(dO[b_i, q_beg:q_end, h_i, :], do)
            for i, j in T.Parallel(BLOCK_Q, D):
                acc[i, j] = o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, dim=1, clear=True)
            T.copy(delta, DELTA[b_i, h_i, q_beg:q_end])

    return kernel


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_sw_sinks_bwd_dsink(
    HQ,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=256,
    BLOCK_Q=256,
):
    B = T.dynamic('B', torch.int32)
    LQ = T.dynamic('LQ', torch.int32)

    @T.prim_func
    def kernel(
        SINKS: T.Tensor((HQ, ), dtype),
        DELTA: T.Tensor((B, HQ, LQ), acc_dtype),
        LSE: T.Tensor((B, HQ, LQ), acc_dtype),
        dSINKS: T.Tensor((B, HQ, LQ), acc_dtype),
    ):
        with T.Kernel(B, HQ, T.ceildiv(LQ, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
            lse = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            delta = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            sinks = T.alloc_fragment((BLOCK_Q, ), dtype)
            dsink = T.alloc_fragment((BLOCK_Q, ), acc_dtype)

            q_beg = q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(LSE[b_i, h_i, q_beg:q_end], lse)
            T.copy(DELTA[b_i, h_i, q_beg:q_end], delta)
            for i in T.Parallel(BLOCK_Q):
                sinks[i] = SINKS[h_i]
            for i in T.Parallel(BLOCK_Q):
                dsink[i] = T.if_then_else(
                    q_beg + i < LQ,
                    -T.exp(sinks[i] - lse[i]) * delta[i],
                    0,
                )
            T.copy(dsink, dSINKS[b_i, h_i, q_beg:q_end])

    return kernel


@tl.jit(pass_configs={
    tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def myfa_sw_sinks_bwd(
    HQ,
    HK,
    D,
    scaling=1.0,
    is_causal=True,
    window_size=None,
    has_q_offsets=False,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    num_stages=2,
    BLOCK_Q=64,
    BLOCK_K=64,
):
    assert D == tl.math.next_power_of_2(D)
    if window_size is not None:
        assert is_causal, "non-causal + sliding window is not supported"
        assert window_size % BLOCK_K == 0
    groups = HQ // HK
    B = T.dynamic('B', torch.int32)
    LQ = T.dynamic('LQ', torch.int32)
    LK = T.dynamic('LK', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, LQ, HQ, D), dtype),
        K: T.Tensor((B, LK, HK, D), dtype),
        V: T.Tensor((B, LK, HK, D), dtype),
        Q_OFFSETS: T.Tensor((B, ), torch.int32),
        LSE: T.Tensor((B, HQ, LQ), acc_dtype),
        dO: T.Tensor((B, LQ, HQ, D), dtype),
        DELTA: T.Tensor((B, HQ, LQ), acc_dtype),
        dQ: T.Tensor((B, LQ, HQ, D), acc_dtype),
        dK: T.Tensor((B, LK, HK, D), acc_dtype),
        dV: T.Tensor((B, LK, HK, D), acc_dtype),
    ):
        with T.Kernel(B, HQ, T.ceildiv(LK, BLOCK_K), threads=threads) as (b_i, h_i, k_idx):
            q = T.alloc_shared((BLOCK_Q, D), dtype)
            k = T.alloc_shared((BLOCK_K, D), dtype)
            v = T.alloc_shared((BLOCK_K, D), dtype)
            lse = T.alloc_shared((BLOCK_Q, ), acc_dtype)
            do = T.alloc_shared((BLOCK_Q, D), dtype)
            delta = T.alloc_shared((BLOCK_Q, ), acc_dtype)

            p = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            p_cast = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            dp = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            ds_cast = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            dq = T.alloc_fragment((BLOCK_Q, D), acc_dtype)
            dk = T.alloc_fragment((BLOCK_K, D), acc_dtype)
            dv = T.alloc_fragment((BLOCK_K, D), acc_dtype)

            k_beg = k_idx * BLOCK_K
            k_end = k_beg + BLOCK_K
            kv_h_i = h_i // groups
            T.copy(K[b_i, k_beg:k_end, kv_h_i, :], k)
            T.copy(V[b_i, k_beg:k_end, kv_h_i, :], v)
            T.clear(dk)
            T.clear(dv)
            if has_q_offsets:
                q_offset = Q_OFFSETS[b_i]
            else:
                q_offset = LK - LQ

            if is_causal:
                q_loop_start = T.max(0, (k_beg - q_offset) // BLOCK_Q)
            else:
                q_loop_start = 0
            if window_size is not None:
                q_loop_end = T.min(
                    T.ceildiv(k_end - q_offset + window_size, BLOCK_Q),
                    T.ceildiv(LQ, BLOCK_Q),
                )
            else:
                q_loop_end = T.ceildiv(LQ, BLOCK_Q)
            loop_range = q_loop_end - q_loop_start

            for q_local in T.Pipelined(loop_range, num_stages=num_stages):
                q_actual = q_local + q_loop_start
                q_beg = q_actual * BLOCK_Q
                q_end = q_beg + BLOCK_Q
                T.copy(Q[b_i, q_beg:q_end, h_i, :], q)

                # S = QK^T * scaling
                T.clear(p)
                T.gemm(q, k, p, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = p[i, j] * scaling

                # P = softmax(S)
                T.copy(LSE[b_i, h_i, q_beg:q_end], lse)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = T.exp(p[i, j] - lse[i])

                # mask
                if is_causal:
                    if window_size is not None:
                        for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                            p[i, j] = T.if_then_else(
                                (k_beg + j < LK) and (q_beg + i < LQ) and
                                (q_beg + i + q_offset >= k_beg + j) and
                                (k_beg + j >= q_beg + i + q_offset - window_size + 1),
                                p[i, j],
                                0.,
                            )
                    else:
                        for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                            p[i, j] = T.if_then_else(
                                (k_beg + j < LK) and (q_beg + i < LQ) and
                                (q_beg + i + q_offset >= k_beg + j),
                                p[i, j],
                                0.,
                            )
                else:
                    for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                        p[i, j] = T.if_then_else(
                            (k_beg + j < LK) and (q_beg + i < LQ),
                            p[i, j],
                            0.,
                        )

                # dP = dO @ V^T
                T.copy(dO[b_i, q_beg:q_end, h_i, :], do)
                T.clear(dp)
                T.gemm(do, v, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # dV = P^T @ dO
                T.copy(p, p_cast)
                T.gemm(p_cast, do, dv, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                # dS = P * (dP - delta), shape [LQ, LK]
                # delta = sum_d(dO * O)
                T.copy(DELTA[b_i, h_i, q_beg:q_end], delta)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    ds_cast[i, j] = p[i, j] * (dp[i, j] - delta[i])
                    ds_cast[i, j] *= scaling

                # dK = (scaling * dS)^T @ Q
                T.gemm(ds_cast, q, dk, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                # dQ = (scaling * dS) @ K^T)
                T.clear(dq)
                T.gemm(ds_cast, k, dq, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BLOCK_Q, D):
                    T.atomic_add(dQ[b_i, q_actual * BLOCK_Q + i, h_i, j], dq[i, j])

            T.atomic_add(dK[b_i, k_beg:k_end, kv_h_i, :], dk)
            T.atomic_add(dV[b_i, k_beg:k_end, kv_h_i, :], dv)

    return kernel


# ---------------------------------------------------------------------------
# autograd wrapper
# ---------------------------------------------------------------------------


class MyfaSwSinks(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, q_offsets, is_causal, window_size):
        b, sq, HQ, d = q.shape
        _, sk, HK, _ = k.shape
        assert is_causal or window_size is None, "non-causal + sliding window is not supported"
        # TODO @astrachang 我晚点会去掉这个限制
        assert sq % 64 == 0, f"seq_len_q={sq} must be divisible by BLOCK_Q (64)"
        assert sk % 64 == 0, f"seq_len_k={sk} must be divisible by BLOCK_K (64)"
        scaling = 1.0 / math.sqrt(d)
        has_sinks = sinks is not None
        has_q_offsets = q_offsets is not None
        if not has_sinks:
            sinks = torch.empty(HQ, dtype=q.dtype, device=q.device)
        if not has_q_offsets:
            q_offsets = torch.empty(b, dtype=torch.int32, device=q.device)
        fwd_kernel = myfa_sw_sinks_fwd.compile(
            HQ=HQ,
            HK=HK,
            D=d,
            scaling=scaling,
            is_causal=is_causal,
            window_size=window_size,
            has_sinks=has_sinks,
            has_q_offsets=has_q_offsets,
        )
        o, lse = fwd_kernel(q, k, v, sinks, q_offsets)
        ctx.save_for_backward(q, k, v, sinks, q_offsets, o, lse)
        ctx.scaling = scaling
        ctx.is_causal = is_causal
        ctx.window_size = window_size
        ctx.has_sinks = has_sinks
        ctx.has_q_offsets = has_q_offsets
        return o

    @staticmethod
    def backward(ctx, dO):
        q, k, v, sinks, q_offsets, o, lse = ctx.saved_tensors
        _, _, HQ, d = q.shape
        HK = k.shape[2]
        scaling = ctx.scaling
        is_causal = ctx.is_causal
        window_size = ctx.window_size
        has_q_offsets = ctx.has_q_offsets
        dt = q.dtype

        bwd_pre_kernel = myfa_sw_sinks_bwd_pre.compile(HQ=HQ, D=d, dtype=dt)
        delta = bwd_pre_kernel(o, dO)

        bwd_kernel = myfa_sw_sinks_bwd.compile(
            HQ=HQ,
            HK=HK,
            D=d,
            scaling=scaling,
            is_causal=is_causal,
            window_size=window_size,
            has_q_offsets=has_q_offsets,
            dtype=dt,
        )
        dQ = torch.zeros_like(q, dtype=torch.float32)
        dK = torch.zeros(
            q.shape[0],
            k.shape[1],
            HK,
            d,
            dtype=torch.float32,
            device=q.device,
        )
        dV = torch.zeros(
            q.shape[0],
            v.shape[1],
            HK,
            d,
            dtype=torch.float32,
            device=q.device,
        )
        bwd_kernel(q, k, v, q_offsets, lse, dO, delta, dQ, dK, dV)
        dQ = dQ.to(q.dtype)
        dK = dK.to(k.dtype)
        dV = dV.to(v.dtype)

        dsinks = None
        if ctx.has_sinks:
            dsink_kernel = myfa_sw_sinks_bwd_dsink.compile(HQ=HQ, dtype=dt)
            dsinks = dsink_kernel(sinks, delta, lse).sum(0).sum(1)

        return dQ, dK, dV, dsinks, None, None, None


def myfa_sw_sinks(q, k, v, sinks=None, q_offsets=None, is_causal=True, window_size=None):
    return MyfaSwSinks.apply(q, k, v, sinks, q_offsets, is_causal, window_size)
