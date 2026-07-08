import math

try:
    import tilelang as tl
    import tilelang.language as T
except ImportError:
    from gfused.fake_tilelang_stub import tilelang_stub as tl, language_stub as T
import torch


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_varlen_sw_sinks_fwd(
    HQ,
    HK,
    D,
    scaling=1.0,
    is_causal=True,
    window_size=None,
    has_sinks=True,
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
    MAX_OUT_L = T.dynamic('MAX_OUT_L', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((LQ, HQ, D), dtype),
        K: T.Tensor((LK, HK, D), dtype),
        V: T.Tensor((LK, HK, D), dtype),
        SINKS: T.Tensor((HQ, ), dtype),
        cu_seqlens_q: T.Tensor((B + 1, ), torch.int32),
        cu_seqlens_k: T.Tensor((B + 1, ), torch.int32),
        max_seqlen: T.int32,
        LSE: T.Tensor((B, HQ, MAX_OUT_L), acc_dtype),
        O: T.Tensor((LQ, HQ, D), dtype),
    ):
        with T.Kernel(B, HQ, T.ceildiv(max_seqlen, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
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
            q_beg_abs = cu_seqlens_q[b_i] + q_idx * BLOCK_Q
            q_end_abs = q_beg_abs + BLOCK_Q
            T.copy(Q[q_beg_abs:q_end_abs, h_i, :], q)

            lq = cu_seqlens_q[b_i + 1] - cu_seqlens_q[b_i]
            lk = cu_seqlens_k[b_i + 1] - cu_seqlens_k[b_i]
            q_offset = lk - lq

            if is_causal:
                if window_size is not None:
                    k_loop_start = T.max(0, (q_offset + q_beg - window_size + 1) // BLOCK_K)
                else:
                    k_loop_start = 0
                k_loop_end = T.min(
                    T.ceildiv(q_offset + q_beg + BLOCK_Q, BLOCK_K),
                    T.ceildiv(lk, BLOCK_K),
                )
            else:
                k_loop_start = 0
                k_loop_end = T.ceildiv(lk, BLOCK_K)
            loop_range = k_loop_end - k_loop_start

            for k_local in T.Pipelined(loop_range, num_stages=num_stages):
                k_actual = k_local + k_loop_start
                k_beg = k_actual * BLOCK_K
                k_beg_abs = cu_seqlens_k[b_i] + k_beg
                k_end_abs = k_beg_abs + BLOCK_K
                T.copy(K[k_beg_abs:k_end_abs, kv_h_i, :], k)

                # mask
                if is_causal:
                    if window_size is not None:
                        for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                            acc_s[i, j] = T.if_then_else(
                                (k_beg + j < lk) and (q_beg + i < lq) and
                                (q_beg + i + q_offset >= k_beg + j) and
                                (k_beg + j >= q_beg + i + q_offset - window_size + 1),
                                0.0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                            acc_s[i, j] = T.if_then_else(
                                (k_beg + j < lk) and (q_beg + i < lq) and
                                (q_beg + i + q_offset >= k_beg + j),
                                0.0,
                                -T.infinity(acc_s.dtype),
                            )
                else:
                    for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                        acc_s[i, j] = T.if_then_else(
                            (k_beg + j < lk) and (q_beg + i < lq),
                            0.0,
                            -T.infinity(acc_s.dtype),
                        )

                # s = qk^T
                T.gemm(q, k, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[k_beg_abs:k_end_abs, kv_h_i, :], v)

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
                T.gemm(acc_s_cast, v, acc_o, policy=T.GemmWarpPolicy.FullRow)

            if has_sinks:
                for i in T.Parallel(BLOCK_Q):
                    se[i] += T.exp(sinks[i] - m[i])

            # o = a / l
            for i, j in T.Parallel(BLOCK_Q, D):
                acc_o[i, j] = acc_o[i, j] / se[i]
                if q_beg + i < lq:
                    O[q_beg_abs + i, h_i, j] = acc_o[i, j]

            # lse
            for i in T.Parallel(BLOCK_Q):
                se[i] = m[i] + T.log(se[i])
                if q_beg + i < lq:
                    LSE[b_i, h_i, q_idx * BLOCK_Q + i] = se[i]

    return kernel


@tl.jit(pass_configs={
    tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, )
def myfa_varlen_sw_sinks_bwd_pre(
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
    MAX_OUT_L = T.dynamic('MAX_OUT_L', torch.int32)

    @T.prim_func
    def kernel(
        O: T.Tensor((LQ, HQ, D), dtype),
        dO: T.Tensor((LQ, HQ, D), dtype),
        cu_seqlens_q: T.Tensor((B + 1, ), torch.int32),
        max_seqlen: T.int32,
        DELTA: T.Tensor((B, HQ, MAX_OUT_L), acc_dtype),
    ):
        with T.Kernel(B, HQ, T.ceildiv(max_seqlen, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
            o = T.alloc_fragment((BLOCK_Q, D), dtype)
            do = T.alloc_fragment((BLOCK_Q, D), dtype)
            acc = T.alloc_fragment((BLOCK_Q, D), acc_dtype)
            delta = T.alloc_fragment((BLOCK_Q, ), acc_dtype)

            # delta = sum_d(o * dO)
            q_beg = cu_seqlens_q[b_i] + q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(O[q_beg:q_end, h_i, :], o)
            T.copy(dO[q_beg:q_end, h_i, :], do)
            for i, j in T.Parallel(BLOCK_Q, D):
                acc[i, j] = o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, dim=1, clear=True)

            for i in T.Parallel(BLOCK_Q):
                if q_beg + i < cu_seqlens_q[b_i + 1]:
                    DELTA[b_i, h_i, q_idx * BLOCK_Q + i] = delta[i]

    return kernel


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_varlen_sw_sinks_bwd_dsink(
    HQ,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=256,
    BLOCK_Q=256,
):
    B = T.dynamic('B', torch.int32)
    MAX_OUT_L = T.dynamic('MAX_OUT_L', torch.int32)

    @T.prim_func
    def kernel(
        SINKS: T.Tensor((HQ, ), dtype),
        DELTA: T.Tensor((B, HQ, MAX_OUT_L), acc_dtype),
        LSE: T.Tensor((B, HQ, MAX_OUT_L), acc_dtype),
        cu_seqlens_q: T.Tensor((B + 1, ), torch.int32),
        max_seqlen: T.int32,
        dSINKS: T.Tensor((B, HQ, MAX_OUT_L), acc_dtype),
    ):
        with T.Kernel(B, HQ, T.ceildiv(max_seqlen, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
            lse = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            delta = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            sinks = T.alloc_fragment((BLOCK_Q, ), dtype)
            dsink = T.alloc_fragment((BLOCK_Q, ), acc_dtype)

            lq = cu_seqlens_q[b_i + 1] - cu_seqlens_q[b_i]

            q_beg = q_idx * BLOCK_Q
            T.copy(LSE[b_i, h_i, q_beg:q_beg + BLOCK_Q], lse)
            T.copy(DELTA[b_i, h_i, q_beg:q_beg + BLOCK_Q], delta)
            for i in T.Parallel(BLOCK_Q):
                sinks[i] = SINKS[h_i]
            for i in T.Parallel(BLOCK_Q):
                dsink[i] = T.if_then_else(
                    q_beg + i < lq,
                    -T.exp(sinks[i] - lse[i]) * delta[i],
                    0,
                )
            T.copy(dsink, dSINKS[b_i, h_i, q_beg:q_beg + BLOCK_Q])

    return kernel


@tl.jit(pass_configs={
    tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def myfa_varlen_sw_sinks_bwd(
    HQ,
    HK,
    D,
    scaling=1.0,
    is_causal=True,
    window_size=None,
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
    MAX_OUT_L = T.dynamic('MAX_OUT_L', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((LQ, HQ, D), dtype),
        K: T.Tensor((LK, HK, D), dtype),
        V: T.Tensor((LK, HK, D), dtype),
        cu_seqlens_q: T.Tensor((B + 1, ), torch.int32),
        cu_seqlens_k: T.Tensor((B + 1, ), torch.int32),
        max_seqlen: T.int32,
        LSE: T.Tensor((B, HQ, MAX_OUT_L), acc_dtype),
        dO: T.Tensor((LQ, HQ, D), dtype),
        DELTA: T.Tensor((B, HQ, MAX_OUT_L), acc_dtype),
        dQ: T.Tensor((LQ, HQ, D), acc_dtype),
        dK: T.Tensor((LK, HK, D), acc_dtype),
        dV: T.Tensor((LK, HK, D), acc_dtype),
    ):
        with T.Kernel(B, HQ, T.ceildiv(max_seqlen, BLOCK_K), threads=threads) as (b_i, h_i, k_idx):
            q = T.alloc_shared((BLOCK_Q, D), dtype)
            k = T.alloc_shared((BLOCK_K, D), dtype)
            v = T.alloc_shared((BLOCK_K, D), dtype)
            lse = T.alloc_shared((BLOCK_Q, ), acc_dtype)
            do = T.alloc_shared((BLOCK_Q, D), dtype)
            delta = T.alloc_shared((BLOCK_Q, ), acc_dtype)

            pT = T.alloc_fragment((BLOCK_K, BLOCK_Q), acc_dtype)
            pT_cast = T.alloc_shared((BLOCK_K, BLOCK_Q), dtype)
            dpT = T.alloc_fragment((BLOCK_K, BLOCK_Q), acc_dtype)
            dsT_cast = T.alloc_shared((BLOCK_K, BLOCK_Q), dtype)
            dq = T.alloc_fragment((BLOCK_Q, D), acc_dtype)
            dk = T.alloc_fragment((BLOCK_K, D), acc_dtype)
            dv = T.alloc_fragment((BLOCK_K, D), acc_dtype)

            k_beg = k_idx * BLOCK_K
            k_beg_abs = cu_seqlens_k[b_i] + k_beg
            k_end_abs = k_beg_abs + BLOCK_K
            kv_h_i = h_i // groups
            T.copy(K[k_beg_abs:k_end_abs, kv_h_i, :], k)
            T.copy(V[k_beg_abs:k_end_abs, kv_h_i, :], v)
            T.clear(dk)
            T.clear(dv)

            lq = cu_seqlens_q[b_i + 1] - cu_seqlens_q[b_i]
            lk = cu_seqlens_k[b_i + 1] - cu_seqlens_k[b_i]
            q_offset = lk - lq

            if is_causal:
                q_loop_start = T.max(0, (k_beg - q_offset) // BLOCK_Q)
            else:
                q_loop_start = 0
            if window_size is not None:
                q_loop_end = T.min(
                    T.ceildiv((k_idx + 1) * BLOCK_K - q_offset + window_size, BLOCK_Q),
                    T.ceildiv(lq, BLOCK_Q),
                )
            else:
                q_loop_end = T.ceildiv(lq, BLOCK_Q)
            loop_range = q_loop_end - q_loop_start

            for q_local in T.Pipelined(loop_range, num_stages=num_stages):
                q_actual = q_local + q_loop_start
                q_beg = q_actual * BLOCK_Q
                q_beg_abs = cu_seqlens_q[b_i] + q_beg
                q_end_abs = q_beg_abs + BLOCK_Q
                T.copy(Q[q_beg_abs:q_end_abs, h_i, :], q)

                # S = QK^T * scaling
                # S^T = K @ Q^T * scaling
                T.clear(pT)
                T.gemm(k, q, pT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(BLOCK_K, BLOCK_Q):
                    pT[i, j] = pT[i, j] * scaling

                # P = softmax(S)
                T.copy(LSE[b_i, h_i, q_beg:q_beg + BLOCK_Q], lse)

                for i, j in T.Parallel(BLOCK_K, BLOCK_Q):
                    pT[i, j] = T.exp(pT[i, j] - lse[j])

                # mask
                if is_causal:
                    if window_size is not None:
                        for i, j in T.Parallel(BLOCK_K, BLOCK_Q):
                            pT[i, j] = T.if_then_else(
                                (k_beg + i < lk) and (q_beg + j < lq) and
                                (q_beg + j + q_offset >= k_beg + i) and
                                (k_beg + i >= q_beg + j + q_offset - window_size + 1),
                                pT[i, j],
                                0.,
                            )
                    else:
                        for i, j in T.Parallel(BLOCK_K, BLOCK_Q):
                            pT[i, j] = T.if_then_else(
                                (k_beg + i < lk) and (q_beg + j < lq) and
                                (q_beg + j + q_offset >= k_beg + i),
                                pT[i, j],
                                0.,
                            )
                else:
                    for i, j in T.Parallel(BLOCK_K, BLOCK_Q):
                        pT[i, j] = T.if_then_else(
                            (k_beg + i < lk) and (q_beg + j < lq),
                            pT[i, j],
                            0.,
                        )

                # dP = dO @ V^T
                # dP^T = V @ dO^T
                T.copy(dO[q_beg_abs:q_end_abs, h_i, :], do)
                T.clear(dpT)
                T.gemm(v, do, dpT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # dV = P^T @ dO
                T.copy(pT, pT_cast)
                T.gemm(pT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                # dS = P * (dP - delta), shape [LQ, LK]
                # dS^T = P^T * (dP^T - delta)
                # delta = sum_d(dO * O)
                T.copy(DELTA[b_i, h_i, q_beg:q_beg + BLOCK_Q], delta)
                for i, j in T.Parallel(BLOCK_K, BLOCK_Q):
                    dsT_cast[i, j] = pT[i, j] * (dpT[i, j] - delta[j])
                    dsT_cast[i, j] *= scaling

                # dK = (scaling * dS)^T @ Q
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                # dQ = (scaling * dS) @ K^T)
                T.clear(dq)
                T.gemm(dsT_cast, k, dq, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                T.atomic_add(dQ[q_beg_abs:q_end_abs, h_i, :], dq)

            T.atomic_add(dK[k_beg_abs:k_end_abs, kv_h_i, :], dk)
            T.atomic_add(dV[k_beg_abs:k_end_abs, kv_h_i, :], dv)

    return kernel


def _get_fwd_block_config(d):
    if d <= 128:
        return 64, 64, 3, 128
    elif d <= 256:
        return 64, 64, 2, 128
    elif d <= 512:
        return 64, 64, 1, 128
    else:
        raise NotImplementedError(f"Unsupported dimension: {d}")


def _get_bwd_block_config(d):
    if d <= 128:
        return 32, 128, 2, 256
    elif d <= 256:
        return 64, 64, 2, 256
    elif d <= 512:
        return 64, 32, 1, 128
    else:
        raise NotImplementedError(f"Unsupported dimension: {d}")


def _round_up(x, m):
    return ((x + m - 1) // m) * m


class MyfaVarlenSwSinks(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, sinks, cu_seqlens_q, cu_seqlens_k, max_seqlen, is_causal, window_size
    ):
        assert is_causal or window_size is None, "non-causal + sliding window is not supported"
        _, HQ, d = q.shape
        HK = k.shape[1]
        scaling = 1.0 / math.sqrt(d)
        has_sinks = sinks is not None
        if not has_sinks:
            sinks = torch.empty(HQ, dtype=q.dtype, device=q.device)

        block_q, block_k, num_stages, threads = _get_fwd_block_config(d)
        if window_size is not None:
            block_k = min(block_k, window_size)
        dt = q.dtype
        fwd_kernel = myfa_varlen_sw_sinks_fwd.compile(
            HQ=HQ,
            HK=HK,
            D=d,
            scaling=scaling,
            is_causal=is_causal,
            window_size=window_size,
            has_sinks=has_sinks,
            dtype=dt,
            BLOCK_Q=block_q,
            BLOCK_K=block_k,
            num_stages=num_stages,
            threads=threads,
        )
        batch_size = cu_seqlens_q.numel() - 1
        max_l = _round_up(max_seqlen, block_q)
        lse = torch.zeros((batch_size, HQ, max_l), device=q.device, dtype=torch.float32)
        o = fwd_kernel(q, k, v, sinks, cu_seqlens_q, cu_seqlens_k, max_seqlen, lse)
        ctx.save_for_backward(q, k, v, sinks, cu_seqlens_q, cu_seqlens_k, o, lse)
        ctx.scaling = scaling
        ctx.max_seqlen = max_seqlen
        ctx.is_causal = is_causal
        ctx.window_size = window_size
        ctx.has_sinks = has_sinks
        return o

    @staticmethod
    def backward(ctx, dO):
        q, k, v, sinks, cu_seqlens_q, cu_seqlens_k, o, lse = ctx.saved_tensors
        _, HQ, d = q.shape
        HK = k.shape[1]
        scaling = ctx.scaling
        max_seqlen = ctx.max_seqlen
        is_causal = ctx.is_causal
        window_size = ctx.window_size
        dt = q.dtype

        block_q_fwd, _, _, _ = _get_fwd_block_config(d)
        bwd_pre_kernel = myfa_varlen_sw_sinks_bwd_pre.compile(
            HQ=HQ, D=d, BLOCK_Q=block_q_fwd, dtype=dt
        )
        delta = torch.zeros_like(lse)
        bwd_pre_kernel(o, dO, cu_seqlens_q, max_seqlen, delta)

        block_q, block_k, num_stages, threads = _get_bwd_block_config(d)
        if window_size is not None:
            block_k = min(block_k, window_size)
        bwd_kernel = myfa_varlen_sw_sinks_bwd.compile(
            HQ=HQ,
            HK=HK,
            D=d,
            scaling=scaling,
            is_causal=is_causal,
            window_size=window_size,
            BLOCK_Q=block_q,
            BLOCK_K=block_k,
            num_stages=num_stages,
            threads=threads,
            dtype=dt,
        )
        dQ = torch.zeros_like(q, dtype=torch.float32)
        dK = torch.zeros(
            q.shape[0] if HQ == HK else k.shape[0], HK, d, dtype=torch.float32, device=q.device
        )
        dV = torch.zeros(
            q.shape[0] if HQ == HK else v.shape[0], HK, d, dtype=torch.float32, device=q.device
        )
        bwd_kernel(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen, lse, dO, delta, dQ, dK, dV)
        dQ = dQ.to(q.dtype)
        dK = dK.to(k.dtype)
        dV = dV.to(v.dtype)

        dsinks = None
        if ctx.has_sinks:
            dsink_kernel = myfa_varlen_sw_sinks_bwd_dsink.compile(HQ=HQ, dtype=dt)
            dsinks = dsink_kernel(sinks, delta, lse, cu_seqlens_q, max_seqlen).sum(0).sum(1)

        return dQ, dK, dV, dsinks, None, None, None, None, None


def myfa_varlen_sw_sinks(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen, sinks=None, is_causal=True, window_size=None
):
    return MyfaVarlenSwSinks.apply(
        q, k, v, sinks, cu_seqlens_q, cu_seqlens_k, max_seqlen, is_causal, window_size
    )
