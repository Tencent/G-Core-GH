# ruff: noqa
# Adapted from miles_plugins/models/glm5/ops/tilelang_indexer_fwd.py for DeepSeek-V4.
# Key differences from GLM-5:
#   - Operates on [seqlen, batch, heads, dim] (SBHD) layout, batch handled externally
#   - Uses causal mask via cu_seqlens instead of variable-length packed sequences
#   - Supports compressed KV (seq_len_kv = seq_len_q / compress_ratio)

import torch

try:
    import tilelang
    from tilelang import language as T
except ImportError:
    from gfused.fake_tilelang_stub import language_stub as T
    from gfused.fake_tilelang_stub import tilelang_stub as tilelang


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, )
def tl_indexer_fwd_impl(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
):
    if block_Q is None:
        block_Q = 128 // heads
    softmax_scale = index_dim**-0.5
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def tl_indexer_fwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(
                T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages
            ):
                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (
                        T.max(s_reshaped[bn_i, bq_i, h_i], 0) * softmax_scale * weights[bq_i, h_i]
                    )

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i,
                                                                                           bq_i]

    return tl_indexer_fwd_kernel


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, )
def tl_indexer_fwd_fp8_impl(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
    fp8_block_size=128,
):
    if block_Q is None:
        block_Q = 128 // heads
    assert index_dim == fp8_block_size
    softmax_scale = index_dim**-0.5
    dtype = torch.float8_e4m3fn
    scale_dtype = torch.float8_e8m0fnu
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_qs_shape = [seq_len * heads, index_dim // fp8_block_size]
    index_k_shape = [seq_len_kv, index_dim]
    index_ks_shape = [seq_len_kv, index_dim // fp8_block_size]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def tl_indexer_fwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexQs: T.Tensor(index_qs_shape, scale_dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        IndexKs: T.Tensor(index_ks_shape, scale_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_qs_shared = T.alloc_shared([block_Q * heads, 1], scale_dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            index_ks_shared = T.alloc_shared([block_N, 1], scale_dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            # block_Q=128 // heads=64 = 2, which is cheap.
            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(IndexQs[seq_len_i * heads, 0], index_qs_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(
                T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages
            ):
                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                T.copy(IndexKs[cu_k_s_min + nbn_i * block_N, 0], index_ks_shared)

                T.clear(s)
                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (
                        T.max(
                            s_reshaped[bn_i, bq_i, h_i] *
                            T.Cast(accum_dtype, index_ks_shared[bn_i, 0]) *
                            T.Cast(accum_dtype, index_qs_shared[bq_i * heads + h_i, 0]),
                            0,
                        ) * softmax_scale * weights[bq_i, h_i]
                    )

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                # https://www.tilelang.com/programming_guides/control_flow.html
                # The LegalizeSafeMemoryAccess pass automatically inserts guards when an access may be out‑of‑bounds,
                # and elides them when proven safe. You can often omit explicit if checks for simple edge handling,
                # but keep them when you need custom logic or clarity.
                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i,
                                                                                           bq_i]

    return tl_indexer_fwd_kernel


@tilelang.jit
def clean_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel


def make_causal_cu_ks_and_cu_ke_for_bshd(
    seq_len_q, seq_len_kv, compress_ratio, device, *, positions=None
):
    """Generate cu_seqlens for causal masking on compressed KV positions.

    For query at position p, valid compressed groups are [0, (p+1) // compress_ratio).

    Parameters
    ----------
    positions : Tensor, optional
        Shape ``[seq_len_q]`` int32. Actual query positions (e.g. CP-offset).
        Defaults to ``arange(seq_len_q)`` when ``None``.
    """
    if positions is None:
        positions = torch.arange(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ks = torch.zeros(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ke = ((positions + 1) // compress_ratio).to(torch.int32)
    return cu_seqlen_ks, cu_seqlen_ke


def indexer_fwd_interface(
    q,
    kv,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
    clean_logits=True,
    *,
    qs=None,
    ks=None,
):
    """Forward interface matching GLM-5's API but for a single batch element.

    Dispatches by `q.dtype`:
      - bf16 / fp16 → `tl_indexer_fwd_impl`
      - float8_e4m3fn → `tl_indexer_fwd_fp8_impl` (requires `qs` / `ks`)

    Args:
        q: [seq_len, heads, index_dim] bf16 or float8_e4m3fn
        kv: [seq_len_kv, index_dim] same dtype as `q`
        weights: [seq_len, heads] fp32
        cu_seqlen_ks: [seq_len] int32 — start of valid KV range per query
        cu_seqlen_ke: [seq_len] int32 — end of valid KV range per query
        qs: [seq_len * heads, 1] float8_e8m0fnu, required when `q` is fp8
        ks: [seq_len_kv, 1] float8_e8m0fnu, required when `q` is fp8

    Returns:
        logits: [seq_len, seq_len_kv] fp32
    """
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]
    assert kv.dtype == q.dtype, f"q/kv dtype mismatch: {q.dtype} vs {kv.dtype}"

    logits = torch.empty([seq_len, seq_len_kv], device=q.device, dtype=torch.float32)
    q_flat = q.view(seq_len * heads, index_dim)
    w = weights.float()

    if q.dtype in (torch.bfloat16, torch.float16):
        assert qs is None and ks is None, "qs/ks only valid for float8_e4m3fn"
        tl_indexer_fwd_impl(heads=heads, index_dim=index_dim)(
            q_flat,
            kv,
            logits,
            w,
            cu_seqlen_ks,
            cu_seqlen_ke,
        )
    elif q.dtype == torch.float8_e4m3fn:
        assert qs is not None and ks is not None, "fp8 path requires qs/ks"
        tl_indexer_fwd_fp8_impl(heads=heads, index_dim=index_dim)(
            q_flat,
            qs,
            kv,
            ks,
            logits,
            w,
            cu_seqlen_ks,
            cu_seqlen_ke,
        )
    else:
        raise TypeError(f"unsupported indexer dtype: {q.dtype}")

    if clean_logits:
        clean_logits_()(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits


def batched_indexer_fwd(
    q, k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True, *, qs=None, ks=None
):
    """Batched forward: loops over batch dim.

    Args:
        q: [seqlen, batch, heads, dim] bf16 or float8_e4m3fn
        k: [seqlen_kv, batch, dim] same dtype as `q`
        weights: [seqlen, batch, heads] fp32
        cu_seqlen_ks: [seqlen] int32
        cu_seqlen_ke: [seqlen] int32
        clean_logits: if False, caller must mask out-of-range before top-k
        qs: [seqlen, batch, heads, 1] float8_e8m0fnu, required when `q` is fp8
        ks: [seqlen_kv, batch, 1] float8_e8m0fnu, required when `q` is fp8

    Returns:
        logits: [batch, seqlen, seqlen_kv] fp32
    """
    seqlen, batch, heads, dim = q.shape
    seq_len_kv = k.shape[0]

    if batch == 1:
        # [S,1,H,D] squeeze 是 view；unsqueeze [S,T] 避免再 copy 一份 [1,S,T]
        all_logits = indexer_fwd_interface(
            q.squeeze(1).contiguous(),
            k.squeeze(1).contiguous(),
            weights.squeeze(1).contiguous(),
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits,
            qs=None if qs is None else qs.squeeze(1).contiguous().view(seqlen * heads, -1),
            ks=None if ks is None else ks.squeeze(1).contiguous().view(seq_len_kv, -1),
        ).unsqueeze(0)
    else:
        all_logits = torch.empty([batch, seqlen, seq_len_kv], device=q.device, dtype=torch.float32)
        for b in range(batch):
            all_logits[b] = indexer_fwd_interface(
                q[:, b, :, :].contiguous(),
                k[:, b, :].contiguous(),
                weights[:, b, :].contiguous(),
                cu_seqlen_ks,
                cu_seqlen_ke,
                clean_logits,
                qs=None if qs is None else qs[:, b, :, :].contiguous().view(seqlen * heads, -1),
                ks=None if ks is None else ks[:, b, :].contiguous().view(seq_len_kv, -1),
            )
    return all_logits
