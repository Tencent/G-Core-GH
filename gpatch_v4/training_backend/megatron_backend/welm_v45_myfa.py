from functools import lru_cache

import torch
import torch.distributed as dist
from einops import rearrange

from gfused.myfa_sw_sinks import (
    myfa_sw_sinks_bwd,
    myfa_sw_sinks_bwd_dsink,
    myfa_sw_sinks_bwd_pre,
    myfa_sw_sinks_fwd,
)
from gfused.myfa_varlen_sw_sinks import (
    myfa_varlen_sw_sinks_backward,
    myfa_varlen_sw_sinks_forward,
)

from megatron.core.extended_models.welm_v4_5.attention import WelmV45Attention
from megatron.core.transformer.dot_product_attention import DotProductAttention

from gpatch_v4.training_backend.megatron_backend.megatron_utils import unwrap_model


def _can_use_welm_v45_myfa(attention: WelmV45Attention, query, key) -> bool:
    cp_group = attention.pg_collection.cp
    assert cp_group is not None, _myfa_preflight_message(attention, query, key)
    cp_size = dist.get_world_size(cp_group)
    assert cp_size > 1, _myfa_preflight_message(attention, query, key)

    assert attention.config.attention_dropout == 0.0, _myfa_preflight_message(attention, query, key)
    assert query.dtype in (torch.float16,
                           torch.bfloat16), _myfa_preflight_message(attention, query, key)
    assert key.dtype == query.dtype, _myfa_preflight_message(attention, query, key)
    assert query.is_cuda and key.is_cuda, _myfa_preflight_message(attention, query, key)

    head_dim = query.shape[-1]
    assert head_dim > 0 and head_dim == 1 << (head_dim - 1).bit_length(), (
        _myfa_preflight_message(attention, query, key)
    )
    assert query.shape[0] == key.shape[0], _myfa_preflight_message(attention, query, key)
    assert query.shape[2] % key.shape[2] == 0, _myfa_preflight_message(attention, query, key)

    local_q = query.shape[1]
    global_k = key.shape[1]
    num_chunks = 2 * cp_size
    assert global_k % num_chunks == 0, _myfa_preflight_message(attention, query, key)
    chunk = global_k // num_chunks
    assert local_q == 2 * chunk, _myfa_preflight_message(attention, query, key)
    assert chunk % 64 == 0 and global_k % 64 == 0, _myfa_preflight_message(attention, query, key)

    window_size = attention._get_layer_window_size()
    assert window_size is None or window_size % 64 == 0, _myfa_preflight_message(
        attention, query, key
    )
    return True


def _zz_order_to_natural(tensor, cp_size):
    batch, seq, heads, head_dim = tensor.shape
    chunk = seq // (2 * cp_size)
    tensor = tensor.reshape(batch, cp_size, 2, chunk, heads, head_dim)
    first = tensor[:, :, 0]
    second = tensor[:, :, 1].flip(1)
    return torch.cat([first, second], dim=1).reshape(batch, seq, heads, head_dim)


def _natural_to_zz_order(tensor, cp_size):
    batch, seq, heads, head_dim = tensor.shape
    chunk = seq // (2 * cp_size)
    tensor = tensor.reshape(batch, 2 * cp_size, chunk, heads, head_dim)
    first = tensor[:, :cp_size]
    second = tensor[:, cp_size:].flip(1)
    return torch.stack([first, second], dim=2).reshape(batch, seq, heads, head_dim)


def _merge_zigzag_chunks(rank_chunks, cp_size):
    """把各 rank 的 zigzag 局部分片还原成自然 token 顺序。

    输入: ``rank_chunks[r]`` 是第 r 个 rank 的局部分片，形状 ``[2*chunk, ...]``，
    内容是 ``[自己的第一段, 自己的第二段]``（zigzag 规则：rank r 拥有第 r 块和
    第 ``2*cp_size-r-1`` 块）。
    返回: 一个 ``[2*cp_size*chunk, ...]`` 的自然序张量。

    例：``cp_size=2``，一条序列切成 4 块 ``[c0,c1,c2,c3]``，
    ``rank_chunks=[[c0,c3],[c1,c2]]``，合并后得到 ``[c0,c1,c2,c3]``。
    """
    if cp_size == 1:
        return rank_chunks[0]
    local_len = rank_chunks[0].shape[0]
    chunk = local_len // 2
    chunks = [None] * (2 * cp_size)
    for rank, local in enumerate(rank_chunks):
        first_idx = rank
        second_idx = 2 * cp_size - rank - 1
        chunks[first_idx] = local[:chunk]
        chunks[second_idx] = local[chunk:]
    return torch.cat(chunks, dim=0)


def _split_thd_by_padded_seqlens(tensor, padded_lens, cp_size):
    """按每条序列的 padded 长度，切分打包后的 THD 张量。

    输入: ``tensor`` 形状 ``[T, ...]``（多条序列首尾拼接）；``padded_lens``
    每条序列的 padded 长度；``cp_size`` 当前 CP 并行度（每条序列本地只占
    ``padded_len // cp_size`` 行，``cp_size=1`` 即完整序列）。
    返回: ``list[Tensor]``，每个元素是一条序列的切片（view，不拷贝）。

    例：``padded_lens=[8,4]``, ``cp_size=2`` -> 返回
    ``[tensor[0:4], tensor[4:6]]``。
    """
    parts = []
    offset = 0
    for padded_len in padded_lens:
        local_len = padded_len // cp_size
        parts.append(tensor[offset:offset + local_len])
        offset += local_len
    assert offset == tensor.shape[0], f"{offset=} != {tensor.shape[0]=}"
    return parts


def _cu_from_lengths(lengths, device):
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    if lengths:
        cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return cu


def _run_varlen_myfa(
    q_parts,
    k_parts,
    v_parts,
    sinks,
    window_size,
    softmax_scale,
):
    """把多个不定长 segment 拼成一次 TileLang varlen kernel 调用。

    输入: ``q_parts``/``k_parts``/``v_parts`` 是等长的 ``list[Tensor]``，每个
    元素是一个 segment 的 q/k/v。
    返回: ``(out_parts, lse)``，``out_parts`` 是每个 segment 对应的输出
    （顺序、长度和输入一致），``lse`` 供 backward 使用。

    注意: kernel 内部用 ``q_offset = len(k_i) - len(q_i)`` 推断每个 token 的
    因果位置（假设 q 是 k 的右对齐后缀），所以 ``k_parts[i]`` 必须精确截断到
    该 segment 允许看到的因果前缀，不能"传更长的 k 图安全"，否则因果边界会算错。
    """
    assert q_parts, "q_parts is empty"
    q_lens = [part.shape[0] for part in q_parts]
    k_lens = [part.shape[0] for part in k_parts]
    q = torch.cat(q_parts, dim=0).contiguous()
    k = torch.cat(k_parts, dim=0).contiguous()
    v = torch.cat(v_parts, dim=0).contiguous()
    device = q.device
    cu_q = _cu_from_lengths(q_lens, device)
    cu_k = _cu_from_lengths(k_lens, device)
    max_seqlen_q = max(q_lens)
    out, lse = myfa_varlen_sw_sinks_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q,
        sinks=sinks,
        is_causal=True,
        window_size=window_size,
        scaling=softmax_scale,
    )
    return list(torch.split(out, q_lens, dim=0)), lse


def _run_varlen_myfa_backward(
    q_parts,
    k_parts,
    v_parts,
    sinks,
    window_size,
    out_parts,
    lse,
    dout_parts,
    softmax_scale,
):
    if not q_parts:
        return [], [], [], None
    q_lens = [part.shape[0] for part in q_parts]
    k_lens = [part.shape[0] for part in k_parts]
    q = torch.cat(q_parts, dim=0).contiguous()
    k = torch.cat(k_parts, dim=0).contiguous()
    v = torch.cat(v_parts, dim=0).contiguous()
    out = torch.cat(out_parts, dim=0).contiguous()
    dout = torch.cat(dout_parts, dim=0).contiguous()
    device = q.device
    cu_q = _cu_from_lengths(q_lens, device)
    cu_k = _cu_from_lengths(k_lens, device)
    max_seqlen_q = max(q_lens)
    dquery, dkey, dvalue, dsinks = myfa_varlen_sw_sinks_backward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q,
        out,
        lse,
        dout,
        sinks=sinks,
        is_causal=True,
        window_size=window_size,
        scaling=softmax_scale,
    )
    return (
        list(torch.split(dquery, q_lens, dim=0)),
        list(torch.split(dkey, k_lens, dim=0)),
        list(torch.split(dvalue, k_lens, dim=0)),
        dsinks,
    )


@lru_cache(maxsize=128)
def _cached_q_offsets(batch, offsets, device_type, device_index):
    if device_index is None:
        device = torch.device(device_type)
    else:
        device = torch.device(device_type, device_index)
    return torch.tensor(offsets, dtype=torch.int32, device=device).repeat_interleave(batch)


def _q_offsets(batch, offsets, device):
    device = torch.device(device)
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return _cached_q_offsets(
        batch,
        tuple(int(offset) for offset in offsets),
        device.type,
        device_index,
    )


def _tensor_shape(tensor) -> tuple:
    return tuple(tensor.shape) if tensor is not None else None


def _myfa_preflight_message(attention: WelmV45Attention, query, key) -> str:
    cp_group = attention.pg_collection.cp
    cp_size = dist.get_world_size(cp_group) if cp_group is not None else None
    window_size = attention._get_layer_window_size()
    return (
        "WeLM v4.5 MyFA preflight failed; refusing to fallback because TileLang is required. "
        f"layer={attention.layer_number}, cp_size={cp_size}, "
        f"query_shape={_tensor_shape(query)}, key_shape={_tensor_shape(key)}, "
        f"query_dtype={getattr(query, 'dtype', None)}, key_dtype={getattr(key, 'dtype', None)}, "
        f"query_cuda={getattr(query, 'is_cuda', None)}, key_cuda={getattr(key, 'is_cuda', None)}, "
        f"attention_dropout={attention.config.attention_dropout}, window_size={window_size}"
    )


@lru_cache(maxsize=None)
def _get_myfa_fwd_kernel(
    heads_q,
    heads_k,
    head_dim,
    softmax_scale,
    is_causal,
    window_size,
    has_sinks,
    has_q_offsets,
    dtype,
):
    return myfa_sw_sinks_fwd.compile(
        HQ=heads_q,
        HK=heads_k,
        D=head_dim,
        scaling=softmax_scale,
        is_causal=is_causal,
        window_size=window_size,
        has_sinks=has_sinks,
        has_q_offsets=has_q_offsets,
        dtype=dtype,
    )


@lru_cache(maxsize=None)
def _get_myfa_bwd_pre_kernel(heads_q, head_dim, dtype):
    return myfa_sw_sinks_bwd_pre.compile(HQ=heads_q, D=head_dim, dtype=dtype)


@lru_cache(maxsize=None)
def _get_myfa_bwd_kernel(
    heads_q,
    heads_k,
    head_dim,
    softmax_scale,
    is_causal,
    window_size,
    has_q_offsets,
    dtype,
):
    return myfa_sw_sinks_bwd.compile(
        HQ=heads_q,
        HK=heads_k,
        D=head_dim,
        scaling=softmax_scale,
        is_causal=is_causal,
        window_size=window_size,
        has_q_offsets=has_q_offsets,
        dtype=dtype,
    )


@lru_cache(maxsize=None)
def _get_myfa_dsink_kernel(heads_q, dtype):
    return myfa_sw_sinks_bwd_dsink.compile(HQ=heads_q, dtype=dtype)


def _myfa_slices_fwd(q, k, v, sinks, q_offsets, window_size, softmax_scale):

    batch, _, heads_q, head_dim = q.shape
    heads_k = k.shape[2]
    sq = q.shape[1]
    sk = k.shape[1]
    assert window_size is None or window_size % 64 == 0
    assert sq % 64 == 0, f"seq_len_q={sq} must be divisible by BLOCK_Q (64)"
    assert sk % 64 == 0, f"seq_len_k={sk} must be divisible by BLOCK_K (64)"
    has_sinks = sinks is not None
    if not has_sinks:
        sinks = torch.empty(heads_q, dtype=q.dtype, device=q.device)
    fwd_kernel = _get_myfa_fwd_kernel(
        heads_q,
        heads_k,
        head_dim,
        softmax_scale,
        True,
        window_size,
        has_sinks,
        True,
        q.dtype,
    )
    q_offsets_t = _q_offsets(batch // len(q_offsets), q_offsets, q.device)
    return fwd_kernel(q, k, v, sinks, q_offsets_t)


def _myfa_slices_bwd(q, k, v, sinks, q_offsets, out, lse, dout, window_size, softmax_scale):

    batch, _, heads_q, head_dim = q.shape
    heads_k = k.shape[2]
    sq = q.shape[1]
    sk = k.shape[1]
    assert window_size is None or window_size % 64 == 0
    assert sq % 64 == 0, f"seq_len_q={sq} must be divisible by BLOCK_Q (64)"
    assert sk % 64 == 0, f"seq_len_k={sk} must be divisible by BLOCK_K (64)"
    dtype = q.dtype
    q_offsets = _q_offsets(batch // len(q_offsets), q_offsets, q.device)

    bwd_pre_kernel = _get_myfa_bwd_pre_kernel(heads_q, head_dim, dtype)
    delta = bwd_pre_kernel(out, dout)

    bwd_kernel = _get_myfa_bwd_kernel(
        heads_q,
        heads_k,
        head_dim,
        softmax_scale,
        True,
        window_size,
        True,
        dtype,
    )
    dquery = torch.zeros_like(q, dtype=torch.float32)
    dkey = torch.zeros(batch, k.shape[1], heads_k, head_dim, dtype=torch.float32, device=q.device)
    dvalue = torch.zeros(batch, v.shape[1], heads_k, head_dim, dtype=torch.float32, device=q.device)
    bwd_kernel(q, k, v, q_offsets, lse, dout, delta, dquery, dkey, dvalue)

    dsinks = None
    if sinks is not None:
        dsink_kernel = _get_myfa_dsink_kernel(heads_q, dtype)
        dsinks = dsink_kernel(sinks, delta, lse).sum(0).sum(1)

    return dquery.to(dtype), dkey.to(dtype), dvalue.to(dtype), dsinks


def _myfa_attention_with_context_parallel(
    attention: WelmV45Attention, query, key, value, attention_mask, sinks, softmax_scale,
    attention_dropout
):
    _can_use_welm_v45_myfa(attention, query, key)

    cp_group = attention.pg_collection.cp
    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    chunk = query.shape[1] // 2
    window_size = attention._get_layer_window_size()

    key = _zz_order_to_natural(key.contiguous(), cp_size)
    value = _zz_order_to_natural(value.contiguous(), cp_size)
    num_chunks = 2 * cp_size
    first_chunk = cp_rank
    second_chunk = num_chunks - cp_rank - 1
    batch = query.shape[0]
    query = rearrange(query, 'b (two s) h d -> (two b) s h d', two=2, s=chunk).contiguous()
    key = key.repeat(2, 1, 1, 1)
    value = value.repeat(2, 1, 1, 1)
    output, lse = _myfa_slices_fwd(
        query,
        key,
        value,
        sinks,
        [first_chunk * chunk, second_chunk * chunk],
        window_size,
        softmax_scale,
    )
    output = rearrange(output, '(two b) s h d -> b (two s) h d', two=2, b=batch)
    lse = rearrange(lse, '(two b) h s -> b h (two s)', two=2, b=batch)
    return output, lse


def _myfa_attention_with_context_parallel_backward(
    attention: WelmV45Attention,
    query,
    key,
    value,
    attention_mask,
    sinks,
    softmax_scale,
    attention_dropout,
    output,
    aux,
    doutput,
):
    cp_group = attention.pg_collection.cp
    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    chunk = query.shape[1] // 2
    window_size = attention._get_layer_window_size()

    key = _zz_order_to_natural(key.contiguous(), cp_size)
    value = _zz_order_to_natural(value.contiguous(), cp_size)
    num_chunks = 2 * cp_size
    first_chunk = cp_rank
    second_chunk = num_chunks - cp_rank - 1

    batch = query.shape[0]
    query = rearrange(query, 'b (two s) h d -> (two b) s h d', two=2, s=chunk).contiguous()
    output = rearrange(output, 'b (two s) h d -> (two b) s h d', two=2, s=chunk).contiguous()
    doutput = rearrange(doutput, 'b (two s) h d -> (two b) s h d', two=2, s=chunk).contiguous()
    aux = rearrange(aux, 'b h (two s) -> (two b) h s', two=2, s=chunk).contiguous()
    key = key.repeat(2, 1, 1, 1)
    value = value.repeat(2, 1, 1, 1)

    dquery, dkey, dvalue, dsinks = _myfa_slices_bwd(
        query,
        key,
        value,
        sinks,
        [first_chunk * chunk, second_chunk * chunk],
        output,
        aux,
        doutput,
        window_size,
        softmax_scale,
    )

    dquery = rearrange(dquery, '(two b) s h d -> b (two s) h d', two=2, b=batch)
    dkey = rearrange(dkey, '(two b) s h d -> two b s h d', two=2, b=batch).sum(0)
    dvalue = rearrange(dvalue, '(two b) s h d -> two b s h d', two=2, b=batch).sum(0)
    dkey = _natural_to_zz_order(dkey, cp_size)
    dvalue = _natural_to_zz_order(dvalue, cp_size)
    return dquery, dkey, dvalue, dsinks


def _myfa_attention_packed_thd_forward(
    attention: WelmV45Attention,
    query,
    key,
    value,
    attention_mask,
    sinks,
    softmax_scale,
    attention_dropout,
    packed_seq_params,
):
    """打包 THD 格式下 dynamic-CP 融合 attention 的前向。

    输入:
        query/key/value: ``[1, T, H, D]``，本 rank 的 zigzag 局部分片
            （``T = sum(padded_len // cp_size)``）。key/value 已经过 CP
            all-gather，本地长度是 ``T * cp_size``（各 rank 分片拼接），还是
            zigzag 顺序，没转成自然序。
        packed_seq_params: 需要 ``cu_seqlens_q_padded``（还原每条子序列的
            padded 长度）和 ``local_cp_size``/``cp_group``。
    返回:
        ``(out, lse)``: ``out`` 形状同 query（``[1, T, H, D]``），``lse`` 是
        softmax 统计量，backward 要用。

    处理步骤（``cp_size>1`` 时）: 把 gather 来的 K/V 按 rank、按序列切开 ->
    用 ``_merge_zigzag_chunks`` 还原成自然序 -> 给本 rank 的两个 zigzag Q 段
    各自配上对应长度的因果 K/V 前缀（zigzag 负载均衡：一段短前缀 + 一段长前缀）
    -> 一次性调用 ``_run_varlen_myfa`` -> 结果按本 rank 原始 THD 顺序拼回去。
    """
    # Packed THD enters here as [1, T, H, D]; drop the dummy batch dim.
    assert attention_mask is None
    assert attention_dropout == 0.0
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4, (
        f"THD MyFA expects [B, T, H, D], got {query.shape=}, {key.shape=}, {value.shape=}"
    )
    assert query.shape[0] == key.shape[0] == value.shape[0] == 1, (
        f"THD MyFA only supports packed batch dim 1, got {query.shape=}, {key.shape=}, "
        f"{value.shape=}"
    )
    assert query.dtype in (torch.float16, torch.bfloat16)
    assert key.dtype == query.dtype and value.dtype == query.dtype
    assert query.is_cuda and key.is_cuda and value.is_cuda
    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)

    # Resolve the dynamic CP group and the global packed sequence boundaries.
    cp_group = packed_seq_params.cp_group if packed_seq_params.cp_group is not None else attention.pg_collection.cp
    cp_size = int(
        packed_seq_params.local_cp_size or
        (dist.get_world_size(cp_group) if cp_group is not None else 1)
    )
    cp_rank = dist.get_rank(cp_group) if cp_size > 1 else 0
    window_size = attention._get_layer_window_size()
    assert window_size is None or window_size % 64 == 0

    padded_lens = packed_seq_params._myfa_padded_lens_cache
    assert padded_lens, "packed THD MyFA requires at least one packed sequence"

    # No CP communication is needed; run varlen attention over packed sequences directly.
    if cp_size == 1:
        out_parts, lse = _run_varlen_myfa(
            _split_thd_by_padded_seqlens(query, padded_lens, 1),
            _split_thd_by_padded_seqlens(key, padded_lens, 1),
            _split_thd_by_padded_seqlens(value, padded_lens, 1),
            sinks,
            window_size,
            softmax_scale,
        )
        return torch.cat(out_parts, dim=0).contiguous().unsqueeze(0), lse

    # K/V arrive from dot-attention CP all-gather as rank-concat zigzag shards.
    local_total = sum(padded_len // cp_size for padded_len in padded_lens)
    k_rank_chunks = list(torch.split(key, local_total, dim=0))
    v_rank_chunks = list(torch.split(value, local_total, dim=0))
    assert len(k_rank_chunks) == cp_size and len(v_rank_chunks) == cp_size, (
        f"Expected gathered K/V for {cp_size=} ranks, got {len(k_rank_chunks)=}, "
        f"{len(v_rank_chunks)=}"
    )
    # to natural by rank
    k_by_rank = [
        _split_thd_by_padded_seqlens(k_rank, padded_lens, cp_size) for k_rank in k_rank_chunks
    ]
    v_by_rank = [
        _split_thd_by_padded_seqlens(v_rank, padded_lens, cp_size) for v_rank in v_rank_chunks
    ]
    q_parts_local = _split_thd_by_padded_seqlens(query, padded_lens, cp_size)

    # Rebuild each packed sequence's full K/V in natural token order.
    k_full_parts = []
    v_full_parts = []
    for seq_idx in range(len(padded_lens)):
        k_full_parts.append(
            _merge_zigzag_chunks([k_by_rank[rank][seq_idx] for rank in range(cp_size)], cp_size)
        )
        v_full_parts.append(
            _merge_zigzag_chunks([v_by_rank[rank][seq_idx] for rank in range(cp_size)], cp_size)
        )

    # For each local zigzag Q chunk, pair it with the causal K/V prefix it may attend to.
    q_segments, k_segments, v_segments, segment_seq_ids, segment_is_second = [], [], [], [], []
    for seq_idx, (q_local, padded_len) in enumerate(zip(q_parts_local, padded_lens)):
        chunk = padded_len // (2 * cp_size)
        assert q_local.shape[0] == 2 * chunk, (
            f"Unexpected THD local chunk size: {q_local.shape[0]=}, {chunk=}, {cp_size=}"
        )
        if chunk == 0:
            continue
        first_start = cp_rank * chunk
        first_end = first_start + chunk
        second_start = (2 * cp_size - cp_rank - 1) * chunk
        second_end = second_start + chunk

        q_segments.append(q_local[:chunk])
        k_segments.append(k_full_parts[seq_idx][:first_end])
        v_segments.append(v_full_parts[seq_idx][:first_end])
        segment_seq_ids.append(seq_idx)
        segment_is_second.append(False)

        q_segments.append(q_local[chunk:])
        k_segments.append(k_full_parts[seq_idx][:second_end])
        v_segments.append(v_full_parts[seq_idx][:second_end])
        segment_seq_ids.append(seq_idx)
        segment_is_second.append(True)

    # Run one varlen kernel call over all local query segments, keeping lse for backward.
    segment_out, lse = _run_varlen_myfa(
        q_segments,
        k_segments,
        v_segments,
        sinks,
        window_size,
        softmax_scale,
    )
    first_by_seq = {}
    second_by_seq = {}
    for seq_idx, is_second, out in zip(segment_seq_ids, segment_is_second, segment_out):
        if is_second:
            second_by_seq[seq_idx] = out
        else:
            first_by_seq[seq_idx] = out

    # Restore this rank's local THD order: first zigzag chunk followed by second chunk.
    out_parts = []
    for seq_idx in range(len(padded_lens)):
        out_parts.append(first_by_seq[seq_idx])
        out_parts.append(second_by_seq[seq_idx])
    return torch.cat(out_parts, dim=0).contiguous().unsqueeze(0), lse


def _myfa_attention_packed_thd_backward(
    attention: WelmV45Attention,
    query,
    key,
    value,
    attention_mask,
    sinks,
    softmax_scale,
    attention_dropout,
    output,
    aux,
    doutput,
    packed_seq_params,
):
    """打包 THD 格式下 dynamic-CP 融合 attention 的反向，与 forward 对称。

    输入:
        query/key/value: 同 forward，``[1, T, H, D]`` 的本 rank 局部分片（key/
            value 是 CP all-gather 后的 zigzag 顺序）。
        output/doutput: forward 的输出和对应梯度，``[1, T, H, D]``。
        aux: forward 返回的 lse。
        packed_seq_params: 同 forward。
    返回:
        ``(dquery, dkey, dvalue, dsinks)``: dquery 形状同 query
        （``[1, T, H, D]``）；dkey/dvalue 形状同 forward 输入的 key/value
        （即 CP all-gather 后的形状，``[1, T*cp_size, H, D]``）。

    处理步骤（``cp_size>1`` 时）: 复现 forward 的切分与 ``_merge_zigzag_chunks``
    还原自然序 K/V -> 对每个 zigzag Q 段调用 ``_run_varlen_myfa_backward`` 算出
    dq/dk/dv，按序列累加到自然序的 dk/dv 缓冲区（同一序列的两个 zigzag 段前缀
    有重叠，需要 ``+=``）-> 把自然序 dk/dv 按 zigzag 规则重新切回各 rank 分片、
    拼接成 all-gather 前的布局 -> dq 按本 rank 原始 THD 顺序拼回去。
    """
    # Match forward's packed THD view and consume the saved varlen lse as aux.
    assert attention_mask is None
    assert attention_dropout == 0.0
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4
    assert query.shape[0] == key.shape[0] == value.shape[0] == 1

    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    output = output.squeeze(0)
    doutput = doutput.squeeze(0)

    # Resolve the same dynamic CP group and sequence boundaries used in forward.
    cp_group = packed_seq_params.cp_group if packed_seq_params.cp_group is not None else attention.pg_collection.cp
    cp_size = int(
        packed_seq_params.local_cp_size or
        (dist.get_world_size(cp_group) if cp_group is not None else 1)
    )
    cp_rank = dist.get_rank(cp_group) if cp_size > 1 else 0
    window_size = attention._get_layer_window_size()
    assert window_size is None or window_size % 64 == 0

    padded_lens = packed_seq_params._myfa_padded_lens_cache
    assert padded_lens, "packed THD MyFA backward requires at least one packed sequence"

    # Single-rank path mirrors forward: split by sequence, run varlen bwd, concatenate.
    if cp_size == 1:
        q_parts = _split_thd_by_padded_seqlens(query, padded_lens, 1)
        k_parts = _split_thd_by_padded_seqlens(key, padded_lens, 1)
        v_parts = _split_thd_by_padded_seqlens(value, padded_lens, 1)
        out_parts = _split_thd_by_padded_seqlens(output, padded_lens, 1)
        dout_parts = _split_thd_by_padded_seqlens(doutput, padded_lens, 1)
        dq_parts, dk_parts, dv_parts, dsinks = _run_varlen_myfa_backward(
            q_parts,
            k_parts,
            v_parts,
            sinks,
            window_size,
            out_parts,
            aux,
            dout_parts,
            softmax_scale,
        )
        return (
            torch.cat(dq_parts, dim=0).contiguous().unsqueeze(0),
            torch.cat(dk_parts, dim=0).contiguous().unsqueeze(0),
            torch.cat(dv_parts, dim=0).contiguous().unsqueeze(0),
            dsinks,
        )

    # Recreate forward's gathered K/V view: rank-concat zigzag -> per-sequence natural.
    local_total = sum(padded_len // cp_size for padded_len in padded_lens)
    k_rank_chunks = list(torch.split(key, local_total, dim=0))
    v_rank_chunks = list(torch.split(value, local_total, dim=0))
    k_by_rank = [
        _split_thd_by_padded_seqlens(k_rank, padded_lens, cp_size) for k_rank in k_rank_chunks
    ]
    v_by_rank = [
        _split_thd_by_padded_seqlens(v_rank, padded_lens, cp_size) for v_rank in v_rank_chunks
    ]
    q_parts_local = _split_thd_by_padded_seqlens(query, padded_lens, cp_size)
    out_parts_local = _split_thd_by_padded_seqlens(output, padded_lens, cp_size)
    dout_parts_local = _split_thd_by_padded_seqlens(doutput, padded_lens, cp_size)

    k_full_parts = []
    v_full_parts = []
    for seq_idx in range(len(padded_lens)):
        k_full_parts.append(
            _merge_zigzag_chunks([k_by_rank[rank][seq_idx] for rank in range(cp_size)], cp_size)
        )
        v_full_parts.append(
            _merge_zigzag_chunks([v_by_rank[rank][seq_idx] for rank in range(cp_size)], cp_size)
        )

    # Rebuild exactly the same Q segment and K/V prefix list used by forward.
    q_segments, k_segments, v_segments = [], [], []
    out_segments, dout_segments = [], []
    segment_seq_ids, segment_is_second = [], []
    for seq_idx, (q_local, out_local, dout_local, padded_len) in enumerate(
        zip(q_parts_local, out_parts_local, dout_parts_local, padded_lens)
    ):
        chunk = padded_len // (2 * cp_size)
        assert q_local.shape[0] == 2 * chunk, (
            f"Unexpected THD local chunk size: {q_local.shape[0]=}, {chunk=}, {cp_size=}"
        )
        if chunk == 0:
            continue
        first_end = cp_rank * chunk + chunk
        second_end = (2 * cp_size - cp_rank - 1) * chunk + chunk

        q_segments.append(q_local[:chunk])
        k_segments.append(k_full_parts[seq_idx][:first_end])
        v_segments.append(v_full_parts[seq_idx][:first_end])
        out_segments.append(out_local[:chunk])
        dout_segments.append(dout_local[:chunk])
        segment_seq_ids.append(seq_idx)
        segment_is_second.append(False)

        q_segments.append(q_local[chunk:])
        k_segments.append(k_full_parts[seq_idx][:second_end])
        v_segments.append(v_full_parts[seq_idx][:second_end])
        out_segments.append(out_local[chunk:])
        dout_segments.append(dout_local[chunk:])
        segment_seq_ids.append(seq_idx)
        segment_is_second.append(True)

    # Run TileLang varlen backward using forward output, lse, and upstream doutput.
    dq_segments, dk_segments, dv_segments, dsinks = _run_varlen_myfa_backward(
        q_segments,
        k_segments,
        v_segments,
        sinks,
        window_size,
        out_segments,
        aux,
        dout_segments,
        softmax_scale,
    )

    # Scatter segment grads back to local Q chunks and natural full-sequence K/V buffers.
    dq_parts_local = [torch.zeros_like(part) for part in q_parts_local]
    dk_full_parts = [torch.zeros_like(part) for part in k_full_parts]
    dv_full_parts = [torch.zeros_like(part) for part in v_full_parts]
    for seq_idx, is_second, dq, dk, dv in zip(
        segment_seq_ids, segment_is_second, dq_segments, dk_segments, dv_segments
    ):
        chunk = padded_lens[seq_idx] // (2 * cp_size)
        if is_second:
            second_end = (2 * cp_size - cp_rank - 1) * chunk + chunk
            dq_parts_local[seq_idx][chunk:] += dq
            dk_full_parts[seq_idx][:second_end] += dk
            dv_full_parts[seq_idx][:second_end] += dv
        else:
            first_end = cp_rank * chunk + chunk
            dq_parts_local[seq_idx][:chunk] += dq
            dk_full_parts[seq_idx][:first_end] += dk
            dv_full_parts[seq_idx][:first_end] += dv

    # Convert natural K/V grads back to rank-concat zigzag for CP reduce-scatter.
    dkey_by_rank = [[] for _ in range(cp_size)]
    dvalue_by_rank = [[] for _ in range(cp_size)]
    for seq_idx, padded_len in enumerate(padded_lens):
        chunk = padded_len // (2 * cp_size)
        for rank in range(cp_size):
            dk_local = torch.zeros_like(k_by_rank[rank][seq_idx])
            dv_local = torch.zeros_like(v_by_rank[rank][seq_idx])
            first_start = rank * chunk
            first_end = first_start + chunk
            second_start = (2 * cp_size - rank - 1) * chunk
            second_end = second_start + chunk
            dk_local[:chunk] = dk_full_parts[seq_idx][first_start:first_end]
            dk_local[chunk:] = dk_full_parts[seq_idx][second_start:second_end]
            dv_local[:chunk] = dv_full_parts[seq_idx][first_start:first_end]
            dv_local[chunk:] = dv_full_parts[seq_idx][second_start:second_end]
            dkey_by_rank[rank].append(dk_local)
            dvalue_by_rank[rank].append(dv_local)

    dquery = torch.cat(dq_parts_local, dim=0).contiguous().unsqueeze(0)
    dkey = torch.cat([torch.cat(rank_parts, dim=0) for rank_parts in dkey_by_rank],
                     dim=0).contiguous().unsqueeze(0)
    dvalue = torch.cat([torch.cat(rank_parts, dim=0) for rank_parts in dvalue_by_rank],
                       dim=0).contiguous().unsqueeze(0)
    return dquery, dkey, dvalue, dsinks


def _install_welm_v45_myfa_hook(attention: WelmV45Attention) -> bool:
    try:
        attention._gcore_welm_v45_myfa_installed
        return False
    except AttributeError:
        pass

    def build_cp_layer_attention_mask(query, key, attention_mask):
        return None

    def custom_attn_fwd_func(*args):
        return _myfa_attention_with_context_parallel(attention, *args)

    def custom_attn_bwd_func(*args):
        return _myfa_attention_with_context_parallel_backward(attention, *args)

    def custom_attn_packed_func(*args):
        return _myfa_attention_packed_thd_forward(attention, *args)

    def custom_attn_packed_bwd_func(*args):
        return _myfa_attention_packed_thd_backward(attention, *args)

    attention._gcore_welm_v45_myfa_installed = True
    attention.core_attention.custom_attn_fwd_func = custom_attn_fwd_func
    attention.core_attention.custom_attn_bwd_func = custom_attn_bwd_func
    attention.core_attention.custom_attn_packed_func = custom_attn_packed_func
    attention.core_attention.custom_attn_packed_bwd_func = custom_attn_packed_bwd_func
    attention._build_cp_layer_attention_mask = build_cp_layer_attention_mask
    return True


def install_welm_v45_myfa_hooks(model, hf_config=None) -> int:
    unwrapped_model = unwrap_model(model)
    if isinstance(unwrapped_model, list):
        model_parts = unwrapped_model
    else:
        model_parts = [unwrapped_model]

    installed = 0
    for model_part in model_parts:
        for module in model_part.modules():
            if isinstance(module, WelmV45Attention
                         ) and isinstance(module.core_attention, DotProductAttention):
                if _install_welm_v45_myfa_hook(module):
                    installed += 1
    return installed
