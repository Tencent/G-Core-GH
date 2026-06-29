from functools import lru_cache

import torch
import torch.distributed as dist
from einops import rearrange

from megatron.core.extended_models.welm_v4_5.attention import WelmV45Attention
from megatron.core.transformer.dot_product_attention import DotProductAttention

from gfused.myfa_sw_sinks import (
    myfa_sw_sinks_bwd,
    myfa_sw_sinks_bwd_dsink,
    myfa_sw_sinks_bwd_pre,
    myfa_sw_sinks_fwd,
)
from gpatch_v4.training_backend.megatron_backend.megatron_utils import unwrap_model


def _can_use_welm_v45_myfa(attention: WelmV45Attention, query, key) -> bool:
    cp_group = attention.pg_collection.cp
    assert cp_group is not None, _myfa_preflight_message(attention, query, key)
    cp_size = dist.get_world_size(cp_group)
    assert cp_size > 1, _myfa_preflight_message(attention, query, key)

    assert attention.config.attention_dropout == 0.0, _myfa_preflight_message(
        attention, query, key
    )
    assert query.dtype in (torch.float16, torch.bfloat16), _myfa_preflight_message(
        attention, query, key
    )
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
    assert chunk % 64 == 0 and global_k % 64 == 0, _myfa_preflight_message(
        attention, query, key
    )

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
    dkey = torch.zeros(
        batch, k.shape[1], heads_k, head_dim, dtype=torch.float32, device=q.device
    )
    dvalue = torch.zeros(
        batch, v.shape[1], heads_k, head_dim, dtype=torch.float32, device=q.device
    )
    bwd_kernel(q, k, v, q_offsets, lse, dout, delta, dquery, dkey, dvalue)

    dsinks = None
    if sinks is not None:
        dsink_kernel = _get_myfa_dsink_kernel(heads_q, dtype)
        dsinks = dsink_kernel(sinks, delta, lse).sum(0).sum(1)

    return dquery.to(dtype), dkey.to(dtype), dvalue.to(dtype), dsinks


def _myfa_attention_with_context_parallel(
    attention: WelmV45Attention, query, key, value, attn_bias, sinks, softmax_scale, attention_dropout
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
    query = rearrange(
        query, 'b (two s) h d -> (two b) s h d', two=2, s=chunk
    ).contiguous()
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
    attn_bias,
    sinks,
    softmax_scale,
    attention_dropout,
    output,
    lse,
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
    query = rearrange(
        query, 'b (two s) h d -> (two b) s h d', two=2, s=chunk
    ).contiguous()
    output = rearrange(
        output, 'b (two s) h d -> (two b) s h d', two=2, s=chunk
    ).contiguous()
    doutput = rearrange(
        doutput, 'b (two s) h d -> (two b) s h d', two=2, s=chunk
    ).contiguous()
    lse = rearrange(lse, 'b h (two s) -> (two b) h s', two=2, s=chunk).contiguous()
    key = key.repeat(2, 1, 1, 1)
    value = value.repeat(2, 1, 1, 1)

    dquery, dkey, dvalue, dsinks = _myfa_slices_bwd(
        query,
        key,
        value,
        sinks,
        [first_chunk * chunk, second_chunk * chunk],
        output,
        lse,
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


def _install_welm_v45_myfa_hook(attention: WelmV45Attention) -> bool:
    try:
        attention._gcore_welm_v45_myfa_installed
        return False
    except AttributeError:
        pass

    def build_cp_layer_attention_mask(query, key, attention_mask):
        return None
    attention._gcore_welm_v45_myfa_installed = True
    attention.core_attention.custom_attn_fwd_func = lambda *args: _myfa_attention_with_context_parallel(
        attention, *args
    )
    attention.core_attention.custom_attn_bwd_func = (
        lambda *args: _myfa_attention_with_context_parallel_backward(attention, *args)
    )
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
            if isinstance(module, WelmV45Attention) and isinstance(
                module.core_attention, DotProductAttention
            ):
                if _install_welm_v45_myfa_hook(module):
                    installed += 1
    return installed
