"""
Cross-DP-rank sequence length balancing utilities.

Rebalance samples across DP ranks before compute_log_probs to minimize padding waste,
then restore original order afterwards.

Communication strategy (optimized):
  - Tensor data: serialized as raw uint8 bytes (zero dtype conversion overhead),
        exchanged via CPU gloo P2P isend/irecv (zero GPU memory).
  - Metadata: lightweight all_gather_object for layouts + non-tensor fields.
  - Phase 1 (rebalance): all_gather seqlens -> KK partition -> P2P tensor exchange
  - Phase 2 (restore): same P2P tensor exchange approach
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.distributed

from megatron.core import mpu

from gpatch_v4.core.seqlen_balancing import get_seqlen_balanced_partitions
from gpatch_v4.utils import log, sync_cuda_and_get_time
from gpatch_v4.utils.training_utils import expand_rollout_batch, expand_rollout_batches


def _compute_partition_plan(
    all_seqlens_list: List[List[int]],
    dp_size: int,
) -> List[List[Tuple[int, int, int]]]:
    """
    Compute deterministic partition plan from global seqlens.
    All ranks call this with the same input, so results are identical.

    Uses Karmarkar-Karp differencing method (via get_seqlen_balanced_partitions)
    for better balanced partitions compared to LPT.

    Returns:
        bins: bins[rank_id] = List[(orig_rank, orig_local_idx, seqlen)]
    """
    # Build flat list of (orig_rank, local_idx, seqlen)
    global_items = []
    for rank_id, rank_seqlens in enumerate(all_seqlens_list):
        for local_idx, sl in enumerate(rank_seqlens):
            global_items.append((rank_id, local_idx, sl))

    total_samples = len(global_items)
    bin_capacity = total_samples // dp_size
    assert bin_capacity * dp_size == total_samples, (
        f"total_samples={total_samples} must be divisible by dp_size={dp_size}"
    )

    # Extract seqlen list for Karmarkar-Karp partitioning
    seqlen_list = [item[2] for item in global_items]

    # Use Karmarkar-Karp algorithm for balanced partitioning
    partitions = get_seqlen_balanced_partitions(
        seqlen_list=seqlen_list,
        k_partitions=dp_size,
        equal_size=True,
    )

    # Map index-based partitions back to (orig_rank, orig_local_idx, seqlen) tuples
    bins = []
    for partition in partitions:
        bin_items = [global_items[idx] for idx in partition]
        # Sort within each bin by seqlen ascending (better packing for fwd pass)
        bin_items.sort(key=lambda x: x[2])
        bins.append(bin_items)

    return bins


def _compute_send_recv_plan(
    bins: List[List[Tuple[int, int, int]]],
    dp_rank: int,
    dp_size: int,
    local_count: int,
) -> Tuple[
    Dict[int, List[int]],
    Dict[int, List[Tuple[int, int]]],
]:
    """
    Compute which samples each rank sends to which other rank.

    Returns:
        send_to: {dest_rank: [local_idx, ...]} — samples this rank sends to dest_rank
        recv_from: {src_rank: [(orig_rank, orig_local_idx), ...]} — samples this rank receives from src_rank
    """
    send_to: Dict[int, List[int]] = {r: [] for r in range(dp_size)}
    recv_from: Dict[int, List[Tuple[int, int]]] = {r: [] for r in range(dp_size)}

    for dest_rank, bin_items in enumerate(bins):
        for (orig_rank, orig_local_idx, _) in bin_items:
            if orig_rank == dp_rank:
                send_to[dest_rank].append(orig_local_idx)
            if dest_rank == dp_rank:
                recv_from[orig_rank].append((orig_rank, orig_local_idx))

    return send_to, recv_from


# ---------------------------------------------------------------------------
# CPU P2P-based tensor exchange helpers
# ---------------------------------------------------------------------------


def _classify_sample_keys(sample: Dict[str, Any],
                          require_keys: List[str] = None) -> Tuple[List[str], List[str]]:
    """Classify sample keys into tensor keys and non-tensor keys.

    Args:
        sample: A sample dict.
        require_keys: If given, only classify these keys.

    Returns:
        tensor_keys: sorted keys whose values are ``torch.Tensor``.
        non_tensor_keys: sorted keys whose values are not.
    """
    tensor_keys = []
    non_tensor_keys = []
    keys_to_check = require_keys if require_keys is not None else list(sample.keys())
    for k in keys_to_check:
        if k not in sample:
            continue
        v = sample[k]
        if torch.is_tensor(v):
            tensor_keys.append(k)
        else:
            non_tensor_keys.append(k)
    tensor_keys.sort()
    non_tensor_keys.sort()
    return tensor_keys, non_tensor_keys


# Map between torch dtype and a serializable string for layout metadata
_DTYPE_TO_STR = {
    torch.float16: 'f16',
    torch.bfloat16: 'bf16',
    torch.float32: 'f32',
    torch.float64: 'f64',
    torch.int8: 'i8',
    torch.int16: 'i16',
    torch.int32: 'i32',
    torch.int64: 'i64',
    torch.bool: 'bool',
    torch.uint8: 'u8',
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}

# Transport dtype for raw byte-level communication. Using uint8 so we can
# pack any dtype into a byte buffer without conversion or precision loss.
_TRANSPORT_DTYPE = torch.uint8


def _serialize_samples_to_buffer(
    samples: List[Dict[str, Any]],
    tensor_keys: List[str],
) -> Tuple[torch.Tensor, List[Dict[str, Tuple[int, int, str, tuple]]]]:
    """Serialize tensor fields from samples into a single contiguous uint8 byte buffer.

    Tensors keep their original dtypes; each is viewed as raw bytes and
    concatenated into one contiguous CPU buffer. Avoids dtype conversion
    (no float64 bloat), keeps everything on CPU.

    Args:
        samples: row-based sample dicts.
        tensor_keys: sorted tensor keys to serialize.

    Returns:
        buffer: 1D uint8 CPU tensor.
        layouts: per-sample dict, ``layouts[i][key] = (byte_offset, numel, dtype_str, shape_tuple)``.
    """
    byte_segments = []
    layouts = []
    byte_offset = 0

    for sample in samples:
        sample_layout = {}
        for key in tensor_keys:
            t = sample[key]
            numel = t.numel()
            dtype_str = _DTYPE_TO_STR.get(t.dtype, str(t.dtype))
            # View as raw bytes: contiguous flat -> reinterpret as uint8
            flat = t.contiguous().reshape(-1)
            byte_view = flat.view(torch.uint8
                                 ) if flat.numel() > 0 else torch.empty(0, dtype=torch.uint8)
            nbytes = byte_view.numel()
            sample_layout[key] = (byte_offset, numel, dtype_str, tuple(t.shape))
            byte_segments.append(byte_view)
            byte_offset += nbytes
        layouts.append(sample_layout)

    if byte_offset == 0:
        return torch.empty(0, dtype=_TRANSPORT_DTYPE), layouts

    buffer = torch.cat(byte_segments)
    return buffer, layouts


def _deserialize_buffer_to_samples(
    buffer: torch.Tensor,
    layouts: List[Dict[str, Tuple[int, int, str, tuple]]],
    non_tensor_data: List[Dict[str, Any]],
    tensor_keys: List[str],
) -> List[Dict[str, Any]]:
    """Reconstruct sample dicts from a uint8 byte buffer + non-tensor metadata.

    Args:
        buffer: 1D uint8 CPU tensor.
        layouts: from ``_serialize_samples_to_buffer``.
        non_tensor_data: per-sample non-tensor key-value dicts.
        tensor_keys: sorted tensor keys.

    Returns:
        samples: row-based sample dicts.
    """
    samples = []
    for i, layout in enumerate(layouts):
        sample = {}
        # Restore non-tensor fields
        if non_tensor_data and i < len(non_tensor_data):
            sample.update(non_tensor_data[i])
        # Restore tensor fields from raw bytes
        for key in tensor_keys:
            byte_offset, numel, dtype_str, orig_shape = layout[key]
            orig_dtype = _STR_TO_DTYPE.get(dtype_str, torch.float32)
            element_size = torch.tensor([], dtype=orig_dtype).element_size()
            nbytes = numel * element_size
            if nbytes > 0:
                raw_bytes = buffer[byte_offset:byte_offset + nbytes]
                # Clone to guarantee storage_offset == 0 so that .view()
                # satisfies the alignment requirement for any target dtype
                # (e.g. int64 requires 8-byte alignment).
                raw_bytes = raw_bytes.clone()
                # Reinterpret uint8 bytes back to original dtype
                flat = raw_bytes.view(orig_dtype)
                sample[key] = flat.reshape(orig_shape)
            else:
                sample[key] = torch.empty(orig_shape, dtype=orig_dtype)
        samples.append(sample)
    return samples


def _exchange_samples_p2p(
    local_samples: List[Dict[str, Any]],
    send_to: Dict[int, List[int]],
    recv_from: Dict[int, List[Any]],
    dp_rank: int,
    dp_size: int,
    dp_group,
    require_keys: List[str] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """Exchange samples across DP ranks using CPU P2P isend/irecv.

    All communication stays on CPU. Tensor data is packed into uint8
    byte buffers (no dtype conversion); non-tensor metadata uses
    ``all_gather_object``.

    Args:
        local_samples: row-based sample dicts on this rank.
        send_to: ``{dest_rank: [local_idx, ...]}``.
        recv_from: ``{src_rank: [(orig_rank, orig_local_idx), ...]}``.
        dp_rank: this rank's DP index.
        dp_size: total DP world size.
        dp_group: DP process group.
        require_keys: if given, only exchange these keys.

    Returns:
        ``{src_rank: list of received sample dicts}``.
    """
    if len(local_samples) == 0:
        return {r: [] for r in range(dp_size)}

    # Classify keys (use first sample as reference)
    tensor_keys, non_tensor_keys = _classify_sample_keys(local_samples[0], require_keys)

    # --- Phase 1: Serialize tensor data per destination rank (CPU, original dtype as bytes) ---
    send_buffers = {}  # dest_rank -> uint8 CPU tensor
    send_layouts = {}  # dest_rank -> layout list
    send_non_tensor = {}  # dest_rank -> list of non-tensor dicts

    for dest_rank in range(dp_size):
        idxs = send_to[dest_rank]
        if not idxs:
            send_buffers[dest_rank] = torch.empty(0, dtype=_TRANSPORT_DTYPE)
            send_layouts[dest_rank] = []
            send_non_tensor[dest_rank] = []
            continue

        dest_samples = [local_samples[i] for i in idxs]
        buf, layouts = _serialize_samples_to_buffer(dest_samples, tensor_keys)
        send_buffers[dest_rank] = buf.contiguous()
        send_layouts[dest_rank] = layouts

        nt_data = []
        for s in dest_samples:
            nt = {k: s[k] for k in non_tensor_keys if k in s}
            nt_data.append(nt)
        send_non_tensor[dest_rank] = nt_data

    # --- Phase 2: Exchange buffer sizes (tiny, via all_gather_object) ---
    send_sizes = [send_buffers[r].numel() for r in range(dp_size)]
    all_send_sizes = [None] * dp_size
    torch.distributed.all_gather_object(all_send_sizes, send_sizes, group=dp_group)
    recv_sizes = [all_send_sizes[src_rank][dp_rank] for src_rank in range(dp_size)]

    # --- Phase 3: CPU P2P isend/irecv for byte buffers ---
    # Allocate recv buffers on CPU
    recv_buffers = {}
    for src_rank in range(dp_size):
        sz = recv_sizes[src_rank]
        recv_buffers[src_rank] = torch.empty(sz, dtype=_TRANSPORT_DTYPE)

    # Translate DP-local ranks to global ranks for P2P
    dp_global_ranks = torch.distributed.get_process_group_ranks(dp_group)

    ops = []
    for peer_rank in range(dp_size):
        peer_global = dp_global_ranks[peer_rank]
        if peer_rank == dp_rank:
            # Local copy: no communication needed
            if send_buffers[dp_rank].numel() > 0:
                recv_buffers[dp_rank] = send_buffers[dp_rank].clone()
            continue

        # Send to peer
        if send_buffers[peer_rank].numel() > 0:
            op = torch.distributed.isend(send_buffers[peer_rank], dst=peer_global, group=dp_group)
            ops.append(op)

        # Recv from peer
        if recv_buffers[peer_rank].numel() > 0:
            op = torch.distributed.irecv(recv_buffers[peer_rank], src=peer_global, group=dp_group)
            ops.append(op)

    # Wait for all P2P ops to complete
    for op in ops:
        op.wait()

    # --- Phase 4: Exchange layouts + non-tensor metadata (lightweight) ---
    my_meta = []
    for dest_rank in range(dp_size):
        my_meta.append(
            {
                'layouts': send_layouts[dest_rank],
                'non_tensor': send_non_tensor[dest_rank],
            }
        )

    gathered_meta = [None] * dp_size
    torch.distributed.all_gather_object(gathered_meta, my_meta, group=dp_group)

    # --- Phase 5: Deserialize received data ---
    recv_data_by_src = {}
    for src_rank in range(dp_size):
        meta = gathered_meta[src_rank][dp_rank]
        recv_samples = _deserialize_buffer_to_samples(
            buffer=recv_buffers[src_rank],
            layouts=meta['layouts'],
            non_tensor_data=meta['non_tensor'],
            tensor_keys=tensor_keys,
        )
        recv_data_by_src[src_rank] = recv_samples

    return recv_data_by_src


def rebalance_across_dp_ranks(
    rollout_batches: List[Dict[str, List[Any]]],
    samples_per_batch: int,
    require_keys: List[str] = None,
) -> Tuple[List[Dict[str, List[Any]]], dict]:
    """Rebalance samples across DP ranks to minimize padding waste in compute_log_probs.

    Uses CPU P2P isend/irecv for tensor data and lightweight
    ``all_gather_object`` for metadata only.

    Args:
        rollout_batches: Column-based, original order.
        samples_per_batch: ``rollout_mbs * keep_n``.
        require_keys: If given, only exchange these keys.

    Returns:
        rebalanced_rollout_batches: Column-based, rebalanced.
        restore_info: Info needed to restore original order.
    """
    t1 = sync_cuda_and_get_time()
    dp_group = mpu.get_data_parallel_group()
    gloo_dp_group = mpu.get_data_parallel_group_gloo()
    dp_size = mpu.get_data_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank()

    # Step 1: Expand to row-based
    local_samples = expand_rollout_batches(rollout_batches)
    local_count = len(local_samples)
    t2 = sync_cuda_and_get_time()

    # Step 2: all_gather seqlens only (lightweight, for partition planning)
    local_seqlens = [int(s['sequence_lengths'].item()) for s in local_samples]
    all_seqlens_list = [None] * dp_size
    torch.distributed.all_gather_object(all_seqlens_list, local_seqlens, group=gloo_dp_group)
    t3 = sync_cuda_and_get_time()

    # Step 3: Compute partition plan (deterministic, same on all ranks)
    bins = _compute_partition_plan(all_seqlens_list, dp_size)
    my_bin = bins[dp_rank]
    t4 = sync_cuda_and_get_time()

    # Step 4: Compute send/recv plan
    send_to, recv_from = _compute_send_recv_plan(bins, dp_rank, dp_size, local_count)
    t5 = sync_cuda_and_get_time()

    # Step 5: Exchange samples via CPU P2P (isend/irecv) + lightweight all_gather_object (metadata)
    recv_data_by_src = _exchange_samples_p2p(
        local_samples,
        send_to,
        recv_from,
        dp_rank,
        dp_size,
        gloo_dp_group,
        require_keys=require_keys,
    )
    t6 = sync_cuda_and_get_time()

    # Step 6: Assemble final samples in my_bin order
    src_counters = {r: 0 for r in range(dp_size)}
    rebalanced_samples = []
    for (orig_rank, orig_local_idx, _) in my_bin:
        idx = src_counters[orig_rank]
        rebalanced_samples.append(recv_data_by_src[orig_rank][idx])
        src_counters[orig_rank] = idx + 1

    # Step 7: Pack back to column-based rollout_batches
    rebalanced_rollout_batches = _pack_to_column_based(rebalanced_samples, samples_per_batch)
    t7 = sync_cuda_and_get_time()

    # Step 8: Build restore_info
    restore_info = {
        'dp_rank': dp_rank,
        'dp_size': dp_size,
        'dp_group': dp_group,
        'gloo_dp_group': gloo_dp_group,
        'bins': bins,
        'my_bin': my_bin,
        'send_to': send_to,
        'recv_from': recv_from,
        'orig_local_count': local_count,
        'samples_per_batch': samples_per_batch,
        'require_keys': require_keys,
    }

    log(
        f"[DP_BALANCE] dp_rank={dp_rank} rebalanced: "
        f"orig_count={local_count}, new_count={len(rebalanced_samples)}, "
        f"orig_max_seqlen={max(local_seqlens)}, "
        f"new_max_seqlen={max(x[2] for x in my_bin)}, "
        f"orig_min_seqlen={min(local_seqlens)}, "
        f"new_min_seqlen={min(x[2] for x in my_bin)}",
        rank=0
    )
    log(
        f"[DP_BALANCE] timing: expand={t2-t1:.4f}, gather_seqlens={t3-t2:.4f}, "
        f"partition={t4-t3:.4f}, plan={t5-t4:.4f}, "
        f"p2p_exchange={t6-t5:.4f}, assemble+pack={t7-t6:.4f}",
        rank=0
    )

    return rebalanced_rollout_batches, restore_info


def restore_original_order(
    rebalanced_rollout_batches: List[Dict[str, List[Any]]],
    restore_info: dict,
) -> List[Dict[str, List[Any]]]:
    """Restore samples to original DP rank distribution and order after compute_log_probs.

    Args:
        rebalanced_rollout_batches: Column-based, rebalanced order
            (now containing logprobs fields).
        restore_info: From ``rebalance_across_dp_ranks``.

    Returns:
        Column-based rollout batches in original order.
    """
    t1 = sync_cuda_and_get_time()
    dp_rank = restore_info['dp_rank']
    dp_size = restore_info['dp_size']
    dp_group = restore_info['dp_group']
    gloo_dp_group = restore_info['gloo_dp_group']
    my_bin = restore_info['my_bin']
    orig_local_count = restore_info['orig_local_count']
    samples_per_batch = restore_info['samples_per_batch']

    # Step 1: Expand to row-based
    rebalanced_samples = expand_rollout_batches(rebalanced_rollout_batches)
    assert len(rebalanced_samples) == len(my_bin), (
        f"rebalanced count {len(rebalanced_samples)} != bin size {len(my_bin)}"
    )

    # Step 2: Build reverse send_to: send samples back to their orig_rank
    restore_send_to: Dict[int, List[int]] = {r: [] for r in range(dp_size)}
    for i, (orig_rank, orig_local_idx, _) in enumerate(my_bin):
        restore_send_to[orig_rank].append(i)

    # Step 3: Compute what we will receive
    restore_recv_from: Dict[int, List[Tuple[int, int]]] = {r: [] for r in range(dp_size)}
    for src_rank, bin_items in enumerate(restore_info['bins']):
        for (orig_rank, orig_local_idx, _) in bin_items:
            if orig_rank == dp_rank:
                restore_recv_from[src_rank].append((orig_rank, orig_local_idx))

    # Step 4: Exchange samples via CPU P2P
    t2 = sync_cuda_and_get_time()
    recv_data_by_src = _exchange_samples_p2p(
        rebalanced_samples,
        restore_send_to,
        restore_recv_from,
        dp_rank,
        dp_size,
        gloo_dp_group,
        require_keys=restore_info.get('require_keys', None),
    )
    t3 = sync_cuda_and_get_time()

    # Step 5: Place samples back in original local order
    restored_samples = [None] * orig_local_count
    src_counters = {r: 0 for r in range(dp_size)}

    for src_rank in range(dp_size):
        for (orig_rank, orig_local_idx) in restore_recv_from[src_rank]:
            assert orig_rank == dp_rank
            idx = src_counters[src_rank]
            assert restored_samples[orig_local_idx] is None, (
                f"duplicate assignment at orig_local_idx={orig_local_idx}"
            )
            restored_samples[orig_local_idx] = recv_data_by_src[src_rank][idx]
            src_counters[src_rank] = idx + 1

    assert all(s is not None for s in restored_samples), "some samples were not restored!"

    # Step 6: Pack back to column-based rollout_batches
    restored_rollout_batches = _pack_to_column_based(restored_samples, samples_per_batch)
    t4 = sync_cuda_and_get_time()

    log(
        f"[DP_BALANCE] dp_rank={dp_rank} restored {len(restored_samples)} samples. "
        f"timing: expand={t2-t1:.4f}, p2p_exchange={t3-t2:.4f}, assemble+pack={t4-t3:.4f}",
        rank=0
    )

    return restored_rollout_batches


# ---------------------------------------------------------------------------
# Helper: row-based -> column-based packing
# ---------------------------------------------------------------------------


def _pack_to_column_based(
    row_samples: List[Dict[str, Any]],
    chunk_size: int,
) -> List[Dict[str, List[Any]]]:
    """Pack row-based samples into column-based rollout_batches by chunk_size."""
    batches = []
    for i in range(0, len(row_samples), chunk_size):
        chunk = row_samples[i:i + chunk_size]
        col_batch = {}
        for key in chunk[0].keys():
            col_batch[key] = [s[key] for s in chunk]
        batches.append(col_batch)
    return batches
