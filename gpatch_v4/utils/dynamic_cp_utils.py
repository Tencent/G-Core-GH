"""Dynamic Context Parallel (Dynamic CP) utilities for gcore V4 training.

Key-wise reroute / packing helpers shared by the per-model
``*_reroute_data_for_dynamic_cp`` implementations (SFT, GRPO, ...): given
per-sample tensors keyed by field name, they schedule samples across DPxCP
ranks and build packed THD microbatches.

Note: the abbreviation "DCP" in this codebase elsewhere refers to PyTorch
``torch.distributed.checkpoint``; this module always uses ``dyn_cp``.
"""
import torch
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Tuple

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.data_schedule_utils import _get_global_seqlens_and_ids

from gpatch_v4.utils import logging_rank0


def _round_up(n: int, divisor: int) -> int:
    if divisor <= 1:
        return n
    return ((n + divisor - 1) // divisor) * divisor


# 2026.05.17 by guanyouhe
# modified from Megatron-LM/megatron/core/datasets/data_schedule_utils.py:_get_global_seqlens_and_ids
def get_global_id_1d_len_by_keys(
    keys: list[str],
    batch: list[dict[str, torch.Tensor]],
    dp_group: torch.distributed.ProcessGroup,
) -> dict[str, list[tuple[int, int]]]:
    """Gather 1-D tensor lengths for multiple keys with fixed DP communication.

    Regardless of ``len(keys)``, performs exactly two collectives on ``dp_group``:
    an ``all_gather`` of per-rank sample counts, then one ``all_gather`` of stacked
    seqlens ``[max_sub_samples, len(keys)]``.

    Parameters
    ----------
    keys
        Field names whose per-sample 1-D length is needed for reroute split sizes.
    batch
        Local subsamples on this DP rank.
    dp_group
        Data-parallel process group.

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        Per key, ``[(global_id, seqlen), ...]`` in global sample order.
    """
    if not keys:
        return {}

    dev = torch.cuda.current_device()
    num_local = len(batch)
    num_keys = len(keys)
    assert num_local > 0 and num_keys > 0

    seqlens_rows = []
    for s in batch:
        row = []
        for key in keys:
            if key in s and s[key] is not None:
                assert s[key].dim() == 1, f"key {key} must be 1-D"
                row.append(s[key].shape[-1])
            else:
                row.append(0)
        seqlens_rows.append(row)

    subsample_seqlens = torch.tensor(seqlens_rows, dtype=torch.int32, device=dev)
    local_len = torch.tensor([num_local], dtype=torch.int32, device=dev)
    dp_subsample_count = [torch.zeros_like(local_len) for _ in range(dp_group.size())]
    torch.distributed.all_gather(dp_subsample_count, local_len, group=dp_group)

    dp_subsample_counts = torch.stack(dp_subsample_count, dim=0).view(-1)
    max_sub_samples = int(dp_subsample_counts.max().item())

    if num_local < max_sub_samples:
        padding = torch.zeros(max_sub_samples - num_local, num_keys, dtype=torch.int32, device=dev)
        subsample_seqlens_padded = torch.cat([subsample_seqlens, padding], dim=0)
    else:
        subsample_seqlens_padded = subsample_seqlens

    seqlens_gathered = [torch.empty_like(subsample_seqlens_padded) for _ in range(dp_group.size())]
    torch.distributed.all_gather(seqlens_gathered, subsample_seqlens_padded, group=dp_group)

    per_rank_tensors = []
    for dp_rank, seqlen in enumerate(seqlens_gathered):
        n = int(dp_subsample_counts[dp_rank].item())
        per_rank_tensors.append(seqlen[:n])

    seqlens_all = torch.cat(per_rank_tensors, dim=0)
    total_samples = seqlens_all.shape[0]
    seqlens_by_col = seqlens_all.t().tolist()

    return {
        key: [(i, seqlens_by_col[ki][i]) for i in range(total_samples)]
        for ki, key in enumerate(keys)
    }


# 2026.05.17 by guanyouhe
# modified from Megatron-LM/megatron/core/datasets/data_schedule_utils.py:reroute_samples_to_dcp_ranks
# global_id_seqlens: list[tuple[int, int]]
#    ->
#    global_id_seqlens_dict: dict[str, list[tuple[int, int]]], [key: list[(id, seqlen)]]
# sample_id_groups: list[list[list[int]]]
#    第一层数组：microbatch
#    第二层数组：dcp
#    第三层数组：global id
# NOTE(guanyouhe): 如果有非 tensor 类型的数据，直接用 torch.all_gather_object 即可
def reroute_samples_to_dcp_ranks_by_keys(
    batch: list[dict[str, torch.Tensor]],
    global_ids_this_rank: torch.Tensor,
    global_id_seqlens_dict: dict[str, list[tuple[int, int]]],
    sample_id_groups: list[list[list[int]]],
    offsets: torch.Tensor,  # 1-D int32 tensor
    dp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
    total_dcp_gpus: int,
    dtype_map: dict[str, torch.dtype] = {},
):
    """
    Reroutes the sub-samples to the correct rank after scheduling.

    For each key in the batch dict, we perform an all-to-all communication
    to transfer the data to the correct ranks.
    """
    def _gid_to_src_rank(gid: int) -> int:
        dp_src_rank = torch.bucketize(gid, offsets[1:] - 1)
        dcp_rank = (
            torch.distributed.get_process_group_ranks(dp_group)[dp_src_rank] // tp_group.size()
        ) % dp_cp_group.size()
        return dcp_rank

    gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
    dcp_rank = dp_cp_group.rank()
    dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
    dp_ranks = [(r // tp_group.size()) % dp_cp_group.size() for r in dp_ranks]

    data_keys = batch[0].keys()

    # Create the send plan
    combined_sample_id_groups: List[List[int]] = [[] for _ in range(total_dcp_gpus)]
    for d in range(total_dcp_gpus):
        for sample_id_group in sample_id_groups:
            combined_sample_id_groups[d].extend(sample_id_group[d])
    for dest_rank in range(total_dcp_gpus):
        combined_sample_id_groups[dest_rank].sort()

    send_ids_sorted = [
        gid for d in dp_ranks for gid in combined_sample_id_groups[d] if gid in global_ids_this_rank
    ]

    send_num_split_dict = {}
    send_lens_split_dict = {}
    for dest_rank in range(total_dcp_gpus):
        for key in data_keys:
            if key not in send_num_split_dict:
                send_num_split_dict[key] = [0] * total_dcp_gpus
                send_lens_split_dict[key] = [0] * total_dcp_gpus

            global_id_seqlens = global_id_seqlens_dict[key]
            if dest_rank in dp_ranks:
                send_seq_lens = [
                    global_id_seqlens[gid][1]
                    for gid in combined_sample_id_groups[dest_rank] if gid in global_ids_this_rank
                ]
                send_num_split_dict[key][dest_rank] = len(send_seq_lens)
                send_lens_split_dict[key][dest_rank] = sum(send_seq_lens)
            else:
                send_lens_split_dict[key][dest_rank] = 0

    # Create the recv plan
    recv_sample_id_groups = [[] for _ in range(total_dcp_gpus)]
    for gid in combined_sample_id_groups[dcp_rank]:
        src_rank = _gid_to_src_rank(gid)
        recv_sample_id_groups[src_rank].append(gid)

    recv_lens_split_dict = {}
    for src_rank in range(total_dcp_gpus):
        for key in data_keys:
            if key not in recv_lens_split_dict:
                recv_lens_split_dict[key] = [0] * total_dcp_gpus

            global_id_seqlens = global_id_seqlens_dict[key]
            recv_lens_split_dict[key][src_rank] = sum(
                [global_id_seqlens[gid][1] for gid in recv_sample_id_groups[src_rank]]
            )

    recv_ids_sorted = [gid for d in range(total_dcp_gpus) for gid in recv_sample_id_groups[d]]
    recv_counts = [len(recv_sample_id_groups[d]) for d in range(total_dcp_gpus)]

    recv_samples = [{k: None for k in data_keys} for _ in range(sum(recv_counts))]

    def _pack_sample_by_key(key: str) -> torch.Tensor:
        flattened_tensors = []
        for gid in send_ids_sorted:
            value = batch[gid2local_id[gid]][key]
            if value is None:
                continue
            t = value.to(torch.cuda.current_device(), non_blocking=True)
            flattened_tensors.append(t.reshape(-1))
        if key in dtype_map:
            dtype = dtype_map[key]
        else:
            dtype = batch[0][key].dtype
        return (
            torch.cat(flattened_tensors, dim=0) if flattened_tensors else
            torch.empty(0, device=torch.cuda.current_device(), dtype=dtype)
        )

    def _unpack_sample_by_key(key: str, recv_tensor: torch.Tensor):
        cursor = 0
        for i, gid in enumerate(recv_ids_sorted):
            sample_len = global_id_seqlens_dict[key][gid][1]
            recv_samples[i][key] = recv_tensor[cursor:cursor + sample_len]
            cursor += sample_len

    for key in data_keys:
        output_split_sizes = recv_lens_split_dict[key]
        input_split_sizes = send_lens_split_dict[key]
        send_tensor = _pack_sample_by_key(key)
        recv_tensor_size = sum(output_split_sizes)
        recv_tensor = torch.empty(
            recv_tensor_size, device=torch.cuda.current_device(), dtype=send_tensor.dtype
        )
        torch.distributed.all_to_all_single(
            output=recv_tensor,
            input=send_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=dp_cp_group,
        )
        _unpack_sample_by_key(key, recv_tensor)

    recv_sample_with_id = {recv_id: recv_samples[i] for i, recv_id in enumerate(recv_ids_sorted)}
    return recv_sample_with_id


# 2026.05.17 by guanyouhe
# modified from Megatron-LM/megatron/core/datasets/data_schedule_utils.py:_pack_sequences
def _pack_sequences_by_keys(
    samples: List,
    padded_lengths: torch.Tensor,
    original_lengths: torch.Tensor,
    local_cp_size: Optional[torch.Tensor],
    dev: torch.device,
    packed_keys: List[str],
    cat_keys: List[str],
) -> Dict[str, torch.Tensor]:
    """Pack multiple samples into a single packed sample."""
    def _pack_tensors(tensors):
        return torch.cat([t.reshape(-1) for t in tensors], dim=0)

    new_sample = {}
    for key in packed_keys:
        packed_tensors = []
        for sample in samples:
            if sample[key] is None:
                continue
            packed_tensors.append(sample[key])
        new_sample[key] = _pack_tensors(packed_tensors)
    for key in cat_keys:
        cat_tensors = []
        for sample in samples:
            if sample[key] is None:
                continue
            cat_tensors.append(sample[key])
        if len(cat_tensors) > 0:
            new_sample[key] = torch.cat(cat_tensors, dim=0)
        else:
            new_sample[key] = None

    padded_lengths = padded_lengths.to(device=dev, dtype=torch.int32, non_blocking=True).reshape(-1)
    cu_seqlens_padded = torch.empty(padded_lengths.numel() + 1, device=dev, dtype=torch.int32)
    cu_seqlens_padded[0] = 0
    cu_seqlens_padded[1:] = torch.cumsum(padded_lengths, dim=0)
    max_seqlen = torch.max(padded_lengths).to(dtype=torch.int32)

    new_sample["cu_seqlens_padded"] = cu_seqlens_padded
    new_sample["max_seqlen"] = max_seqlen

    original_lengths = original_lengths.to(device=dev, dtype=torch.int32,
                                           non_blocking=True).reshape(-1)
    cu_seqlens = torch.empty(original_lengths.numel() + 1, device=dev, dtype=torch.int32)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = torch.cumsum(original_lengths, dim=0).reshape(-1)
    new_sample["cu_seqlens"] = cu_seqlens

    if local_cp_size is not None:
        new_sample["local_cp_size"] = local_cp_size

    return new_sample


# 2026.05.17 by guanyouhe
# modified from Megatron-LM/megatron/core/datasets/data_schedule_utils.py:build_packed_microbatches
def build_packed_microbatches_by_keys(
    samples_this_rank_with_id: Dict[int, Dict[str, torch.Tensor]],
    sample_id_groups: List[List[List[int]]],
    dcp_rank: int,
    dev: torch.device,
    is_dynamic_cp: bool = False,
    packed_keys: Optional[List[str]] = None,
    cat_keys: Optional[List[str]] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Build packed samples for each microbatch.

    Args:
        samples_this_rank_with_id: Mapping from global sample ID to sample dict.
        sample_id_groups: Per-microbatch, per-rank lists of global sample IDs.
        dcp_rank: Index within the DP×CP group.
        dev: Target device.
        is_dynamic_cp: Whether dynamic context parallel is enabled.
    """
    assert packed_keys is not None and cat_keys is not None

    num_micro_batches = len(sample_id_groups)
    seg_starts: List[int] = [0]
    original_lens_tensors = []
    padded_lens_tensors = []

    grouped_samples = [
        [
            samples_this_rank_with_id[sub_sample_id]
            for sub_sample_id in sample_id_groups[i][dcp_rank]
        ] for i in range(num_micro_batches)
    ]

    local_cp_sizes_gpu = None
    if is_dynamic_cp:
        local_cp_sizes_cpu: List[int] = []
        for i in range(num_micro_batches):
            sample_ids_this_group = sample_id_groups[i][dcp_rank]
            local_cp_sizes_cpu.append(
                len(
                    [
                        1 for sample_ids in sample_id_groups[i]
                        if sample_ids_this_group[0] in sample_ids
                    ]
                )
            )
        local_cp_sizes_gpu = torch.tensor(local_cp_sizes_cpu, dtype=torch.int32, device=dev)

    for i in range(num_micro_batches):
        samples = grouped_samples[i]
        seg_starts.append(seg_starts[-1] + len(samples))
        original_lens_tensors.extend([s["original_seq_len"].reshape(-1) for s in samples])
        padded_lens_tensors.extend([s["padded_seq_len"].reshape(-1) for s in samples])

    padded_lens_all_gpu = torch.cat(padded_lens_tensors, dim=0).to(dtype=torch.int32)
    original_lens_all_gpu = torch.cat(original_lens_tensors, dim=0).to(dtype=torch.int32)

    new_samples: List[Dict[str, torch.Tensor]] = []
    for i in range(num_micro_batches):
        samples = grouped_samples[i]
        lens_padded = padded_lens_all_gpu[seg_starts[i]:seg_starts[i + 1]]
        lens_original = original_lens_all_gpu[seg_starts[i]:seg_starts[i + 1]]
        local_cp_size = local_cp_sizes_gpu[i] if is_dynamic_cp else None
        new_sample = _pack_sequences_by_keys(
            samples, lens_padded, lens_original, local_cp_size, dev, packed_keys, cat_keys
        )
        new_samples.append(new_sample)

    return new_samples


def dyn_cp_schedule_default(
    gbs_batches: List[Dict[str, Any]],
    dp_group,
    tp_group,
    dp_cp_group,
    cp_size: int,
    dp_size: int,
    dist_config,
    dev,
    packed_keys: list[str],
    cat_keys: list[str],
    global_id_seqlens_keys: list[str],
    dtype_map: Dict[str, torch.dtype],
) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float]:
    """Default dynamic CP: global sorting + all-to-all redistribution."""
    total_dyn_cp_gpus = dp_cp_group.size()

    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=dist_config.max_seqlen_per_dp_cp_rank,
        cp_size=cp_size,
        dp_size=dp_size,
        microbatch_group_size_per_vp_stage=None,
        min_cp_size=dist_config.min_dynamic_context_parallel_size,
    )

    subsample_seqlens = torch.tensor(
        [s["tokens"].shape[-1] for s in gbs_batches],
        dtype=torch.int32,
        device=dev,
    )
    global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered = (
        _get_global_seqlens_and_ids(subsample_seqlens, dp_group)
    )
    sample_id_groups = scheduler.get_groups_and_subsamples(global_id_seqlens)

    global_id_seqlens_dict = get_global_id_1d_len_by_keys(
        global_id_seqlens_keys,
        gbs_batches,
        dp_group,
    )
    # both are list[tuple[int, int]] with pure Python ints (via .tolist())
    assert global_id_seqlens_dict['tokens'] == global_id_seqlens

    samples_this_rank_with_id = reroute_samples_to_dcp_ranks_by_keys(
        gbs_batches,
        global_ids_this_rank,
        global_id_seqlens_dict,
        sample_id_groups,
        offsets,
        dp_group,
        tp_group,
        dp_cp_group,
        total_dyn_cp_gpus,
        dtype_map,
    )

    dyn_cp_rank = dp_cp_group.rank()
    num_micro_batches = len(sample_id_groups)

    new_samples = build_packed_microbatches_by_keys(
        samples_this_rank_with_id,
        sample_id_groups,
        dyn_cp_rank,
        dev,
        scheduler.is_dynamic_cp,
        packed_keys=packed_keys,
        cat_keys=cat_keys,
    )

    # Move packed microbatches to CPU to reduce peak GPU memory when GBS is
    # large.  Each microbatch will be moved back to GPU lazily during forward.
    for sample in new_samples:
        for k, v in list(sample.items()):
            if isinstance(v, torch.Tensor) and v.is_cuda:
                sample[k] = v.to('cpu', non_blocking=True)
    torch.cuda.current_stream().synchronize()

    seqlen_sum = float(sum(seqlens_gathered))
    seqlen_sq_sum = float(sum(s**2 for s in seqlens_gathered))

    return new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum


def _compute_smart_padding_dyn_cp_params(
    gbs_max_seqlen: int,
    max_seqlen_per_dp_cp_rank: int,
    config_cp_size: int,
) -> Tuple[int, int]:
    """Determine ``local_cp_size`` and ``num_packed`` for smart-padding dynamic CP.

    Returns
    -------
    local_cp_size : int
        Power-of-2 CP group size for this GBS.
    num_packed : int
        Number of samples to pack per microbatch.
    """
    if gbs_max_seqlen <= max_seqlen_per_dp_cp_rank:
        local_cp_size = 1
        num_packed = max_seqlen_per_dp_cp_rank // gbs_max_seqlen
    else:
        local_cp_size = 1
        while local_cp_size * max_seqlen_per_dp_cp_rank < gbs_max_seqlen:
            local_cp_size *= 2
        assert local_cp_size <= config_cp_size, (
            f"gbs_max_seqlen ({gbs_max_seqlen}) requires cp_size={local_cp_size} "
            f"which exceeds context_parallel_size={config_cp_size}. "
            f"Increase context_parallel_size or reduce max sequence length."
        )
        num_packed = (local_cp_size * max_seqlen_per_dp_cp_rank) // gbs_max_seqlen
    return local_cp_size, max(num_packed, 1)


def dyn_cp_schedule_smart_padding(
    gbs_batches: List[Dict[str, Any]],
    dp_group,
    config_cp_size: int,
    dist_config,
    dev,
    packed_keys: list[str],
    cat_keys: list[str],
) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float]:
    """Smart-padding-aware dynamic CP: local scheduling, no all-to-all.

    Exploits the fact that smart padding produces similar-length samples
    within a GBS.  Each rank independently determines cp_size / num_packed
    and selects its own sample subset.  The only communication is a single
    scalar all-reduce(MAX) across the DP group.
    """
    max_seqlen_per_dp_cp_rank = dist_config.max_seqlen_per_dp_cp_rank

    local_seqlens = [s["tokens"].shape[-1] for s in gbs_batches]
    local_stats = torch.tensor(
        [max(local_seqlens),
         sum(local_seqlens),
         sum(s**2 for s in local_seqlens)],
        dtype=torch.int64,
        device=dev,
    )
    stats_gathered = [torch.empty_like(local_stats) for _ in range(dp_group.size())]
    dist.all_gather(stats_gathered, local_stats, group=dp_group)
    all_stats = torch.stack(stats_gathered, dim=0)
    gbs_max_seqlen = int(all_stats[:, 0].max().item())
    seqlen_sum = float(all_stats[:, 1].sum().item())
    seqlen_sq_sum = float(all_stats[:, 2].sum().item())

    local_cp_size, num_packed = _compute_smart_padding_dyn_cp_params(
        gbs_max_seqlen,
        max_seqlen_per_dp_cp_rank,
        config_cp_size,
    )

    num_cp_groups = config_cp_size // local_cp_size
    cp_rank = parallel_state.get_context_parallel_rank()
    my_group_idx = cp_rank // local_cp_size

    N = len(gbs_batches)
    assert N % num_cp_groups == 0
    M = N // num_cp_groups
    my_samples = gbs_batches[my_group_idx * M:(my_group_idx + 1) * M]

    # Keep samples on CPU; they will be moved to GPU lazily per-microbatch
    # during the forward pass to reduce peak GPU memory when GBS is large.
    cpu_dev = torch.device('cpu')

    num_packed = min(num_packed, M)
    num_microbatches = (M + num_packed - 1) // num_packed
    base_size = M // num_microbatches
    remainder = M % num_microbatches
    local_cp_size_t = torch.tensor(local_cp_size, dtype=torch.int32)

    new_samples: List[Dict[str, torch.Tensor]] = []
    offset = 0
    for mb_idx in range(num_microbatches):
        size = base_size + (1 if mb_idx < remainder else 0)
        mb_samples = my_samples[offset:offset + size]
        offset += size

        padded_lens = torch.tensor(
            [s["padded_seq_len"].item() for s in mb_samples],
            dtype=torch.int32,
        )
        original_lens = torch.tensor(
            [s["original_seq_len"].item() for s in mb_samples],
            dtype=torch.int32,
        )
        packed = _pack_sequences_by_keys(
            mb_samples,
            padded_lens,
            original_lens,
            local_cp_size_t,
            cpu_dev,
            packed_keys,
            cat_keys,
        )
        new_samples.append(packed)
    assert len(new_samples) == num_microbatches
    assert len(my_samples) == offset

    logging_rank0(
        f"[SmartPadDynCP] gbs_max_seqlen={gbs_max_seqlen} "
        f"local_cp_size={local_cp_size} num_packed={num_packed} "
        f"num_microbatches={num_microbatches} "
    )

    return new_samples, num_microbatches, seqlen_sum, seqlen_sq_sum
