"""Dynamic Context Parallel (Dynamic CP) utilities for gcore V4 training.

Key-wise reroute / packing helpers shared by the per-model
``*_reroute_data_for_dynamic_cp`` implementations (SFT, GRPO, ...): given
per-sample tensors keyed by field name, they schedule samples across DPxCP
ranks and build packed THD microbatches.

Note: the abbreviation "DCP" in this codebase elsewhere refers to PyTorch
``torch.distributed.checkpoint``; this module always uses ``dyn_cp``.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.data_schedule_utils import _get_global_seqlens_and_ids

from gpatch_v4.utils import logging_rank0, logging_with_rank_and_datetime

# 2026.07.02 by guanyouhe
# Env var switch to enable the cross-rank ``all_to_all_single`` legality
# check below. Off by default since it costs one extra ``all_gather_object``
# per call; turn on when debugging a dyn_cp hang.
A2A_CHECK_ENV_VAR = "GPATCH_DYN_CP_CHECK_A2A"


def _a2a_check_enabled() -> bool:
    return os.environ.get(A2A_CHECK_ENV_VAR, "0") == "1"


def check_all_to_all_single_legal(
    name: str,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    input_split_sizes: List[int],
    output_split_sizes: List[int],
    group: dist.ProcessGroup,
) -> None:
    """Validate that an upcoming ``all_to_all_single`` call is globally legal.

    No-op unless ``GPATCH_DYN_CP_CHECK_A2A=1``, since it performs an extra
    ``all_gather_object`` across ``group`` to cross-check what every rank
    believes it is about to send/receive. This is meant to turn a silent
    NCCL hang into an immediate, actionable ``AssertionError``.

    Catches (at least) two known dyn_cp hang root causes:
      - Different ranks using a different dtype for the *same* collective
        (e.g. one rank packs real data with its true dtype while another
        rank, having nothing to send this round, packs an empty tensor
        with a merely-guessed dtype).
      - Ranks disagreeing on send/recv split sizes for a given (src, dst)
        pair -- e.g. two ranks both believe they are the "compute rank"
        for the same sample and both send, while the destination only
        budgeted to receive one sender's worth of data (or nobody sends
        what the receiver expects).

    Parameters
    ----------
    name
        Human-readable label for the call site, used in error messages.
    input_tensor, output_tensor
        The tensors this rank is about to pass as ``input``/``output`` to
        ``torch.distributed.all_to_all_single``.
    input_split_sizes, output_split_sizes
        This rank's per-destination send sizes / per-source recv sizes.
    group
        The process group the collective will run on.
    """
    if not _a2a_check_enabled():
        return

    world_size = group.size()
    my_rank = group.rank()

    local_info: Dict[str, Any] = {
        "rank": my_rank,
        "dtype": str(input_tensor.dtype),
        "input_numel": int(input_tensor.numel()),
        "output_numel": int(output_tensor.numel()),
        "input_split_sizes": [int(x) for x in input_split_sizes],
        "output_split_sizes": [int(x) for x in output_split_sizes],
    }

    gathered: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(gathered, local_info, group=group)

    errors: List[str] = []
    for info in gathered:
        r = info["rank"]
        if len(info["input_split_sizes"]) != world_size:
            errors.append(
                f"rank {r}: len(input_split_sizes)={len(info['input_split_sizes'])} "
                f"!= world_size={world_size}"
            )
        if len(info["output_split_sizes"]) != world_size:
            errors.append(
                f"rank {r}: len(output_split_sizes)={len(info['output_split_sizes'])} "
                f"!= world_size={world_size}"
            )
        if sum(info["input_split_sizes"]) != info["input_numel"]:
            errors.append(
                f"rank {r}: sum(input_split_sizes)={sum(info['input_split_sizes'])} "
                f"!= input_numel={info['input_numel']}"
            )
        if sum(info["output_split_sizes"]) != info["output_numel"]:
            errors.append(
                f"rank {r}: sum(output_split_sizes)={sum(info['output_split_sizes'])} "
                f"!= output_numel={info['output_numel']}"
            )

    dtypes = {info["dtype"] for info in gathered}
    if len(dtypes) > 1:
        errors.append(
            "dtype mismatch across ranks: "
            f"{ {info['rank']: info['dtype'] for info in gathered} }"
        )

    for src in range(world_size):
        src_info = gathered[src]
        if len(src_info["input_split_sizes"]) != world_size:
            continue  # already reported above
        for dst in range(world_size):
            dst_info = gathered[dst]
            if len(dst_info["output_split_sizes"]) != world_size:
                continue  # already reported above
            declared_send = src_info["input_split_sizes"][dst]
            declared_recv = dst_info["output_split_sizes"][src]
            if declared_send != declared_recv:
                errors.append(
                    f"send/recv split-size mismatch: rank {src} declares sending "
                    f"{declared_send} elem(s) to rank {dst}, but rank {dst} expects "
                    f"{declared_recv} elem(s) from rank {src}"
                )

    if errors:
        report = "\n  ".join(errors)
        logging_with_rank_and_datetime(
            f"[dyn_cp][all_to_all_single check FAILED] name={name!r}\n  {report}",
            rank=0,
        )
        raise AssertionError(
            f"[dyn_cp] all_to_all_single(name={name!r}) is not legal: "
            f"found {len(errors)} inconsistency(ies) across ranks "
            f"(full report logged on rank0). First few: {errors[:5]}"
        )


def check_gid_to_compute_rank_consistent(
    gid_to_compute_rank: Dict[int, int],
    cp_group: dist.ProcessGroup,
) -> None:
    """Cross-check that ``gid_to_compute_rank`` agrees across CP siblings.

    Every CP sibling of a DP index independently recomputes the *same*
    dyn_cp schedule from replicated rollout data (see ``docs/source/dynamic_cp.md``);
    ``reverse_reroute_logprobs`` relies on this to fan a single all-to-all
    out to every CP sibling correctly. If a future change ever breaks the
    "rollout data is replicated across CP siblings" invariant, the two
    siblings' independently-computed ``gid_to_compute_rank`` dicts would
    silently diverge -- this turns that into an immediate ``AssertionError``
    instead of a reverse-reroute hang. No-op unless ``GPATCH_DYN_CP_CHECK_A2A=1``.
    """
    if not _a2a_check_enabled():
        return
    if cp_group.size() <= 1:
        return

    gathered: List[Optional[Dict[int, int]]] = [None] * cp_group.size()
    dist.all_gather_object(gathered, gid_to_compute_rank, group=cp_group)

    my_rank = cp_group.rank()
    mine = gathered[my_rank]
    mismatched_ranks = [r for r, other in enumerate(gathered) if other != mine]
    if mismatched_ranks:
        logging_with_rank_and_datetime(
            f"[dyn_cp][gid_to_compute_rank check FAILED] this CP group's ranks "
            f"{mismatched_ranks} disagree with rank {my_rank} on gid_to_compute_rank "
            f"-- the 'rollout data replicated across CP siblings' invariant that "
            f"reverse_reroute_logprobs depends on has been broken.",
            rank=0,
        )
        raise AssertionError(
            f"[dyn_cp] gid_to_compute_rank is not identical across CP siblings: "
            f"rank {my_rank} disagrees with CP-group rank(s) {mismatched_ranks}."
        )


def _gather_int_metadata_by_key(
    batch: list[dict[str, torch.Tensor]],
    key: str,
    dp_group: torch.distributed.ProcessGroup,
    dev,
) -> List[Tuple[int, int]]:
    """Gather per-sample scalar metadata values across DP ranks.

    Unlike ``get_global_id_1d_len_by_keys``, this reads the stored *value*
    (e.g. ``original_seq_len``) rather than the 1-D tensor width used for
    all-to-all byte counts during reroute.
    """
    local_vals = torch.tensor(
        [int(s[key].reshape(-1)[0].item()) for s in batch],
        dtype=torch.int32,
        device=dev,
    )
    gathered = [torch.empty_like(local_vals) for _ in range(dp_group.size())]
    torch.distributed.all_gather(gathered, local_vals, group=dp_group)

    all_vals = torch.cat(gathered, dim=0).tolist()
    return [(i, int(v)) for i, v in enumerate(all_vals)]


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

    data_keys = sorted(batch[0].keys())  # sort to ensure consistent order across ranks

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
        check_all_to_all_single_legal(
            name=f"reroute_samples_to_dcp_ranks_by_keys[key={key}]",
            input_tensor=send_tensor,
            output_tensor=recv_tensor,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
            group=dp_cp_group,
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
    max_seqlen_per_dp_cp_rank: Optional[int] = None,
    need_routing_info: bool = True,
) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float, Optional[Dict[str, Any]]]:
    """Default dynamic CP: global sorting + all-to-all redistribution.

    Parameters
    ----------
    need_routing_info : bool
        If False, skip building the reverse-routing metadata and return
        ``None`` for ``routing_info``.  This saves one collective
        (``_gather_int_metadata_by_key``) and some CPU work when the caller
        only needs the packed microbatches (e.g. training forward-backward).

    Returns
    -------
    new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum, routing_info

    ``routing_info`` (when requested) contains scheduling metadata for reverse
    communication:
      - ``global_ids_this_rank``: Tensor of global IDs originally on this rank
      - ``offsets``: cumulative sample counts per rank
      - ``global_id_logprob_lens``: list[(gid, logprob_len)] for all samples
      - ``gid_to_compute_rank``: dict mapping gid → dcp_rank that computes it
      - ``gid_to_orig_dcp_rank``: dict mapping gid → list of dcp_ranks (all
        CP siblings of the originating DP index) that need the restored
        result
    """
    total_dyn_cp_gpus = dp_cp_group.size()
    assert max_seqlen_per_dp_cp_rank is not None
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=max_seqlen_per_dp_cp_rank,
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

    # Embed global sample IDs in each packed microbatch for reverse mapping.
    for i, sample in enumerate(new_samples):
        sample["_dyn_cp_sample_ids"] = torch.tensor(
            sample_id_groups[i][dyn_cp_rank], dtype=torch.int64, device=dev
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

    if not need_routing_info:
        return new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum, None

    # Build gid → compute_rank mapping from scheduling.
    gid_to_compute_rank: Dict[int, int] = {}
    for group in sample_id_groups:
        for rank_idx, gids in enumerate(group):
            for gid in gids:
                gid_to_compute_rank[gid] = rank_idx

    # Defensive (debug-only): confirm every CP sibling computed the exact
    # same schedule from the replicated rollout data. See docstring on
    # ``check_gid_to_compute_rank_consistent``.
    check_gid_to_compute_rank_consistent(
        gid_to_compute_rank, parallel_state.get_context_parallel_group()
    )

    # Map each global ID back to *every* dcp rank that needs the restored
    # result, i.e. all CP siblings of its originating DP index -- not just
    # the single rank this call's own (CP-slice-scoped) ``dp_group`` happens
    # to resolve.
    #
    # Rollout data is broadcast identically to every CP sibling of a DP
    # index (see ``is_mp_and_cp_head`` + ``broadcast_object_within_mp_and_cp``),
    # so every one of those ``cp_size`` physical ranks independently computes
    # this same routing with an *identical* ``gid -> compute_rank`` mapping
    # (a pure function of the replicated seqlen content) but a *different*,
    # self-referential ``dp_group`` (its own CP slice) when resolving "who is
    # the original owner". Resolving to a single rank here would silently
    # drop the reverse-reroute send/recv for every other CP sibling that
    # also needs the result -- exactly the all-to-all deadlock this function
    # was hardened against. Instead, expand to the full CP-sibling group so
    # every rank sends/receives a copy directly via the same all-to-all
    # (ranks are laid out CP-fastest-within-DP: dcp_rank = dp_idx*cp_size + cp_idx).
    dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
    orig_dcp_rank_by_dp = [(r // tp_group.size()) % total_dyn_cp_gpus for r in dp_ranks]
    total_samples = len(seqlens_gathered)
    gid_to_orig_dcp_rank: Dict[int, List[int]] = {}
    for gid in range(total_samples):
        dp_src_rank = int(torch.bucketize(torch.tensor(gid), offsets[1:] - 1))
        representative = orig_dcp_rank_by_dp[dp_src_rank]
        dp_idx = representative // cp_size
        gid_to_orig_dcp_rank[gid] = [dp_idx * cp_size + c for c in range(cp_size)]

    # Logprob length = shifted token count (stored value, not tensor width).
    global_id_logprob_lens = _gather_int_metadata_by_key(
        gbs_batches, "original_seq_len", dp_group, dev
    )

    routing_info = {
        "global_ids_this_rank": global_ids_this_rank.cpu(),
        "offsets": offsets.cpu(),
        "global_id_logprob_lens": global_id_logprob_lens,
        "gid_to_compute_rank": gid_to_compute_rank,
        "gid_to_orig_dcp_rank": gid_to_orig_dcp_rank,
    }

    return new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum, routing_info


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
    max_seqlen_per_dp_cp_rank: Optional[int] = None,
) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float]:
    """Smart-padding-aware dynamic CP: local scheduling, no all-to-all.

    Exploits the fact that smart padding produces similar-length samples
    within a GBS.  Each rank independently determines cp_size / num_packed
    and selects its own sample subset.  The only communication is a single
    scalar all-reduce(MAX) across the DP group.
    """
    assert max_seqlen_per_dp_cp_rank is not None

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


def reverse_reroute_logprobs(
    per_sample_logprobs: Dict[int, torch.Tensor],
    global_ids_this_rank: torch.Tensor,
    global_id_logprob_lens: List[Tuple[int, int]],
    gid_to_compute_rank: Dict[int, int],
    gid_to_orig_dcp_rank: Dict[int, List[int]],
    dp_cp_group: dist.ProcessGroup,
) -> Dict[int, torch.Tensor]:
    """Reverse the dynamic CP all-to-all with a single data exchange.

    All send/recv split sizes and per-sample lengths are pre-computed from
    globally-known scheduling info, so only ONE ``all_to_all_single`` call
    (for the actual log-prob data) is needed.

    Parameters
    ----------
    per_sample_logprobs : dict[int, Tensor]
        Global-ID → log-prob tensor (1D) computed on this rank.
    global_ids_this_rank : Tensor
        Global IDs that ORIGINALLY belonged to this rank (before reroute).
    global_id_logprob_lens : list[tuple[int, int]]
        ``(global_id, logprob_length)`` for ALL samples globally.
    gid_to_compute_rank : dict[int, int]
        Global-ID → the DCP rank that computed its log-probs (derived from
        ``sample_id_groups`` during scheduling).
    gid_to_orig_dcp_rank : dict[int, list[int]]
        Global-ID → *every* DCP rank that needs the restored result, i.e.
        all CP siblings of the sample's originating DP index. Rollout data
        is replicated identically across CP siblings (see
        ``is_mp_and_cp_head`` broadcast), so every one of them independently
        expects its own copy back -- not just a single "canonical" owner.
    dp_cp_group : ProcessGroup
        The DP×CP process group used for all-to-all.

    Returns
    -------
    dict[int, Tensor]
        Global-ID → log-prob tensor for all samples originally on this rank.
    """
    total_dcp_gpus = dp_cp_group.size()
    my_dcp_rank = dp_cp_group.rank()
    dev = torch.cuda.current_device()

    gid_to_len = {gid: length for gid, length in global_id_logprob_lens}

    # --- Sender: group computed logprobs by destination (original owner). ---
    # When local_cp_size > 1, every CP collaborator runs forward and holds the
    # same gids in per_sample_logprobs after CP all_reduce, but only the
    # designated compute rank (gid_to_compute_rank) may send each gid.
    # Each gid is duplicated to *every* CP sibling in gid_to_orig_dcp_rank[gid]
    # since all of them independently expect their own copy back. A gid whose
    # compute rank IS one of its own CP siblings (dest == my_dcp_rank) is kept
    # locally instead of being routed through the all-to-all -- it would only
    # ever be looped back to ourselves.
    local_results: Dict[int, torch.Tensor] = {}
    send_by_dest: List[List[Tuple[int, torch.Tensor]]] = [[] for _ in range(total_dcp_gpus)]
    for gid, lp in per_sample_logprobs.items():
        if gid_to_compute_rank[gid] != my_dcp_rank:
            continue
        assert lp.numel() == gid_to_len[gid], (
            f"logprob length mismatch for gid={gid}: got {lp.numel()}, "
            f"expected {gid_to_len[gid]}"
        )
        for dest in gid_to_orig_dcp_rank[gid]:
            if dest == my_dcp_rank:
                local_results[gid] = lp.to(device=dev, dtype=torch.float32)
            else:
                send_by_dest[dest].append((gid, lp))
    for dest_list in send_by_dest:
        dest_list.sort(key=lambda x: x[0])

    send_split_sizes = [sum(lp.numel() for _, lp in dest_list) for dest_list in send_by_dest]

    # --- Receiver: pre-compute recv sizes from global knowledge. ---
    # For each source rank S, the samples it sends to me are: my original gids
    # that were computed on rank S, sorted by gid (excluding gids resolved
    # locally above).
    my_gids = [int(g) for g in global_ids_this_rank.tolist()]
    recv_by_src: List[List[int]] = [[] for _ in range(total_dcp_gpus)]
    for gid in my_gids:
        if gid in local_results:
            continue
        src = gid_to_compute_rank[gid]
        recv_by_src[src].append(gid)
    for src_list in recv_by_src:
        src_list.sort()

    recv_split_sizes = [sum(gid_to_len[gid] for gid in src_list) for src_list in recv_by_src]

    # --- Single all-to-all for log-prob data. ---
    send_data = torch.cat(
        [
            lp.to(device=dev, dtype=torch.float32).reshape(-1) for dest_list in send_by_dest
            for _, lp in dest_list
        ]
    ) if sum(send_split_sizes) > 0 else torch.empty(0, device=dev, dtype=torch.float32)

    recv_data = torch.empty(sum(recv_split_sizes), device=dev, dtype=torch.float32)
    check_all_to_all_single_legal(
        name="reverse_reroute_logprobs",
        input_tensor=send_data,
        output_tensor=recv_data,
        input_split_sizes=send_split_sizes,
        output_split_sizes=recv_split_sizes,
        group=dp_cp_group,
    )
    dist.all_to_all_single(
        recv_data,
        send_data,
        output_split_sizes=recv_split_sizes,
        input_split_sizes=send_split_sizes,
        group=dp_cp_group,
    )

    # --- Unpack recv_data using known per-sample lengths and ordering. ---
    result: Dict[int, torch.Tensor] = dict(local_results)
    data_cursor = 0
    for src_rank in range(total_dcp_gpus):
        for gid in recv_by_src[src_rank]:
            length = gid_to_len[gid]
            result[gid] = recv_data[data_cursor:data_cursor + length]
            data_cursor += length

    return result
