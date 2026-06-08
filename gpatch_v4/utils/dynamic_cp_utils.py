"""Dynamic Context Parallel (Dynamic CP) utilities for gcore V4 training.

Generic scheduling + batch-prep infrastructure shared by all training types
(SFT, GRPO, DPO, distill, ...). Each training type supplies:

1. A *converter* turning raw per-sample dicts into the Dynamic CP standard
   format (tokens, labels, loss_mask, position_ids + any extra fields).
2. A *forward_step* closure computing the task-specific loss.

Note: the abbreviation "DCP" in this codebase elsewhere refers to PyTorch
``torch.distributed.checkpoint``; this module always uses ``dyn_cp``.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.data_schedule_utils import (
    _get_global_seqlens_and_ids,
    broadcast_scalars,
    broadcast_tensor,
    broadcast_to_pp_group,
    build_packed_microbatches,
    reroute_samples_to_dcp_ranks,
)
from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.rerun_state_machine import RerunDataIterator

from gpatch_v4.utils import logging_rank0

_SFT_KEYS = frozenset({"tokens", "labels", "loss_mask", "position_ids"})
_META_KEYS = frozenset(
    {
        "original_seq_len",
        "padded_seq_len",
        "cu_seqlens",
        "cu_seqlens_padded",
        "max_seqlen",
        "local_cp_size",
    }
)

RL_TOKEN_KEYS: Tuple[str, ...] = (
    "advantages",
    "prev_log_probs",
    "ref_log_probs",
    "rollout_log_probs",
    "sample_mask",
)

# ---------------------------------------------------------------------------
#  Padding helpers
# ---------------------------------------------------------------------------


def _get_total_pad_divisor() -> int:
    """Return the per-sample padding divisor ``(2 * dp * cp) * tp_size``.

    ``2 * dp * cp`` keeps THD CP partitioning legal for any ``local_cp_size``
    in ``[1, dp * cp]``. We always multiply by ``tp_size`` so the same divisor
    works whether or not ``sequence_parallel`` is on; the cost is at most one
    extra padded chunk per sample.
    """
    dp_cp = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    cp_pad = 2 * dp_cp if dp_cp > 1 else 1
    tp_pad = tp_size if tp_size > 1 else 1
    return cp_pad * tp_pad


def _round_up(n: int, divisor: int) -> int:
    if divisor <= 1:
        return n
    return ((n + divisor - 1) // divisor) * divisor


def _pad_1d(t: torch.Tensor, target_len: int, value: float = 0) -> torch.Tensor:
    """Right-pad (or truncate) a 1-D tensor to ``target_len``."""
    cur = t.shape[0]
    if cur >= target_len:
        return t[:target_len]
    return torch.nn.functional.pad(t, (0, target_len - cur), value=value)


def convert_rl_samples_to_dyn_cp_format(
    samples: List[Dict[str, Any]],
) -> List[Dict[str, torch.Tensor]]:
    """Convert per-sample RL dicts (GRPO / PPO) into the Dynamic CP standard format.

    这里预先做next-token shift，因为seq-pack之后再做roll(-1)会导致sub-seq边界的label错误
    """
    converted: List[Dict[str, torch.Tensor]] = []
    pad_div = _get_total_pad_divisor()
    for s in samples:
        tokens = s["tokens"]
        assert tokens.dim() == 1, f"RL tokens must be 1-D, got shape {tuple(tokens.shape)}"
        seq_len = tokens.shape[0]
        assert seq_len >= 2, (
            f"RL sample seq_len must be >= 2 (per-token fields would be empty otherwise), "
            f"got seq_len={seq_len}"
        )
        dev = tokens.device
        d: Dict[str, torch.Tensor] = {}

        shifted_tokens = tokens[:-1]
        shifted_labels = tokens[1:]
        actual_len = shifted_tokens.shape[0]
        padded_len = _round_up(actual_len, pad_div)
        assert padded_len % pad_div == 0, (
            f"padded_len={padded_len} not aligned to divisor={pad_div}"
        )

        d["tokens"] = _pad_1d(shifted_tokens, padded_len, 0)
        d["labels"] = _pad_1d(shifted_labels, padded_len, 0)
        d["position_ids"] = torch.arange(padded_len, dtype=torch.int64, device=dev)

        # RL per-token fields are already seq_len - 1 long, matching the shift.
        d["loss_mask"] = _pad_1d(s["mask"], padded_len, 0).to(torch.float32)
        d["advantages"] = _pad_1d(s["advantages"], padded_len, 0).to(torch.float32)
        d["prev_log_probs"] = _pad_1d(s["logprobs"], padded_len, 0).to(torch.float32)
        d["ref_log_probs"] = _pad_1d(s["ref_logprobs"], padded_len, 0).to(torch.float32)

        if "rollout_log_probs" in s and s["rollout_log_probs"] is not None:
            d["rollout_log_probs"] = _pad_1d(s["rollout_log_probs"], padded_len,
                                             0).to(torch.float32)

        if "sample_mask" in s and s["sample_mask"] is not None:
            sm = s["sample_mask"]
            if sm.dim() == 0:
                sm = sm.unsqueeze(0)
            sm_full = sm.expand(actual_len).to(torch.float32).contiguous()
            d["sample_mask"] = _pad_1d(sm_full, padded_len, 0)

        d["original_seq_len"] = torch.tensor([actual_len], dtype=torch.int32)
        d["padded_seq_len"] = torch.tensor([padded_len], dtype=torch.int32)

        converted.append(d)
    return converted


# ---------------------------------------------------------------------------
#  Generic Dynamic CP scheduling pipeline
# ---------------------------------------------------------------------------


def _augment_extra_fields(
    new_samples: List[Dict[str, torch.Tensor]],
    samples_with_id: Dict[int, Dict[str, torch.Tensor]],
    sample_id_groups: List[List[List[int]]],
    dyn_cp_rank: int,
) -> None:
    """Pack any extra per-token fields beyond SFT keys into packed microbatches."""
    if not samples_with_id:
        return
    first_sample = next(iter(samples_with_id.values()))
    extra_keys = [k for k in first_sample.keys() if k not in _SFT_KEYS and k not in _META_KEYS]
    if not extra_keys:
        return
    for i, packed in enumerate(new_samples):
        sample_ids = sample_id_groups[i][dyn_cp_rank]
        for key in extra_keys:
            tensors = [samples_with_id[sid][key].reshape(-1) for sid in sample_ids]
            packed[key] = torch.cat(tensors, dim=0) if tensors else torch.empty(0)


def run_dyn_cp_schedule(
    samples: List[Dict[str, torch.Tensor]],
    num_microbatches: int,
    max_seqlen_per_dp_cp_rank: int,
    min_cp_size: int = 1,
) -> Tuple[Any, int, float, float]:
    """Run the full Dynamic CP scheduling pipeline.

    Extra per-token fields beyond the SFT four-tuple are auto-discovered
    and packed.

    Returns
    -------
    data_iterator : RerunDataIterator or None
        Packed-microbatch iterator on TP rank 0, else None.
    num_micro_batches : int
    seqlen_sum, seqlen_sq_sum : float
        Sum and squared-sum of original sequence lengths over the global
        batch, broadcast to every PP+TP rank; used for FLOPs accounting.
    """
    dp_group = parallel_state.get_data_parallel_group()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

    dev = torch.cuda.current_device()
    dp_size = dp_group.size()
    cp_size = dp_cp_group.size() // dp_size
    total_dyn_cp_gpus = dp_cp_group.size()

    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=max_seqlen_per_dp_cp_rank,
        cp_size=cp_size,
        dp_size=dp_size,
        microbatch_group_size_per_vp_stage=None,
        min_cp_size=min_cp_size,
    )

    if tp_group.rank() == 0 and (pp_group.rank() == 0 or pp_group.rank() == pp_group.size() - 1):
        batch = samples
        subsample_seqlens = torch.cat([s["padded_seq_len"]
                                       for s in batch]).to(dtype=torch.int32, device=dev)

        global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered = (
            _get_global_seqlens_and_ids(subsample_seqlens, dp_group)
        )

        sample_id_groups = scheduler.get_groups_and_subsamples(global_id_seqlens)

        samples_this_rank_with_id = reroute_samples_to_dcp_ranks(
            batch,
            global_ids_this_rank,
            global_id_seqlens,
            sample_id_groups,
            offsets,
            dp_group,
            tp_group,
            dp_cp_group,
            total_dyn_cp_gpus,
        )

        dyn_cp_rank = dp_cp_group.rank()
        num_micro_batches = len(sample_id_groups)

        new_samples = build_packed_microbatches(
            samples_this_rank_with_id, sample_id_groups, dyn_cp_rank, dev, scheduler.is_dynamic_cp
        )

        _augment_extra_fields(new_samples, samples_this_rank_with_id, sample_id_groups, dyn_cp_rank)

        seqlen_sum = float(sum(seqlens_gathered))
        seqlen_sq_sum = float(sum(s**2 for s in seqlens_gathered))
    else:
        new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum = None, None, None, None

    if tp_group.rank() == 0:
        new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum = broadcast_to_pp_group(
            new_samples,
            num_micro_batches,
            seqlen_sum,
            seqlen_sq_sum,
            pp_group,
            dev,
            is_dynamic_cp=scheduler.is_dynamic_cp,
        )

    num_micro_batches, seqlen_sum, seqlen_sq_sum = broadcast_scalars(
        [num_micro_batches, seqlen_sum, seqlen_sq_sum],
        tp_group,
        dev,
    )
    num_micro_batches = int(num_micro_batches)

    if tp_group.rank() == 0:
        new_data_iterator = RerunDataIterator(iter(new_samples))
    else:
        new_data_iterator = None

    return new_data_iterator, num_micro_batches, float(seqlen_sum), float(seqlen_sq_sum)


# ---------------------------------------------------------------------------
#  Generic per-microbatch batch fetcher
# ---------------------------------------------------------------------------


def get_batch_for_dyn_cp(
    data_iterator,
    dynamic_cp: bool = True,
    extra_token_keys: Sequence[str] = (),
) -> Tuple[Dict[str, torch.Tensor], PackedSeqParams]:
    """Fetch one packed microbatch, apply CP slicing / TP alignment / TP broadcast.

    Parameters
    ----------
    extra_token_keys
        Additional per-token float32 fields beyond the SFT four-tuple
        (e.g. RL advantages, logprobs). Pass ``()`` for SFT.
    """
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    tp_src_rank = torch.distributed.get_process_group_ranks(tp_group)[0]

    is_tp_rank_0 = tp_group.rank() == 0
    is_first_stage = pp_group.rank() == 0
    is_last_stage = pp_group.rank() == pp_group.size() - 1
    dev = torch.cuda.current_device()
    tp_size = tp_group.size()

    # -- 1. 在 TP rank 0 取一条 packed batch --
    if is_tp_rank_0:
        assert data_iterator is not None, "TP rank 0 must have a data_iterator"
        batch = next(data_iterator)

        required_meta = ["cu_seqlens", "cu_seqlens_padded", "max_seqlen"]
        if is_first_stage or is_last_stage:
            required_meta += ["tokens", "position_ids"]
        if is_last_stage:
            required_meta += ["labels", "loss_mask"]
        for k in required_meta:
            assert k in batch, f"required key '{k}' missing from packed batch"
        if dynamic_cp:
            assert "local_cp_size" in batch, "dynamic_cp=True but 'local_cp_size' missing"
    else:
        assert data_iterator is None, "Non TP-rank-0 must not own a data_iterator"
        batch = {}

    # -- 2. 算出本 microbatch 的 CP 子组 --
    if dynamic_cp and is_tp_rank_0:
        lcp = batch.get("local_cp_size")
        if lcp is not None:
            lcp_val = lcp.item() if isinstance(lcp, torch.Tensor) else int(lcp)
            cp_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=lcp_val)
        else:
            cp_group = parallel_state.get_context_parallel_group()
    else:
        cp_group = parallel_state.get_context_parallel_group()

    # -- 3. 在 TP=0 上做 CP 切分（THD 负载均衡） --
    if is_tp_rank_0 and (is_first_stage or is_last_stage):
        cp_size = cp_group.size()
        if cp_size > 1:
            cp_rank = cp_group.rank()
            total_tokens = batch["tokens"].size(0)
            # Pass cu_seqlens_padded as cu_seqlens to work around a TE bug
            # in thd_get_partitioned_indices.
            cu_seqlens_for_partition = batch["cu_seqlens_padded"]
            index = get_thd_partitioned_indices(
                cu_seqlens_for_partition, total_tokens, cp_size, cp_rank
            )
            per_token_keys = list(_SFT_KEYS)
            for k in extra_token_keys:
                if k in batch:
                    per_token_keys.append(k)
            for key in per_token_keys:
                if key in batch:
                    batch[key] = batch[key].index_select(0, index)

        if tp_size > 1:
            post_cp_len = batch["tokens"].size(0)
            assert post_cp_len % tp_size == 0, (
                f"post-CP tokens ({post_cp_len}) not aligned to tp_size={tp_size}"
            )

    # -- 4. Broadcast metadata sizes --
    if is_tp_rank_0:
        cu_seqlen_size = torch.tensor(batch["cu_seqlens"].size(0), dtype=torch.int32, device=dev)
    else:
        cu_seqlen_size = torch.empty(1, dtype=torch.int32, device=dev)
    broadcast_tensor(cu_seqlen_size, tp_src_rank, tp_group)
    cu_seqlen_size = cu_seqlen_size.item()

    total_tokens_val = None
    if is_first_stage or is_last_stage:
        if is_tp_rank_0:
            total_tokens_val = torch.tensor(batch["tokens"].size(0), dtype=torch.int32, device=dev)
        else:
            total_tokens_val = torch.empty(1, dtype=torch.int32, device=dev)
        broadcast_tensor(total_tokens_val, tp_src_rank, tp_group)
        total_tokens_val = total_tokens_val.item()

    # -- 5. Prepare fields on all TP ranks --
    if is_first_stage:
        if is_tp_rank_0:
            assert batch["tokens"].dtype == torch.int64, (
                f"tokens dtype must be int64, got {batch['tokens'].dtype}"
            )
            assert batch["position_ids"].dtype == torch.int64, (
                f"position_ids dtype must be int64, got {batch['position_ids'].dtype}"
            )
            batch["tokens"] = batch["tokens"].view(1, total_tokens_val).contiguous()
            batch["position_ids"] = batch["position_ids"].view(1, total_tokens_val).contiguous()
        else:
            batch["tokens"] = torch.empty([1, total_tokens_val], dtype=torch.int64, device=dev)
            batch["position_ids"] = torch.empty(
                [1, total_tokens_val], dtype=torch.int64, device=dev
            )
    else:
        batch["tokens"] = None
        batch["position_ids"] = None

    # Discover which extra keys are present via bitmask.
    if is_last_stage and extra_token_keys:
        if is_tp_rank_0:
            extra_flags = torch.tensor(
                [1 if k in batch and batch[k] is not None else 0 for k in extra_token_keys],
                dtype=torch.int32,
                device=dev,
            )
        else:
            extra_flags = torch.empty(len(extra_token_keys), dtype=torch.int32, device=dev)
        broadcast_tensor(extra_flags, tp_src_rank, tp_group)
        # NOTE: must keep deterministic order across TP ranks. A set
        # comprehension would iterate in PYTHONHASHSEED-dependent order.
        # Use a tuple ordered by ``extra_token_keys``.
        extra_present = tuple(k for k, flag in zip(extra_token_keys, extra_flags.tolist()) if flag)
    else:
        extra_present = ()

    if is_last_stage:
        if is_tp_rank_0:
            assert batch["labels"].dtype == torch.int64, (
                f"labels dtype must be int64, got {batch['labels'].dtype}"
            )
            assert batch["loss_mask"].dtype == torch.float32, (
                f"loss_mask dtype must be float32, got {batch['loss_mask'].dtype}"
            )
            batch["labels"] = batch["labels"].view(1, total_tokens_val).contiguous()
            batch["loss_mask"] = batch["loss_mask"].view(1, total_tokens_val).contiguous()
            for k in extra_present:
                batch[k] = batch[k].view(1, total_tokens_val).contiguous()
        else:
            batch["labels"] = torch.empty([1, total_tokens_val], dtype=torch.int64, device=dev)
            batch["loss_mask"] = torch.empty([1, total_tokens_val], dtype=torch.float32, device=dev)
            for k in extra_present:
                batch[k] = torch.empty([1, total_tokens_val], dtype=torch.float32, device=dev)
    else:
        batch["labels"] = None
        batch["loss_mask"] = None
        for k in extra_token_keys:
            batch[k] = None

    # cu_seqlens / max_seqlen / local_cp_size
    if not is_tp_rank_0:
        batch["cu_seqlens"] = torch.empty([cu_seqlen_size], dtype=torch.int32, device=dev)
        batch["cu_seqlens_padded"] = torch.empty([cu_seqlen_size], dtype=torch.int32, device=dev)
        batch["max_seqlen"] = torch.empty(1, dtype=torch.int32, device=dev)
    else:
        assert batch["cu_seqlens"].dtype == torch.int32, (
            f"cu_seqlens dtype must be int32, got {batch['cu_seqlens'].dtype}"
        )
        assert batch["cu_seqlens_padded"].dtype == torch.int32, (
            f"cu_seqlens_padded dtype must be int32, got {batch['cu_seqlens_padded'].dtype}"
        )
        assert batch["cu_seqlens"].dim(
        ) == 1, (f"cu_seqlens must be 1-D, got {batch['cu_seqlens'].dim()}")
        assert batch["cu_seqlens_padded"].dim(
        ) == 1, (f"cu_seqlens_padded must be 1-D, got {batch['cu_seqlens_padded'].dim()}")
        if isinstance(batch["max_seqlen"], int):
            batch["max_seqlen"] = torch.tensor(batch["max_seqlen"], dtype=torch.int32, device=dev)
        else:
            assert batch["max_seqlen"].dtype == torch.int32, (
                f"max_seqlen dtype must be int32, got {batch['max_seqlen'].dtype}"
            )
            assert batch["max_seqlen"].numel(
            ) == 1, (f"max_seqlen must be scalar, got numel={batch['max_seqlen'].numel()}")

    if dynamic_cp:
        if not is_tp_rank_0:
            batch["local_cp_size"] = torch.empty(1, dtype=torch.int32, device=dev)
        else:
            lcp = batch.get("local_cp_size")
            if lcp is not None and isinstance(lcp, int):
                batch["local_cp_size"] = torch.tensor(lcp, dtype=torch.int32, device=dev)
            else:
                assert isinstance(lcp, torch.Tensor
                                 ), (f"local_cp_size must be int or torch.Tensor, got {type(lcp)}")
                assert lcp.dtype == torch.int32, (
                    f"local_cp_size dtype must be int32, got {lcp.dtype}"
                )
                assert lcp.numel() == 1, (f"local_cp_size must be scalar, got numel={lcp.numel()}")

    # -- 6. TP broadcast --
    broadcast_tensor(batch.get("tokens"), tp_src_rank, tp_group)
    broadcast_tensor(batch.get("position_ids"), tp_src_rank, tp_group)
    broadcast_tensor(batch.get("labels"), tp_src_rank, tp_group)
    broadcast_tensor(batch.get("loss_mask"), tp_src_rank, tp_group)
    broadcast_tensor(batch["cu_seqlens"], tp_src_rank, tp_group)
    broadcast_tensor(batch["cu_seqlens_padded"], tp_src_rank, tp_group)
    broadcast_tensor(batch["max_seqlen"], tp_src_rank, tp_group)
    if dynamic_cp:
        broadcast_tensor(batch.get("local_cp_size"), tp_src_rank, tp_group)

    if is_last_stage:
        for k in extra_present:
            broadcast_tensor(batch.get(k), tp_src_rank, tp_group)

    # -- 7. Resolve CP group on non-TP-rank-0 --
    if dynamic_cp and not is_tp_rank_0:
        lcp = batch.get("local_cp_size")
        if lcp is not None:
            cp_group = parallel_state.get_dynamic_data_context_parallel_groups(
                group_size=lcp.item()
            )

    # -- 8. Build PackedSeqParams --
    cu_seqlens_padded = batch["cu_seqlens_padded"]
    max_seqlen = batch["max_seqlen"].item()

    local_cp_size = None
    if dynamic_cp and batch.get("local_cp_size") is not None:
        local_cp_size = batch["local_cp_size"].item()

    # All four cu_seqlens* fields use cu_seqlens_padded as a workaround for
    # a TE bug; revert once TE is fixed.
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        local_cp_size=local_cp_size,
        cp_group=cp_group,
    )

    return batch, packed_seq_params


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


def sft_dyn_cp_schedule_default(
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


def sft_dyn_cp_schedule_smart_padding(
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

    for s in my_samples:
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                s[k] = v.to(dev, non_blocking=True)

    num_packed = min(num_packed, M)
    num_microbatches = (M + num_packed - 1) // num_packed
    base_size = M // num_microbatches
    remainder = M % num_microbatches
    local_cp_size_t = torch.tensor(local_cp_size, dtype=torch.int32, device=dev)

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
            dev,
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
