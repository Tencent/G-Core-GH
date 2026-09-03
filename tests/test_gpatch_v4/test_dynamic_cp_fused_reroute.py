"""Fused dyn-CP reroute: real multi-process NCCL correctness + efficiency.

Uses ``torch.multiprocessing.spawn`` (2 processes / 2 GPUs) so
``reroute_samples_to_dcp_ranks_by_keys`` hits genuine ``nccl:all_to_all``.
"""
from __future__ import annotations

import os
import socket
import time
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpatch_v4.utils.dynamic_cp_utils import (
    group_keys_by_dtype_and_len_sig,
    reroute_samples_to_dcp_ranks_by_keys,
)

WORLD_SIZE = 2
TIMING_ITERS = 20
WARMUP_ITERS = 3
# Long-tensor payload per sample base length (scaled by gid); keep big enough
# that NCCL launch count dominates over Python pack cost.
BASE_LEN = 4096


class _TpSizeOne:
    """TP is unused for a2a; only ``size()==1`` is required by rank mapping."""

    def size(self) -> int:
        return 1


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _sample(gid: int, device: torch.device) -> dict:
    n = BASE_LEN + gid * 128
    base = gid * 1_000_000
    long_int = ["tokens", "labels", "position_ids"]
    long_fp = ["loss_mask", "advantages", "prev_log_probs"]
    out = {}
    for key in long_int:
        out[key] = torch.arange(base, base + n, device=device, dtype=torch.int64)
    for key in long_fp:
        out[key] = (
            torch.arange(base, base + n, device=device, dtype=torch.float32) + 0.25 * gid
        )
    out["original_seq_len"] = torch.tensor([n - 1 + gid], device=device, dtype=torch.int32)
    out["padded_seq_len"] = torch.tensor([n + gid], device=device, dtype=torch.int32)
    return out


def _world_fixture(rank: int, device: torch.device):
    lens = {gid: BASE_LEN + gid * 128 for gid in range(4)}
    batches_by_rank = [
        [_sample(0, device), _sample(1, device)],
        [_sample(2, device), _sample(3, device)],
    ]
    global_ids_by_rank = [[0, 1], [2, 3]]
    offsets = torch.tensor([0, 2, 4], dtype=torch.int32, device="cpu")
    # Cross-rank shuffle: rank0 <- {1,2}, rank1 <- {0,3}
    sample_id_groups = [[[1, 2], [0, 3]]]

    long_keys = [
        "tokens",
        "labels",
        "position_ids",
        "loss_mask",
        "advantages",
        "prev_log_probs",
    ]
    scalar_keys = ["original_seq_len", "padded_seq_len"]
    global_id_seqlens_dict = {
        **{k: [(gid, lens[gid]) for gid in range(4)] for k in long_keys},
        **{k: [(gid, 1) for gid in range(4)] for k in scalar_keys},
    }
    data_keys = sorted(batches_by_rank[0][0].keys())
    dtypes = {k: batches_by_rank[0][0][k].dtype for k in data_keys}
    key_groups = group_keys_by_dtype_and_len_sig(
        data_keys,
        lambda k: dtypes[k],
        lambda k: tuple(seqlen for _, seqlen in global_id_seqlens_dict[k]),
    )
    return {
        "batch": batches_by_rank[rank],
        "batches_by_rank": batches_by_rank,
        "global_ids": torch.tensor(global_ids_by_rank[rank], dtype=torch.int64),
        "global_ids_by_rank": global_ids_by_rank,
        "offsets": offsets,
        "sample_id_groups": sample_id_groups,
        "global_id_seqlens_dict": global_id_seqlens_dict,
        "data_keys": data_keys,
        "key_groups": key_groups,
    }


def _oracle_for_rank(
    batches_by_rank: List[List[dict]],
    global_ids_by_rank: List[List[int]],
    sample_id_groups: List[List[List[int]]],
    dest_rank: int,
) -> Dict[int, dict]:
    gid_to_sample = {}
    for batch, gids in zip(batches_by_rank, global_ids_by_rank):
        for local_i, gid in enumerate(gids):
            gid_to_sample[gid] = {
                k: v.detach().cpu().clone() for k, v in batch[local_i].items()
            }
    combined: List[List[int]] = [[] for _ in range(len(batches_by_rank))]
    for mb in sample_id_groups:
        for d, gids in enumerate(mb):
            combined[d].extend(gids)
    for d in range(len(combined)):
        combined[d].sort()
    return {gid: gid_to_sample[gid] for gid in combined[dest_rank]}


def _reroute_per_key(
    batch: list[dict[str, torch.Tensor]],
    global_ids_this_rank: torch.Tensor,
    global_id_seqlens_dict: dict[str, list[tuple[int, int]]],
    sample_id_groups: list[list[list[int]]],
    offsets: torch.Tensor,
    dp_group,
    tp_group,
    dp_cp_group,
    total_dcp_gpus: int,
) -> Dict[int, dict]:
    """Pre-fuse baseline: one ``all_to_all_single`` per key (real NCCL)."""

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
    dp_rank_set = set(dp_ranks)
    data_keys = sorted(batch[0].keys())

    combined_sample_id_groups: List[List[int]] = [[] for _ in range(total_dcp_gpus)]
    for d in range(total_dcp_gpus):
        for sample_id_group in sample_id_groups:
            combined_sample_id_groups[d].extend(sample_id_group[d])
    for dest_rank in range(total_dcp_gpus):
        combined_sample_id_groups[dest_rank].sort()

    send_ids_sorted = [
        gid for d in dp_ranks for gid in combined_sample_id_groups[d] if gid in gid2local_id
    ]
    recv_sample_id_groups = [[] for _ in range(total_dcp_gpus)]
    for gid in combined_sample_id_groups[dcp_rank]:
        recv_sample_id_groups[_gid_to_src_rank(gid)].append(gid)
    recv_ids_sorted = [gid for d in range(total_dcp_gpus) for gid in recv_sample_id_groups[d]]
    recv_samples = [{k: None for k in data_keys} for _ in range(len(recv_ids_sorted))]

    device = torch.cuda.current_device()
    for key in data_keys:
        global_id_seqlens = global_id_seqlens_dict[key]
        dtype = batch[0][key].dtype
        input_split_sizes = [0] * total_dcp_gpus
        for dest_rank in range(total_dcp_gpus):
            if dest_rank not in dp_rank_set:
                continue
            input_split_sizes[dest_rank] = sum(
                global_id_seqlens[gid][1]
                for gid in combined_sample_id_groups[dest_rank]
                if gid in gid2local_id
            )
        output_split_sizes = [
            sum(global_id_seqlens[gid][1] for gid in recv_sample_id_groups[src_rank])
            for src_rank in range(total_dcp_gpus)
        ]
        flats = []
        for gid in send_ids_sorted:
            flats.append(
                batch[gid2local_id[gid]][key].reshape(-1).to(
                    device=device, dtype=dtype, non_blocking=True
                )
            )
        send_tensor = (
            torch.cat(flats, dim=0) if flats else torch.empty(0, device=device, dtype=dtype)
        )
        recv_tensor = torch.empty(sum(output_split_sizes), device=device, dtype=dtype)
        dist.all_to_all_single(
            output=recv_tensor,
            input=send_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=dp_cp_group,
        )
        cursor = 0
        for i, gid in enumerate(recv_ids_sorted):
            sample_len = global_id_seqlens[gid][1]
            recv_samples[i][key] = recv_tensor[cursor:cursor + sample_len]
            cursor += sample_len

    return {recv_id: recv_samples[i] for i, recv_id in enumerate(recv_ids_sorted)}


def _install_a2a_counter() -> Tuple[dict, Callable[..., Any]]:
    counter = {"n": 0}
    real = dist.all_to_all_single

    def counted(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    dist.all_to_all_single = counted  # type: ignore[assignment]
    return counter, real


def _worker(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    try:
        world = _world_fixture(rank, device)
        dp_cp_group = dist.group.WORLD
        dp_group = dist.group.WORLD
        tp_group = _TpSizeOne()

        counter, real_a2a = _install_a2a_counter()
        try:
            counter["n"] = 0
            fused = reroute_samples_to_dcp_ranks_by_keys(
                world["batch"],
                world["global_ids"],
                world["global_id_seqlens_dict"],
                world["sample_id_groups"],
                world["offsets"],
                dp_group,
                tp_group,
                dp_cp_group,
                world_size,
            )
            fused_calls = counter["n"]

            expected = _oracle_for_rank(
                world["batches_by_rank"],
                world["global_ids_by_rank"],
                world["sample_id_groups"],
                dest_rank=rank,
            )
            assert set(fused.keys()) == set(expected.keys())
            for gid, exp_sample in expected.items():
                for key, exp_t in exp_sample.items():
                    torch.testing.assert_close(fused[gid][key].cpu(), exp_t)

            counter["n"] = 0
            per_key = _reroute_per_key(
                world["batch"],
                world["global_ids"],
                world["global_id_seqlens_dict"],
                world["sample_id_groups"],
                world["offsets"],
                dp_group,
                tp_group,
                dp_cp_group,
                world_size,
            )
            per_key_calls = counter["n"]

            assert set(per_key.keys()) == set(fused.keys())
            for gid in fused:
                for key in fused[gid]:
                    torch.testing.assert_close(fused[gid][key], per_key[gid][key])

            n_keys = len(world["data_keys"])
            n_groups = len(world["key_groups"])
            assert n_groups < n_keys
            assert fused_calls == n_groups, (
                f"fused a2a calls={fused_calls}, expected groups={n_groups}"
            )
            assert per_key_calls == n_keys, (
                f"per-key a2a calls={per_key_calls}, expected keys={n_keys}"
            )

            # --- wall-clock: real NCCL, fused should beat per-key ---
            for _ in range(WARMUP_ITERS):
                reroute_samples_to_dcp_ranks_by_keys(
                    world["batch"],
                    world["global_ids"],
                    world["global_id_seqlens_dict"],
                    world["sample_id_groups"],
                    world["offsets"],
                    dp_group,
                    tp_group,
                    dp_cp_group,
                    world_size,
                )
                _reroute_per_key(
                    world["batch"],
                    world["global_ids"],
                    world["global_id_seqlens_dict"],
                    world["sample_id_groups"],
                    world["offsets"],
                    dp_group,
                    tp_group,
                    dp_cp_group,
                    world_size,
                )
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(TIMING_ITERS):
                reroute_samples_to_dcp_ranks_by_keys(
                    world["batch"],
                    world["global_ids"],
                    world["global_id_seqlens_dict"],
                    world["sample_id_groups"],
                    world["offsets"],
                    dp_group,
                    tp_group,
                    dp_cp_group,
                    world_size,
                )
            torch.cuda.synchronize()
            fused_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            for _ in range(TIMING_ITERS):
                _reroute_per_key(
                    world["batch"],
                    world["global_ids"],
                    world["global_id_seqlens_dict"],
                    world["sample_id_groups"],
                    world["offsets"],
                    dp_group,
                    tp_group,
                    dp_cp_group,
                    world_size,
                )
            torch.cuda.synchronize()
            per_key_s = time.perf_counter() - t0

            if rank == 0:
                print(
                    f"[fused_reroute_nccl] keys={n_keys} groups={n_groups} "
                    f"fused_calls={fused_calls} per_key_calls={per_key_calls} "
                    f"fused_s={fused_s:.4f} per_key_s={per_key_s:.4f} "
                    f"iters={TIMING_ITERS}",
                    flush=True,
                )
            assert fused_s < per_key_s, (
                f"fused NCCL wall {fused_s:.4f}s not faster than per-key {per_key_s:.4f}s "
                f"(iters={TIMING_ITERS})"
            )
        finally:
            dist.all_to_all_single = real_a2a  # type: ignore[assignment]
    finally:
        dist.destroy_process_group()


def test_group_keys_by_dtype_and_len_sig_merges_matching_keys():
    data_keys = [
        "advantages",
        "labels",
        "loss_mask",
        "original_seq_len",
        "padded_seq_len",
        "position_ids",
        "prev_log_probs",
        "tokens",
    ]
    dtypes = {
        "tokens": torch.int64,
        "labels": torch.int64,
        "position_ids": torch.int64,
        "loss_mask": torch.float32,
        "advantages": torch.float32,
        "prev_log_probs": torch.float32,
        "original_seq_len": torch.int32,
        "padded_seq_len": torch.int32,
    }
    long_sig = (4, 8, 2)
    scalar_sig = (1, 1, 1)
    len_sigs = {k: long_sig for k in data_keys}
    len_sigs["original_seq_len"] = scalar_sig
    len_sigs["padded_seq_len"] = scalar_sig

    groups = group_keys_by_dtype_and_len_sig(
        data_keys, lambda k: dtypes[k], lambda k: len_sigs[k]
    )
    assert groups == [
        ["advantages", "loss_mask", "prev_log_probs"],
        ["labels", "position_ids", "tokens"],
        ["original_seq_len", "padded_seq_len"],
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"need >= {WORLD_SIZE} CUDA devices for real NCCL spawn",
)
def test_fused_reroute_nccl_multiprocess_correctness_and_speed():
    """2-proc NCCL: fused == oracle == per-key; fewer a2a; lower wall time."""
    port = _find_free_port()
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, port),
        nprocs=WORLD_SIZE,
        join=True,
    )
