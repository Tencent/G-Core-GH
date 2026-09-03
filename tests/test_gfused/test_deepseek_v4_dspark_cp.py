from __future__ import annotations

import dataclasses
import socket
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpatch_v4.models.deepseek_v4.dspark import (
    DSparkBatch,
    build_dspark_contiguous_cp_buffers,
    build_dspark_sparse_topk,
    prepare_dspark_batch,
    shard_dspark_batch_for_contiguous_cp,
)
from gpatch_v4.models.deepseek_v4.thd import (
    PackedSeqParams,
    cp_slice_layout,
    make_packed_seq_layout,
)
from gpatch_v4.training_backend.fsdp2_backend.dspark_loss import dspark_loss_denominator


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _halo_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )
    try:
        local = torch.arange(rank * 4, rank * 4 + 4).view(1, 4)
        target, positions, teacher, prefix_length = build_dspark_contiguous_cp_buffers(
            target_hidden_states=local.unsqueeze(-1).float(),
            target_last_hidden_states=local.unsqueeze(-1).float(),
            position_ids=local,
            sliding_window=2,
            block_size=3,
            cp_group=dist.group.WORLD,
        )
        if rank == 0:
            torch.testing.assert_close(target[:, :, 0], torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
            torch.testing.assert_close(positions, torch.tensor([[0, 1, 2, 3]]))
            torch.testing.assert_close(
                teacher[:, :, 0],
                torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]),
            )
            assert prefix_length == 0
        elif rank == 1:
            torch.testing.assert_close(
                target[:, :, 0],
                torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]),
            )
            torch.testing.assert_close(positions, torch.tensor([[2, 3, 4, 5, 6, 7]]))
            torch.testing.assert_close(
                teacher[:, :, 0],
                torch.tensor([[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]),
            )
            assert prefix_length == 2
        else:
            torch.testing.assert_close(
                target[:, :, 0],
                torch.tensor([[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]]),
            )
            torch.testing.assert_close(positions, torch.tensor([[6, 7, 8, 9, 10, 11]]))
            torch.testing.assert_close(
                teacher[:, :, 0],
                torch.tensor([[8.0, 9.0, 10.0, 11.0]]),
            )
            assert prefix_length == 2
    finally:
        dist.destroy_process_group()


def test_dspark_cp_owner_masks_partition_global_slots() -> None:
    eval_mask = torch.ones((1, 5, 3), dtype=torch.bool)
    eval_mask[:, -1] = False
    batch = DSparkBatch(
        anchor_positions=torch.tensor([[1, 2, 3, 14, 0]]),
        block_keep_mask=torch.tensor([[True, True, True, True, False]]),
        target_ids=torch.arange(15).reshape(1, 5, 3),
        eval_mask=eval_mask,
        prev_token_ids=torch.arange(15).reshape(1, 5, 3),
        target_hidden_indices=torch.arange(15).reshape(1, 5, 3),
    )
    shards = [
        shard_dspark_batch_for_contiguous_cp(
            batch,
            sequence_length=16,
            cp_rank=rank,
            cp_size=2,
        )
        for rank in range(2)
    ]

    torch.testing.assert_close(
        shards[0].block_keep_mask,
        torch.tensor([[True, True, True, False, False]]),
    )
    torch.testing.assert_close(
        shards[1].block_keep_mask,
        torch.tensor([[False, False, False, True, False]]),
    )
    owner_count = torch.stack([shard.block_keep_mask for shard in shards]).sum(dim=0)
    torch.testing.assert_close(owner_count, batch.block_keep_mask.long())
    eval_count = torch.stack([shard.eval_mask for shard in shards]).sum(dim=0)
    torch.testing.assert_close(eval_count, batch.eval_mask.long())
    global_denominator = dspark_loss_denominator(
        batch,
        block_size=3,
        loss_decay_gamma=1.0,
    )
    local_denominator = sum(
        dspark_loss_denominator(
            shard,
            block_size=3,
            loss_decay_gamma=1.0,
        )
        for shard in shards
    )
    torch.testing.assert_close(local_denominator, global_denominator)
    for shard in shards:
        torch.testing.assert_close(shard.anchor_positions, batch.anchor_positions)


def test_dspark_cp_sparse_topk_maps_global_context_to_left_halo() -> None:
    batch = DSparkBatch(
        anchor_positions=torch.tensor([[4]]),
        block_keep_mask=torch.tensor([[True]]),
        target_ids=torch.zeros((1, 1, 2), dtype=torch.long),
        eval_mask=torch.ones((1, 1, 2), dtype=torch.bool),
        prev_token_ids=torch.zeros((1, 1, 2), dtype=torch.long),
        target_hidden_indices=torch.tensor([[[4, 5]]]),
    )
    topk = build_dspark_sparse_topk(
        batch,
        seq_len=8,
        block_size=2,
        sliding_window=2,
        sequence_start=4,
        context_prefix_len=2,
        target_context_len=6,
    )

    expected = torch.tensor([[[0, 1, 6, 7], [0, 1, 6, 7]]], dtype=torch.int32)
    torch.testing.assert_close(topk, expected)


def test_dspark_cp_sparse_topk_keeps_safe_draft_kv_for_inactive_slots() -> None:
    batch = DSparkBatch(
        anchor_positions=torch.tensor([[1, 6]]),
        block_keep_mask=torch.tensor([[False, False]]),
        target_ids=torch.zeros((1, 2, 2), dtype=torch.long),
        eval_mask=torch.zeros((1, 2, 2), dtype=torch.bool),
        prev_token_ids=torch.zeros((1, 2, 2), dtype=torch.long),
        target_hidden_indices=torch.zeros((1, 2, 2), dtype=torch.long),
    )
    topk = build_dspark_sparse_topk(
        batch,
        seq_len=8,
        block_size=2,
        sliding_window=2,
        sequence_start=4,
        context_prefix_len=2,
        target_context_len=6,
    )

    expected = torch.tensor(
        [[
            [-1, -1, 6, 7],
            [-1, -1, 6, 7],
            [-1, -1, 8, 9],
            [-1, -1, 8, 9],
        ]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(topk, expected)


def test_dspark_cp_sparse_topk_masks_cross_segment_halo() -> None:
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int64)
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        max_seqlen_q=4,
        total_seqlen=8,
    )
    layout = make_packed_seq_layout(
        packed_seq_params,
        SimpleNamespace(compress_rates={}, sliding_window=2),
    )
    local_layout = cp_slice_layout(
        layout,
        cp_rank=1,
        cp_size=2,
        total_seqlen=8,
    )
    packed_seq_params = dataclasses.replace(packed_seq_params, layout=local_layout)
    batch = DSparkBatch(
        anchor_positions=torch.tensor([[4]]),
        block_keep_mask=torch.tensor([[True]]),
        target_ids=torch.zeros((1, 1, 2), dtype=torch.long),
        eval_mask=torch.ones((1, 1, 2), dtype=torch.bool),
        prev_token_ids=torch.zeros((1, 1, 2), dtype=torch.long),
        target_hidden_indices=torch.tensor([[[4, 5]]]),
    )

    topk = build_dspark_sparse_topk(
        batch,
        seq_len=8,
        block_size=2,
        sliding_window=2,
        packed_seq_params=packed_seq_params,
        sequence_start=4,
        context_prefix_len=2,
        target_context_len=6,
    )

    expected = torch.tensor([[[-1, -1, 6, 7], [-1, -1, 6, 7]]], dtype=torch.int32)
    torch.testing.assert_close(topk, expected)


def test_dspark_anchor_generator_is_reproducible() -> None:
    input_ids = torch.arange(12).view(1, -1)
    labels = input_ids + 1
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    first_generator = torch.Generator().manual_seed(1234)
    second_generator = torch.Generator().manual_seed(1234)

    first = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=4,
        block_size=3,
        rng=first_generator,
    )
    second = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=4,
        block_size=3,
        rng=second_generator,
    )

    torch.testing.assert_close(first.anchor_positions, second.anchor_positions)
    torch.testing.assert_close(first.block_keep_mask, second.block_keep_mask)


def test_dspark_contiguous_cp_halos() -> None:
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    mp.spawn(_halo_worker, args=(3, init_method), nprocs=3, join=True)
