#!/usr/bin/env python3
# coding=utf-8

from __future__ import annotations

import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from gpatch_v4.models.deepseek_v4.mtp import _roll_tensor_cp
from gpatch_v4.training_backend.fsdp2_backend.mtp_loss import (
    calculate_mtp_loss,
    mtp_per_depth_valid_count,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _roll_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )
    group = dist.group.WORLD

    bsz = 2
    s_full = 8
    s_local = s_full // world_size
    start = rank * s_local
    end = (rank + 1) * s_local
    full = torch.arange(bsz * s_full, dtype=torch.long).view(bsz, s_full)
    local = full[:, start:end].clone()

    rolled1, _ = _roll_tensor_cp(local, cp_group=group)
    rolled2, _ = _roll_tensor_cp(rolled1, cp_group=group)

    ref1 = torch.roll(full, shifts=-1, dims=1)
    ref1[:, -1] = 0
    ref2 = torch.roll(ref1, shifts=-1, dims=1)
    ref2[:, -1] = 0
    assert torch.equal(rolled1, ref1[:, start:end]), (
        f"rank={rank} roll1 mismatch:\n"
        f"got={rolled1}\nexp={ref1[:, start:end]}"
    )
    assert torch.equal(rolled2, ref2[:, start:end]), (
        f"rank={rank} roll2 mismatch:\n"
        f"got={rolled2}\nexp={ref2[:, start:end]}"
    )
    dist.destroy_process_group()


def _mtp_loss_parity_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )
    group = dist.group.WORLD

    torch.manual_seed(20260601)
    bsz, s_full, hidden_size, vocab = 2, 8, 6, 13
    scale = 0.1
    depth = 2
    s_local = s_full // world_size
    start = rank * s_local
    end = (rank + 1) * s_local

    labels_full = torch.randint(0, vocab, (bsz, s_full), dtype=torch.long)
    labels_full[:, 0] = -100
    loss_mask_full = torch.ones(bsz, s_full, dtype=torch.float32)
    loss_mask_full[:, 0] = 0.0

    hidden_full = [
        torch.randn(bsz, s_full, hidden_size, dtype=torch.float32)
        for _ in range(depth)
    ]
    hidden_local = [h[:, start:end, :].contiguous() for h in hidden_full]
    labels_local = labels_full[:, start:end].contiguous()
    loss_mask_local = loss_mask_full[:, start:end].contiguous()

    lm_head = nn.Linear(hidden_size, vocab, bias=False)
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    # 1. numerator: CP-split calculate_mtp_loss vs single-rank reference
    nums_local = calculate_mtp_loss(
        mtp_per_depth_h=hidden_local,
        labels=labels_local,
        lm_head=lm_head,
        loss_fct=loss_fct,
        loss_mask=loss_mask_local,
        cp_group=group,
    )
    nums_global = []
    for n_local in nums_local:
        n = n_local.detach().clone()
        dist.all_reduce(n)
        nums_global.append(n)

    # 2. denominator: mtp_per_depth_valid_count CP-split vs single-rank
    dens_cp_local = mtp_per_depth_valid_count(
        labels_full, loss_mask_full,
        num_depth=depth, cp_size=world_size, cp_rank=rank,
    )
    dens_cp_global = dens_cp_local.clone()
    dist.all_reduce(dens_cp_global)

    if rank == 0:
        nums_ref = calculate_mtp_loss(
            mtp_per_depth_h=hidden_full,
            labels=labels_full,
            lm_head=lm_head,
            loss_fct=loss_fct,
            loss_mask=loss_mask_full,
            cp_group=None,
        )
        dens_ref = mtp_per_depth_valid_count(
            labels_full, loss_mask_full,
            num_depth=depth, cp_size=1, cp_rank=0,
        )

        for i, (ng, nr) in enumerate(zip(nums_global, nums_ref)):
            assert torch.allclose(ng, nr, atol=1e-6, rtol=0), (
                f"depth={i} num mismatch: cp={ng.item()} ref={nr.item()}"
            )
        for i in range(depth):
            assert torch.allclose(dens_cp_global[i], dens_ref[i], atol=1e-6, rtol=0), (
                f"depth={i} den mismatch: cp={dens_cp_global[i].item()} ref={dens_ref[i].item()}"
            )

        per_depth_cp = [ng / dens_cp_global[i].clamp_min(1.0) for i, ng in enumerate(nums_global)]
        per_depth_ref = [nr / dens_ref[i].clamp_min(1.0) for i, nr in enumerate(nums_ref)]
        total_cp = torch.stack(per_depth_cp).sum() * (scale / len(per_depth_cp))
        total_ref = torch.stack(per_depth_ref).sum() * (scale / len(per_depth_ref))
        assert torch.allclose(total_cp, total_ref, atol=1e-6, rtol=0), (
            f"total mismatch: cp={total_cp.item()} ref={total_ref.item()}"
        )
    dist.destroy_process_group()


def _brute_force_mtp_valid_count(
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    num_depth: int,
    cp_size: int,
    cp_rank: int,
    ignore_index: int = -100,
) -> list[int]:
    """逐 depth 手动 roll + tail-mask + CP-chunk 算 valid count，作为 oracle。"""
    _, s_full = labels.shape
    s_local = s_full // cp_size
    lo, hi = cp_rank * s_local, (cp_rank + 1) * s_local
    counts = []
    rolled_labels = labels.clone()
    rolled_mask = loss_mask.clone()
    for d in range(num_depth):
        rolled_labels = torch.roll(rolled_labels, shifts=-1, dims=1)
        rolled_mask = torch.roll(rolled_mask, shifts=-1, dims=1)
        # 尾部 d+1 个全局位置无效
        rolled_labels[:, -(d + 1):] = ignore_index
        rolled_mask[:, -(d + 1):] = 0
        valid = (rolled_labels != ignore_index) & (rolled_mask != 0)
        counts.append(int(valid[:, lo:hi].sum().item()))
    return counts


class TestMtpPerDepthValidCount(unittest.TestCase):
    """单进程测试 mtp_per_depth_valid_count（纯 tensor，不需要 dist）。"""

    def test_cp_size_1(self) -> None:
        """cp_size=1：helper 结果 == 手动 oracle。"""
        torch.manual_seed(42)
        bsz, s_full, num_depth = 3, 16, 3
        labels = torch.randint(0, 100, (bsz, s_full), dtype=torch.long)
        # 散布一些 -100
        labels[:, 0] = -100
        labels[:, 5] = -100
        loss_mask = torch.ones(bsz, s_full, dtype=torch.float32)
        loss_mask[:, 0] = 0.0
        loss_mask[:, 7] = 0.0

        result = mtp_per_depth_valid_count(
            labels, loss_mask, num_depth=num_depth, cp_size=1, cp_rank=0,
        )
        expected = _brute_force_mtp_valid_count(
            labels, loss_mask, num_depth=num_depth, cp_size=1, cp_rank=0,
        )
        self.assertEqual(result.tolist(), expected)

    def test_cp_size_4_all_ranks(self) -> None:
        """cp_size=4：每个 cp_rank 的 local count 之和 == cp_size=1 的全局 count。"""
        torch.manual_seed(99)
        bsz, s_full, num_depth, cp_size = 2, 32, 2, 4
        labels = torch.randint(0, 50, (bsz, s_full), dtype=torch.long)
        labels[:, :3] = -100
        labels[:, 15] = -100
        loss_mask = torch.ones(bsz, s_full, dtype=torch.float32)
        loss_mask[:, :3] = 0.0

        global_counts = mtp_per_depth_valid_count(
            labels, loss_mask, num_depth=num_depth, cp_size=1, cp_rank=0,
        )
        sum_local = torch.zeros(num_depth)
        for cp_rank in range(cp_size):
            local = mtp_per_depth_valid_count(
                labels, loss_mask, num_depth=num_depth,
                cp_size=cp_size, cp_rank=cp_rank,
            )
            # 每个 rank 的结果跟手动 oracle 一致
            expected = _brute_force_mtp_valid_count(
                labels, loss_mask, num_depth=num_depth,
                cp_size=cp_size, cp_rank=cp_rank,
            )
            self.assertEqual(local.tolist(), expected, f"cp_rank={cp_rank}")
            sum_local += local.float()
        # 各 rank 加起来 == 全局
        self.assertEqual(sum_local.tolist(), global_counts.float().tolist())

    def test_all_masked(self) -> None:
        """全部 -100 / loss_mask=0 时每个 depth count 都是 0。"""
        bsz, s_full, num_depth = 2, 8, 2
        labels = torch.full((bsz, s_full), -100, dtype=torch.long)
        loss_mask = torch.zeros(bsz, s_full, dtype=torch.float32)
        result = mtp_per_depth_valid_count(
            labels, loss_mask, num_depth=num_depth, cp_size=1, cp_rank=0,
        )
        self.assertEqual(result.tolist(), [0, 0])

    def test_deeper_depth_fewer_tokens(self) -> None:
        """depth 越大，尾部 mask 掉的越多，valid count 应递减（或相等）。"""
        torch.manual_seed(7)
        bsz, s_full, num_depth = 2, 64, 5
        labels = torch.randint(0, 100, (bsz, s_full), dtype=torch.long)
        loss_mask = torch.ones(bsz, s_full, dtype=torch.float32)
        result = mtp_per_depth_valid_count(
            labels, loss_mask, num_depth=num_depth, cp_size=1, cp_rank=0,
        )
        for d in range(1, num_depth):
            self.assertGreaterEqual(
                result[d - 1].item(), result[d].item(),
                f"depth {d-1} count ({result[d-1].item()}) < depth {d} count ({result[d].item()})",
            )


class TestDsv4MtpCpRoll(unittest.TestCase):
    def test_roll_cp_size_1_fallback(self) -> None:
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        rolled, _ = _roll_tensor_cp(x, cp_group=None)
        exp = torch.tensor([[2, 3, 4, 0]], dtype=torch.long)
        self.assertTrue(torch.equal(rolled, exp))

    def test_roll_cp_size_2(self) -> None:
        port = _free_port()
        init_method = f"tcp://127.0.0.1:{port}"
        mp.spawn(_roll_worker, args=(2, init_method), nprocs=2, join=True)

    def test_roll_cp_size_4(self) -> None:
        port = _free_port()
        init_method = f"tcp://127.0.0.1:{port}"
        mp.spawn(_roll_worker, args=(4, init_method), nprocs=4, join=True)

    def test_mtp_loss_cp_parity(self) -> None:
        port = _free_port()
        init_method = f"tcp://127.0.0.1:{port}"
        mp.spawn(_mtp_loss_parity_worker, args=(2, init_method), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
