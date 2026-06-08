#!/usr/bin/env python3
# coding=utf-8

from __future__ import annotations

import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from gpatch_v4.models.deepseek_v4.mtp import mtp_roll_tensor_cp
from gpatch_v4.training_backend.fsdp2_backend.mtp_loss import calculate_mtp_loss


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

    rolled1, _ = mtp_roll_tensor_cp(local, shifts=-1, dim=1, cp_group=group)
    rolled2, _ = mtp_roll_tensor_cp(rolled1, shifts=-1, dim=1, cp_group=group)

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

    _, _, nums_local, dens_local = calculate_mtp_loss(
        mtp_per_depth_h=hidden_local,
        labels=labels_local,
        lm_head=lm_head,
        loss_fct=loss_fct,
        loss_mask=loss_mask_local,
        cp_group=group,
        scaling_factor=scale,
    )
    nums_global = []
    dens_global = []
    for n_local, d_local in zip(nums_local, dens_local):
        n = n_local.detach().clone()
        d = d_local.detach().clone()
        dist.all_reduce(n)
        dist.all_reduce(d)
        nums_global.append(n)
        dens_global.append(d)

    if rank == 0:
        _, _, nums_ref, dens_ref = calculate_mtp_loss(
            mtp_per_depth_h=hidden_full,
            labels=labels_full,
            lm_head=lm_head,
            loss_fct=loss_fct,
            loss_mask=loss_mask_full,
            cp_group=None,
            scaling_factor=scale,
        )
        for i, (ng, dg, nr, dr) in enumerate(zip(nums_global, dens_global, nums_ref, dens_ref)):
            assert torch.allclose(ng, nr, atol=1e-6, rtol=0), (
                f"depth={i} num mismatch: cp={ng.item()} ref={nr.item()}"
            )
            assert torch.allclose(dg, dr, atol=1e-6, rtol=0), (
                f"depth={i} den mismatch: cp={dg.item()} ref={dr.item()}"
            )
        per_depth_cp = [ng / dg.clamp_min(1.0) for ng, dg in zip(nums_global, dens_global)]
        per_depth_ref = [nr / dr.clamp_min(1.0) for nr, dr in zip(nums_ref, dens_ref)]
        total_cp = torch.stack(per_depth_cp).sum() * (scale / len(per_depth_cp))
        total_ref = torch.stack(per_depth_ref).sum() * (scale / len(per_depth_ref))
        assert torch.allclose(total_cp, total_ref, atol=1e-6, rtol=0), (
            f"total mismatch: cp={total_cp.item()} ref={total_ref.item()}"
        )
    dist.destroy_process_group()


class TestDsv4MtpCpRoll(unittest.TestCase):
    def test_roll_cp_size_1_fallback(self) -> None:
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        rolled, _ = mtp_roll_tensor_cp(x, shifts=-1, dim=1, cp_group=None)
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
