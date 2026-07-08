"""Distributed runtime helpers shared across training scripts."""

from __future__ import annotations

import socket

import torch
import torch.distributed as dist


def test_allreduce(device: torch.device) -> None:
    """Verify inter-rank communication via a SUM all-reduce sanity check.

    Each rank contributes ``1``; after all-reduce the sum must equal ``world_size``.
    Per-rank status and hostnames are gathered so rank-0 can pinpoint and report
    any machine whose collective failed, raising ``RuntimeError`` if so.

    Intended to be called right after process-group / device-mesh init and before
    loading any model, to fail fast on broken inter-node links.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Each rank sends 1; after allreduce-sum the result should equal world_size
    local_val = torch.tensor([1], dtype=torch.int64, device=device)
    summed = local_val.clone()
    dist.all_reduce(summed, op=dist.ReduceOp.SUM)

    # Gather per-rank status: 1 = OK, 0 = failed
    ok = 1 if summed.item() == world_size else 0
    status_tensor = torch.tensor([ok], dtype=torch.int64, device=device)
    all_status = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_status, status_tensor)

    # Gather hostname from every rank so rank-0 can report which machine failed
    hostname = socket.gethostname()
    hostname_bytes = hostname.encode('utf-8')
    max_len = 256
    padded = hostname_bytes[:max_len].ljust(max_len, b'\x00')
    hostname_tensor = torch.tensor(list(padded), dtype=torch.uint8, device=device)
    all_hostnames = [
        torch.zeros(max_len, dtype=torch.uint8, device=device) for _ in range(world_size)
    ]
    dist.all_gather(all_hostnames, hostname_tensor)

    if rank == 0:
        failed_ranks = []
        for r in range(world_size):
            if all_status[r].item() != 1:
                hn = bytes(all_hostnames[r].cpu().tolist()).decode('utf-8').rstrip('\x00')
                failed_ranks.append((r, hn))
        if failed_ranks:
            print(f"[test_allreduce] FAILED ranks:")
            for r, hn in failed_ranks:
                print(f"  Rank {r} on host {hn}")
            raise RuntimeError(
                f"test_allreduce failed: {len(failed_ranks)} rank(s) have communication issues"
            )
        else:
            print(f"[test_allreduce] All {world_size} ranks passed allreduce check successfully")
