"""NCCL collective sanity tests (allreduce, all_to_all, allgather).

Spawns 16 Ray actors across 2 nodes (8 GPUs each), initialises
``torch.distributed`` with NCCL backend, and exercises each collective on
GPU tensors.

主要是防 TCCL。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=300 tests/test_gpatch_v4/test_nccl.py
"""

import os
import socket
import unittest

import ray
import torch

from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLD_SIZE = 16  # 2 nodes x 8 GPUs

# ---------------------------------------------------------------------------
# Ray actor
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
class NcclWorker:
    """A single-GPU worker that participates in ``torch.distributed`` collectives."""
    def init_dist(self, rank: int, world_size: int, master_addr: str, master_port: int):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NCCL_CUMEM_ENABLE"] = "0"

        # Use the first visible GPU (Ray already sets CUDA_VISIBLE_DEVICES).
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cuda:0")

    def get_master_addr_and_port(self):
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        # Pick a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return ip, port

    # -- collectives --------------------------------------------------------

    def allreduce(self):
        t = torch.ones(64, device=self.device) * (self.rank + 1)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        torch.cuda.synchronize()
        return t.cpu().tolist()

    def allgather(self):
        t = torch.full((4, ), float(self.rank), device=self.device)
        gathered = [torch.zeros(4, device=self.device) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, t)
        torch.cuda.synchronize()
        return [g.cpu().tolist() for g in gathered]

    def all_to_all(self):
        send_tensors = [
            torch.full((4, ), float(self.rank * 100 + j), device=self.device)
            for j in range(self.world_size)
        ]
        recv_tensors = [torch.zeros(4, device=self.device) for _ in range(self.world_size)]
        torch.distributed.all_to_all(recv_tensors, send_tensors)
        torch.cuda.synchronize()
        return [r.cpu().tolist() for r in recv_tensors]

    # -- teardown -----------------------------------------------------------

    def destroy(self):
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_and_init_workers(world_size: int):
    """Create *world_size* ``NcclWorker`` actors and init dist."""
    workers = [NcclWorker.remote() for _ in range(world_size)]

    # Rank 0 picks master address.
    master_addr, master_port = ray.get(workers[0].get_master_addr_and_port.remote())

    # Init distributed in parallel.
    init_refs = [
        w.init_dist.remote(rank, world_size, master_addr, master_port)
        for rank, w in enumerate(workers)
    ]
    ray.get(init_refs)
    return workers


def _destroy_workers(workers):
    ray.get([w.destroy.remote() for w in workers])


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class NcclCollectiveTest(unittest.TestCase):
    """Verify NCCL allreduce / allgather / all_to_all across 2 nodes (16 GPUs)."""
    @classmethod
    def setUpClass(cls):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"Need >= {WORLD_SIZE} GPUs across the cluster, only {total_gpus} available"
            )
        cls.workers = _create_and_init_workers(WORLD_SIZE)

    @classmethod
    def tearDownClass(cls):
        _destroy_workers(cls.workers)
        kill_all_actors_and_shutdown_ray()

    # -- allreduce ----------------------------------------------------------

    def test_allreduce(self):
        refs = [w.allreduce.remote() for w in self.workers]
        results = ray.get(refs)

        expected_val = float(sum(range(1, WORLD_SIZE + 1)))
        for rank, vals in enumerate(results):
            for v in vals:
                self.assertAlmostEqual(
                    v,
                    expected_val,
                    places=4,
                    msg=f"rank {rank}: expected {expected_val}, got {v}",
                )

    # -- allgather ----------------------------------------------------------

    def test_allgather(self):
        refs = [w.allgather.remote() for w in self.workers]
        results = ray.get(refs)

        for rank, gathered in enumerate(results):
            assert len(gathered) == WORLD_SIZE, (
                f"rank {rank}: expected {WORLD_SIZE} chunks, got {len(gathered)}"
            )
            for src_rank, chunk in enumerate(gathered):
                for v in chunk:
                    self.assertAlmostEqual(
                        v,
                        float(src_rank),
                        places=4,
                        msg=f"rank {rank}: chunk from {src_rank} wrong",
                    )

    # -- all_to_all ---------------------------------------------------------

    def test_all_to_all(self):
        refs = [w.all_to_all.remote() for w in self.workers]
        results = ray.get(refs)

        for rank, recv_chunks in enumerate(results):
            assert len(recv_chunks) == WORLD_SIZE
            for src_rank, chunk in enumerate(recv_chunks):
                expected_val = float(src_rank * 100 + rank)
                for v in chunk:
                    self.assertAlmostEqual(
                        v,
                        expected_val,
                        places=4,
                        msg=f"rank {rank}: from {src_rank} expected "
                        f"{expected_val}, got {v}",
                    )
