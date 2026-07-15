# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""Measure NCCL all2all launch overhead and large allgather latency.

Case A sends 100 million float32 values once per rank. Case B sends the same
per-rank total in ``NUM_ALL_TO_ALL_CALLS`` equal all2all calls.

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$PYTHONPATH"
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    pytest -v -s --timeout=1800 tests/test_gfused/test_nccl_bench.py
"""

from __future__ import annotations

import os
import socket
import statistics
import time
import unittest
from dataclasses import dataclass

import ray
import torch
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.orches.placement_group import _create_placement_group

WORLD_SIZE = 32
NUM_FLOATS_PER_RANK = 128 * 128 * 512 * 20
NUM_ALL_TO_ALL_CALLS = 20
WARMUP_ITERS = 2
BENCH_ITERS = 10
ALL_GATHER_BATCH_SIZE = 1
ALL_GATHER_SEQ_LEN = 1_000_000
ALL_GATHER_HIDDEN_DIM = 1024


@ray.remote(num_cpus=0, num_gpus=0)
def _get_master_endpoint() -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return ray.util.get_node_ip_address(), s.getsockname()[1]


@dataclass
class _BenchResult:
    rank: int
    one_call_ms: float
    multi_call_ms: float


@dataclass
class _AllGatherBenchResult:
    rank: int
    avg_ms: float


@ray.remote(num_gpus=1)
class _AllToAllBenchWorker:
    def run(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        num_floats_per_rank: int,
        num_all_to_all_calls: int,
    ) -> _BenchResult:
        assert num_floats_per_rank % world_size == 0
        assert num_floats_per_rank % num_all_to_all_calls == 0
        assert (num_floats_per_rank // num_all_to_all_calls) % world_size == 0

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        try:
            device = torch.device("cuda:0")
            send_one = torch.empty(num_floats_per_rank, dtype=torch.float32, device=device)
            recv_one = torch.empty_like(send_one)
            floats_per_call = num_floats_per_rank // num_all_to_all_calls
            send_small = torch.empty(floats_per_call, dtype=torch.float32, device=device)
            recv_small = torch.empty_like(send_small)

            def one_call() -> None:
                dist.all_to_all_single(recv_one, send_one)

            def multi_call() -> None:
                for _ in range(num_all_to_all_calls):
                    dist.all_to_all_single(recv_small, send_small)

            def time_fn(fn) -> float:
                for _ in range(WARMUP_ITERS):
                    fn()
                dist.barrier()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(BENCH_ITERS):
                    fn()
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0 / BENCH_ITERS
                dist.barrier()
                return elapsed_ms

            return _BenchResult(
                rank=rank,
                one_call_ms=time_fn(one_call),
                multi_call_ms=time_fn(multi_call),
            )
        finally:
            dist.destroy_process_group()


@ray.remote(num_gpus=1)
class _AllGatherBenchWorker:
    def run(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> _AllGatherBenchResult:
        assert ALL_GATHER_SEQ_LEN % world_size == 0

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        try:
            device = torch.device("cuda:0")
            local_seq_len = ALL_GATHER_SEQ_LEN // world_size
            send = torch.empty(
                (ALL_GATHER_BATCH_SIZE, local_seq_len, ALL_GATHER_HIDDEN_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            gathered = torch.empty(
                (world_size, local_seq_len, ALL_GATHER_HIDDEN_DIM),
                dtype=send.dtype,
                device=device,
            )

            for _ in range(WARMUP_ITERS):
                dist.all_gather_into_tensor(gathered, send)
            dist.barrier()
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(BENCH_ITERS):
                dist.all_gather_into_tensor(gathered, send)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / BENCH_ITERS
            dist.barrier()
            return _AllGatherBenchResult(rank=rank, avg_ms=elapsed_ms)
        finally:
            dist.destroy_process_group()


def _run_benchmark(
    world_size: int,
    num_floats_per_rank: int,
    num_all_to_all_calls: int,
) -> list[_BenchResult]:
    pg, bundle_indices = _create_placement_group(world_size)
    workers = []
    try:
        master_addr, master_port = ray.get(
            _get_master_endpoint.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_indices[0],
                )
            ).remote()
        )
        for rank in range(world_size):
            workers.append(
                _AllToAllBenchWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_indices[rank],
                    )
                ).remote()
            )
        return ray.get([
            worker.run.remote(
                rank,
                world_size,
                master_addr,
                master_port,
                num_floats_per_rank,
                num_all_to_all_calls,
            )
            for rank, worker in enumerate(workers)
        ])
    finally:
        for worker in workers:
            ray.kill(worker)
        remove_placement_group(pg)


def _run_all_gather_benchmark(world_size: int) -> list[_AllGatherBenchResult]:
    pg, bundle_indices = _create_placement_group(world_size)
    workers = []
    try:
        master_addr, master_port = ray.get(
            _get_master_endpoint.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_indices[0],
                )
            ).remote()
        )
        for rank in range(world_size):
            workers.append(
                _AllGatherBenchWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_indices[rank],
                    )
                ).remote()
            )
        return ray.get([
            worker.run.remote(rank, world_size, master_addr, master_port)
            for rank, worker in enumerate(workers)
        ])
    finally:
        for worker in workers:
            ray.kill(worker)
        remove_placement_group(pg)


class AllToAllOverheadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"Need >= {WORLD_SIZE} GPUs, only {total_gpus} available")

    @classmethod
    def tearDownClass(cls):
        kill_all_actors_and_shutdown_ray()

    def test_fixed_volume_all_to_all_overhead(self):
        results = _run_benchmark(
            WORLD_SIZE,
            NUM_FLOATS_PER_RANK,
            NUM_ALL_TO_ALL_CALLS,
        )
        one_call_ms = [result.one_call_ms for result in results]
        multi_call_ms = [result.multi_call_ms for result in results]
        floats_per_call = NUM_FLOATS_PER_RANK // NUM_ALL_TO_ALL_CALLS
        ratio = statistics.mean(multi_call_ms) / statistics.mean(one_call_ms)
        overhead_ms = statistics.mean(multi_call_ms) - statistics.mean(one_call_ms)

        print(f"\n[world={WORLD_SIZE}, per-rank payload={NUM_FLOATS_PER_RANK:,} float32]")
        print(
            f"  case A: 1 x all2all({NUM_FLOATS_PER_RANK:,})"
            f"  mean={statistics.mean(one_call_ms):.3f} ms"
            f"  max={max(one_call_ms):.3f} ms"
        )
        print(
            f"  case B: {NUM_ALL_TO_ALL_CALLS} x all2all({floats_per_call:,})"
            f"  mean={statistics.mean(multi_call_ms):.3f} ms"
            f"  max={max(multi_call_ms):.3f} ms"
        )
        print(f"  multi/single={ratio:.3f}x  overhead={overhead_ms:.3f} ms")

    def test_bf16_all_gather_1m_1024(self):
        '''
        [world=32, allgather output=[1, 1,000,000, 1024] bf16]
        per-rank input=[1, 31,250, 1024] (61.0 mib)
        gathered=1.91 gib  mean=6.290 ms  max=6.301 ms
        '''
        results = _run_all_gather_benchmark(WORLD_SIZE)
        all_gather_ms = [result.avg_ms for result in results]
        local_seq_len = ALL_GATHER_SEQ_LEN // WORLD_SIZE
        local_mib = (
            ALL_GATHER_BATCH_SIZE
            * local_seq_len
            * ALL_GATHER_HIDDEN_DIM
            * torch.empty((), dtype=torch.bfloat16).element_size()
            / 1024**2
        )
        gathered_gib = local_mib * WORLD_SIZE / 1024

        print(
            f"\n[world={WORLD_SIZE}, allgather output="
            f"[{ALL_GATHER_BATCH_SIZE}, {ALL_GATHER_SEQ_LEN:,}, {ALL_GATHER_HIDDEN_DIM}] bf16]"
        )
        print(
            f"  per-rank input=[{ALL_GATHER_BATCH_SIZE}, {local_seq_len:,}, "
            f"{ALL_GATHER_HIDDEN_DIM}] ({local_mib:.1f} MiB)"
        )
        print(
            f"  gathered={gathered_gib:.2f} GiB"
            f"  mean={statistics.mean(all_gather_ms):.3f} ms"
            f"  max={max(all_gather_ms):.3f} ms"
        )
        for result in results:
            self.assertGreater(result.avg_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
