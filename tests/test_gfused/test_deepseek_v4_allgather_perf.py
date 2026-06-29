"""Benchmark NCCL all_gather on DeepSeek-V4-Pro head-shaped tensors.

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_deepseek_v4_allgather_perf.py

Environment variables:

    DSV4_AG_WORLD_SIZE     default: 16
    DSV4_AG_SEQ_LEN        default: 128
    DSV4_AG_WARMUP         default: 5
    DSV4_AG_ITERS          default: 20
    DSV4_AG_CONFIG_PATH    default: /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/deepseek-ai/DeepSeek-V4-Pro/config.json

    # indexer causal head/tail benchmark (test_indexer_causal_head_tail)
    DSV4_IDX_SEQ_LEN       default: 16384  (local query length S)
    DSV4_IDX_CONTEXT_LEN   default: 524288 (raw token context length)
"""

import json
import os
import socket
import unittest
from pathlib import Path
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from gpatch_v4.orches.placement_group import _create_placement_group
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray


CONFIG_PATH = Path(
    os.environ.get(
        "DSV4_AG_CONFIG_PATH",
        "hf-hub/deepseek-ai/DeepSeek-V4-Pro/config.json",
    )
)
WORLD_SIZE = int(os.environ.get("DSV4_AG_WORLD_SIZE", "32"))
FULL_SEQ_LEN = 512 * 1024
SEQ_LEN = FULL_SEQ_LEN // WORLD_SIZE
WARMUP = int(os.environ.get("DSV4_AG_WARMUP", "5"))
ITERS = int(os.environ.get("DSV4_AG_ITERS", "20"))
IDX_SEQ_LEN = int(os.environ.get("DSV4_IDX_SEQ_LEN", str(SEQ_LEN)))
IDX_CONTEXT_LEN = int(os.environ.get("DSV4_IDX_CONTEXT_LEN", str(FULL_SEQ_LEN)))

# AG
# s   1k: avg_ms per rank: mean=0.281, min=0.280, max=0.285
# s  16k: avg_ms per rank: mean=0.334, min=0.332, max=0.336
# s  64k: avg_ms per rank: mean=0.491, min=0.489, max=0.493
# s 512k: avg_ms per rank: mean=2.251, min=2.247, max=2.255
# s   1m: avg_ms per rank: mean=4.106, min=4.101, max=4.113

# indexer (512k)
# DeepSeek-V4-Pro indexer causal head/tail perf: q=[16384, 1, 128], context=524288, k_compressed=[131072, 128], compress_rate=4, dtype=torch.bfloat16, logits=8192.0 MiB, warmup=5, iters=20
# clean_logits=True head avg_ms per rank: mean=2.483, min=2.469, max=2.499
# clean_logits=True tail avg_ms per rank: mean=8.529, min=8.465, max=8.595
# clean_logits=True tail/head mean ratio: 3.44x
# clean_logits=False head avg_ms per rank: mean=0.203, min=0.201, max=0.205
# clean_logits=False tail avg_ms per rank: mean=7.763, min=7.696, max=7.840
# clean_logits=False tail/head mean ratio: 38.20x

# 能看出来两件事情，nccl 通信也很昂贵，耗时和 indexer 在一个数量级；而且 indexer 的 tail 的 head 耗时相差巨大。


def _load_deepseek_v4_pro_shape(config_path: Path) -> tuple[int, int, torch.dtype]:
    with config_path.open() as f:
        config = json.load(f)

    num_heads = 1
    head_dim = int(config["head_dim"])
    torch_dtype = torch.bfloat16
    return num_heads, head_dim, torch_dtype


def _load_deepseek_v4_indexer_shape(config_path: Path) -> tuple[int, int, int]:
    """Load DeepSeek-V4-Pro Lightning-Indexer (CSA) shapes from config.

    Returns ``(index_n_heads, index_head_dim, compress_rate_csa)``. The CSA
    compress rate is the smallest non-zero entry of the legacy
    ``compress_ratios`` list (= 4 for V4-Pro), mirroring
    ``config.compress_rates["compressed_sparse_attention"]``.
    """
    with config_path.open() as f:
        config = json.load(f)

    index_n_heads = 1
    index_head_dim = int(config["index_head_dim"])
    ratios = [int(r) for r in config.get("compress_ratios", []) if int(r) != 0]
    compress_rate = min(ratios) if ratios else 4
    return index_n_heads, index_head_dim, compress_rate


@ray.remote(num_gpus=1)
class AllGatherPerfWorker:
    def get_master_addr_and_port(self) -> tuple[str, int]:
        ip = socket.gethostbyname(socket.gethostname())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return ip, s.getsockname()[1]

    def init_dist(self, rank: int, world_size: int, master_addr: str, master_port: int) -> None:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cuda:0")

    def benchmark(
        self,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        warmup: int,
        iters: int,
    ) -> dict[str, Any]:
        tensor = torch.empty((seq_len, num_heads, head_dim), dtype=dtype, device=self.device)
        tensor.fill_(self.rank)
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]

        for _ in range(warmup):
            dist.all_gather(gathered, tensor)
        torch.cuda.synchronize()
        dist.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            dist.all_gather(gathered, tensor)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iters
        return {
            "rank": self.rank,
            "host": socket.gethostname(),
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "world_size": self.world_size,
            "local_bytes": tensor.numel() * tensor.element_size(),
            "gathered_bytes": tensor.numel() * tensor.element_size() * self.world_size,
            "avg_ms": avg_ms,
        }

    def benchmark_indexer(
        self,
        local_s: int,
        kv_len: int,
        num_heads: int,
        head_dim: int,
        compress_rate: int,
        dtype: torch.dtype,
        warmup: int,
        iters: int,
    ) -> dict[str, Any]:
        """Benchmark the Lightning-Indexer kernel under causal HEAD vs TAIL.

        Mirrors the ``batched_indexer_fwd(q_sbhd, k_sbd, w_sbh, cu_ks, cu_ke)``
        call in ``DeepseekV4Indexer.forward`` (modeling_deepseek_v4.py). Both
        cases share identical tensor shapes ``[S, B, H, D]`` / ``[T, B, D]`` and
        differ only in the per-query causal range ``[cu_ks, cu_ke)`` built with
        ``_make_causal_cu_seqlens``:

          * **head** — queries sit at the very start of the context
            (``positions = arange(S)``); early queries attend to almost no
            compressed KV, so each ``block_Q`` tile touches a tiny KV span.
          * **tail** — queries sit at the end of the context
            (``positions = arange(S) + T*m - S``); every query attends to nearly
            the full ``T`` compressed entries, so each tile scans the whole KV.
        """
        from gpatch_v4.models.deepseek_v4.kernel.tilelang_indexer_fwd import (
            _make_causal_cu_seqlens,
            clean_logits_,
            tl_indexer_fwd_impl,
        )

        device = self.device
        batch = 1
        assert kv_len % compress_rate == 0, (
            f"context_len ({kv_len}) must be divisible by compress_rate ({compress_rate})"
        )
        assert kv_len >= local_s, (
            f"context_len ({kv_len}) must be >= seq_len_q ({local_s}) "
            "so the tail chunk fits inside the context"
        )
        compressed_kv_len = kv_len // compress_rate
        q = torch.empty((local_s, batch, num_heads, head_dim), dtype=dtype, device=device).normal_()
        k = torch.empty((compressed_kv_len, batch, head_dim), dtype=dtype, device=device).normal_()
        weights = torch.empty(
            (local_s, batch, num_heads), dtype=torch.float32, device=device
        ).normal_()
        q_single = q[:, 0, :, :].contiguous()
        k_single = k[:, 0, :].contiguous()
        weights_single = weights[:, 0, :].contiguous()
        logits = torch.empty((local_s, compressed_kv_len), dtype=torch.float32, device=device)
        q_flat = q_single.view(local_s * num_heads, head_dim)
        clean_logits_kernel = clean_logits_()
        tl_indexer_fwd_kernel = tl_indexer_fwd_impl(heads=num_heads, index_dim=head_dim)

        positions_head = torch.arange(local_s, device=device, dtype=torch.int32)
        positions_tail = positions_head + (kv_len - local_s)
        cu_ks_head, cu_ke_head = _make_causal_cu_seqlens(
            local_s, compressed_kv_len, compress_rate, device, positions=positions_head,
        )
        cu_ks_tail, cu_ke_tail = _make_causal_cu_seqlens(
            local_s, compressed_kv_len, compress_rate, device, positions=positions_tail,
        )

        @torch.no_grad()
        def _time(cu_ks: torch.Tensor, cu_ke: torch.Tensor, clean_logits: bool) -> float:
            for _ in range(warmup):
                tl_indexer_fwd_kernel(q_flat, k_single, logits, weights_single, cu_ks, cu_ke)
                if clean_logits:
                    clean_logits_kernel(logits, cu_ks, cu_ke)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                tl_indexer_fwd_kernel(q_flat, k_single, logits, weights_single, cu_ks, cu_ke)
                if clean_logits:
                    clean_logits_kernel(logits, cu_ks, cu_ke)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters

        head_clean_ms = _time(cu_ks_head, cu_ke_head, clean_logits=True)
        tail_clean_ms = _time(cu_ks_tail, cu_ke_tail, clean_logits=True)
        head_raw_ms = _time(cu_ks_head, cu_ke_head, clean_logits=False)
        tail_raw_ms = _time(cu_ks_tail, cu_ke_tail, clean_logits=False)
        return {
            "rank": self.rank,
            "host": socket.gethostname(),
            "seq_len_q": local_s,
            "context_len": kv_len,
            "seq_len_kv": compressed_kv_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "compress_rate": compress_rate,
            "dtype": str(dtype),
            "head_clean_avg_ms": head_clean_ms,
            "tail_clean_avg_ms": tail_clean_ms,
            "head_raw_avg_ms": head_raw_ms,
            "tail_raw_avg_ms": tail_raw_ms,
        }

    def destroy(self) -> None:
        dist.destroy_process_group()


def _create_and_init_workers(
    world_size: int,
    pg: tuple[Any, list[int]],
) -> list[ray.actor.ActorHandle]:
    pg_obj, bundle_indices = pg
    workers = [
        AllGatherPerfWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg_obj,
                placement_group_bundle_index=bundle_indices[rank],
            )
        ).remote()
        for rank in range(world_size)
    ]
    master_addr, master_port = ray.get(workers[0].get_master_addr_and_port.remote())
    ray.get(
        [
            worker.init_dist.remote(rank, world_size, master_addr, master_port)
            for rank, worker in enumerate(workers)
        ]
    )
    return workers


def _destroy_workers(workers: list[ray.actor.ActorHandle]) -> None:
    ray.get([worker.destroy.remote() for worker in workers])


class DeepSeekV4AllGatherPerfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not CONFIG_PATH.exists():
            raise unittest.SkipTest(f"DeepSeek-V4-Pro config not found: {CONFIG_PATH}")
        if ITERS <= 0:
            raise unittest.SkipTest(f"DSV4_AG_ITERS must be positive, got {ITERS}")
        if WARMUP < 0:
            raise unittest.SkipTest(f"DSV4_AG_WARMUP must be non-negative, got {WARMUP}")

        cls.num_heads, cls.head_dim, cls.dtype = _load_deepseek_v4_pro_shape(CONFIG_PATH)
        cls.index_n_heads, cls.index_head_dim, cls.compress_rate = _load_deepseek_v4_indexer_shape(CONFIG_PATH)

        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"Need >= {WORLD_SIZE} GPUs across the cluster, only {total_gpus} available"
            )
        cls.pg = _create_placement_group(WORLD_SIZE)
        cls.workers = _create_and_init_workers(WORLD_SIZE, cls.pg)

    @classmethod
    def tearDownClass(cls) -> None:
        _destroy_workers(cls.workers)
        remove_placement_group(cls.pg[0])
        kill_all_actors_and_shutdown_ray()

    def test_allgather_avg_time(self) -> None:
        results = ray.get(
            [
                worker.benchmark.remote(
                    SEQ_LEN,
                    self.num_heads,
                    self.head_dim,
                    self.dtype,
                    WARMUP,
                    ITERS,
                )
                for worker in self.workers
            ]
        )

        avg_times = [result["avg_ms"] for result in results]
        local_mib = results[0]["local_bytes"] / 1024**2
        gathered_mib = results[0]["gathered_bytes"] / 1024**2
        print(
            "\nDeepSeek-V4-Pro all_gather perf: "
            f"shape=[{SEQ_LEN}, {self.num_heads}, {self.head_dim}], "
            f"dtype={results[0]['dtype']}, world_size={WORLD_SIZE}, "
            f"local={local_mib:.2f} MiB, gathered={gathered_mib:.2f} MiB, "
            f"warmup={WARMUP}, iters={ITERS}"
        )
        print(
            "avg_ms per rank: "
            f"mean={sum(avg_times) / len(avg_times):.3f}, "
            f"min={min(avg_times):.3f}, max={max(avg_times):.3f}"
        )
        for result in sorted(results, key=lambda item: item["rank"]):
            print(
                f"rank={result['rank']:03d} host={result['host']} "
                f"avg_ms={result['avg_ms']:.3f}"
            )

        for result in results:
            self.assertGreater(result["avg_ms"], 0.0)

    def test_indexer_causal_head_tail(self) -> None:
        """Bench Lightning-Indexer ``batched_indexer_fwd`` for causal HEAD vs TAIL.

        Same tensor shapes in both cases; only the per-query causal range
        ``[cu_ks, cu_ke)`` differs (see ``benchmark_indexer``). Head queries
        touch a small KV span, tail queries touch nearly the full KV, so this
        isolates the cost impact of causal position on the indexer kernel.
        """
        results = ray.get(
            [
                worker.benchmark_indexer.remote(
                    IDX_SEQ_LEN,
                    IDX_CONTEXT_LEN,
                    self.index_n_heads,
                    self.index_head_dim,
                    self.compress_rate,
                    self.dtype,
                    WARMUP,
                    ITERS,
                )
                for worker in self.workers
            ]
        )

        head_clean_times = [r["head_clean_avg_ms"] for r in results]
        tail_clean_times = [r["tail_clean_avg_ms"] for r in results]
        head_raw_times = [r["head_raw_avg_ms"] for r in results]
        tail_raw_times = [r["tail_raw_avg_ms"] for r in results]
        seq_len_kv = results[0]["seq_len_kv"]
        logits_mib = IDX_SEQ_LEN * seq_len_kv * 4 / 1024**2
        head_clean_mean = sum(head_clean_times) / len(head_clean_times)
        tail_clean_mean = sum(tail_clean_times) / len(tail_clean_times)
        head_raw_mean = sum(head_raw_times) / len(head_raw_times)
        tail_raw_mean = sum(tail_raw_times) / len(tail_raw_times)
        print(
            "\nDeepSeek-V4-Pro indexer causal head/tail perf: "
            f"q=[{IDX_SEQ_LEN}, {self.index_n_heads}, {self.index_head_dim}], "
            f"context={IDX_CONTEXT_LEN}, "
            f"k_compressed=[{seq_len_kv}, {self.index_head_dim}], "
            f"compress_rate={self.compress_rate}, dtype={results[0]['dtype']}, "
            f"logits={logits_mib:.1f} MiB, warmup={WARMUP}, iters={ITERS}"
        )
        print(
            "clean_logits=True head avg_ms per rank: "
            f"mean={head_clean_mean:.3f}, "
            f"min={min(head_clean_times):.3f}, max={max(head_clean_times):.3f}"
        )
        print(
            "clean_logits=True tail avg_ms per rank: "
            f"mean={tail_clean_mean:.3f}, "
            f"min={min(tail_clean_times):.3f}, max={max(tail_clean_times):.3f}"
        )
        print(f"clean_logits=True tail/head mean ratio: {tail_clean_mean / head_clean_mean:.2f}x")
        print(
            "clean_logits=False head avg_ms per rank: "
            f"mean={head_raw_mean:.3f}, "
            f"min={min(head_raw_times):.3f}, max={max(head_raw_times):.3f}"
        )
        print(
            "clean_logits=False tail avg_ms per rank: "
            f"mean={tail_raw_mean:.3f}, "
            f"min={min(tail_raw_times):.3f}, max={max(tail_raw_times):.3f}"
        )
        print(f"clean_logits=False tail/head mean ratio: {tail_raw_mean / head_raw_mean:.2f}x")
        for result in sorted(results, key=lambda item: item["rank"]):
            print(
                f"rank={result['rank']:03d} host={result['host']} "
                f"head_clean_ms={result['head_clean_avg_ms']:.3f} "
                f"tail_clean_ms={result['tail_clean_avg_ms']:.3f} "
                f"head_raw_ms={result['head_raw_avg_ms']:.3f} "
                f"tail_raw_ms={result['tail_raw_avg_ms']:.3f}"
            )

        for result in results:
            self.assertGreater(result["head_clean_avg_ms"], 0.0)
            self.assertGreater(result["tail_clean_avg_ms"], 0.0)
            self.assertGreater(result["head_raw_avg_ms"], 0.0)
            self.assertGreater(result["tail_raw_avg_ms"], 0.0)
