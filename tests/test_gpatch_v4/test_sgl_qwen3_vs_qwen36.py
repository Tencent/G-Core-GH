"""Throughput comparison: Qwen3-30B-A3B vs Qwen3.6-35B-A3B (SGLang).

Runs identical random-token workloads on both models sequentially and
prints a side-by-side comparison of decode tok/s and total tok/s.

Both engines use TP=8, EP=8, ``mem_fraction_static=0.7``.  Model paths
default to ``hf-hub/Qwen/Qwen3.5-30B-A3B`` and
``hf-hub/Qwen/Qwen3.6-35B-A3B`` and can be overridden via env vars
``QWEN35_MODEL_PATH`` / ``QWEN36_MODEL_PATH``.

Manual-run only — not in CI.  Requires:
- 8 GPUs visible to this process
- Both model checkpoints downloaded

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=3600 \\
        tests/test_gpatch_v4/test_sgl_qwen35_vs_qwen36.py

Env-tunable knobs::

    QWEN35_MODEL_PATH   default: (see test body)
    QWEN36_MODEL_PATH   default: (see test body)
    CMP_BATCH            default: 8
    CMP_PROMPT_TOKENS    default: 127*1024
    CMP_GEN_TOKENS       default: 1024
    CMP_ITERS            default: 2
    CMP_CONTEXT_LENGTH   default: 131072  (override model's native limit)
    CMP_MEM_FRACTION     default: 0.7
    CMP_TP_SIZE          default: 8
    CMP_EP_SIZE          default: 8
"""

'''
don't change mamba cache
```
[2026-05-13 15:13:54 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.00, mamba usage: 0.00, #running-req: 0, #queue-req: 31, cuda graph: False, input throughput (token/s): 0.00
[2026-05-13 15:13:54 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.01, mamba usage: 0.00, #running-req: 0, #queue-req: 31, cuda graph: False, input throughput (token/s): 60079.16
...
[2026-05-13 15:15:55 TP0 EP0] Decode batch, #running-req: 24, #full token: 3118656, full token usage: 0.99, mamba num: 48, mamba usage: 0.01, cuda graph: True, gen throughput (token/s): 625.38, #queue-req: 8
[2026-05-13 15:15:57 TP0 EP0] Decode batch, #running-req: 24, #full token: 3119616, full token usage: 0.99, mamba num: 48, mamba usage: 0.01, cuda graph: True, gen throughput (token/s): 624.76, #queue-req: 8
[2026-05-13 15:15:58 TP0 EP0] Decode batch, #running-req: 24, #full token: 3120576, full token usage: 0.99, mamba num: 48, mamba usage: 0.01, cuda graph: True, gen throughput (token/s): 625.94, #queue-req: 8
...
[2026-05-13 15:16:44 TP0 EP0] Decode batch, #running-req: 8, #full token: 1039688, full token usage: 0.33, mamba num: 16, mamba usage: 0.00, cuda graph: True, gen throughput (token/s): 499.24, #queue-req: 0
[2026-05-13 15:16:45 TP0 EP0] Decode batch, #running-req: 8, #full token: 1040008, full token usage: 0.33, mamba num: 16, mamba usage: 0.00, cuda graph: True, gen throughput (token/s): 499.39, #queue-req: 0
[2026-05-13 15:16:45 TP0 EP0] Decode batch, #running-req: 8, #full token: 1040328, full token usage: 0.33, mamba num: 16, mamba usage: 0.00, cuda graph: True, gen throughput (token/s): 498.41, #queue-req: 0
[qwen3.6] iter=1 elapsed=172.78s total_tok/s=24085.6
[qwen3.6] summary: total_tok/s=[24085.57030424254]
```

change mamba cache
```
[2026-05-13 15:09:13 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.00, mamba usage: 0.00, #running-req: 0, #queue-req: 31, cuda graph: False, input throughput (token/s): 0.00
[2026-05-13 15:09:14 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.00, mamba usage: 0.01, #running-req: 0, #queue-req: 31, cuda graph: False, input throughput (token/s): 59947.69
...
[2026-05-13 15:12:15 TP0 EP0] Decode batch, #running-req: 32, #full token: 4159488, full token usage: 0.73, mamba num: 64, mamba usage: 0.18, cuda graph: True, gen throughput (token/s): 455.75, #queue-req: 0
[2026-05-13 15:12:17 TP0 EP0] Decode batch, #running-req: 32, #full token: 4160768, full token usage: 0.73, mamba num: 64, mamba usage: 0.18, cuda graph: True, gen throughput (token/s): 455.88, #queue-req: 0
[qwen3.6] iter=1 elapsed=187.15s total_tok/s=22236.1
[qwen3.6] summary: total_tok/s=[22236.103594983426]
```
'''

import json
import logging
import os
import random
import time
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pynvml
import pytest
import torch

sgl = pytest.importorskip("sglang")

_LOG = logging.getLogger(__name__)
_GIB = 1 << 30


# ---------------------------------------------------------------------------
# pynvml helpers (copied from test_sgl_qwen36moe_release_resume_v4)
# ---------------------------------------------------------------------------

_NVML_INITED = False


def _ensure_nvml() -> None:
    global _NVML_INITED
    if not _NVML_INITED:
        pynvml.nvmlInit()
        _NVML_INITED = True


@dataclass(frozen=True)
class GpuMemSnapshot:
    label: str
    iter: Optional[int]
    gpu: int
    free_gib: float
    used_gib: float
    total_gib: float


def _snapshot(label: str, iter_idx: Optional[int] = None) -> List[GpuMemSnapshot]:
    _ensure_nvml()
    snaps: List[GpuMemSnapshot] = []
    for gpu in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        snap = GpuMemSnapshot(
            label=label, iter=iter_idx, gpu=gpu,
            free_gib=mem.free / _GIB, used_gib=mem.used / _GIB,
            total_gib=mem.total / _GIB,
        )
        snaps.append(snap)
        iter_str = f"iter={iter_idx}" if iter_idx is not None else "iter=-"
        print(
            f"[snapshot] phase={label} {iter_str} gpu={gpu} "
            f"free_gib={snap.free_gib:.2f} used_gib={snap.used_gib:.2f} "
            f"total_gib={snap.total_gib:.2f}",
            flush=True,
        )
    return snaps


def _print_gpu_processes(label: str) -> None:
    _ensure_nvml()
    for gpu in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        ents = ", ".join(
            f"pid={p.pid} used_gib={(p.usedGpuMemory or 0) / _GIB:.2f}"
            for p in procs
        )
        print(f"[procs] phase={label} gpu={gpu} [{ents}]", flush=True)


def _all_gpu_pids() -> Set[int]:
    _ensure_nvml()
    pids: Set[int] = set()
    for gpu in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            pids.add(p.pid)
    return pids


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Per-model throughput result
# ---------------------------------------------------------------------------


@dataclass
class ThroughputResult:
    model_name: str
    batch: int
    prompt_tokens: int
    gen_tokens: int
    total_tps_per_iter: List[float] = field(default_factory=list)
    elapsed_per_iter: List[float] = field(default_factory=list)

    @property
    def best_total_tps(self) -> float:
        return max(self.total_tps_per_iter) if self.total_tps_per_iter else 0.0


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def _run_throughput(
    model_path: str,
    model_tag: str,
    batch: int,
    prompt_tokens: int,
    gen_tokens: int,
    iters: int,
    tp_size: int,
    ep_size: int,
    mem_fraction: float,
    context_length: Optional[int] = None,
) -> ThroughputResult:
    """Load model, run random-token throughput, shutdown, return results.

    Parameters
    ----------
    context_length : int, optional
        Override the model's max context length.  When set to a value
        larger than the model's native limit (e.g. Qwen3's 40960),
        ``SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`` is automatically
        set so SGLang accepts the override.
    """

    if context_length is not None:
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    config_path = os.path.join(model_path, "config.json")
    assert os.path.isfile(config_path), (
        f"missing model config at {config_path}; "
        f"download {model_path} first"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    vocab_size = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size")
    assert isinstance(vocab_size, int) and vocab_size > 1, (
        f"cannot find a valid vocab_size in {config_path}"
    )

    rng = random.Random(20260512)

    ctx_str = f"  context_length={context_length}" if context_length else ""
    print(f"\n{'='*72}", flush=True)
    print(f"  Model: {model_tag}  ({model_path})", flush=True)
    print(f"  batch={batch}  prompt_tokens={prompt_tokens}  "
          f"gen_tokens={gen_tokens}  iters={iters}", flush=True)
    print(f"  vocab_size={vocab_size}  tp={tp_size}  ep={ep_size}  "
          f"mem_frac={mem_fraction}{ctx_str}", flush=True)
    print(f"{'='*72}", flush=True)

    port = 10000 + os.getpid() % 1000
    from gpatch_v4.generation_backend.sglang_engine import filter_server_args_kwargs
    server_args = sgl.ServerArgs(
        **filter_server_args_kwargs(
            dict(
                model_path=model_path,
                tp_size=tp_size,
                ep_size=ep_size,
                dp_size=1,
                pp_size=1,
                enable_dp_attention=False,
                dist_init_addr=f"127.0.0.1:{port}",
                nnodes=1,
                node_rank=0,
                base_gpu_id=0,
                mem_fraction_static=mem_fraction,
                trust_remote_code=True,
                context_length=context_length,
                mamba_full_memory_ratio=0.05,
                disable_cuda_graph=bool(os.environ.get("SGLANG_PROFILE_LAYERS")),
                disable_piecewise_cuda_graph=bool(
                    os.environ.get("SGLANG_PROFILE_LAYERS")
                ),
            )
        )
    )
    llm = sgl.Engine(server_args=server_args)

    result = ThroughputResult(
        model_name=model_tag,
        batch=batch,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
    )

    try:
        time.sleep(1)
        _snapshot(f"after_engine_init_{model_tag}")

        total_prompt = prompt_tokens * batch
        total_gen = gen_tokens * batch

        for it in range(1, iters + 1):
            batched_input_ids = [
                torch.randint(0, vocab_size, (prompt_tokens,)).tolist()
                for _ in range(batch)
            ]
            batched_sp = [
                {
                    "temperature": 0.0,
                    "min_new_tokens": gen_tokens,
                    "max_new_tokens": gen_tokens,
                    "ignore_eos": True,
                }
                for _ in range(batch)
            ]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = llm.generate(
                input_ids=batched_input_ids,
                sampling_params=batched_sp,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            assert isinstance(outputs, list) and len(outputs) == batch, (
                f"[{model_tag}] iter {it}: expected {batch} outputs, "
                f"got {len(outputs) if isinstance(outputs, list) else type(outputs)}"
            )
            for oi, output in enumerate(outputs):
                got = len(output["output_ids"])
                assert abs(got - gen_tokens) <= 1, (
                    f"[{model_tag}] iter {it} output[{oi}]: expected "
                    f"~{gen_tokens} tokens, got {got}"
                )

            total_tps = (total_prompt + total_gen) / elapsed
            result.total_tps_per_iter.append(total_tps)
            result.elapsed_per_iter.append(elapsed)

            print(
                f"[{model_tag}] iter={it} elapsed={elapsed:.2f}s "
                f"total_tok/s={total_tps:.1f}",
                flush=True,
            )

        print(
            f"[{model_tag}] summary: "
            f"total_tok/s={result.total_tps_per_iter}",
            flush=True,
        )

    finally:
        engine_pids = _all_gpu_pids() - {os.getpid()}
        llm.shutdown()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            live = _all_gpu_pids() & engine_pids
            if not live:
                break
            time.sleep(0.5)
        _snapshot(f"after_shutdown_{model_tag}")
        _print_gpu_processes(f"after_shutdown_{model_tag}")

    return result


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------


def _print_comparison(r35: ThroughputResult, r36: ThroughputResult) -> None:
    W = 72
    print(f"\n{'='*W}", flush=True)
    print("  Throughput Comparison: Qwen3 vs Qwen3.6", flush=True)
    print(f"{'='*W}", flush=True)

    hdr = (f"{'metric':>20} | {'Qwen3':>14} | {'Qwen3.6':>14} | "
           f"{'ratio (3.6/3)':>16}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    pairs = [
        ("best total tok/s", r35.best_total_tps, r36.best_total_tps),
    ]
    for i, (t35, t36) in enumerate(
        zip(r35.total_tps_per_iter, r36.total_tps_per_iter)
    ):
        pairs.append((f"total tok/s iter {i+1}", t35, t36))
    for i, (e35, e36) in enumerate(
        zip(r35.elapsed_per_iter, r36.elapsed_per_iter)
    ):
        pairs.append((f"elapsed (s) iter {i+1}", e35, e36))

    for label, v35, v36 in pairs:
        ratio = v36 / v35 if v35 > 0 else float("inf")
        print(
            f"{label:>20} | {v35:>14.1f} | {v36:>14.1f} | {ratio:>13.2f}x",
            flush=True,
        )
    print(flush=True)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    torch.cuda.device_count() >= 8,
    f"needs 8 GPUs, have {torch.cuda.device_count()}",
)
class Qwen3vsQwen36ThroughputTest(unittest.TestCase):
    """Side-by-side throughput comparison with random-token workloads."""

    def test_throughput_comparison(self):
        model_3 = _env_str("QWEN3_MODEL_PATH", "hf-hub/Qwen/Qwen3-30B-A3B")
        model_36 = _env_str("QWEN36_MODEL_PATH", "hf-hub/Qwen/Qwen3.6-35B-A3B")
        batch = _env_int("CMP_BATCH", 8)
        prompt_tokens = _env_int("CMP_PROMPT_TOKENS", 126 * 1024)
        gen_tokens = _env_int("CMP_GEN_TOKENS", 1024)
        iters = _env_int("CMP_ITERS", 1)
        mem_fraction = _env_float("CMP_MEM_FRACTION", 0.7)
        tp_size = _env_int("CMP_TP_SIZE", 4)
        ep_size = _env_int("CMP_EP_SIZE", 4)
        context_length = _env_int("CMP_CONTEXT_LENGTH", 131072)

        common_kw = dict(
            batch=batch,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            iters=iters,
            tp_size=tp_size,
            ep_size=ep_size,
            mem_fraction=mem_fraction,
            context_length=context_length,
        )

        # r35 = _run_throughput(model_3, "qwen3", **common_kw)
        r36 = _run_throughput(model_36, "qwen3.6", **common_kw)
        # _print_comparison(r35, r36)


if __name__ == "__main__":
    unittest.main()
