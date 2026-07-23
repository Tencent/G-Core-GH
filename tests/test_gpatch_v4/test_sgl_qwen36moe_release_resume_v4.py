"""Per-iteration GPU memory leak / budget probe for SGLang + Qwen3.6-35B-A3B.

This test loads ``Qwen/Qwen3.6-35B-A3B`` (MoE) under SGLang with TP=8, EP=8,
and ``mem_fraction_static=0.7`` then runs a release/resume cycle around
``QWEN36_MEM_TEST_ITERS`` iterations (default 2) of
(64k-token prompt x batch=4, forced 16k-token decode).
After each phase we capture per-GPU memory via NVML (driver-level, not
torch process-level, because the SGLang Engine spawns subprocesses that
the driver process cannot introspect).

Four classes of hard assertion fire on regressions:

1. **Decode validity** -- every request in every iter (incl. warmup)
   must produce exactly ``GEN_TOKENS`` output tokens. An EOS-driven
   short decode would deflate KV pressure and invalidate the leak
   signal, so we fail fast instead.
2. **Release effectiveness** -- each iter's release must drop per-GPU
   ``used`` by at least ``RELEASE_MIN_DROP_GIB``.
3. **Per-iter growth** -- per-GPU ``used`` on each of ``post_gen``,
   ``post_resume``, and ``post_release`` snapshots must never exceed
   the running high-water mark by more than ``GROWTH_TOL_GIB``.
   Only upward drift is flagged; transient under-readings (e.g. a
   ``post_resume`` sample taken before sglang's async KV-pool
   reallocation finishes on every TP rank) are treated as benign noise
   and ignored. Sustained leaks still trip the assertion because each
   step contributes at most ``GROWTH_TOL_GIB`` to the high-water mark.
4. **Budget overrun** -- each ``post_gen``/``post_resume`` snapshot must
   stay within ``baseline.free * mem_fraction_static + budget_tol``
   per GPU. ``mem_fraction_static`` is defined relative to *free at
   engine init* (see
   ``sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:171-173``),
   not total VRAM, hence the assertion is expressed as a delta from the
   ``baseline`` snapshot taken before ``sgl.Engine(...)``.

Output is a one-line-per-snapshot ``[snapshot] phase=... iter=... gpu=...
free_gib=... used_gib=... total_gib=...`` format, plus per-device
running PIDs from ``nvmlDeviceGetComputeRunningProcesses`` for triage.

NOTE: WeLM-v4-only ServerArgs (``enable_kv_mirror``,
``enable_over_encoding``, ``enable_return_routed_experts``,
``disable_piecewise_cuda_graph``) are intentionally omitted; they belong
to the WeLM internal pipeline, not vanilla Qwen3 MoE inference.

Manual-run only -- not in CI. Requires:
- 8 GPUs visible to this process
- ~70 GiB pinned host RAM (``enable_weights_cpu_backup=True`` parks
  weights to pinned host memory on release).
- Wall-clock ~12-20 minutes.

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=3600 \\
        tests/test_gpatch_v4/test_sgl_qwen36moe_release_resume_v4.py
"""

import json
import logging
import os
import random
import time
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import pynvml
import pytest
import torch
from transformers import AutoTokenizer

from random_text import generate_words

sgl = pytest.importorskip("sglang")

_LOG = logging.getLogger(__name__)
_GIB = 1 << 30

# ---------------------------------------------------------------------------
# pynvml helpers
# ---------------------------------------------------------------------------

_NVML_INITED = False


def _ensure_nvml() -> None:
    """Lazy-init pynvml; never shut down (process-local, reaped on exit)."""
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
    """Capture per-GPU NVML memory and emit a grep-able log line per device.

    Iterates ``range(torch.cuda.device_count())`` so ``CUDA_VISIBLE_DEVICES``
    remapping is honored (NVML index = visible index for this process).
    """
    _ensure_nvml()
    snaps: List[GpuMemSnapshot] = []
    n = torch.cuda.device_count()
    for gpu in range(n):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        snap = GpuMemSnapshot(
            label=label,
            iter=iter_idx,
            gpu=gpu,
            free_gib=mem.free / _GIB,
            used_gib=mem.used / _GIB,
            total_gib=mem.total / _GIB,
        )
        snaps.append(snap)
        iter_str = f"iter={iter_idx}" if iter_idx is not None else "iter=-"
        line = (
            f"[snapshot] phase={label} {iter_str} gpu={gpu} "
            f"free_gib={snap.free_gib:.2f} used_gib={snap.used_gib:.2f} "
            f"total_gib={snap.total_gib:.2f}"
        )
        print(line, flush=True)
        _LOG.info(line)
    return snaps


def _print_gpu_processes(label: str) -> None:
    """Print per-device running compute PIDs for triage."""
    _ensure_nvml()
    n = torch.cuda.device_count()
    for gpu in range(n):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        ents = ", ".join(f"pid={p.pid} used_gib={(p.usedGpuMemory or 0) / _GIB:.2f}" for p in procs)
        print(f"[procs] phase={label} gpu={gpu} [{ents}]", flush=True)


def _all_gpu_pids() -> Set[int]:
    _ensure_nvml()
    pids = set()
    for gpu in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            pids.add(p.pid)
    return pids


# ---------------------------------------------------------------------------
# Env-tunable knobs
# ---------------------------------------------------------------------------


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    torch.cuda.device_count() >= 8,
    f"needs 8 GPUs, have {torch.cuda.device_count()}",
)
class Qwen36MoEReleaseResumeMemoryLeakTest(unittest.TestCase):
    """Per-iter NVML probe of SGLang release/resume on Qwen3.6-35B-A3B."""

    REPO_ID = "hf-hub/Qwen/Qwen3.6-35B-A3B"
    MEM_FRACTION_STATIC = 0.7

    def test_release_resume_memory_leak(self):
        iters = _env_int("QWEN36_MEM_TEST_ITERS", 2)
        batch = _env_int("QWEN36_MEM_TEST_BATCH", 4)
        prompt_tokens = _env_int("QWEN36_MEM_TEST_PROMPT_TOKENS", 65536)
        gen_tokens = _env_int("QWEN36_MEM_TEST_GEN_TOKENS", 16384)
        growth_tol = _env_float("QWEN36_MEM_TEST_GROWTH_TOL_GIB", 3.0)
        budget_tol = _env_float("QWEN36_MEM_TEST_BUDGET_TOL_GIB", 8.0)
        release_min_drop = _env_float("QWEN36_MEM_TEST_RELEASE_MIN_DROP_GIB", 15.0)

        assert iters >= 1, f"ITERS must be >= 1, got {iters}"

        config_path = os.path.join(self.REPO_ID, "config.json")
        assert os.path.isfile(config_path), (
            f"missing model config at {config_path}; download "
            f"{self.REPO_ID} into hf-hub/ first"
        )

        # Must be set before sgl.Engine(...) so cuda graph buffers are
        # tracked by the memory saver and freed on release_memory_occupation.
        # Subprocesses inherit os.environ at spawn.
        _prev_mem_saver_cg = os.environ.get("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = "1"

        baseline_snaps = _snapshot("baseline")
        baseline_free = {s.gpu: s.free_gib for s in baseline_snaps}

        tokenizer = AutoTokenizer.from_pretrained(self.REPO_ID, trust_remote_code=True)

        words = generate_words(128 * 1024)
        ids = tokenizer(words, add_special_tokens=False)["input_ids"]
        assert len(ids) >= prompt_tokens, (
            f"generate_words produced only {len(ids)} tokens; need {prompt_tokens}. "
            f"Increase the word count."
        )
        prompt_ids: List[int] = ids[:prompt_tokens]
        sampling_params: Dict = {
            "temperature": 1.0,
            "top_p": 0.9,
            "min_new_tokens": gen_tokens,
            "max_new_tokens": gen_tokens,
            # Force exact GEN_TOKENS decode: ignore_eos disables EOS-stop
            # entirely. min_new_tokens alone is insufficient because its
            # penalizer only masks tokenizer.eos_token_id (singular),
            # while Qwen3 generation_config typically lists multiple EOS
            # ids (e.g. <|im_end|>=151645) that would otherwise short-decode.
            "ignore_eos": True,
        }
        batched_input_ids: List[List[int]] = [prompt_ids] * batch
        batched_sp: List[Dict] = [dict(sampling_params) for _ in range(batch)]

        port = 10000 + os.getpid() % 1000
        server_args = sgl.ServerArgs(
            model_path=self.REPO_ID,
            tp_size=8,
            ep_size=8,
            dp_size=1,
            pp_size=1,
            enable_dp_attention=False,
            dist_init_addr=f"127.0.0.1:{port}",
            nnodes=1,
            node_rank=0,
            base_gpu_id=0,
            enable_memory_saver=True,
            enable_weights_cpu_backup=True,
            mem_fraction_static=self.MEM_FRACTION_STATIC,
            trust_remote_code=True,
        )
        llm = sgl.Engine(server_args=server_args)

        try:
            time.sleep(2)  # let NCCL/cuda graph capture settle before snapshot
            _snapshot("after_engine_init")
            _print_gpu_processes("after_engine_init")

            outputs = llm.generate(input_ids=batched_input_ids, sampling_params=batched_sp)
            assert isinstance(
                outputs, list
            ), (f"expected list of dicts from batched generate; got {type(outputs)}")
            assert len(outputs) == batch, (f"expected {batch} outputs, got {len(outputs)}")
            for oi, output in enumerate(outputs):
                got = len(output["output_ids"])
                assert got == gen_tokens, (
                    f"warmup output[{oi}]: expected {gen_tokens} tokens, got {got}"
                )
            _snapshot("after_warmup_first_gen")

            post_gen: Dict[int, List[GpuMemSnapshot]] = {}
            post_release: Dict[int, List[GpuMemSnapshot]] = {}
            post_resume: Dict[int, List[GpuMemSnapshot]] = {}

            for it in range(1, iters + 1):
                _snapshot("pre_gen", it)

                outputs = llm.generate(input_ids=batched_input_ids, sampling_params=batched_sp)
                assert len(outputs
                          ) == batch, (f"iter {it}: expected {batch} outputs, got {len(outputs)}")
                for oi, output in enumerate(outputs):
                    got = len(output["output_ids"])
                    assert got == gen_tokens, (
                        f"iter {it} output[{oi}]: expected {gen_tokens} "
                        f"tokens, got {got} (EOS short-decode would invalidate "
                        f"the leak signal)"
                    )

                post_gen[it] = _snapshot("post_gen", it)

                llm.release_memory_occupation()
                time.sleep(1.0)  # let allocator settle (cumem unmap is sync but cheap)
                post_release[it] = _snapshot("post_release", it)

                llm.resume_memory_occupation()
                # ``resume_memory_occupation`` returns once every TP rank
                # has acked, but the cuda alloc backing the KV pool may
                # still be in flight at the driver level (NVML reads the
                # driver, not the cuda runtime); sample after a short
                # settle so we don't catch a transient under-reading.
                time.sleep(2.0)
                post_resume[it] = _snapshot("post_resume", it)

            self._assert_release_drop(post_gen, post_release, release_min_drop)
            self._assert_growth(post_gen, "post_gen", growth_tol)
            self._assert_growth(post_release, "post_release", growth_tol)
            self._assert_growth(post_resume, "post_resume", growth_tol)
            self._assert_budget(post_gen, post_resume, baseline_free, budget_tol)

        finally:
            # Restore env to avoid polluting subsequent tests in the same
            # pytest process (SGLANG_MEMORY_SAVER_CUDA_GRAPH=1 would cause
            # later SGLang engines to attempt pauseable CUDA graph capture
            # without the required LD_PRELOAD, crashing the scheduler).
            if _prev_mem_saver_cg is None:
                os.environ.pop("SGLANG_MEMORY_SAVER_CUDA_GRAPH", None)
            else:
                os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = _prev_mem_saver_cg

            engine_pids = _all_gpu_pids() - {os.getpid()}
            llm.shutdown()
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                live = _all_gpu_pids() & engine_pids
                if not live:
                    break
                time.sleep(0.5)
            _snapshot("after_shutdown")
            _print_gpu_processes("after_shutdown")

    # -- assertions --------------------------------------------------------

    def _assert_release_drop(
        self,
        post_gen: Dict[int, List[GpuMemSnapshot]],
        post_release: Dict[int, List[GpuMemSnapshot]],
        min_drop: float,
    ) -> None:
        """Each release must reclaim at least ``min_drop`` GiB per GPU."""
        for it in sorted(post_gen):
            for g_snap, r_snap in zip(post_gen[it], post_release[it]):
                assert g_snap.gpu == r_snap.gpu
                drop = g_snap.used_gib - r_snap.used_gib
                assert drop >= min_drop, (
                    f"iter {it} gpu {g_snap.gpu}: release freed only "
                    f"{drop:.2f} GiB; expected >= {min_drop:.2f}. "
                    f"post_gen.used={g_snap.used_gib:.2f} "
                    f"post_release.used={r_snap.used_gib:.2f}"
                )

    def _assert_growth(
        self,
        snaps_by_iter: Dict[int, List[GpuMemSnapshot]],
        axis_name: str,
        growth_tol: float,
    ) -> None:
        """Per-GPU ``used`` on this axis must not exceed running max + tol.

        We only flag *upward* drift -- a single low NVML sample is treated
        as benign noise (e.g. ``post_resume`` snapshots can be taken
        before sglang's async KV-pool reallocation across all TP ranks
        finishes, producing transient under-readings; this is not a leak).

        For each GPU we walk iters in order and require::

            used[it] <= max(used[1..it]) + growth_tol

        i.e. the high-water mark may grow by at most ``growth_tol``
        between consecutive iters. This is symmetric in the sense that
        sustained upward drift across many iters still trips the
        assertion (each step contributes at most ``growth_tol``), while
        isolated dips are ignored.
        """
        # Per-GPU running max across iters.
        gpu_running_max: Dict[int, float] = {}
        for it in sorted(snaps_by_iter):
            for snap in snaps_by_iter[it]:
                prev_max = gpu_running_max.get(snap.gpu, snap.used_gib)
                allowed = prev_max + growth_tol
                assert snap.used_gib <= allowed, (
                    f"{axis_name} leak: iter {it} gpu {snap.gpu} used "
                    f"{snap.used_gib:.2f} GiB exceeds prior high-water "
                    f"{prev_max:.2f} + tol {growth_tol:.2f} = "
                    f"{allowed:.2f} GiB"
                )
                gpu_running_max[snap.gpu] = max(prev_max, snap.used_gib)

    def _assert_budget(
        self,
        post_gen: Dict[int, List[GpuMemSnapshot]],
        post_resume: Dict[int, List[GpuMemSnapshot]],
        baseline_free: Dict[int, float],
        budget_tol: float,
    ) -> None:
        """Used delta from baseline must stay within mem_fraction_static budget.

        ``mem_fraction_static`` budgets ``(weights + KV pool)`` against
        free-at-init (cf.
        ``sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:171-173``).
        Activations + cuda graph + NCCL workspace + CUDA context add on
        top, hence ``budget_tol`` (default 8 GiB; empirical, env-tunable
        if real overhead differs).
        """
        for snaps_by_iter, label in [(post_gen, "post_gen"), (post_resume, "post_resume")]:
            for it in sorted(snaps_by_iter):
                for snap in snaps_by_iter[it]:
                    delta_used = baseline_free[snap.gpu] - snap.free_gib
                    budget = baseline_free[snap.gpu] * self.MEM_FRACTION_STATIC + budget_tol
                    assert delta_used <= budget, (
                        f"budget overrun: {label} iter {it} gpu {snap.gpu} "
                        f"consumed {delta_used:.2f} GiB > "
                        f"{self.MEM_FRACTION_STATIC}*{baseline_free[snap.gpu]:.2f} "
                        f"+ {budget_tol:.2f} = {budget:.2f} GiB"
                    )


if __name__ == "__main__":
    unittest.main()
