"""Full-run sync vs async (staleness=0) alignment comparison.

Run separately — each test needs its own Ray cluster:
  pytest -v -s --timeout=86400 tests/test_gpatch_v4/test_fullrun_comparison.py::FullRunTest::test_sync
  pytest -v -s --timeout=86400 tests/test_gpatch_v4/test_fullrun_comparison.py::FullRunTest::test_async

test_async will automatically invoke my_test_compare() at the end to verify
bit-exact alignment of rollout data and training metrics.
"""
import math
import os
import shutil
import unittest

import ray
import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.trainer import GrpoSingleCtrlTrainer, GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)

METRICS_DIR = "/tmp/fullrun_metrics"
DATASET_N_LINES = 4


def _prepare_dataset(n_lines=DATASET_N_LINES):
    """Truncate GSM8K training data to *n_lines* for faster iteration."""
    tmp_dir = "tests/test_gpatch_v4/gsm8k_small"
    os.makedirs(tmp_dir, exist_ok=True)
    src = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
    dst = os.path.join(tmp_dir, "train.jsonl")
    with open(src, 'r') as fin, open(dst, 'w') as fout:
        for i, line in enumerate(fin):
            if i >= n_lines:
                break
            fout.write(line)


def _save_metrics(name, metrics):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, f"{name}.pt")
    torch.save(metrics, path)
    print(f"\nMetrics saved to {path} ({len(metrics[0])} steps)")


def _rename_debug_dir(tag):
    """Rename debug-tmp/ -> debug-tmp-{tag}/ to separate sync/async outputs."""
    src = "debug-tmp"
    dst = f"debug-tmp-{tag}"
    if os.path.exists(src):
        shutil.rmtree(dst, ignore_errors=True)
        os.rename(src, dst)
        print(f"Renamed {src}/ -> {dst}/")


def _compare_values(sv, av):
    """Compare two values element-wise.

    Returns
    -------
    bool
        True if identical.
    """
    if isinstance(sv, torch.Tensor) and isinstance(av, torch.Tensor):
        return torch.equal(sv, av)
    if isinstance(sv, list) and isinstance(av, list):
        if len(sv) != len(av):
            return False
        return all(_compare_values(s, a) for s, a in zip(sv, av))
    return sv == av


def _metrics_close(sv, av, rtol=1e-6, atol=1e-8):
    """Approximate comparison for scalar training metrics.

    Allows tiny floating-point drift from non-deterministic reduction order
    across ranks, while still catching real divergences.
    """
    if isinstance(sv, float) and isinstance(av, float):
        return math.isclose(sv, av, rel_tol=rtol, abs_tol=atol)
    return sv == av


@requires_sglang
class FullRunTest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        if ray.is_initialized():
            kill_all_actors_and_shutdown_ray()

    async def run_sync(self, backend="sglang"):
        config = load_config('test_math_rl_sync_fullrun', RlConfig)
        config.sampler.backend = backend
        assert config.placement_type == "colocate"

        _prepare_dataset()
        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        shutil.rmtree("debug-tmp", ignore_errors=True)
        try:
            trainer = GrpoTrainer()
            metrics = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
            _rename_debug_dir("sync")

        _save_metrics("fullrun_sync", metrics)

    async def run_async(self, backend="sglang"):
        config = load_config('test_math_rl_async_fullrun', RlConfig)
        config.sampler.backend = backend
        assert config.placement_type == "disaggregated"
        assert config.training.async_rollout
        assert config.training.rollout_max_staleness == 0

        _prepare_dataset()
        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        shutil.rmtree("debug-tmp", ignore_errors=True)
        try:
            trainer = GrpoSingleCtrlTrainer()
            metrics = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
            _rename_debug_dir("async")

        _save_metrics("fullrun_async", metrics)

    async def test_cmp(self):
        # This test verifies bit-exact alignment between sync and async modes.
        # Only sglang is tested here because the comparison requires deterministic
        # inference results across colocate/disaggregated placements, which vllm
        # does not guarantee.
        for d in ("debug-tmp", "debug-tmp-sync", "debug-tmp-async"):
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(METRICS_DIR, ignore_errors=True)

        await self.run_sync()
        kill_all_actors_and_shutdown_ray()
        await self.run_async()
        self.my_test_compare()

    def my_test_compare(self):
        """Compare rollout and metrics data saved by run_sync and run_async.

        Collects all rollout .pt files per PPO step, merges samples across
        ranks/batches, sorts by token content, then compares.  This handles
        the fact that sync and async may distribute samples across ranks in
        different orders.

        Also compares training metrics (loss, grad_norm, etc.) step by step.
        """
        sync_dir = "debug-tmp-sync"
        async_dir = "debug-tmp-async"
        assert os.path.isdir(sync_dir
                            ), (f"{sync_dir}/ not found - run_sync did not produce debug output")
        assert os.path.isdir(async_dir
                            ), (f"{async_dir}/ not found - run_async did not produce debug output")

        mismatches = []

        # --- Compare rollout_batches: merge per step, sort, then compare ---
        def _collect_step_samples(directory, step_prefix):
            """Load all rollout files for a step and flatten into a list of
            per-sample dicts, sorted by token content for order-invariant
            comparison."""
            files = sorted(
                f for f in os.listdir(directory) if f.startswith(step_prefix) and f.endswith(".pt")
            )
            samples = []
            for fname in files:
                batches = torch.load(
                    os.path.join(directory, fname),
                    map_location="cpu",
                    weights_only=False,
                )
                for batch in batches:
                    keys = sorted(batch.keys())
                    n = len(batch[keys[0]])
                    for i in range(n):
                        sample = {k: batch[k][i] for k in keys}
                        samples.append(sample)
            return samples

        def _sort_key(sample):
            """Sort key based on token content."""
            tok = sample.get("tokens")
            if isinstance(tok, torch.Tensor):
                return tuple(tok.tolist())
            return tuple(tok) if tok is not None else ()

        # Find all PPO steps present
        sync_files = sorted(os.listdir(sync_dir))
        step_prefixes = sorted(
            set(
                "_".join(f.split("_")[:3])  # "rollout_batches_N"
                for f in sync_files if f.startswith("rollout_batches_")
            )
        )
        assert step_prefixes, f"no rollout_batches_*.pt in {sync_dir}/"

        for prefix in step_prefixes:
            sync_samples = sorted(_collect_step_samples(sync_dir, prefix), key=_sort_key)
            async_samples = sorted(_collect_step_samples(async_dir, prefix), key=_sort_key)

            if len(sync_samples) != len(async_samples):
                mismatches.append(
                    f"{prefix}: sample count mismatch {len(sync_samples)} vs {len(async_samples)}"
                )
                continue

            step_ok = True
            for si, (ss, as_) in enumerate(zip(sync_samples, async_samples)):
                common_keys = sorted(set(ss.keys()) & set(as_.keys()))
                for key in common_keys:
                    if not _compare_values(ss[key], as_[key]):
                        step_ok = False
                        mismatches.append(f"{prefix} sample {si} key {key}")

            print(f"  {'OK' if step_ok else 'FAIL'} {prefix} ({len(sync_samples)} samples)")

        # --- Compare training metrics from saved .pt ---
        sync_pt = os.path.join(METRICS_DIR, "fullrun_sync.pt")
        async_pt = os.path.join(METRICS_DIR, "fullrun_async.pt")
        if os.path.exists(sync_pt) and os.path.exists(async_pt):
            sync_m = torch.load(sync_pt, map_location="cpu", weights_only=False)
            async_m = torch.load(async_pt, map_location="cpu", weights_only=False)
            # Compare dp_rank 0 metrics step by step
            for step_i, (sm, am) in enumerate(zip(sync_m[0], async_m[0])):
                for key in sorted(sm.keys()):
                    if key.startswith("time_perf/"):
                        continue  # skip timing
                    if not _metrics_close(sm[key], am[key]):
                        mismatches.append(
                            f"metrics step {step_i} {key}: sync={sm[key]} async={am[key]}"
                        )
            print(
                f"  {'OK' if not any('metrics step' in m for m in mismatches) else 'FAIL'} training metrics"
            )

            # Sanity check: advantages should be non-trivial (warn only, small
            # datasets may produce all-identical rewards for some steps)
            for step_i, sm in enumerate(sync_m[0]):
                adv_std = sm.get("ppo-metrics/global_advantages_std", 0)
                if adv_std == 0:
                    print(
                        f"  WARN sync step {step_i}: advantages_std == 0, "
                        f"sampling may be producing identical outputs"
                    )
            for step_i, am in enumerate(async_m[0]):
                adv_std = am.get("ppo-metrics/global_advantages_std", 0)
                if adv_std == 0:
                    print(
                        f"  WARN async step {step_i}: advantages_std == 0, "
                        f"sampling may be producing identical outputs"
                    )

        # --- Summary ---
        print(f"\n{'='*60}")
        if mismatches:
            print(f"FAILED: {len(mismatches)} mismatches:")
            for m in mismatches:
                print(f"   - {m}")
        else:
            print("ALL IDENTICAL - sync and async outputs match")
        print(f"{'='*60}")

        assert not mismatches, (
            f"{len(mismatches)} field(s) differ between sync and async: " +
            ", ".join(mismatches[:10])
        )
