# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com

from __future__ import annotations

import shutil
import subprocess
import time
import unittest
from pathlib import Path

import ray
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.orches.placement_group import _create_placement_group
from gpatch_v4.orches.utils import build_actor_env_vars

from test_sp_overlap import (
    ULYSSES_CP_SIZES,
    ULYSSES_WARMUP_STEPS,
    WORLD_SIZE,
    _FA3_AVAILABLE,
    _assert_ulysses_worker_results,
    _get_master_endpoint,
    _profile_ulysses_worker,
)

_NSYS_MARKER = "ulysses_e2e_overlap"


def _wait_for_report(profile_dir: Path, timeout: float = 60.0) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reports = list(profile_dir.glob("ulysses_rank0_*.nsys-rep"))
        if len(reports) == 1:
            return reports[0]
        time.sleep(0.5)
    raise AssertionError(f"expected one nsys report under {profile_dir}")


@unittest.skipUnless(_FA3_AVAILABLE, "flash_attn_interface (FA3) not installed")
class Test1(unittest.TestCase):
    def setUp(self) -> None:
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"need >= {WORLD_SIZE} GPUs in Ray cluster, only {total_gpus}")

    def tearDown(self) -> None:
        kill_all_actors_and_shutdown_ray()

    def test_ulysses_e2e_timeline(self) -> None:
        # yum cuda 12.9 自带的 nsys 有问题（TargetProfilingFailed），需手动升级到 2026.3。
        self.skipTest("nsys TargetProfilingFailed on current cluster CUDA image")
        if shutil.which("nsys") is None:
            self.skipTest("nsys is unavailable")
        self.assertGreaterEqual(ULYSSES_WARMUP_STEPS, 0)
        profile_dir = Path.cwd() / "ulysses_sp_overlap"
        profile_dir.mkdir(exist_ok=True)
        for path in profile_dir.glob("ulysses_rank0_*"):
            path.unlink()
        placement_group, bundle_indices = _create_placement_group(WORLD_SIZE)

        try:
            master_addr, master_port = ray.get(
                _get_master_endpoint.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[0],
                    )
                ).remote()
            )
            nsight_config = {
                "t": "cuda,nccl,cudnn,cublas,nvtx,osrt",
                "capture-range": "cudaProfilerApi",
                "capture-range-end": "stop",
                "flush-on-cudaprofilerstop": "true",
                "force-overwrite": "true",
                "o": str(profile_dir / "ulysses_rank0_%p"),
            }
            actor_env_vars = build_actor_env_vars()
            futures = [
                _profile_ulysses_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[rank],
                    ),
                    runtime_env={
                        "env_vars": actor_env_vars,
                        **({"nsight": nsight_config} if rank == 0 else {}),
                    },
                ).remote(
                    rank,
                    WORLD_SIZE,
                    master_addr,
                    master_port,
                    True,
                ) for rank in range(WORLD_SIZE)
            ]
            worker_results = ray.get(futures)
            report_path = _wait_for_report(profile_dir)
        finally:
            remove_placement_group(placement_group)

        self.assertTrue(report_path.is_file())
        self.assertGreater(report_path.stat().st_size, 0)
        stats = subprocess.run(
            [
                "nsys",
                "stats",
                "--report=cuda_gpu_kern_sum,nvtx_sum",
                "--format=csv",
                "--force-export=true",
                str(report_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        stats_output = stats.stdout + stats.stderr
        if stats.returncode != 0 or "TargetProfilingFailed" in stats_output:
            self.skipTest(
                f"nsys stats/profile failed on this cluster image: {stats_output[:500]}"
            )
        # nsys may mention "SKIPPED:" mid-line while converting sqlite; only
        # treat a line that starts with SKIPPED: as a real report skip.
        skipped_lines = [
            ln for ln in stats_output.splitlines() if ln.lstrip().startswith("SKIPPED:")
        ]
        self.assertFalse(skipped_lines, stats_output)
        self.assertIn(f"{_NSYS_MARKER}_cp{ULYSSES_CP_SIZES[-1]}", stats_output)
        self.assertIn("nccl", stats_output.lower())

        _assert_ulysses_worker_results(self, worker_results)
        print(f"  nsys: {report_path}")
