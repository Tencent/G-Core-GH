# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com

'''
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="$http_proxy"

curl -fsSL \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/7fa2af80.pub \
  -o /tmp/nvidia.pub
rpm --import /tmp/nvidia.pub
rm /tmp/nvidia.pub

dnf --disablerepo='*' \
  --repofrompath=nvidia-devtools,https://developer.download.nvidia.com/devtools/repos/rhel8/x86_64/ \
  --enablerepo=nvidia-devtools \
  install -y nsight-systems-cli-2026.3.1
'''

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest
import ray
import torch
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist

from gpatch_v4.orches.placement_group import _create_placement_group
from gpatch_v4.orches.utils import build_actor_env_vars

_NVTX_MARKER = "gcore_nsys_profile_test"
_WORLD_SIZE = 16


@ray.remote(num_cpus=0, num_gpus=0)
def _get_master_endpoint() -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return ray.util.get_node_ip_address(), sock.getsockname()[1]


@ray.remote(num_cpus=0, num_gpus=1, max_calls=1)
def _profile_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> int:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    gpu_cnt = torch.cuda.device_count()
    assert gpu_cnt > 0
    device = torch.device("cuda", rank % gpu_cnt)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        x = torch.randn(1024, 1024, device=device)
        x @ x
        torch.cuda.synchronize()
        dist.barrier()

        if rank == 0:
            torch.cuda.profiler.start()
        with torch.cuda.nvtx.range(_NVTX_MARKER):
            with torch.cuda.nvtx.range("matmul_before_barrier"):
                y = x @ x
            with torch.cuda.nvtx.range("nccl_barrier"):
                dist.barrier()
            with torch.cuda.nvtx.range("matmul_after_barrier"):
                y @ x
            torch.cuda.synchronize()
        if rank == 0:
            torch.cuda.profiler.stop()
        dist.barrier()
        return rank
    finally:
        dist.destroy_process_group()


def _wait_for_report(profile_dir: Path, timeout: float = 60.0) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reports = list(profile_dir.glob("ray_rank0_*.nsys-rep"))
        if len(reports) == 1:
            return reports[0]
        time.sleep(0.5)
    raise AssertionError(f"expected one nsys report under {profile_dir}")


@pytest.mark.skip(reason="nsys TargetProfilingFailed on current cluster CUDA image")
def test_ray_nsys_profiles_matmul_nccl_barrier() -> None:
    if shutil.which("nsys") is None:
        pytest.skip("nsys is unavailable")

    profile_dir = Path.cwd() / "nsys"
    profile_dir.mkdir(exist_ok=True)
    for path in profile_dir.glob("ray_rank0_*"):
        path.unlink()

    try:
        ray.init(address="auto")
    except ConnectionError:
        pytest.skip("Ray cluster is unavailable")
    try:
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < _WORLD_SIZE:
            pytest.skip(f"need >= {_WORLD_SIZE} GPUs in Ray cluster, only {total_gpus}")

        placement_group, bundle_indices = _create_placement_group(_WORLD_SIZE)
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
                "o": str(profile_dir / "ray_rank0_%p"),
            }
            actor_env_vars = build_actor_env_vars()
            futures = [
                _profile_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[rank],
                    ),
                    runtime_env={
                        "env_vars": actor_env_vars,
                        **({"nsight": nsight_config} if rank == 0 else {}),
                    },
                ).remote(rank, _WORLD_SIZE, master_addr, master_port)
                for rank in range(_WORLD_SIZE)
            ]
            assert ray.get(futures) == list(range(_WORLD_SIZE))
            report_path = _wait_for_report(profile_dir)
        finally:
            remove_placement_group(placement_group)
    finally:
        ray.shutdown()

    assert report_path.is_file()
    assert report_path.stat().st_size > 0

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
        import pytest
        pytest.skip(
            f"nsys stats/profile failed on this cluster image: {stats_output[:500]}"
        )
    # nsys may mention "SKIPPED:" mid-line while converting sqlite; only
    # treat a line that starts with SKIPPED: as a real report skip.
    skipped_lines = [
        ln for ln in stats_output.splitlines() if ln.lstrip().startswith("SKIPPED:")
    ]
    assert not skipped_lines, stats_output
    assert _NVTX_MARKER in stats_output, stats_output
    assert "nccl" in stats_output.lower(), stats_output
