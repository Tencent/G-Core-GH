"""Lightweight profiling helpers (device-agnostic torch/NPU profiler)."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from ..core.device import get_device_perf_activity, get_profiler_module


class DummyProfiler:
    """No-op profiler used when profiling is disabled (matches the profiler API)."""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def step(self):
        pass


def profile_ctx(args):
    """Return a profiler context manager driven by ``args``.

    When ``args.profile`` is truthy, returns a device-appropriate
    ``torch.profiler``/``torch_npu.profiler`` profiler that exports the first few
    steps as Chrome traces under ``args.profile_dir`` (default ``./profile_traces``),
    one file per rank. Otherwise returns a :class:`DummyProfiler` no-op.
    """
    profile_module = get_profiler_module()

    if args.profile:

        def trace_handler(p):
            if p.step_num < 20:
                profile_dir = getattr(args, 'profile_dir', './profile_traces')
                os.makedirs(profile_dir, exist_ok=True)
                p.export_chrome_trace(
                    f"{profile_dir}/trace_rank_{dist.get_rank()}_step_{p.step_num}.json"
                )

        activities = [get_device_perf_activity(), torch.profiler.ProfilerActivity.CPU]

        return profile_module.profile(
            activities=activities,
            schedule=profile_module.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=trace_handler,
        )

    return DummyProfiler()
