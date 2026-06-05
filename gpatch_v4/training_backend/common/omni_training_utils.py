"""
Shared training utilities used by both Bagel and WGOv3 pipelines/trainers.
"""

import gc
import os

import torch
import torch.distributed as dist

from gpatch_v4.core.device import get_device_module
from gpatch_v4.utils import logging_rank0


class LoggerAdaptor:
    """Lightweight logger that delegates to logging_rank0."""
    def info(self, *args):
        logging_rank0(*args)

    def error(self, *args):
        logging_rank0(*args)

    def warning(self, *args):
        logging_rank0(*args)

    def debug(self, *args):
        logging_rank0(*args)


class DummyProfiler:
    """No-op profiler context manager."""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def step(self):
        pass


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def get_latest_ckpt(checkpoint_dir: str):
    """Return path to the latest numbered checkpoint subdirectory, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None
    step_dirs = [
        d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    return os.path.join(checkpoint_dir, step_dirs[-1])


def detect_peak_tflops(default_tflops: float) -> float:
    """Guess per-device BF16 TFLOPs from GPU name; fall back to *default_tflops* when unknown."""
    try:
        device_name = get_device_module().get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    try:
        from gpatch_v4.training_backend.common.omni_training_utils_priv import PEAK_TFLOPS_TABLE
    except ImportError:
        return default_tflops
    name = device_name.upper()
    for tag, tflops in PEAK_TFLOPS_TABLE.items():
        if tag in name:
            return tflops
    return default_tflops
