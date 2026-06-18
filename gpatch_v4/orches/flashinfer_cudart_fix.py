# copyright (c) 2026 tencent inc. all rights reserved.
"""Avoid flashinfer binding tilelang's libcudart stub.

tilelang ships ``libcudart_stub.so`` — a compile-time linker stub that lacks
runtime symbols like ``cudaDeviceReset``.  flashinfer's ``CudaRTLibrary``
discovers it via file-system search and binds to it, causing
``AttributeError: undefined symbol: cudaDeviceReset``.

Two fixes live here:

* **vLLM path** — set ``fi_allreduce_fusion_max_size_mb`` explicitly so
  ``PassConfig`` never imports ``flashinfer.comm``.
* **sglang path** — monkey-patch ``ctypes.CDLL`` while sglang is imported,
  redirecting any load of ``libcudart_stub`` to the real ``libcudart.so``.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import os


def _find_real_cudart() -> str | None:
    for candidate in [
        os.environ.get("CUDART_LIBRARY_PATH"),
        "/usr/local/cuda/lib64/libcudart.so",
        *sorted(glob.glob("/usr/local/cuda-*/targets/x86_64-linux/lib/libcudart.so")),
        *sorted(glob.glob("/usr/local/cuda-*/lib64/libcudart.so")),
    ]:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


@contextlib.contextmanager
def patch_ctypes_for_cudart_stub():
    """Redirect ``ctypes.CDLL("…libcudart_stub…")`` to the real libcudart."""
    real_cudart = _find_real_cudart()
    if real_cudart is None:
        yield
        return

    _orig_init = ctypes.CDLL.__init__

    def _patched_init(self, name, *args, **kwargs):
        if name and "libcudart_stub" in str(name):
            print(f"redirect {name} to {real_cudart}", flush=True)
            name = real_cudart
        _orig_init(self, name, *args, **kwargs)

    ctypes.CDLL.__init__ = _patched_init
    try:
        yield
    finally:
        ctypes.CDLL.__init__ = _orig_init

# Sync with vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py
# FI_ALLREDUCE_FUSION_MAX_SIZE_MB
_FI_ALLREDUCE_FUSION_MAX_SIZE_MB: dict[int, dict[int, float]] = {
    90: {
        2: 64,
        4: 2,
        8: 0.5,
    },
    100: {
        2: 64,
        4: 32,
        8: 1,
    },
    103: {
        2: 64,
        4: 64,
        8: 2,
    },
}

_FI_SUPPORTED_WORLD_SIZES = frozenset({2, 4, 8})


def resolve_fi_allreduce_fusion_max_size_mb(tensor_parallel_size: int) -> float | None:
    """Mirror ``PassConfig.default_fi_allreduce_fusion_max_size_mb()[world_size]``.

    Parameters
    ----------
    tensor_parallel_size : int
        TP world size (same as vLLM ``flashinfer_max_size`` input).

    Returns
    -------
    float or None
        Threshold in MB, or *None* when fusion is not defined for this
        platform / world size (caller should still pass a numeric value to
        avoid the ``flashinfer.comm`` import — see
        :func:`build_compilation_config_patch`).
    """
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return None
    capability = current_platform.get_device_capability()
    if capability is None:
        return None
    per_world = _FI_ALLREDUCE_FUSION_MAX_SIZE_MB.get(capability.to_int())
    if per_world is None:
        return None
    if tensor_parallel_size not in _FI_SUPPORTED_WORLD_SIZES:
        return None
    return per_world.get(tensor_parallel_size)


def build_compilation_config_patch(tensor_parallel_size: int) -> dict:
    """Build ``compilation_config`` dict for ``AsyncEngineArgs``.

    Uses vLLM's FI allreduce fusion table + local device capability. Any
    explicit ``fi_allreduce_fusion_max_size_mb`` prevents config init from
    importing ``flashinfer.comm``.

    Parameters
    ----------
    tensor_parallel_size : int

    Returns
    -------
    dict
    """
    max_size_mb = resolve_fi_allreduce_fusion_max_size_mb(tensor_parallel_size)
    # PassConfig treats None as "call default_fi_..." which imports flashinfer.comm.
    # Use 0.0 when vLLM has no table entry (e.g. TP=1): flashinfer_max_size -> 0.
    # fi_allreduce_fusion_max_size_mb 表示：通信 tensor 小于多少 MB 时，才用 flashinfer 的 fused allreduce
    if max_size_mb is None:
        max_size_mb = 0.0
    return {
        "pass_config": {
            "fi_allreduce_fusion_max_size_mb": float(max_size_mb),
        },
    }
