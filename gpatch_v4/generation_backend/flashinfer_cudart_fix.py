# copyright (c) 2026 tencent inc. all rights reserved.
"""Avoid flashinfer binding tilelang's libcudart stub during vLLM config init.

vLLM ``PassConfig.default_fi_allreduce_fusion_max_size_mb()`` imports
``flashinfer.comm`` (which binds the first ``libcudart`` in ``/proc/self/maps``,
often tilelang's stub). Setting ``fi_allreduce_fusion_max_size_mb`` explicitly
skips that import.

The lookup table is copied from vLLM (do not import ``allreduce_rms_fusion`` —
that module also imports ``flashinfer.comm`` at load time).
"""

from __future__ import annotations

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
