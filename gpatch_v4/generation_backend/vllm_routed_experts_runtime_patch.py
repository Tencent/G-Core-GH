# copyright (c) 2026 tencent inc. all rights reserved.

"""Runtime compatibility patch for vLLM routed-experts capture.

vLLM 0.19.1 sizes the routed-experts shared-memory buffer as if every
hybrid KV-cache group owned a disjoint slice of the physical block pool.  The
groups actually share the full pool, so a valid full-attention ``slot_mapping``
can exceed that undersized buffer and terminate EngineCore with ``IndexError``.

The writer-side patch is imported by :mod:`vllm_worker_extension` inside every
vLLM TP worker, before the concrete GPU worker is constructed.  The reader
lives in the separately spawned EngineCore process, so its process entry point
is wrapped as well.  This keeps both sides consistent without modifying the
vLLM installation in the container image.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_gcore_routed_experts_full_kv_pool_patch"
_READER_PATCHED_ATTR = "_gcore_routed_experts_full_shm_view_patch"
_SCHEDULER_PATCHED_ATTR = "_gcore_routed_experts_full_attention_gid_patch"
_ENGINE_CORE_PATCHED_ATTR = "_gcore_routed_experts_engine_core_entry_patch"
_ORIGINAL_ENGINE_CORE_RUN_ATTR = "_gcore_original_run_engine_core"
_SUPPORTED_VLLM_VERSION = "0.19.1"
VLLM_R3_ENGINE_CORE_PATCH_ENV = "GPATCH_VLMM_R3_ENGINE_CORE"


def is_vllm_r3_engine_core_patch_enabled() -> bool:
    """Return whether the opt-in vLLM routed-experts patch is enabled."""
    value = os.environ.get(VLLM_R3_ENGINE_CORE_PATCH_ENV)
    return value is not None and value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _base_version(version: str) -> str:
    """Strip common local/build suffixes from a package version."""
    return version.split("+", 1)[0]


def _is_full_attention_kv_cache_spec(
    spec: Any,
    *,
    full_attention_spec_cls: type,
    uniform_type_kv_cache_specs_cls: type,
) -> bool:
    """Return whether a KV spec is, or exclusively wraps, full attention."""
    if isinstance(spec, uniform_type_kv_cache_specs_cls):
        wrapped_specs = tuple(spec.kv_cache_specs.values())
        return bool(wrapped_specs) and all(
            isinstance(wrapped_spec, full_attention_spec_cls)
            for wrapped_spec in wrapped_specs
        )
    return isinstance(spec, full_attention_spec_cls)


def _get_routed_experts_kv_cache_gid(
    kv_cache_config: Any,
    *,
    full_attention_spec_cls: type,
    uniform_type_kv_cache_specs_cls: type,
) -> int:
    """Select the stable full-attention slot layout used by router replay."""
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        if _is_full_attention_kv_cache_spec(
            group.kv_cache_spec,
            full_attention_spec_cls=full_attention_spec_cls,
            uniform_type_kv_cache_specs_cls=uniform_type_kv_cache_specs_cls,
        ):
            return gid
    raise RuntimeError(
        "moe_router_replay requires a full-attention KV-cache group, "
        "but none was found"
    )


def _patch_gpu_model_runner(
    *,
    vllm_version: str,
    gpu_model_runner_cls: type,
    full_attention_spec_cls: type,
    uniform_type_kv_cache_specs_cls: type,
    routed_experts_capturer_cls: type,
) -> bool:
    """Patch a vLLM 0.19.1 ``GPUModelRunner`` class in place.

    Dependency injection keeps the capacity calculation independently
    testable without importing vLLM in unit-test environments.

    Returns
    -------
    bool
        ``True`` when this call applied the patch, otherwise ``False``.
    """
    if getattr(gpu_model_runner_cls, _PATCHED_ATTR, False):
        return False

    if _base_version(vllm_version) != _SUPPORTED_VLLM_VERSION:
        logger.info(
            "Skipping GCore routed-experts KV-pool patch for vLLM %s; "
            "the patch targets vLLM %s only",
            vllm_version,
            _SUPPORTED_VLLM_VERSION,
        )
        return False

    required_methods = (
        "_bind_routed_experts_capturer",
        "init_routed_experts_capturer",
    )
    missing = [
        name for name in required_methods
        if not hasattr(gpu_model_runner_cls, name)
    ]
    if missing:
        raise RuntimeError(
            "Cannot apply the vLLM routed-experts KV-pool patch: "
            f"GPUModelRunner is missing {missing}"
        )

    def get_full_attention_kv_cache_gid(self) -> int:
        """Select the KV group whose physical slots back full attention."""
        return _get_routed_experts_kv_cache_gid(
            self.kv_cache_config,
            full_attention_spec_cls=full_attention_spec_cls,
            uniform_type_kv_cache_specs_cls=uniform_type_kv_cache_specs_cls,
        )

    def init_routed_experts_capturer(self) -> None:
        capturer = routed_experts_capturer_cls.create()
        self.routed_experts_attn_gid = get_full_attention_kv_cache_gid(self)

        attn_group = self.kv_cache_config.kv_cache_groups[
            self.routed_experts_attn_gid
        ]
        block_size = int(attn_group.kv_cache_spec.block_size)
        num_blocks = int(self.kv_cache_config.num_blocks)

        # All hybrid KV-cache groups draw block IDs from the complete physical
        # pool.  Do not divide num_blocks by the number of groups (the v0.19.1
        # bug); slot_mapping is block_id * full_attention_block_size + offset.
        self.max_num_kv_tokens = num_blocks * block_size

        parallel_config = self.vllm_config.parallel_config
        dcp_size = int(parallel_config.decode_context_parallel_size)
        pcp_size = int(parallel_config.prefill_context_parallel_size)
        context_parallel_size = dcp_size * pcp_size
        if context_parallel_size > 1:
            # Preserve vLLM 0.19.1's existing DCP/PCP capacity adjustment.
            self.max_num_kv_tokens *= context_parallel_size

        hf_config: Any = self.vllm_config.model_config.hf_text_config
        num_layers = int(hf_config.num_hidden_layers)
        topk = int(hf_config.num_experts_per_tok)
        shared_memory_bytes = self.max_num_kv_tokens * num_layers * topk * 4

        logger.warning(
            "Applied GCore vLLM %s routed-experts KV-pool patch: "
            "full_attention_gid=%d num_blocks=%d block_size=%d slots=%d "
            "shared_memory_capacity=%.2f GiB (allocated by TP0 only)",
            vllm_version,
            self.routed_experts_attn_gid,
            num_blocks,
            block_size,
            self.max_num_kv_tokens,
            shared_memory_bytes / (1024**3),
        )

        capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=self.max_num_kv_tokens,
            vllm_config=self.vllm_config,
        )
        self._bind_routed_experts_capturer(capturer)
        self.routed_experts_initialized = True

    # Keep the helper coherent for any other v0.19.1 call site, and replace
    # capturer initialization with the corrected full-pool capacity.
    gpu_model_runner_cls._get_attention_kv_cache_gid = get_full_attention_kv_cache_gid
    gpu_model_runner_cls.init_routed_experts_capturer = init_routed_experts_capturer
    setattr(gpu_model_runner_cls, _PATCHED_ATTR, True)
    return True


def _patch_scheduler(
    *,
    vllm_version: str,
    scheduler_cls: type,
    full_attention_spec_cls: type,
    uniform_type_kv_cache_specs_cls: type,
) -> bool:
    """Make EngineCore read routed experts through the worker's KV group."""
    if getattr(scheduler_cls, _SCHEDULER_PATCHED_ATTR, False):
        return False

    if _base_version(vllm_version) != _SUPPORTED_VLLM_VERSION:
        logger.info(
            "Skipping GCore routed-experts scheduler patch for vLLM %s; "
            "the patch targets vLLM %s only",
            vllm_version,
            _SUPPORTED_VLLM_VERSION,
        )
        return False

    original_init = scheduler_cls.__init__

    def init_with_full_attention_gid(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if not self.vllm_config.model_config.enable_return_routed_experts:
            return

        old_gid = self.routed_experts_attn_gid
        self.routed_experts_attn_gid = _get_routed_experts_kv_cache_gid(
            self.kv_cache_config,
            full_attention_spec_cls=full_attention_spec_cls,
            uniform_type_kv_cache_specs_cls=uniform_type_kv_cache_specs_cls,
        )
        if old_gid != self.routed_experts_attn_gid:
            logger.warning(
                "Applied GCore vLLM %s routed-experts scheduler gid patch: "
                "old_gid=%d full_attention_gid=%d",
                vllm_version,
                old_gid,
                self.routed_experts_attn_gid,
            )

    scheduler_cls.__init__ = init_with_full_attention_gid
    setattr(scheduler_cls, _SCHEDULER_PATCHED_ATTR, True)
    return True


def _patch_routed_experts_reader(
    *,
    vllm_version: str,
    routed_experts_reader_cls: type,
) -> bool:
    """Make EngineCore view the complete writer-created shared memory.

    vLLM 0.19.1 passes the same undersized ``max_num_kv_tokens`` calculation
    to :class:`RoutedExpertsReader`.  The writer-side fix creates a larger
    shared-memory object, but NumPy otherwise exposes only the smaller prefix
    requested by EngineCore.  Rebuilding the view from ``SharedMemory.size``
    makes the reader use the exact capacity created by TP0 and cannot drift
    from the writer's allocation formula.
    """
    if getattr(routed_experts_reader_cls, _READER_PATCHED_ATTR, False):
        return False

    if _base_version(vllm_version) != _SUPPORTED_VLLM_VERSION:
        logger.info(
            "Skipping GCore routed-experts reader patch for vLLM %s; "
            "the patch targets vLLM %s only",
            vllm_version,
            _SUPPORTED_VLLM_VERSION,
        )
        return False

    if not hasattr(routed_experts_reader_cls, "attach_buffer"):
        raise RuntimeError(
            "Cannot apply the vLLM routed-experts reader patch: "
            "RoutedExpertsReader is missing attach_buffer"
        )

    original_attach_buffer = routed_experts_reader_cls.attach_buffer

    def attach_full_shared_memory_view(
        self,
        max_num_kv_tokens: int,
        vllm_config: Any,
    ) -> None:
        original_attach_buffer(
            self,
            max_num_kv_tokens=max_num_kv_tokens,
            vllm_config=vllm_config,
        )

        shm = self._shm
        host_view = self._host_buffer_view
        if shm is None or host_view is None:
            raise RuntimeError(
                "RoutedExpertsReader did not attach its shared-memory buffer"
            )

        hf_config: Any = vllm_config.model_config.hf_text_config
        num_layers = int(hf_config.num_hidden_layers)
        topk = int(hf_config.num_experts_per_tok)
        bytes_per_token = num_layers * topk * int(host_view.dtype.itemsize)
        shared_memory_bytes = int(shm.size)
        if shared_memory_bytes % bytes_per_token != 0:
            raise RuntimeError(
                "Routed-experts shared-memory size is not divisible by one "
                "token row: "
                f"size={shared_memory_bytes} bytes_per_token={bytes_per_token}"
            )

        shared_memory_slots = shared_memory_bytes // bytes_per_token
        configured_slots = int(max_num_kv_tokens)
        if shared_memory_slots < configured_slots:
            raise RuntimeError(
                "Routed-experts shared memory is smaller than EngineCore's "
                "configured view: "
                f"shared_memory_slots={shared_memory_slots} "
                f"configured_slots={configured_slots}"
            )
        if shared_memory_slots == int(host_view.shape[0]):
            return

        import numpy as np

        self._host_buffer_view = np.ndarray(
            (shared_memory_slots, num_layers, topk),
            dtype=host_view.dtype,
            buffer=shm.buf,
        )
        logger.warning(
            "Applied GCore vLLM %s routed-experts reader patch: "
            "configured_slots=%d shared_memory_slots=%d "
            "shared_memory_capacity=%.2f GiB",
            vllm_version,
            configured_slots,
            shared_memory_slots,
            shared_memory_bytes / (1024**3),
        )

    routed_experts_reader_cls.attach_buffer = attach_full_shared_memory_view
    setattr(routed_experts_reader_cls, _READER_PATCHED_ATTR, True)
    return True


def _run_engine_core_with_routed_experts_patch(*args: Any, **kwargs: Any) -> Any:
    """Pickle-safe EngineCore entry point that installs the reader patch."""
    apply_vllm_routed_experts_reader_patch()

    from vllm.v1.engine.core import EngineCoreProc

    # With ``spawn`` the class is freshly imported and therefore unpatched;
    # with ``fork`` it retains the saved original entry point from the parent.
    original_run = getattr(
        EngineCoreProc,
        _ORIGINAL_ENGINE_CORE_RUN_ATTR,
        EngineCoreProc.run_engine_core,
    )
    if original_run is _run_engine_core_with_routed_experts_patch:
        raise RuntimeError("Lost the original vLLM EngineCore entry point")
    return original_run(*args, **kwargs)


def _patch_engine_core_proc(
    *,
    vllm_version: str,
    engine_core_proc_cls: type,
) -> bool:
    """Route spawned EngineCore processes through the reader-patch wrapper."""
    if getattr(engine_core_proc_cls, _ENGINE_CORE_PATCHED_ATTR, False):
        return False

    if _base_version(vllm_version) != _SUPPORTED_VLLM_VERSION:
        logger.info(
            "Skipping GCore routed-experts EngineCore entry patch for vLLM %s; "
            "the patch targets vLLM %s only",
            vllm_version,
            _SUPPORTED_VLLM_VERSION,
        )
        return False

    if not hasattr(engine_core_proc_cls, "run_engine_core"):
        raise RuntimeError(
            "Cannot apply the vLLM routed-experts EngineCore entry patch: "
            "EngineCoreProc is missing run_engine_core"
        )

    setattr(
        engine_core_proc_cls,
        _ORIGINAL_ENGINE_CORE_RUN_ATTR,
        engine_core_proc_cls.run_engine_core,
    )
    engine_core_proc_cls.run_engine_core = staticmethod(
        _run_engine_core_with_routed_experts_patch
    )
    setattr(engine_core_proc_cls, _ENGINE_CORE_PATCHED_ATTR, True)
    return True


def apply_vllm_routed_experts_capacity_patch() -> bool:
    """Apply the compatibility patch in the current vLLM worker process.

    Importing this helper outside a vLLM runtime is intentionally a no-op so
    lightweight tooling and unit tests can still import GCore modules.
    """
    try:
        import vllm
    except ModuleNotFoundError as exc:
        if exc.name == "vllm":
            return False
        raise

    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCapturer,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        UniformTypeKVCacheSpecs,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    return _patch_gpu_model_runner(
        vllm_version=vllm.__version__,
        gpu_model_runner_cls=GPUModelRunner,
        full_attention_spec_cls=FullAttentionSpec,
        uniform_type_kv_cache_specs_cls=UniformTypeKVCacheSpecs,
        routed_experts_capturer_cls=RoutedExpertsCapturer,
    )


def apply_vllm_routed_experts_reader_patch() -> bool:
    """Patch the routed-experts reader and KV-group selection in EngineCore."""
    import vllm
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsReader,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    reader_patched = _patch_routed_experts_reader(
        vllm_version=vllm.__version__,
        routed_experts_reader_cls=RoutedExpertsReader,
    )
    scheduler_patched = _patch_scheduler(
        vllm_version=vllm.__version__,
        scheduler_cls=Scheduler,
        full_attention_spec_cls=FullAttentionSpec,
        uniform_type_kv_cache_specs_cls=UniformTypeKVCacheSpecs,
    )
    return reader_patched or scheduler_patched


def apply_vllm_routed_experts_engine_core_patch() -> bool:
    """Ensure spawned EngineCore processes install the reader-side patch."""
    import vllm
    from vllm.v1.engine.core import EngineCoreProc

    # Also covers an in-process or forked EngineCore.  A spawned EngineCore
    # installs the same patch again from the pickle-safe wrapper above.
    apply_vllm_routed_experts_reader_patch()
    return _patch_engine_core_proc(
        vllm_version=vllm.__version__,
        engine_core_proc_cls=EngineCoreProc,
    )


__all__ = [
    "VLLM_R3_ENGINE_CORE_PATCH_ENV",
    "apply_vllm_routed_experts_capacity_patch",
    "apply_vllm_routed_experts_engine_core_patch",
    "apply_vllm_routed_experts_reader_patch",
    "is_vllm_r3_engine_core_patch_enabled",
]
