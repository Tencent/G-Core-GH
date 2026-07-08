"""MoE expert parallelism (internal helpers; use ``parallelize_model`` to shard models)."""

from .bind_parallel_experts import bind_ep_experts, bind_ep_experts_model
from .parallel_experts_alltoall import AllToAllEPExpertsMixin
from .parellel_experts_alltoall_npu import AllToAllEPExpertsMixinNPU

__all__ = [
    "bind_ep_experts",
    "bind_ep_experts_model",
    "AllToAllEPExpertsMixin",
    "AllToAllEPExpertsMixinNPU",
]
