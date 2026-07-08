"""FSDP2 + EP + CP lightweight training framework.

Primary API: ``parallelize_model``.
"""

from .parallelizer import (
    ep_full_placement_hooks,
    parallelize_model,
    share_fsdp2_comm_ctx,
)
from .parallelizer.fsdp import _fsdp_mixed_precision_policy
from .parallelizer.fsdp_patch import (
    apply_fsdp2_post_forward_patch,
    apply_fsdp2_reduce_scatter_ring_buffer_patch,
    enable_rs_ring,
    fully_shard,
    mark_rs_ring_eligible,
)

__all__ = [
    "parallelize_model",
    "share_fsdp2_comm_ctx",
    "ep_full_placement_hooks",
    "apply_fsdp2_post_forward_patch",
    "apply_fsdp2_reduce_scatter_ring_buffer_patch",
    "enable_rs_ring",
    "fully_shard",
    "mark_rs_ring_eligible",
    "_fsdp_mixed_precision_policy",
]

__version__ = "0.1.0"
