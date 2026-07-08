"""Device mesh and process-group management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .config import ParallelConfig

_STATE: Optional["ParallelState"] = None


def init_ep_mesh_matrix(ep_size: int, ep_fsdp_size: int, ep_outside: bool = False) -> torch.Tensor:
    if ep_outside:
        return torch.arange(ep_size * ep_fsdp_size, dtype=torch.int).view(ep_size, ep_fsdp_size)
    return (
        torch.arange(ep_size * ep_fsdp_size, dtype=torch.int).view(ep_fsdp_size,
                                                                   ep_size).transpose(0, 1)
    )


def init_ep_hsdp_mesh_matrix(
    ep_size: int,
    ep_fsdp_replicate_size: int,
    ep_fsdp_shard_size: int,
    ep_outside: bool = False,
) -> torch.Tensor:
    """Build ``(ep, ep_fsdp_replicate, ep_fsdp_shard)`` rank grid for EP HSDP.

    Mesh dim order matches dense HSDP ``(dp_replicate, dp_shard)``: replicate is
    outer, shard is inner (fastest-varying). ``ep_outside`` only changes how
    global ranks map into the grid (same role as ``init_ep_mesh_matrix``), not
    the replicate/shard ordering required by ``fully_shard``.
    """
    total = ep_size * ep_fsdp_replicate_size * ep_fsdp_shard_size
    if ep_outside:
        return torch.arange(total, dtype=torch.int).view(
            ep_size, ep_fsdp_replicate_size, ep_fsdp_shard_size
        )
    return (
        torch.arange(total,
                     dtype=torch.int).view(ep_fsdp_replicate_size, ep_fsdp_shard_size,
                                           ep_size).permute(2, 0, 1).contiguous()
    )


@dataclass
class ParallelState:
    config: ParallelConfig
    device_mesh: DeviceMesh
    ep_device_mesh: Optional[DeviceMesh] = None
    ep_hsdp_enabled: bool = False
    world_size: int = 1

    @property
    def rank(self) -> int:
        return dist.get_rank()

    @property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", self.rank))

    @property
    def fsdp_mesh(self) -> DeviceMesh:
        """FSDP / HSDP mesh for decoder and other non-expert modules."""
        if self.config.cp_enabled:
            if self.config.hsdp_enabled:
                return self.device_mesh["dp_replicate", "dp_shard_cp"]
            return self.device_mesh["dp_shard_cp"]
        if self.config.hsdp_enabled:
            return self.device_mesh["dp_replicate", "dp_shard"]
        return self.device_mesh["dp_shard"]

    @property
    def hsdp_enabled(self) -> bool:
        return self.config.hsdp_enabled

    @property
    def dp_replicate_mesh(self) -> DeviceMesh | None:
        if not self.config.hsdp_enabled:
            return None
        return self.device_mesh["dp_replicate"]

    @property
    def dp_replicate_group(self) -> dist.ProcessGroup | None:
        if not self.config.hsdp_enabled:
            return None
        return self.device_mesh.get_group("dp_replicate")

    @property
    def loss_group(self) -> dist.ProcessGroup:
        mesh_names = set(self.device_mesh.mesh_dim_names)
        if "dp_cp" in mesh_names:
            return self.device_mesh["dp_cp"].get_group()
        return self.device_mesh["dp"].get_group()

    @property
    def fsdp_group(self) -> dist.ProcessGroup:
        return self.loss_group

    @property
    def dp_group(self) -> dist.ProcessGroup:
        return self.device_mesh["dp"].get_group()

    @property
    def cp_group(self) -> Optional[dist.ProcessGroup]:
        if not self.config.cp_enabled:
            return None
        return self.device_mesh["cp"].get_group()

    @property
    def ep_group(self) -> Optional[dist.ProcessGroup]:
        if self.ep_device_mesh is None:
            return None
        return self.ep_device_mesh["ep"].get_group()

    @property
    def ep_mesh(self) -> Optional[DeviceMesh]:
        if self.ep_device_mesh is None:
            return None
        return self.ep_device_mesh["ep"]

    @property
    def ep_fsdp_mesh(self) -> Optional[DeviceMesh]:
        if self.ep_device_mesh is None:
            return None
        if self.ep_hsdp_enabled:
            return self.ep_device_mesh["ep_fsdp_replicate", "ep_fsdp_shard"]
        return self.ep_device_mesh["ep_fsdp"]

    @property
    def ep_rank(self) -> int:
        if self.ep_device_mesh is None:
            return 0
        return self.ep_device_mesh.get_local_rank("ep")

    @property
    def ep_fsdp_size(self) -> int:
        if self.ep_device_mesh is None:
            return self.world_size
        return self.ep_device_mesh["ep_fsdp"].size()

    @property
    def ep_gradient_divide_factor(self) -> int:
        """Global microbatch count for EP expert ``set_gradient_divide_factor``.

        EP FSDP uses the ``ep_fsdp`` mesh (no CP dim). When CP is enabled, ranks
        in the same CP group share one microbatch (sequence shards only), so
        ``world_size`` over-counts by ``cp_size``; use ``dp_size * ep_size`` instead.
        When CP is off, ``world_size`` matches VeOmni (``dp_size * ep_size``).
        """
        return self.world_size


def build_parallel_state(
    config: ParallelConfig,
    *,
    device_type: str,
    world_size: Optional[int] = None,
    register: bool = True,
) -> ParallelState:
    """Build ``ParallelState`` from ``ParallelConfig`` without touching process groups.

    Assumes ``dist.init_process_group`` and device binding were already done by the
    caller (e.g. trainer ``init_distributed``). ``device_type`` is passed through to
    ``init_device_mesh`` so callers can supply ``get_device_backend_name()`` for
    NPU/MLU instead of hard-coding ``"cuda"``.
    """
    global _STATE

    world_size = world_size if world_size is not None else dist.get_world_size()
    config.validate(world_size)

    mesh_shape: list[int] = []
    mesh_names: list[str] = []
    for size, name in [
        (config.dp_replicate_size, "dp_replicate"),
        (config.dp_shard_size, "dp_shard"),
        (config.cp_size, "cp"),
    ]:
        if size > 1 or name == "dp_shard":
            mesh_shape.append(size)
            mesh_names.append(name)

    device_mesh = init_device_mesh(device_type, tuple(mesh_shape), mesh_dim_names=tuple(mesh_names))

    dp_dims = [n for n in ("dp_replicate", "dp_shard") if n in mesh_names]
    if dp_dims:
        device_mesh[tuple(dp_dims)]._flatten("dp")

    fsdp_dims = list(dp_dims)
    if config.cp_enabled:
        fsdp_dims.append("cp")
    if fsdp_dims:
        device_mesh[tuple(fsdp_dims)]._flatten("dp_shard_cp")

    loss_dims = list(dp_dims)
    if config.cp_enabled:
        loss_dims.append("cp")
    if loss_dims:
        device_mesh[tuple(loss_dims)]._flatten("dp_cp")

    ep_device_mesh = None
    ep_hsdp_enabled = False
    if config.ep_enabled:
        ep_fsdp_total = world_size // config.ep_size
        ep_fsdp_shard = config.ep_fsdp_shard_size_for(world_size)
        ep_hsdp_enabled = config.ep_hsdp_active_for(world_size)
        if ep_hsdp_enabled:
            ep_matrix = init_ep_hsdp_mesh_matrix(
                config.ep_size,
                config.ep_fsdp_replicate_size,
                ep_fsdp_shard,
                config.ep_outside,
            )
            ep_device_mesh = DeviceMesh(
                device_type,
                ep_matrix,
                mesh_dim_names=("ep", "ep_fsdp_replicate", "ep_fsdp_shard"),
            )
            ep_device_mesh[tuple(["ep_fsdp_replicate", "ep_fsdp_shard"])]._flatten("ep_fsdp")
        else:
            ep_matrix = init_ep_mesh_matrix(config.ep_size, ep_fsdp_total, config.ep_outside)
            ep_device_mesh = DeviceMesh(device_type, ep_matrix, mesh_dim_names=("ep", "ep_fsdp"))

    state = ParallelState(
        config=config,
        device_mesh=device_mesh,
        ep_device_mesh=ep_device_mesh,
        ep_hsdp_enabled=ep_hsdp_enabled,
        world_size=world_size,
    )
    if register:
        _STATE = state
    return state


def init_distributed(config: ParallelConfig, backend: str = "nccl") -> ParallelState:
    global _STATE
    if _STATE is not None:
        return _STATE

    if not dist.is_initialized():
        dist.init_process_group(backend)

    local_rank = int(
        os.environ.get("LOCAL_RANK",
                       dist.get_rank() % max(torch.cuda.device_count(), 1))
    )
    torch.cuda.set_device(local_rank)

    return build_parallel_state(config, device_type="cuda", register=True)


def get_parallel_state() -> ParallelState:
    if _STATE is None:
        raise RuntimeError("Call init_distributed() first.")
    return _STATE
