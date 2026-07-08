"""Parallel topology configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ParallelConfig:
    """Lightweight parallel config: FSDP2 / HSDP + optional CP + optional EP.

    Dense / decoder FSDP mesh (HSDP when ``dp_replicate_size > 1``):
        world_size = dp_replicate_size * dp_shard_size * cp_size

    Expert parallel mesh (orthogonal, VeOmni-style):
        world_size = ep_size * ep_fsdp_size
        ep_fsdp_size = world_size // ep_size
        Optional EP HSDP: ep_fsdp_size = ep_fsdp_replicate_size * ep_fsdp_shard_size
    """

    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    ep_fsdp_replicate_size: int = 1
    ep_fsdp_shard_size: Optional[int] = None
    ep_outside: bool = False
    ep_dispatch: Literal["alltoall", "deepep"] = "alltoall"
    mixed_precision: Optional[str] = "bf16"
    cp_comm_strategy: str = "allgather"

    def __post_init__(self) -> None:
        for name, value in (
            ("dp_replicate_size", self.dp_replicate_size),
            ("dp_shard_size", self.dp_shard_size),
            ("cp_size", self.cp_size),
            ("ep_size", self.ep_size),
            ("ep_fsdp_replicate_size", self.ep_fsdp_replicate_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

        if self.ep_fsdp_shard_size is not None and self.ep_fsdp_shard_size < 1:
            raise ValueError(f"ep_fsdp_shard_size must be >= 1, got {self.ep_fsdp_shard_size}")

        if self.dp_replicate_size > 1 and self.dp_shard_size <= 1:
            raise ValueError("HSDP requires dp_shard_size > 1 when dp_replicate_size > 1")

        if not self.ep_enabled:
            if self.ep_fsdp_replicate_size != 1:
                raise ValueError("ep_fsdp_replicate_size > 1 requires ep_size > 1")
            if self.ep_fsdp_shard_size is not None and self.ep_fsdp_shard_size != 1:
                raise ValueError("ep_fsdp_shard_size > 1 requires ep_size > 1")
        elif (
            self.ep_fsdp_replicate_size > 1 and self.ep_fsdp_shard_size is not None and
            self.ep_fsdp_shard_size <= 1
        ):
            raise ValueError(
                "EP HSDP requires ep_fsdp_shard_size > 1 when ep_fsdp_replicate_size > 1"
            )

        if self.mixed_precision is not None:
            self.mixed_precision = self.mixed_precision.strip() or None

    def validate(self, world_size: int) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        dp = self.dp_replicate_size * self.dp_shard_size * self.cp_size
        if dp != world_size:
            raise ValueError(
                f"dp_replicate({self.dp_replicate_size}) * dp_shard({self.dp_shard_size}) * "
                f"cp({self.cp_size}) = {dp} != world_size({world_size})"
            )
        if self.ep_enabled and world_size % self.ep_size != 0:
            raise ValueError(f"ep_size({self.ep_size}) must divide world_size({world_size})")
        if self.ep_enabled:
            ep_fsdp_total = world_size // self.ep_size
            ep_fsdp_shard = self.ep_fsdp_shard_size_for(world_size)
            if self.ep_fsdp_replicate_size * ep_fsdp_shard != ep_fsdp_total:
                raise ValueError(
                    f"ep_fsdp_replicate({self.ep_fsdp_replicate_size}) * "
                    f"ep_fsdp_shard({ep_fsdp_shard}) = "
                    f"{self.ep_fsdp_replicate_size * ep_fsdp_shard} != "
                    f"world_size({world_size}) // ep_size({self.ep_size}) = {ep_fsdp_total}"
                )
            if self.ep_fsdp_replicate_size > 1 and ep_fsdp_shard <= 1:
                raise ValueError(
                    "EP HSDP requires ep_fsdp_shard_size > 1 when ep_fsdp_replicate_size > 1"
                )

    @property
    def dp_size(self) -> int:
        return self.dp_replicate_size * self.dp_shard_size

    @property
    def hsdp_enabled(self) -> bool:
        return self.dp_replicate_size > 1 and self.dp_shard_size > 1

    @property
    def fsdp_enabled(self) -> bool:
        return self.dp_shard_size > 1 or self.cp_enabled

    @property
    def cp_enabled(self) -> bool:
        return self.cp_size > 1

    @property
    def ep_enabled(self) -> bool:
        return self.ep_size > 1

    @property
    def ep_hsdp_enabled(self) -> bool:
        return self.ep_fsdp_replicate_size > 1

    def ep_fsdp_shard_size_for(self, world_size: int) -> int:
        """Resolve EP-FSDP shard width (defaults to ``world // ep // replicate``)."""
        if not self.ep_enabled:
            return world_size
        ep_fsdp_total = world_size // self.ep_size
        if self.ep_fsdp_shard_size is not None:
            return self.ep_fsdp_shard_size
        return ep_fsdp_total // self.ep_fsdp_replicate_size

    def ep_fsdp_size_for(self, world_size: int) -> int:
        """Return ``world_size // ep_size`` (total EP-FSDP mesh width)."""
        if not self.ep_enabled:
            return world_size
        return world_size // self.ep_size

    def ep_hsdp_active_for(self, world_size: int) -> bool:
        """True when experts use 2-D EP-FSDP hybrid sharding."""
        return (
            self.ep_enabled and self.ep_fsdp_replicate_size > 1 and
            self.ep_fsdp_shard_size_for(world_size) > 1
        )
