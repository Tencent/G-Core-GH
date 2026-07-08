"""Ulysses-style context-parallel primitives.

Two reusable APIs:
- ``shard_tensor(x, group, dim)``: split a tensor along ``dim`` by rank.
- ``all_to_all(x, group, scatter_dim, gather_dim)``: autograd-aware all-to-all
  that splits along ``scatter_dim`` and concatenates along ``gather_dim``.

``ulysses_attention`` composes these for ``[B, S, H, D]`` self-attention:
scatter heads / gather seq before SDPA, then reverse on the output.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from ..core.parallel_state import ParallelState, get_parallel_state


def shard_tensor(x: Tensor, group: dist.ProcessGroup, dim: int = 1) -> Tensor:
    """Split ``x`` along ``dim`` into ``world_size`` chunks; return this rank's chunk."""
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x
    rank = dist.get_rank(group)
    return torch.chunk(x, world_size, dim=dim)[rank].contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all that scatters ``scatter_dim`` and gathers ``gather_dim``.

    Backward reverses the exchange (swap scatter/gather dims).
    """
    @staticmethod
    def forward(
        ctx: Any, group: dist.ProcessGroup, x: Tensor, scatter_dim: int, gather_dim: int
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return x
        if x.size(scatter_dim) % world_size != 0:
            raise ValueError(
                f"all_to_all scatter dim {scatter_dim} (size {x.size(scatter_dim)}) "
                f"must be divisible by world_size ({world_size})."
            )
        inputs = [t.contiguous() for t in torch.chunk(x, world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(inputs[0]) for _ in range(world_size)]
        dist.all_to_all(outputs, inputs, group=group)
        return torch.cat(outputs, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        grad = _AllToAll.apply(ctx.group, grad_output, ctx.gather_dim, ctx.scatter_dim)
        return None, grad, None, None


def all_to_all(x: Tensor, group: dist.ProcessGroup, scatter_dim: int, gather_dim: int) -> Tensor:
    """Autograd-aware all-to-all: split along ``scatter_dim``, concat along ``gather_dim``."""
    return _AllToAll.apply(group, x, scatter_dim, gather_dim)


class _AllGatherTensor(torch.autograd.Function):
    """All-gather with backward = scatter (take local chunk from full grad)."""
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, dim: int) -> Tensor:
        world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.dim = dim
        ctx.world_size = world_size
        if world_size == 1:
            return input
        gather_list = [torch.empty_like(input) for _ in range(world_size)]
        dist.all_gather(gather_list, input.contiguous(), group=group)
        return torch.cat(gather_list, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor, None]:
        if ctx.world_size == 1:
            return None, grad_output, None
        rank = dist.get_rank(ctx.group)
        grad_chunks = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)
        return None, grad_chunks[rank].contiguous(), None


def all_gather_tensor(x: Tensor, group: dist.ProcessGroup, dim: int = 1) -> Tensor:
    """Autograd-aware all-gather along ``dim`` (backward scatters the local chunk)."""
    return _AllGatherTensor.apply(group, x, dim)


class UlyssesContext:
    """Ulysses context-parallel process group state.

    ``active`` is set inside ``with UlyssesContext():`` so attention dispatchers
    can tell Ulysses comm is in progress and skip redundant seq sharding.
    """

    cp_group = None
    enabled = False
    active = False

    @classmethod
    def init(cls, cp_group: Optional[dist.ProcessGroup] = None) -> None:
        if cp_group is None:
            cp_group = get_parallel_state().cp_group
        cls.cp_group = cp_group
        cls.enabled = cp_group is not None

    @classmethod
    def get_cp_rank(cls) -> int:
        return dist.get_rank(cls.cp_group)

    @classmethod
    def get_cp_world_size(cls) -> int:
        return dist.get_world_size(cls.cp_group)

    @classmethod
    def shard_tensor(cls, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if cls.is_enabled():
            return shard_tensor(x, cls.cp_group, dim=dim)
        return x

    @classmethod
    def all_gather_tensor(cls, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if cls.is_enabled():
            return all_gather_tensor(x, cls.cp_group, dim=dim)
        return x

    @classmethod
    def get_cp_group(cls) -> dist.ProcessGroup | None:
        return cls.cp_group

    @classmethod
    def is_active(cls) -> bool:
        return cls.is_enabled() and cls.active

    @classmethod
    def is_enabled(cls) -> bool:
        return cls.enabled and cls.get_cp_world_size() > 1

    def __enter__(self) -> UlyssesContext:
        self.prev_active = type(self).active
        type(self).active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        type(self).active = self.prev_active


def _local_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    # Ulysses tensors are [B, S, H, D]; SDPA expects [B, H, S, D].
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    ).transpose(1, 2).contiguous()


def ulysses_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    cp_group: dist.ProcessGroup | None = None,
    state: ParallelState | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Ulysses self-attention for sequence-sharded ``[B, S, H, D]`` q/k/v.

    Each rank passes its local sequence shard. Falls back to local SDPA when CP
    is disabled or ``world_size == 1``.
    """
    state = state or get_parallel_state()
    group = cp_group or state.cp_group
    if group is None or dist.get_world_size(group) == 1:
        return _local_sdpa(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa
        )

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("ulysses_attention expects q/k/v in B,S,H,D layout.")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shape mismatch: {q.shape}, {k.shape}, {v.shape}")

    # head-split / seq-gather: [B, S/p, H, D] -> [B, S, H/p, D]
    q = all_to_all(q, group, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, group, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, group, scatter_dim=2, gather_dim=1)
    out = _local_sdpa(
        q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa
    )
    # seq-split / head-gather: [B, S, H/p, D] -> [B, S/p, H, D]
    return all_to_all(out, group, scatter_dim=1, gather_dim=2)
