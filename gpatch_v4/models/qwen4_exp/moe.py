# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Expert-parallel MoE for Qwen3.8-Flash-Next.

512 routed experts × 48 layers is ~120.8 B of the 125 B backbone, so the experts must be
partitioned across ranks. Each rank owns a contiguous slice of
``num_experts / ep_size`` experts; a forward ships each (token, top-k slot) pair to the
rank owning its expert, computes locally, and ships the result back.

Structure follows :class:`gpatch_v4.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts`
``_forward_eager`` / ``fwd_gmm``, which is the validated pattern in this repo and uses the
identical 3-D stacked weight layout (``gate_up_proj [E, 2I, H]``, ``down_proj [E, H, I]``).
Differences: no FP8/FP4 QAT paths, and no ``swiglu_limit`` clamp — upstream
``Qwen4ExpTextExperts`` applies a plain ``act_fn(gate) * up``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.integrations.moe import _grouped_linear
from typing_extensions import override

from .a2a import all_to_all_uneven
from .modeling_qwen4_exp import Qwen4ExpTextExperts

__all__ = ["Qwen4ExpEPExperts"]


class _EmptyExpertsWithGrad(torch.autograd.Function):
    """Zero-token stub that keeps expert parameters in the autograd graph.

    Grouped MM never touches the weights when there are no rows, which would leave the
    expert parameters out of the graph so FSDP2 may not produce zero grads for them.
    Ported from DSV4, where the same failure mode was hit under activation checkpointing.
    """
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ) -> torch.Tensor:
        ctx.gate_up_meta = (gate_up_proj.shape, gate_up_proj.dtype, gate_up_proj.device)
        ctx.down_meta = (down_proj.shape, down_proj.dtype, down_proj.device)
        return torch.zeros_like(hidden_states)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        def zeros(meta):
            shape, dtype, device = meta
            return torch.zeros(shape, dtype=dtype, device=device)

        return grad_output, zeros(ctx.gate_up_meta), zeros(ctx.down_meta)


class Qwen4ExpEPExperts(Qwen4ExpTextExperts):
    """Expert-parallel replacement for upstream's dense expert loop.

    Parameter names, shapes and dtypes are inherited unchanged, so the checkpoint FQNs
    ``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj`` still apply — after
    :func:`configure_ep` the leading dimension holds only this rank's slice.

    With ``ep_size == 1`` the forward delegates to upstream's implementation; that is the
    correct behaviour when EP is off, not a fallback.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.ep_size = 1
        self.ep_rank = 0
        self.ep_group: Optional[dist.ProcessGroup] = None
        self.num_local_experts = self.num_experts

    def configure_ep(self, ep_group: Optional[dist.ProcessGroup]) -> None:
        """Record the expert-parallel group and this rank's local expert count.

        Does **not** slice the weights — the caller (``apply_hp``) owns that, because it
        must also distribute them onto the FSDP2 mesh as DTensors.

        Raises
        ------
        ValueError
            If ``num_experts`` is not divisible by the group size.
        """
        if ep_group is None:
            self.ep_size, self.ep_rank, self.ep_group = 1, 0, None
            self.num_local_experts = self.num_experts
            return

        ep_size = dist.get_world_size(ep_group)
        if self.num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by ep_size ({ep_size})."
            )
        self.ep_size = ep_size
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_group = ep_group
        self.num_local_experts = self.num_experts // ep_size

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to expert owners, compute locally, combine back.

        Parameters
        ----------
        hidden_states : torch.Tensor
            ``[N, H]`` flat token activations (``N = B * S``).
        top_k_index : torch.Tensor
            ``[N, K]`` int64 global expert ids.
        top_k_weights : torch.Tensor
            ``[N, K]`` routing weights.

        Returns
        -------
        torch.Tensor
            ``[N, H]``, summed over the top-k slots.
        """
        if self.ep_size == 1:
            return super().forward(hidden_states, top_k_index, top_k_weights)

        num_tokens, hidden_dim = hidden_states.shape
        top_k = top_k_index.shape[-1]

        # Each (token, slot) pair is an independent dispatch unit.
        flat_expert = top_k_index.reshape(-1)
        flat_weight = top_k_weights.reshape(-1)
        flat_hidden = hidden_states.repeat_interleave(top_k, dim=0)

        # Group by owning rank so each rank's payload is contiguous.
        target_rank = torch.div(flat_expert, self.num_local_experts, rounding_mode="floor")
        order = torch.argsort(target_rank, stable=True)
        send_counts = torch.bincount(target_rank, minlength=self.ep_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        recv_hidden = all_to_all_uneven(
            flat_hidden[order], send_splits, recv_splits, self.ep_group
        )
        recv_expert = all_to_all_uneven(
            flat_expert[order], send_splits, recv_splits, self.ep_group
        )
        local_expert_id = recv_expert - self.ep_rank * self.num_local_experts
        local_output = self._local_experts(recv_hidden, local_expert_id)

        combined = all_to_all_uneven(
            local_output, recv_splits, send_splits, self.ep_group
        )
        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(order.numel(), device=order.device)
        # Weight after combine to avoid routing one scalar per expert assignment twice.
        combined = (
            combined[inverse_order] * flat_weight.unsqueeze(-1)
        ).to(hidden_states.dtype)
        return combined.view(num_tokens, top_k, hidden_dim).sum(dim=1)

    def _local_experts(
        self,
        hidden_states: torch.Tensor,
        local_expert_id: torch.Tensor,
    ) -> torch.Tensor:
        """Grouped SwiGLU over this rank's experts.

        Each row is already assigned 1:1 to one local expert, so unlike upstream's
        implementation there is no top-k expansion or sentinel masking to undo.

        Parameters
        ----------
        hidden_states : torch.Tensor
            ``[S, H]`` rows received by this rank.
        local_expert_id : torch.Tensor
            ``[S]`` int64 ids in ``[0, num_local_experts)``.
        Returns
        -------
        torch.Tensor
            ``[S, H]`` expert outputs, in the input row order.
        """
        if hidden_states.shape[0] == 0:
            return _EmptyExpertsWithGrad.apply(
                hidden_states, self.gate_up_proj, self.down_proj
            )

        # Grouped MM needs rows sorted by expert so each expert owns a contiguous span.
        sorted_ids, perm = torch.sort(local_expert_id)
        rows = hidden_states[perm]

        # `bincount` rather than DSV4's `histc`: histc has no integer CPU kernel
        # ("histogram_cpu not implemented for 'Int'"), which would confine these tests to
        # GPUs, and bincount is exact integer counting instead of float bucketing. An
        # out-of-range local id makes this longer than `num_local_experts`, which
        # `_grouped_linear` then rejects — a loud failure rather than silent corruption.
        tokens_per_expert = torch.bincount(sorted_ids, minlength=self.num_local_experts)
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

        gate_up = _grouped_linear(
            rows.to(self.gate_up_proj.dtype), self.gate_up_proj, offsets, bias=None,
            is_transposed=False
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self.act_fn(gate) * up
        projected = _grouped_linear(
            activated.to(self.down_proj.dtype), self.down_proj, offsets, bias=None,
            is_transposed=False
        )

        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(perm.shape[0], device=perm.device)
        return projected[inverse_perm].to(hidden_states.dtype)
