"""EP-aware Experts mixin base (shared grouped GEMM helpers)."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from packaging.version import Version as _V

from ..core.parallel_state import get_parallel_state

# torch._grouped_mm (CUTLASS) backward hangs / produces invalid strides when any
# expert group has split_size == 0.  This was fixed upstream in torch 2.10.
# See pytorch/pytorch#152668, pytorch/pytorch#172439.
_GROUPED_MM_EMPTY_GROUP_FIXED = _V(torch.__version__.split("+")[0]) >= _V("2.10.0")


class EPExpertsMixin:
    """Mixin for HuggingFace ``@use_experts_implementation`` Experts modules."""

    _fsdp_ep_cp_dispatch: str = "base"

    def _resolve_ep_group(self) -> Optional[dist.ProcessGroup]:
        ep_group = getattr(self, "_ep_group_override", None)
        if ep_group is not None:
            return ep_group
        try:
            return get_parallel_state().ep_group
        except RuntimeError:
            return None

    def _ep_enabled(self) -> bool:
        ep_group = self._resolve_ep_group()
        return ep_group is not None and dist.get_world_size(ep_group) > 1

    @staticmethod
    def _tokens_to_offsets(tokens_per_expert: torch.Tensor, device: torch.device) -> torch.Tensor:
        counts = tokens_per_expert.to(device=device, dtype=torch.int64)
        return torch.cumsum(counts, dim=0, dtype=torch.int32)

    @staticmethod
    def _pad_empty_groups(
        tokens: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Pad zero-length expert groups with one zero row each (when needed).

        ``torch._grouped_mm`` (CUTLASS) hangs / produces invalid-stride gradients
        in backward when any expert group has 0 tokens (``split_size == 0``).
        This was fixed in torch >= 2.10 (pytorch/pytorch#152668, #172439).

        On torch < 2.10 we insert one dummy zero row per empty expert so every
        group has >= 1 token.  On torch >= 2.10 no padding is needed.

        Returns ``(tokens, offs, dest)`` where ``offs`` is the int32 cumulative
        count tensor and ``dest`` is an index that maps each original token to its
        row in the (possibly padded) tensor; ``dest`` is ``None`` when no padding
        was applied.  After the down-projection call, ``out.index_select(0, dest)``
        recovers the original token order.  The inserted dummy rows are constant
        zeros so they receive a correct zero weight gradient in backward.
        """
        counts = tokens_per_expert.to(device=tokens.device, dtype=torch.long)
        # Check version first (CPU constant, no GPU sync) before the all() check
        # (which forces a GPU→CPU sync to evaluate the boolean).
        if _GROUPED_MM_EMPTY_GROUP_FIXED or bool((counts > 0).all()):
            return tokens, torch.cumsum(counts, dim=0, dtype=torch.int32), None

        num_experts = counts.numel()
        n_real = tokens.shape[0]
        device = tokens.device
        padded_counts = counts.clamp(min=1)

        expert_ids = torch.repeat_interleave(torch.arange(num_experts, device=device), counts)
        src_start = torch.cumsum(counts, dim=0) - counts
        within = torch.arange(n_real, device=device) - src_start[expert_ids]
        dst_start = torch.cumsum(padded_counts, dim=0) - padded_counts
        dest = dst_start[expert_ids] + within

        padded_total = int(padded_counts.sum().item())
        padded_tokens = tokens.new_zeros(padded_total, tokens.shape[-1]).index_copy(0, dest, tokens)
        offs = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
        return padded_tokens, offs, dest

    def _handle_empty_tokens(
        self,
        tokens: torch.Tensor,
        *,
        permuted_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Handle the empty-tokens case while keeping ``gate_up_proj`` /
        ``down_proj`` (and optional gate biases) in the autograd graph.

        When a rank receives 0 routed tokens, we still need every expert weight
        to appear in the forward graph so DDP / FSDP reducers do not treat them
        as "unused parameters" (which would either error out or break the
        all-reduce schedule). The mathematically-correct gradient for these
        weights *in this batch* is zero.

        Implementation note — why we do NOT use ``_grouped_linear`` here:
        ``torch._grouped_mm`` backward expands the upstream scalar gradient
        (from ``.sum()``) back to the operand shape using stride=0 broadcasting.
        The resulting non-contiguous tensor (strides ``[0, 0]``) is then passed
        back into grouped_mm as mat_a, which the CUTLASS kernel rejects with
        "Invalid strides/sizes". This cannot be fixed by padding because the
        pathological strides originate from autograd's `.sum()` backward, not
        from empty expert groups.

        Instead we register each parameter in the graph via a plain scalar
        ``.sum() * 0.0`` — no matmul, no stride issues, zero gradient. Adding
        the resulting scalars to the (empty) output tensor broadcasts correctly
        and keeps the grad_fn chain alive so every parameter gets a zero gradient
        rather than ``None``.
        """
        # Touch each expert parameter without going through grouped_mm.
        # .sum() * 0.0  →  scalar 0 that is differentiable w.r.t. the weight
        # (grad will be zeros_like(weight), not None — satisfies DDP/FSDP reducer).
        zero = self.gate_up_proj.sum() * 0.0 + self.down_proj.sum() * 0.0
        if getattr(self, "has_bias", False):
            if hasattr(self, "gate_up_proj_bias"):
                zero = zero + self.gate_up_proj_bias.sum() * 0.0
            if hasattr(self, "down_proj_bias"):
                zero = zero + self.down_proj_bias.sum() * 0.0

        if permuted_probs is not None:
            zero = zero + permuted_probs.sum() * 0.0

        # tokens is (0, H); adding a scalar broadcasts to (0, H) with no elements,
        # so the output is numerically zero and correctly shaped.
        return tokens + zero * 0.0

    def local_grouped_forward(
        self,
        tokens: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        apply_gate: Callable[[torch.Tensor], torch.Tensor],
        *,
        permuted_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 检查是否是空tensor
        if tokens.numel() == 0 or tokens_per_expert.sum() == 0:
            return self._handle_empty_tokens(tokens, permuted_probs=permuted_probs)

        from transformers.integrations.moe import _grouped_linear

        is_transposed = getattr(self, "is_transposed", False)
        # Pad zero-length expert groups (some experts get no tokens) so the real
        # CUTLASS grouped-GEMM never sees split_size==0, whose backward hangs /
        # produces invalid grads on torch < 2.10. Padded rows are dropped via
        # ``dest`` after the down projection.
        tokens, offs, dest = self._pad_empty_groups(tokens, tokens_per_expert)
        if dest is not None and permuted_probs is not None:
            permuted_probs = permuted_probs.new_zeros(tokens.shape[0],
                                                      *permuted_probs.shape[1:]).index_copy(
                                                          0, dest, permuted_probs
                                                      )
        gate_up_out = _grouped_linear(
            tokens, self.gate_up_proj, offs=offs, is_transposed=is_transposed
        )
        if permuted_probs is not None:
            hidden = apply_gate(gate_up_out)
            hidden = hidden * permuted_probs
        else:
            hidden = apply_gate(gate_up_out)
        out = _grouped_linear(hidden, self.down_proj, offs=offs, is_transposed=is_transposed)
        if dest is not None:
            out = out.index_select(0, dest)
        return out

    def _get_apply_gate(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self._apply_gate if getattr(self, "has_gate", True) else self.act_fn

    def _ep_fallback_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.integrations.moe import batched_mm_experts_forward

        return batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self._ep_enabled():
            return self._ep_fallback_forward(hidden_states, top_k_index, top_k_weights)
        return self._ep_forward(hidden_states, top_k_index, top_k_weights)

    def _ep_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
