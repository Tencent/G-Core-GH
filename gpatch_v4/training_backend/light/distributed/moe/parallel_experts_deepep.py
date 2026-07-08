"""DeepEP token dispatch mixin (Automodel-style)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from .parallel_experts_mixin import EPExpertsMixin

try:
    from nemo_automodel.components.moe.megatron.token_dispatcher import (
        MoEFlexTokenDispatcher,
        TokenDispatcherConfig,
    )
except ImportError:
    MoEFlexTokenDispatcher = None  # type: ignore[assignment,misc]
    TokenDispatcherConfig = None  # type: ignore[assignment,misc]

DEEPEP_AVAILABLE = MoEFlexTokenDispatcher is not None
DEEPEP_IMPORT_ERROR = (
    "DeepEP dispatch requires nemo_automodel with DeepEP installed. "
    "Use ep_dispatch='alltoall' or install DeepEP."
)


class DeepEPEPExpertsMixin(EPExpertsMixin):
    """EP experts forward via DeepEP / Automodel ``MoEFlexTokenDispatcher``."""

    _fsdp_ep_cp_dispatch = "deepep"
    _token_dispatcher: Optional[object] = None

    def init_ep_dispatcher(self) -> None:
        """Build DeepEP token dispatcher on the current EP group."""
        if not DEEPEP_AVAILABLE:
            raise ImportError(DEEPEP_IMPORT_ERROR)

        ep_group = self._resolve_ep_group()
        if ep_group is None:
            raise RuntimeError("DeepEP dispatch requires an initialized EP process group.")

        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
        num_local_experts = self.num_experts // ep_size
        top_k = getattr(self.config, "num_experts_per_tok",
                        None) or getattr(self.config, "top_k", 2)

        config = TokenDispatcherConfig(
            moe_router_topk=top_k,
            num_moe_experts=self.num_experts,
            moe_permute_fusion=True,
            moe_enable_deepep=True,
            moe_flex_dispatcher_backend="deepep",
        )
        offset = ep_rank * num_local_experts
        local_indices = [offset + i for i in range(num_local_experts)]
        self._token_dispatcher = MoEFlexTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_indices,
            config=config,
            ep_group=ep_group,
        )
        self._ep_size = ep_size

    def _ep_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._token_dispatcher is None:
            self.init_ep_dispatcher()

        assert self.num_experts % self._ep_size == 0
        token_mask = torch.ones(
            hidden_states.size(0), device=hidden_states.device, dtype=torch.bool
        )
        indices = top_k_index.masked_fill(~token_mask.unsqueeze(-1), -1)

        permuted, tokens_per_expert, permuted_probs = self._token_dispatcher.token_permutation2(
            hidden_states=hidden_states,
            num_local_tokens=hidden_states.size(0),
            token_probs=top_k_weights.float(),
            token_indices=indices,
        )
        permuted_probs = permuted_probs.unsqueeze(-1)

        if torch.count_nonzero(tokens_per_expert) == 0:
            dummy = torch.matmul(hidden_states[:1], self.gate_up_proj[0:1].transpose(-1, -2))
            return hidden_states + dummy.sum() * 0.0

        expert_out = self.local_grouped_forward(
            permuted,
            tokens_per_expert.to(permuted.device),
            self._get_apply_gate(),
            permuted_probs=permuted_probs,
        )
        return self._token_dispatcher.token_unpermutation(expert_out)
