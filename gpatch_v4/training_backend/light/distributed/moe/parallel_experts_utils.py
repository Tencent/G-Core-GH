"""MoE permute / unpermute helpers (VeOmni-style)."""

from __future__ import annotations

import torch


def permute(tokens: torch.Tensor, routing_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = torch.arange(num_tokens,
                                 device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)
    return tokens.index_select(0, sorted_indices), sorted_indices


def unpermute(
    tokens: torch.Tensor,
    routing_weights: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
) -> torch.Tensor:
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]
    out = torch.zeros(hidden_states_shape, device=tokens.device, dtype=torch.float32)
    out.scatter_add_(0, permutation_mapping.unsqueeze(1).expand(-1, hidden_dim), tokens.float())
    return out.to(tokens.dtype)


def generate_weights_idx(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    num_tokens, _ = routing_weights.shape
    weights_idx = torch.zeros(
        (num_tokens, num_experts), dtype=routing_weights.dtype, device=routing_weights.device
    )
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx


def sort_chunks_by_idxs(
    input_tensor: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: list[int]
) -> torch.Tensor:
    chunks = torch.split(input_tensor, split_sizes.tolist(), dim=0)
    return torch.cat([chunks[i] for i in sorted_idxs], dim=0)
