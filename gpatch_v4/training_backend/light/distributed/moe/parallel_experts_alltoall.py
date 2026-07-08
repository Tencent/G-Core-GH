"""All-to-all EP dispatch: comm primitives, token routing, and Experts mixin."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .parallel_experts_mixin import EPExpertsMixin
from .parallel_experts_utils import (
    generate_weights_idx,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input_tensor, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        if dist.get_world_size(group=group) == 1:
            return input_tensor

        input_tensor = input_tensor.contiguous()
        if output_split_sizes is None:
            output = torch.empty_like(input_tensor)
        else:
            output = torch.empty(
                (sum(output_split_sizes), input_tensor.size(1)),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
        dist.all_to_all_single(
            output,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


def all_to_all(
    group: dist.ProcessGroup,
    input_tensor: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
) -> torch.Tensor:
    return _AllToAll.apply(group, input_tensor, output_split_sizes, input_split_sizes)


def preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> Tuple[list[int], list[int], torch.Tensor, torch.Tensor]:
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))

    input_splits = num_local_tokens_per_expert.reshape(ep_size,
                                                       num_local_experts).sum(dim=1).tolist()

    num_global_tokens_per_expert = torch.zeros(
        ep_size,
        num_local_tokens_per_expert.size(0),
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(
        num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group
    )

    start_idx = rank * num_local_experts
    end_idx = (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:,
                                                                      start_idx:end_idx].contiguous(
                                                                      )
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    sum_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0
                                                                 ).to("cpu", non_blocking=True)
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
        -1, num_local_experts
    ).to("cpu", non_blocking=True)
    return input_splits, output_splits, num_global_tokens_per_local_expert, sum_per_local_expert


def token_pre_all2all(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits: list[int],
    output_splits: list[int],
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    # 处理空tensor的情况
    if hidden_states.numel() == 0:
        hidden_dim = hidden_states.size(-1) if hidden_states.dim() > 0 else 0
        org_shape = torch.Size([0, hidden_dim])
        routing_map = expert_mask.sum(dim=1)
        local_perm_mapping = torch.empty(0, dtype=torch.long, device=hidden_states.device)
        return hidden_states.reshape(-1, hidden_dim), routing_map, local_perm_mapping, org_shape

    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)

    local_permuted, local_perm_mapping = permute(hidden_states, routing_map)
    global_permuted = all_to_all(ep_group, local_permuted, output_splits, input_splits)

    num_local_experts = num_experts // ep_group.size()
    permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    global_permuted = sort_chunks_by_idxs(
        global_permuted,
        num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )
    return global_permuted, routing_map, local_perm_mapping, org_shape


def tokens_post_all2all(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: int,
    num_experts: int,
    input_splits: list[int],
    output_splits: list[int],
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_map: torch.Tensor,
    local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    # 即使expert_outputs是空的，也必须参与all_to_all通信
    # 否则其他rank会在等待时卡住

    num_local_experts = num_experts // ep_group.size()
    unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()

    # sort_chunks_by_idxs需要处理空tensor
    expert_outputs = sort_chunks_by_idxs(
        expert_outputs,
        num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )

    # all_to_all必须调用，即使输入是空的
    unpermute_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)

    # 如果unpermute_outputs是空的，尝试调用unpermute
    # 如果unpermute不能处理空tensor，我们需要创建一个与计算图有连接的零张量
    if unpermute_outputs.numel() == 0:
        # 创建一个与计算图有连接的零张量
        # 方法：对unpermute_outputs进行无操作计算，然后创建正确形状的零张量
        zero_sum = unpermute_outputs.sum()  # 对空tensor求和得到0
        zero_scalar = zero_sum * 0.0  # 乘以0，但保持梯度连接

        # 创建正确形状的零张量
        zero_tensor = torch.zeros(
            org_hidden_states_shape, device=unpermute_outputs.device, dtype=unpermute_outputs.dtype
        )

        # 将zero_tensor与zero_scalar相乘，保持梯度连接
        result = zero_tensor * zero_scalar
        result.requires_grad_(unpermute_outputs.requires_grad)
        return result

    return unpermute(
        unpermute_outputs,
        routing_weights,
        org_hidden_states_shape,
        local_input_permutation_mapping,
        routing_map,
    )


class AllToAllEPExpertsMixin(EPExpertsMixin):
    """EP experts forward via ``dist.all_to_all`` token dispatch."""

    _fsdp_ep_cp_dispatch = "alltoall"

    @staticmethod
    def _build_expert_mask(top_k_index: torch.Tensor, num_experts: int) -> torch.Tensor:
        return F.one_hot(top_k_index,
                         num_classes=num_experts).to(dtype=torch.int32).permute(2, 1, 0)

    def _ep_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        ep_group = self._resolve_ep_group()
        assert ep_group is not None

        num_experts = self.num_experts
        expert_mask = self._build_expert_mask(top_k_index, num_experts)
        input_splits, output_splits, num_global_per_local, tokens_per_expert = preprocess(
            expert_mask, num_experts, ep_group
        )
        permuted, routing_map, perm_mapping, org_shape = token_pre_all2all(
            hidden_states,
            expert_mask,
            num_experts,
            input_splits,
            output_splits,
            num_global_per_local,
            ep_group,
        )

        expert_out = self.local_grouped_forward(
            permuted,
            tokens_per_expert,
            self._get_apply_gate(),
        )
        weights_idx = generate_weights_idx(top_k_weights, top_k_index, num_experts)
        return tokens_post_all2all(
            expert_out,
            weights_idx,
            top_k_index.size(-1),
            num_experts,
            input_splits,
            output_splits,
            num_global_per_local,
            routing_map,
            perm_mapping,
            org_shape,
            ep_group,
        )
