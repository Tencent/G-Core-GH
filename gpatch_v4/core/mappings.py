# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.

from typing import List

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def _all_reduce_of_cp(input_):
    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_context_parallel_group())
    return input_


class _AllReduceOfContextParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _all_reduce_of_cp(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def reduce_from_context_parallel_region(input_):
    return _AllReduceOfContextParallel.apply(input_)


class _AllGatherToContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, seq_dim, bwd_op):
        assert seq_dim in [0, 1] and input_.dim() > seq_dim
        cp_size = get_context_parallel_world_size()
        # Split local_logits into two parts
        input_ = input_.view(
            *input_.shape[0:seq_dim],
            2,
            input_.shape[seq_dim] // 2,
            *input_.shape[(seq_dim + 1):],
        )

        gathered_logits = [torch.zeros_like(input_) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered_logits, input_, group=get_context_parallel_group())

        reorded_logits = [None for _ in range(2 * cp_size)]
        if seq_dim == 1:
            for rank in range(cp_size):
                reorded_logits[rank] = gathered_logits[rank][:, 0]
                reorded_logits[2 * cp_size - rank - 1] = gathered_logits[rank][:, 1]
        elif seq_dim == 0:
            for rank in range(cp_size):
                reorded_logits[rank] = gathered_logits[rank][0]
                reorded_logits[2 * cp_size - rank - 1] = gathered_logits[rank][1]
        else:
            assert False
        gathered_logits = torch.cat(reorded_logits, dim=seq_dim)

        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        ctx.cp_group = get_context_parallel_group()
        ctx.local_shape = input_.shape
        ctx.bwd_op = bwd_op

        return gathered_logits

    @staticmethod
    def backward(ctx, grad_output):
        seq_dim = ctx.seq_dim
        assert seq_dim in [0, 1]
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        local_shape = ctx.local_shape
        bwd_op = ctx.bwd_op

        # Reshape grad_output to match the forward pass
        grad_output = grad_output.view(
            *grad_output.shape[0:seq_dim],
            2 * cp_size,
            grad_output.shape[seq_dim] // (2 * cp_size),
            *grad_output.shape[(seq_dim + 1):],
        )

        reordered_indices = []
        for rank in range(cp_size):
            reordered_indices.append(rank)
            reordered_indices.append(2 * cp_size - rank - 1)
        if seq_dim == 1:
            grad_output = grad_output[:, reordered_indices, :]
        elif seq_dim == 0:
            grad_output = grad_output[reordered_indices, :]
        else:
            assert False

        split_tensors = torch.split(grad_output, grad_output.size(seq_dim) // cp_size, dim=seq_dim)
        grad_list = [t.contiguous() for t in split_tensors]
        assert split_tensors[0].shape == local_shape

        local_grad = torch.empty(
            local_shape, dtype=grad_output.dtype, device=torch.cuda.current_device()
        )
        torch.distributed.reduce_scatter(local_grad, grad_list, op=bwd_op, group=cp_group)

        local_grad = local_grad.view(
            *local_grad.shape[0:seq_dim],
            -1,
            *local_grad.shape[(seq_dim + 2):],
        )
        return local_grad, None, None


def _gather_along_any_dim(input_, gather_dim, comm_group=None):
    """Gather tensors and concatinate along the last dimension."""

    if comm_group is not None:
        world_size = torch.distributed.get_world_size(group=comm_group)
    else:
        world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    assert gather_dim < input_.dim() and gather_dim >= 0, "Invalid dimension to gather along."
    if comm_group is not None:
        rank = torch.distributed.get_rank(group=comm_group)
    else:
        rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    if comm_group is None:
        comm_group = get_tensor_model_parallel_group()
    torch.distributed.all_gather(tensor_list, input_, group=comm_group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=gather_dim).contiguous()

    return output


def split_tensor_along_any_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
    any_dim=0,
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Args:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    assert any_dim < tensor.dim() and any_dim >= 0, "Invalid dimension"
    # Get the size and dimension.
    any_dim_size = divide(tensor.size()[any_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, any_dim_size, dim=any_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def _split_along_any_dim(input_, split_dim, comm_group=None):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    assert split_dim >= 0 and split_dim < input_.dim()

    if comm_group is not None:
        world_size = torch.distributed.get_world_size(group=comm_group)
    else:
        world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_any_dim(input_, world_size, any_dim=split_dim)

    # Note: torch.split does not create contiguous tensors by default.
    if comm_group is not None:
        rank = torch.distributed.get_rank(group=comm_group)
    else:
        rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _reduce_scatter_along_any_dim(input_, dim, comm_group=None):
    """Reduce-scatter the input tensor across model parallel group."""
    assert dim < input_.dim() and dim >= 0, "Invalid dimension to reduce-scatter along."
    if comm_group is not None:
        world_size = torch.distributed.get_world_size(group=comm_group)
    else:
        world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert (
        dim_size[dim] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    dim_size[dim] = dim_size[dim] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    if comm_group is None:
        comm_group = get_tensor_model_parallel_group()
    torch.distributed.reduce_scatter_tensor(output, input_.contiguous(), group=comm_group)
    return output


class _GatherFromAnyDim(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""
    @staticmethod
    def symbolic(graph, input_, gather_dim=1, tensor_parallel_output_grad=True, comm_group=None):
        return _gather_along_any_dim(input_, gather_dim=gather_dim, comm_group=comm_group)

    @staticmethod
    def forward(ctx, input_, gather_dim=1, tensor_parallel_output_grad=True, comm_group=None):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.gather_dim = gather_dim
        ctx.comm_group = comm_group
        return _gather_along_any_dim(input_, gather_dim=gather_dim, comm_group=comm_group)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad
        gather_dim = ctx.gather_dim
        comm_group = ctx.comm_group
        if tensor_parallel_output_grad:
            return _reduce_scatter_along_any_dim(
                grad_output, dim=gather_dim, comm_group=comm_group
            ), None, None, None
        else:
            return _split_along_any_dim(
                grad_output, split_dim=gather_dim, comm_group=comm_group
            ), None, None, None


def all_gather_from_context_parallel_region(
    local_tensor, gather_dim=1, bwd_op=torch.distributed.ReduceOp.AVG
):
    return _AllGatherToContextParallelRegion.apply(local_tensor, gather_dim, bwd_op)


# todo zz: invert the shared CP partition map, including per-segment THD
class _NoZigzagAllGatherToCPRegion(torch.autograd.Function):
    """All-gather across CP ranks with contiguous rank-order concat (dsv4 / non-zigzag).

    Inverse of :func:`gpatch_v4.models.deepseek_v4.cp.cp_chunk_data`.
    No block interleaving — rank r's chunk goes straight to positions
    ``[r * local_s, (r+1) * local_s]`` of the output.
    """
    @staticmethod
    def forward(ctx, input_, gather_dim, bwd_op):
        assert gather_dim in [0, 1] and input_.dim() > gather_dim
        cp_size = get_context_parallel_world_size()

        gathered_list = [torch.zeros_like(input_) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered_list, input_, group=get_context_parallel_group())

        output = torch.cat(gathered_list, dim=gather_dim)

        ctx.cp_size = cp_size
        ctx.gather_dim = gather_dim
        ctx.cp_group = get_context_parallel_group()
        ctx.bwd_op = bwd_op

        return output

    @staticmethod
    def backward(ctx, grad_output):
        gather_dim = ctx.gather_dim
        cp_size = ctx.cp_size
        cp_group = ctx.cp_group
        bwd_op = ctx.bwd_op

        chunk_size = grad_output.shape[gather_dim] // cp_size
        split_tensors = torch.split(grad_output, chunk_size, dim=gather_dim)
        grad_list = [t.contiguous() for t in split_tensors]

        local_grad = torch.empty_like(grad_list[0])
        torch.distributed.reduce_scatter(local_grad, grad_list, op=bwd_op, group=cp_group)

        return local_grad, None, None


def all_gather_from_context_parallel_region_no_zigzag(
    local_tensor, gather_dim=1, bwd_op=torch.distributed.ReduceOp.AVG
):
    return _NoZigzagAllGatherToCPRegion.apply(local_tensor, gather_dim, bwd_op)


def gather_from_any_dim(input_, gather_dim=1, tensor_parallel_output_grad=True, comm_group=None):
    return _GatherFromAnyDim.apply(input_, gather_dim, tensor_parallel_output_grad, comm_group)


class AllGatherVisionEmbeddings(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, seqlens_on_cp_ranks):
        outputs = []
        for i in range(len(seqlens_on_cp_ranks)):
            o = torch.zeros(
                (seqlens_on_cp_ranks[i].sum(), *input.shape[1:]),
                device=input.device,
                dtype=input.dtype,
                layout=input.layout,
            )
            outputs.append(o)
        torch.distributed.all_gather(
            outputs, input, group=parallel_state.get_context_parallel_group()
        )
        cp_rank = parallel_state.get_context_parallel_rank()
        ctx.cp_rank = cp_rank
        ctx.save_for_backward(*seqlens_on_cp_ranks)

        output = torch.cat(outputs, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cp_rank = ctx.cp_rank
        seqlens_on_cp_ranks = ctx.saved_tensors
        start_idx = torch.cat(seqlens_on_cp_ranks[:cp_rank]).sum() if cp_rank != 0 else 0
        end_idx = start_idx + seqlens_on_cp_ranks[cp_rank].sum()
        grad_output = grad_output[start_idx:end_idx]
        return grad_output, None
