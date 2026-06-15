# coding=utf-8
# Adapted from:
#   Megatron-LM/megatron/core/transformer/moe/fused_a2a.py
# for WeChat-YATT DeepSeek-V4.

from __future__ import annotations

import torch
import torch.distributed as dist

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    Buffer = None
    EventHandle = None
    EventOverlap = None
    HAVE_DEEP_EP = False

_buffer = None


def _require_deepep() -> None:
    if not HAVE_DEEP_EP:
        raise ImportError("DeepEP backend requested, but `deep_ep` is not installed.")


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: dist.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    _require_deepep()
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or
        _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""
    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        previous_event = EventOverlap(EventHandle()) if async_finish else None
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(
            num_recv_tokens_per_expert_list,
            dtype=torch.int32,
            device=recv_x.device,
        )
        return recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        previous_event = EventOverlap(EventHandle()) if ctx.async_finish else None
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            ctx.handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""
    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        previous_event = EventOverlap(EventHandle()) if async_finish else None
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


def fused_dispatch(
    x: torch.Tensor,
    token_indices: torch.Tensor,
    token_probs: torch.Tensor,
    num_experts: int,
    group: dist.ProcessGroup,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
):
    """Perform fused dispatch operation if deep_ep is available.

    Args:
        x: Input tensor [num_tokens, hidden_size]
        token_indices: Token routing indices [num_tokens, topk]
        token_probs: Token routing probabilities [num_tokens, topk]
        num_experts: Number of experts
        group: Process group
        previous_event: Previous CUDA event

    Returns:
        Result of FusedDispatch
    """
    _require_deepep()
    assert token_indices.dtype == torch.int64, (
        f"DeepEP requires token_indices dtype torch.int64, got {token_indices.dtype}"
    )
    assert token_indices.dim() == 2, (
        f"DeepEP requires token_indices shape [num_tokens, top_k], got {tuple(token_indices.shape)}"
    )
    return FusedDispatch.apply(
        x.contiguous(),
        token_indices.contiguous(),
        token_probs.float().contiguous(),
        num_experts,
        group,
        async_finish,
        allocate_on_comm_stream,
    )


def fused_combine(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    handle,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
) -> torch.Tensor:
    """Perform fused combine operation if deep_ep is available.

    Args:
        x: Input tensor
        group: Process group
        handle: Communication handle
        previous_event: Previous CUDA event

    Returns:
        Result of FusedCombine
    """
    _require_deepep()
    return FusedCombine.apply(
        x.contiguous(),
        group,
        handle,
        async_finish,
        allocate_on_comm_stream,
    )


def set_deepep_num_sms(num_sms: int) -> None:
    """Sets the number of SMs to use for DeepEP"""
    _require_deepep()
    Buffer.set_num_sms(num_sms)
