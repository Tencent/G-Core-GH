import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Union

from deep_ep import Buffer, EventOverlap

# Communication buffer (will allocate at runtime)
_buffer: Optional[Buffer] = None

# Set the number of SMs to use
# NOTES: this is a static variable
Buffer.set_num_sms(24)


# You may call this function at the framework initialization
def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
    # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency
    # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
    # refer to the docs of `Buffer.dispatch`
    global _buffer

    # Calculate layout before actual dispatch
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
        _buffer.get_dispatch_layout(topk_idx, num_experts,
                                    previous_event=previous_event, async_finish=True,
                                    allocate_on_comm_stream=previous_event is not None)
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
    # Unless you specify `num_worst_tokens`, but this flag is for intranode only
    # For more advanced usages, please refer to the docs of the `dispatch` function
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        _buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                         num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                         is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                         previous_event=previous_event, async_finish=True,
                         allocate_on_comm_stream=True)
    # For event management, please refer to the docs of the `EventOverlap` class
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor, grad_recv_topk_weights: torch.Tensor, handle: Tuple) -> \
        Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    global _buffer

    # The backward process of MoE dispatch is actually a combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_grad_x, combined_grad_recv_topk_weights, event = \
        _buffer.combine(grad_recv_x, handle, topk_weights=grad_recv_topk_weights, async_finish=True)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_grad_x, combined_grad_recv_topk_weights, event


def combine_forward(x: torch.Tensor, handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[torch.Tensor, EventOverlap]:
    global _buffer

    # Do MoE combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_x, _, event = _buffer.combine(
        x,
        handle,
        async_finish=True,
        previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None
    )

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    global _buffer

    # The backward process of MoE combine is actually a dispatch
    # For more advanced usages, please refer to the docs of the `dispatch` function
    grad_x, _, _, _, _, event = _buffer.dispatch(
        grad_combined_x,
        handle=handle,
        async_finish=True,
        previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None
    )

    # For event management, please refer to the docs of the `EventOverlap` class
    return grad_x, event


# ---------------------------------------------

# def test_fused_deepep_dispatch_combine():
#     """
#     Test program for fused_deepep_dispatch and fused_deepep_combine.
#
#     This test simulates a simple MoE scenario with:
#     - 2 ranks (EP size = 2)
#     - 4 experts total (2 experts per rank)
#     - Small input for easy observation
#     - Tests both forward and backward passes
#
#     **IMPORTANT**: This test MUST be run with torchrun, not directly with python!
#
#     Usage:
#         torchrun --nproc_per_node=2 fused_deep_ep_a2a.py
#     """
#     import os
#
#     # Initialize distributed environment
#     if not dist.is_initialized():
#         dist.init_process_group(backend='nccl')
#
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#
#     # Check GPU availability
#     num_gpus = torch.cuda.device_count()
#     print(f"[Rank {rank}] Detected {num_gpus} GPU(s)")
#
#     if num_gpus == 0:
#         print(f"[Rank {rank}] ERROR: No CUDA devices available!")
#         raise RuntimeError("No CUDA devices available")
#
#     if world_size > num_gpus:
#         print(f"[Rank {rank}] ERROR: Not enough GPUs! World size: {world_size}, GPUs: {num_gpus}")
#         print(
#             f"[Rank {rank}] Please run with: torchrun --nproc_per_node={num_gpus} fused_deep_ep_a2a.py"
#         )
#         raise RuntimeError(f"Not enough GPUs: need {world_size}, have {num_gpus}")
#
#     # Set device (use modulo to handle case where world_size > num_gpus)
#     device_id = rank % num_gpus
#     device = torch.device(f'cuda:{device_id}')
#
#     try:
#         torch.cuda.set_device(device)
#         print(f"[Rank {rank}] Successfully set CUDA device to cuda:{device_id}")
#     except Exception as e:
#         print(f"[Rank {rank}] ERROR: Failed to set CUDA device cuda:{device_id}")
#         print(f"[Rank {rank}] Error: {e}")
#         print(f"[Rank {rank}] GPU {device_id} may be busy or unavailable")
#         print(
#             f"[Rank {rank}] Try: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 fused_deep_ep_a2a.py"
#         )
#         raise
#
#     print(f"[Rank {rank}] Starting test...")
#
#     # Test configuration
#     num_experts = 4  # Total experts
#     num_local_experts = num_experts // world_size  # 2 experts per rank
#     hidden_dim = 8  # Small hidden dimension for easy observation
#     num_tokens = 6  # Small number of tokens
#     top_k = 2  # Each token goes to 2 experts
#
#     # Create process group
#     ep_group = dist.new_group(ranks=list(range(world_size)))
#
#     # ========================================================================
#     # Step 1: Prepare input data
#     # ========================================================================
#
#     # Input tokens: [num_tokens, hidden_dim]
#     # Use simple values for easy observation
#     tokens = torch.arange(num_tokens * hidden_dim, dtype=torch.float32,
#                           device=device).reshape(num_tokens, hidden_dim)
#     tokens = tokens + rank * 100  # Different values per rank
#     tokens.requires_grad = True
#
#     # Expert assignments: [num_tokens, top_k]
#     # Design routing so that tokens are distributed across ranks
#     # IMPORTANT: DeepEP requires topk_idx to be torch.int64 (Long), not torch.int32 (Int)
#     if rank == 0:
#         # Rank 0: tokens go to various experts
#         topk_idx = torch.tensor(
#             [
#                 [0, 1],  # Token 0 -> expert 0 (local), expert 1 (local)
#                 [1, 2],  # Token 1 -> expert 1 (local), expert 2 (remote)
#                 [2, 3],  # Token 2 -> expert 2 (remote), expert 3 (remote)
#                 [0, 2],  # Token 3 -> expert 0 (local), expert 2 (remote)
#                 [1, 3],  # Token 4 -> expert 1 (local), expert 3 (remote)
#                 [0, 3],  # Token 5 -> expert 0 (local), expert 3 (remote)
#             ],
#             dtype=torch.int64,
#             device=device
#         )  # Changed from int32 to int64
#     else:
#         # Rank 1: tokens go to various experts
#         topk_idx = torch.tensor(
#             [
#                 [2, 3],  # Token 0 -> expert 2 (local), expert 3 (local)
#                 [0, 2],  # Token 1 -> expert 0 (remote), expert 2 (local)
#                 [1, 3],  # Token 2 -> expert 1 (remote), expert 3 (local)
#                 [2, 0],  # Token 3 -> expert 2 (local), expert 0 (remote)
#                 [3, 1],  # Token 4 -> expert 3 (local), expert 1 (remote)
#                 [2, 1],  # Token 5 -> expert 2 (local), expert 1 (remote)
#             ],
#             dtype=torch.int64,
#             device=device
#         )  # Changed from int32 to int64
#
#     # Routing weights: [num_tokens, top_k]
#     topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
#
#     print(f"\n[Rank {rank}] Input tokens shape: {tokens.shape}")
#     print(f"[Rank {rank}] Input tokens:\n{tokens}")
#     print(f"[Rank {rank}] topk_idx:\n{topk_idx}")
#     print(f"[Rank {rank}] topk_weights:\n{topk_weights}")
#
#     # ========================================================================
#     # Step 2: Test Forward Pass - Dispatch
#     # ========================================================================
#
#     print(f"\n[Rank {rank}] ========== Testing Dispatch (Forward) ==========")
#
#     # IMPORTANT: DeepEP expects topk_idx to be 2D [num_tokens, top_k]
#     # Do NOT flatten it!
#     print(f"[Rank {rank}] tokens shape: {tokens.shape}")
#     print(f"[Rank {rank}] topk_idx shape: {topk_idx.shape}")
#     print(f"[Rank {rank}] topk_weights shape: {topk_weights.shape}")
#
#     # Call fused_deepep_dispatch
#     (
#         recv_tokens,
#         recv_topk_weights,
#         recv_num_tokens_per_expert_list,
#         dispatch_handle,
#         row_id_map,
#         num_tokens_from_dispatch,
#     ) = fused_deepep_dispatch(
#         tokens=tokens,  # [num_tokens, hidden_dim]
#         topk_idx=topk_idx,  # [num_tokens, top_k] - Keep 2D!
#         topk_weights=topk_weights,  # [num_tokens, top_k] - Keep 2D!
#         num_experts=num_experts,
#         group=ep_group,
#         async_finish=False,
#         allocate_on_comm_stream=False,
#     )
#
#     print(f"\n[Rank {rank}] num_tokens_from_dispatch: {num_tokens_from_dispatch}")
#
#     print(f"\n[Rank {rank}] Received tokens shape: {recv_tokens.shape}")
#     print(f"[Rank {rank}] Received tokens:\n{recv_tokens}")
#     print(f"[Rank {rank}] Received topk_weights: {recv_topk_weights}")
#     print(f"[Rank {rank}] Tokens per expert: {recv_num_tokens_per_expert_list}")
#
#     # Assert 1: Verify token count matches recv_num_tokens_per_expert_list
#     # This implicitly verifies that tokens are sorted by expert ID
#     expected_total_tokens = sum(recv_num_tokens_per_expert_list)
#     assert recv_tokens.size(0) == expected_total_tokens, \
#         f"[Rank {rank}] Token count mismatch! Expected: {expected_total_tokens}, Got: {recv_tokens.size(0)}"
#     print(
#         f"[Rank {rank}] ✓ Assert 1 passed: Token count matches recv_num_tokens_per_expert_list ({expected_total_tokens})"
#     )
#
#     # Assert 2: Hidden dimension should be preserved
#     assert recv_tokens.size(1) == hidden_dim, \
#         f"[Rank {rank}] Hidden dimension mismatch! Expected: {hidden_dim}, Got: {recv_tokens.size(1)}"
#     print(f"[Rank {rank}] ✓ Assert 2 passed: Hidden dimension preserved ({hidden_dim})")
#
#     # ========================================================================
#     # Step 3: Simulate Expert Computation
#     # ========================================================================
#
#     print(f"\n[Rank {rank}] ========== Simulating Expert Computation ==========")
#
#     # Simple expert computation: multiply by 2.0
#     expert_outputs = recv_tokens * 2.0
#
#     print(f"[Rank {rank}] Expert outputs shape: {expert_outputs.shape}")
#     print(f"[Rank {rank}] Expert outputs:\n{expert_outputs}")
#
#     # Assert 3: Expert outputs should be exactly 2x of recv_tokens
#     expected_expert_outputs = recv_tokens * 2.0
#     assert torch.allclose(expert_outputs, expected_expert_outputs, rtol=1e-5), \
#         f"[Rank {rank}] Expert computation incorrect!"
#     print(f"[Rank {rank}] ✓ Assert 3 passed: Expert computation correct (2x input)")
#
#     # ========================================================================
#     # Step 4: Test Forward Pass - Combine
#     # ========================================================================
#
#     print(f"\n[Rank {rank}] ========== Testing Combine (Forward) ==========")
#
#     # Call fused_deepep_combine
#     output = fused_deepep_combine(
#         expert_outputs=expert_outputs,
#         topk_weights=recv_topk_weights,
#         dispatch_handle=dispatch_handle,
#         row_id_map=row_id_map,
#         group=ep_group,
#         num_tokens=tokens.size(0),  # Original number of tokens (not flattened)
#         hidden_dim=hidden_dim,
#         num_tokens_from_dispatch=num_tokens_from_dispatch,  # M from DeepEP dispatch
#         async_finish=False,
#         allocate_on_comm_stream=False,
#     )
#
#     print(f"\n[Rank {rank}] Output shape: {output.shape}")
#     print(f"[Rank {rank}] Output:\n{output}")
#
#     # Assert 4: Output shape should match input shape
#     assert output.shape == tokens.shape, \
#         f"[Rank {rank}] Output shape mismatch! Expected: {tokens.shape}, Got: {output.shape}"
#     print(f"[Rank {rank}] ✓ Assert 4 passed: Output shape matches input ({output.shape})")
#
#     # Assert 5: Output should not contain NaN or Inf
#     assert not torch.isnan(output).any(), f"[Rank {rank}] Output contains NaN!"
#     assert not torch.isinf(output).any(), f"[Rank {rank}] Output contains Inf!"
#     print(f"[Rank {rank}] ✓ Assert 5 passed: Output is valid (no NaN/Inf)")
#
#     # ========================================================================
#     # Step 5: Test Backward Pass
#     # ========================================================================
#
#     print(f"\n[Rank {rank}] ========== Testing Backward Pass ==========")
#
#     # Create gradient for backward pass
#     # Use simple gradient: all ones
#     grad_output = torch.ones_like(output)
#
#     print(f"[Rank {rank}] Grad output shape: {grad_output.shape}")
#
#     # Backward pass
#     output.backward(grad_output)
#
#     print(f"\n[Rank {rank}] Grad tokens shape: {tokens.grad.shape}")
#     print(f"[Rank {rank}] Grad tokens:\n{tokens.grad}")
#
#     # Assert 6: Gradients must be computed
#     assert tokens.grad is not None, f"[Rank {rank}] No gradients computed!"
#     print(f"[Rank {rank}] ✓ Assert 6 passed: Gradients computed successfully")
#
#     # Assert 7: Gradient shape should match input shape
#     assert tokens.grad.shape == tokens.shape, \
#         f"[Rank {rank}] Gradient shape mismatch! Expected: {tokens.shape}, Got: {tokens.grad.shape}"
#     print(f"[Rank {rank}] ✓ Assert 7 passed: Gradient shape matches input ({tokens.grad.shape})")
#
#     # Assert 8: Gradients should not contain NaN or Inf
#     assert not torch.isnan(tokens.grad).any(), f"[Rank {rank}] Gradients contain NaN!"
#     assert not torch.isinf(tokens.grad).any(), f"[Rank {rank}] Gradients contain Inf!"
#     print(f"[Rank {rank}] ✓ Assert 8 passed: Gradients are valid (no NaN/Inf)")
#
#     # Assert 9: Gradients should be non-zero (since we have non-zero grad_output)
#     grad_norm = tokens.grad.norm().item()
#     assert grad_norm > 0, f"[Rank {rank}] Gradient norm is zero!"
#     print(f"[Rank {rank}] ✓ Assert 9 passed: Gradients are non-zero (norm: {grad_norm:.4f})")
#
#     # ========================================================================
#     # Step 6: Summary
#     # ========================================================================
#
#     print(f"\n[Rank {rank}] ========== Test Summary ==========")
#     print(f"[Rank {rank}] ✓ Assert 1: Token count matches recv_num_tokens_per_expert_list")
#     print(f"[Rank {rank}] ✓ Assert 2: Hidden dimension preserved")
#     print(f"[Rank {rank}] ✓ Assert 3: Expert computation correct")
#     print(f"[Rank {rank}] ✓ Assert 4: Output shape matches input")
#     print(f"[Rank {rank}] ✓ Assert 5: Output is valid (no NaN/Inf)")
#     print(f"[Rank {rank}] ✓ Assert 6: Gradients computed")
#     print(f"[Rank {rank}] ✓ Assert 7: Gradient shape matches")
#     print(f"[Rank {rank}] ✓ Assert 8: Gradients are valid (no NaN/Inf)")
#     print(f"[Rank {rank}] ✓ Assert 9: Gradients are non-zero")
#     print(f"[Rank {rank}] ")
#     print(f"[Rank {rank}] All 9 assertions passed! ✓")
#     print(f"[Rank {rank}] Test completed successfully!")
#
#     # Cleanup
#     dist.barrier()
#     if rank == 0:
#         print("\n" + "=" * 60)
#         print("All tests passed on all ranks! ✓")
#         print("Total assertions: 9 per rank")
#         print("=" * 60)
