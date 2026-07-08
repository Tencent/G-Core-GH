"""All-to-all EP dispatch for NPU.

Inherits from ``AllToAllEPExpertsMixin`` and overrides three parts that benefit
from NPU-specific fused ops:

  1. ``_ep_forward``          : replaces CPU ``permute / unpermute`` with
                                ``npu_moe_token_permute / npu_moe_token_unpermute``
                                and uses a ``bincount``-based split preprocess
                                (avoids constructing a full one-hot expert mask).
  2. ``local_grouped_forward``: replaces HF ``_grouped_linear`` + cumsum with
                                ``npu_grouped_matmul`` (``group_list_type=1``,
                                raw counts) + ``npu_swiglu``.

Everything else (``all_to_all`` primitive, ``_resolve_ep_group``,
``_get_apply_gate``, ``_ep_fallback_forward``, ``forward``, …) is inherited
from ``AllToAllEPExpertsMixin`` / ``EPExpertsMixin``.

Weight shape convention (is_transposed=True / VeOmni style):
  gate_up_proj : (E, 2*I, H)  — (num_experts, out_features, in_features)
  down_proj    : (E,   H, I)  — same convention
``npu_grouped_matmul`` computes ``x @ W``, so weights are transposed to
(E, in, out) before the kernel call when is_transposed=True.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import torch_npu
    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False

# Reuse all_to_all primitive and GPU mixin from the GPU implementation.
from .parallel_experts_alltoall import AllToAllEPExpertsMixin, all_to_all
from .parallel_experts_utils import sort_chunks_by_idxs

# ─────────────────────────────────────────────────────────────────────────────
# NPU grouped GEMM autograd function
# ─────────────────────────────────────────────────────────────────────────────


class NpuGmmFunction(torch.autograd.Function):
    """Autograd wrapper for ``torch_npu.npu_grouped_matmul``.

    Computes ``y = x @ W`` per expert group where:
      x          : (total_tokens, in_features)
      weight     : (num_local_experts, in_features, out_features)  already transposed
      group_list : (num_local_experts,) raw token counts, on device

    Uses ``group_list_type=1`` (raw counts) and ``group_type=0`` (A variable-length).
    """
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, group_list)
        return torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, group_list = ctx.saved_tensors

        # dgrad: grad_x = grad_output @ W^T
        grad_x = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight.transpose(1, 2)],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

        # wgrad: grad_W = x^T @ grad_output
        grad_weight = None
        if weight.requires_grad:
            grad_weight = torch_npu.npu_grouped_matmul(
                [x.T],
                [grad_output],
                bias=None,
                group_list=group_list,
                split_item=3,
                group_type=2,
                group_list_type=1,
            )[0]

        return grad_x, grad_weight, None


def npu_group_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    """``y = x @ W`` grouped by expert on NPU.

    Args:
        x          : (total_tokens, in_features)
        weight     : (num_local_experts, in_features, out_features)
        group_list : (num_local_experts,) raw token counts per expert, on device
    Returns:
        (total_tokens, out_features)
    """
    return NpuGmmFunction.apply(x, weight, group_list)


# ─────────────────────────────────────────────────────────────────────────────
# NPU dispatch helpers (replace the CPU one-hot preprocess / permute / unpermute)
# ─────────────────────────────────────────────────────────────────────────────


def _dispatch_preprocess(
    selected_experts: torch.Tensor,
    num_global_experts: int,
    ep_group: Optional[dist.ProcessGroup],
) -> Tuple[List[int], List[int], torch.Tensor, torch.Tensor]:
    """Compute all-to-all split sizes using ``bincount`` (no one-hot mask).

    Returns:
        input_splits  : tokens this rank sends to each EP peer.
        output_splits : tokens this rank receives from each EP peer.
        num_global_tokens_per_local_expert : CPU tensor (ep_size, num_local_experts)
            for ``sort_chunks_by_idxs``; moved non-blocking, sync before use.
        tokens_per_expert : device tensor (num_local_experts,) raw counts
            passed directly to ``npu_grouped_matmul`` as group_list.
    """
    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    ep_rank = 0 if ep_group is None else dist.get_rank(ep_group)
    num_local_experts = num_global_experts // ep_size

    # NPU 上 bincount 效率/支持较差，用 histc 等价替换：取值在 [0, num_global_experts-1]
    # 的整数 id，bins=num_global_experts、min=0、max=num_global_experts 时桶宽为 1.0，
    # 整数 i 恰好落入第 i 个桶，计数与 bincount(minlength=...) 完全一致；再转回 int64。
    num_local_tokens_per_expert = torch.histc(
        selected_experts.view(-1).to(torch.int64),
        bins=num_global_experts,
        min=0,
        max=num_global_experts - 1,
    )

    if ep_size <= 1:
        num_global_tokens_per_expert = num_local_tokens_per_expert.unsqueeze(0)
    else:
        num_global_tokens_per_expert = torch.zeros(
            ep_size,
            num_global_experts,
            dtype=num_local_tokens_per_expert.dtype,
            device=num_local_tokens_per_expert.device,
        )
        dist.all_gather_into_tensor(
            num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group
        )

    start = ep_rank * num_local_experts
    num_global_tokens_per_local_expert = (
        num_global_tokens_per_expert[:, start:start + num_local_experts].contiguous()
    )

    input_splits = num_local_tokens_per_expert.reshape(ep_size,
                                                       num_local_experts).sum(dim=1).tolist()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()

    tokens_per_expert = num_global_tokens_per_local_expert.sum(dim=0)  # on device
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.to(
        torch.device("cpu"), non_blocking=True
    )
    return input_splits, output_splits, num_global_tokens_per_local_expert, tokens_per_expert


def _alltoall_dispatch(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    input_splits: List[int],
    output_splits: List[int],
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """npu_moe_token_permute → all_to_all → sort_chunks_by_idxs."""
    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    num_local_experts = num_global_experts // ep_size

    # 处理空tensor的情况
    if hidden_states.numel() == 0:
        # 创建空的row_ids_map，形状与selected_experts匹配
        row_ids_map = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
        return hidden_states, row_ids_map

    hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states, selected_experts.to(torch.int32)
    )
    hidden_states = all_to_all(ep_group, hidden_states, output_splits, input_splits)

    # Sync so that CPU split_sizes are ready for sort_chunks_by_idxs
    torch.npu.current_stream().synchronize()

    permute_order = torch.arange(num_global_experts).reshape(-1,
                                                             num_local_experts).T.ravel().tolist()
    hidden_states = sort_chunks_by_idxs(
        hidden_states, num_global_tokens_per_local_expert.ravel(), permute_order
    )
    return hidden_states, row_ids_map


def _alltoall_combine(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    row_ids_map: torch.Tensor,
    input_splits: List[int],
    output_splits: List[int],
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """sort_chunks_by_idxs → all_to_all → npu_moe_token_unpermute."""
    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    num_local_experts = num_global_experts // ep_size

    # 即使expert_outputs是空的，也必须参与all_to_all通信
    # 否则其他rank会在等待时卡住

    unpermute_order = torch.arange(num_global_experts).reshape(num_local_experts,
                                                               -1).T.ravel().tolist()

    # sort_chunks_by_idxs需要处理空tensor
    expert_outputs = sort_chunks_by_idxs(
        expert_outputs, num_global_tokens_per_local_expert.T.ravel(), unpermute_order
    )

    # all_to_all必须调用，即使输入是空的
    expert_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)

    # 如果expert_outputs是空的，尝试调用npu_moe_token_unpermute
    # 如果npu_moe_token_unpermute能处理空tensor，那就直接使用
    # 如果不能，我们需要创建一个与计算图有连接的零张量

    # 首先尝试调用npu_moe_token_unpermute
    # 但我们需要确保即使expert_outputs是空的，也能正确调用
    if expert_outputs.numel() == 0:
        if row_ids_map.numel() > 0:
            # 从row_ids_map推断原始token数
            num_original_tokens = row_ids_map.max().item() + 1
            hidden_dim = expert_outputs.shape[-1] if expert_outputs.dim() > 0 else 0
            if hidden_dim > 0:
                # 创建一个零张量，但需要确保与计算图连接
                # 方法：对expert_outputs进行无操作计算，然后扩展形状
                # 即使expert_outputs是空的，我们也可以保持它
                # 然后创建一个形状正确的零张量

                # 创建一个零张量，形状为 (num_original_tokens, hidden_dim)
                # 但我们需要确保这个张量与输入有计算关系
                # 我们可以通过 expert_outputs.sum() * 0 来创建一个零张量
                # 这样即使expert_outputs是空的，sum()会返回0，然后乘以0还是0
                # 这样梯度就能从expert_outputs传播过来
                zero_sum = expert_outputs.sum()  # + routing_weights.sum() # 对空tensor求和得到0
                zero_scalar = zero_sum * 0.0  # 乘以0，但保持梯度连接

                # 创建一个零张量，并与zero_scalar有计算关系
                zero_tensor = torch.zeros(
                    num_original_tokens,
                    hidden_dim,
                    device=expert_outputs.device,
                    dtype=expert_outputs.dtype
                )

                # 将zero_tensor与zero_scalar相乘，保持梯度连接
                # 注意：我们需要确保zero_scalar是一个标量，但zero_tensor是一个矩阵
                # 我们可以使用广播机制：zero_tensor * zero_scalar
                result = zero_tensor * zero_scalar
                result.requires_grad_(expert_outputs.requires_grad)
                return result
            else:
                # hidden_dim为0，返回空tensor，但保持梯度连接
                zero_sum = expert_outputs.sum()  #+ routing_weights.sum() # 对空tensor求和得到0
                zero_scalar = zero_sum * 0.0  # 乘以0，但保持梯度连接

                # 创建一个空的tensor，并与zero_scalar有计算关系
                empty_tensor = torch.zeros(
                    0, device=expert_outputs.device, dtype=expert_outputs.dtype
                )
                result = empty_tensor * zero_scalar
                result.requires_grad_(expert_outputs.requires_grad)
                return result
        else:
            # row_ids_map也是空的，直接返回expert_outputs
            return expert_outputs

    return torch_npu.npu_moe_token_unpermute(expert_outputs, row_ids_map, probs=routing_weights)


# ─────────────────────────────────────────────────────────────────────────────
# NPU mixin
# ─────────────────────────────────────────────────────────────────────────────


class AllToAllEPExpertsMixinNPU(AllToAllEPExpertsMixin):
    """EP experts forward via all_to_all on NPU.

    Inherits from ``AllToAllEPExpertsMixin`` and overrides:
      - ``local_grouped_forward``: uses ``npu_grouped_matmul`` + ``npu_swiglu``.
      - ``_ep_forward``          : uses ``npu_moe_token_permute / unpermute``
                                   and a ``bincount``-based split preprocess.

    Everything else (``all_to_all``, ``_resolve_ep_group``, ``_get_apply_gate``,
    ``_ep_fallback_forward``, ``forward``, …) is inherited unchanged.
    """

    _fsdp_ep_cp_dispatch = "alltoall_npu"

    # ── local grouped GEMM ────────────────────────────────────────────────────

    # _handle_empty_tokens is intentionally NOT overridden here.
    #
    # The old NPU implementation ran npu_group_gemm + npu_swiglu on a single
    # dummy token and used `dummy_out.sum() * 0.0` to attach weights to the
    # graph. That pattern is unsafe: the backward of `.sum()` produces a
    # stride-0 (broadcast) gradient tensor that `npu_grouped_matmul` may not
    # handle correctly, just like torch._grouped_mm on GPU.
    #
    # The base-class implementation (EPExpertsMixin._handle_empty_tokens) does
    # `weight.sum() * 0.0` directly on the parameter — no matmul, no stride
    # issues — and is device-agnostic.  It correctly registers gate_up_proj and
    # down_proj (and optional biases) in the autograd graph with zero gradient.

    def local_grouped_forward(
        self,
        tokens: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        apply_gate,
        *,
        permuted_probs=None,
    ) -> torch.Tensor:
        """NPU grouped GEMM replacing HF ``_grouped_linear`` + cumsum.

        Args:
            tokens            : (total_recv_tokens, hidden_dim).
            tokens_per_expert : (num_local_experts,) raw counts on device,
                                from ``_dispatch_preprocess`` ``tokens_per_expert``.
            apply_gate        : fallback gate callable (GPU path or non-SwiGLU).
            permuted_probs    : optional per-token expert weight.

        Weight attributes on ``self``:
            gate_up_proj : (E, 2I, H) if is_transposed else (E, H, 2I)
            down_proj    : (E,  H, I) if is_transposed else (E, I,  H)
            is_transposed: bool, default True (VeOmni / HF patched experts)
            has_gate     : bool, default True  → SwiGLU activation
            act_fn       : fallback when has_gate=False
        """
        if not _HAS_TORCH_NPU:
            return super().local_grouped_forward(
                tokens, tokens_per_expert, apply_gate, permuted_probs=permuted_probs
            )

        # 检查是否是空tensor
        if tokens.numel() == 0:
            return self._handle_empty_tokens(tokens, permuted_probs=permuted_probs)

        is_transposed = getattr(self, "is_transposed", False)
        has_gate = getattr(self, "has_gate", True)

        # npu_grouped_matmul computes x @ W → W must be (E, in, out)
        if not is_transposed:  # stored (E, out, in)
            w_up = self.gate_up_proj.transpose(1, 2)  # → (E, H, 2I)
            w_down = self.down_proj.transpose(1, 2)  # → (E, I,  H)
        else:  # already (E, in, out)
            w_up = self.gate_up_proj
            w_down = self.down_proj
        gate_up_out = npu_group_gemm(tokens, w_up, tokens_per_expert)  # (T, 2I)

        # TODO: use the more general apply_gate method
        # hidden = apply_gate(gate_up_out)

        if has_gate:
            hidden = torch_npu.npu_swiglu(gate_up_out, dim=-1)  # (T, I)
        else:
            hidden = getattr(self, "act_fn", F.silu)(gate_up_out)

        if permuted_probs is not None:
            hidden = hidden * permuted_probs

        return npu_group_gemm(hidden, w_down, tokens_per_expert)  # (T, H)

    # ── EP forward ────────────────────────────────────────────────────────────

    def _ep_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not _HAS_TORCH_NPU:
            raise RuntimeError(
                "AllToAllEPExpertsMixinNPU requires torch_npu. "
                "Use AllToAllEPExpertsMixin on CUDA."
            )

        ep_group = self._resolve_ep_group()
        assert ep_group is not None

        num_experts = self.num_experts
        hidden_states_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

        # Step 1: split sizes (bincount, no one-hot mask)
        input_splits, output_splits, num_global_per_local, tokens_per_expert = (
            _dispatch_preprocess(top_k_index, num_experts, ep_group)
        )

        # Step 2: dispatch (npu_moe_token_permute → all_to_all → reorder)
        permuted, row_ids_map = _alltoall_dispatch(
            hidden_states_flat,
            top_k_index,
            input_splits,
            output_splits,
            num_experts,
            num_global_per_local,
            ep_group,
        )

        # Step 3: local grouped GEMM (npu_grouped_matmul + npu_swiglu)
        expert_out = self.local_grouped_forward(permuted, tokens_per_expert, self._get_apply_gate())

        # Step 4: combine (reorder → all_to_all → npu_moe_token_unpermute)
        out = _alltoall_combine(
            expert_out,
            top_k_weights,
            row_ids_map,
            input_splits,
            output_splits,
            num_experts,
            num_global_per_local,
            ep_group,
        )
        return out.reshape(hidden_states.shape)
