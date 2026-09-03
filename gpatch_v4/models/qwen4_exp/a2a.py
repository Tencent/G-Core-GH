# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Autograd-aware all-to-all primitives.

Copied from :mod:`gpatch_v4.models.deepseek_v4.a2a` to avoid coupling the model forks.
Used by both the owner-sharded Engram lookup and expert-parallel MoE dispatch.
"""

import torch
import torch.distributed as dist


class _AllToAllUneven(torch.autograd.Function):
    """Uneven all-to-all along dim 0 with explicit per-rank split sizes.

    Forward
    -------
    Sends ``input[sum(in_splits[:i]) : sum(in_splits[:i+1])]`` to rank ``i`` for every
    ``i``. Receives from rank ``i`` a slice of length ``out_splits[i]``; concatenates
    received slices in rank order to form the output.

    Backward
    --------
    The transpose of an uneven all-to-all swaps ``in_splits`` and ``out_splits``
    (sender becomes receiver and vice versa), so backward applies the same op with the
    splits swapped.
    """
    @staticmethod
    def forward(ctx, group, input, in_splits, out_splits):
        ctx.group = group
        ctx.in_splits = in_splits
        ctx.out_splits = out_splits
        out = torch.empty(sum(out_splits), *input.shape[1:], dtype=input.dtype, device=input.device)
        dist.all_to_all_single(
            out,
            input.contiguous(),
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=group
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        return None, _AllToAllUneven.apply(
            ctx.group, grad, ctx.out_splits, ctx.in_splits
        ), None, None


def all_to_all_uneven(
    x: torch.Tensor,
    in_splits: list[int],
    out_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Uneven all-to-all along dim 0.

    Parameters
    ----------
    x : torch.Tensor
        ``x.shape[0]`` must equal ``sum(in_splits)``.
    in_splits : list[int]
        Dim-0 slice sizes to send to each rank.
    out_splits : list[int]
        Dim-0 slice sizes this rank receives from each rank.
        Must satisfy: ``in_splits[j]`` on rank ``i`` == ``out_splits[i]`` on rank ``j``.
    group : dist.ProcessGroup

    Returns
    -------
    torch.Tensor
        Shape ``[sum(out_splits), *x.shape[1:]]``.
    """
    return _AllToAllUneven.apply(group, x, in_splits, out_splits)
