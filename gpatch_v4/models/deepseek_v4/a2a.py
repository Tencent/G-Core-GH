# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""All-to-all primitives for Expert Parallelism (EP).

Two autograd-aware ``all_to_all`` variants for MoE token dispatch:

* :func:`all_to_all` — even split along dim 0 (``x.shape[0]`` must be
  divisible by ``world_size``).
* :func:`all_to_all_uneven` — explicit ``in_splits`` / ``out_splits`` per
  rank; used by the dynamic-routing dispatch path.

The backward of each variant is the transposed all-to-all (``in_splits`` and
``out_splits`` swapped for the uneven case; the same op for the even case).

Intentionally a verbatim copy of the ``qwen3_5_moe`` version to keep the two
MoE forks symmetric without cross-import coupling.
"""

import torch
import torch.distributed as dist


class _AllToAll(torch.autograd.Function):
    """Even-split all-to-all along dim 0.

    Forward
    -------
    ``out[r * S/W : (r+1) * S/W]`` on this rank ← ``x[this_rank * S/W :
    (this_rank+1) * S/W]`` on rank ``r``, where ``S = x.shape[0]`` and
    ``W = world_size``. ``S`` must be divisible by ``W``.

    Backward
    --------
    The dual of an even all-to-all is the same op: re-apply on ``grad``.
    """
    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group
        out = torch.empty_like(input)
        dist.all_to_all_single(out, input.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        # ``None`` for the non-tensor ``group`` arg; recursive apply for the
        # tensor — the transpose of an even all-to-all is the same op.
        return None, _AllToAll.apply(ctx.group, grad)


def all_to_all(x, group):
    """Even-split all-to-all along dim 0.

    Parameters
    ----------
    x : torch.Tensor
        ``x.shape[0]`` must be divisible by ``group.size()``.
    group : dist.ProcessGroup

    Returns
    -------
    torch.Tensor
        Same shape as ``x``.
    """
    return _AllToAll.apply(group, x)


class _AllToAllUneven(torch.autograd.Function):
    """Uneven all-to-all along dim 0 with explicit per-rank split sizes.

    Forward
    -------
    Sends ``input[sum(in_splits[:i]) : sum(in_splits[:i+1])]`` to rank
    ``i`` for every ``i``. Receives from rank ``i`` a slice of length
    ``out_splits[i]``; concatenates received slices in rank order to form
    the output.

    Splits are passed explicitly because MoE token routing produces a
    ragged distribution that depends on the gate decisions; the caller
    (typically the MoE dispatch code) computes ``in_splits`` from the
    local routing table and obtains ``out_splits`` via a prior all-to-all
    of the split sizes themselves.

    Backward
    --------
    The transpose of an uneven all-to-all swaps ``in_splits`` and
    ``out_splits`` (sender becomes receiver and vice versa), so
    backward applies the same op with the splits swapped.
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
        # Transpose: gradient on chunk-i-sent-to-rank-r flows back to rank
        # r as a chunk of size in_splits[i]. So in backward, what was
        # ``in_splits`` (sizes we sent) becomes ``out_splits`` (sizes we
        # receive), and vice versa.
        return None, _AllToAllUneven.apply(
            ctx.group, grad, ctx.out_splits, ctx.in_splits
        ), None, None


def all_to_all_uneven(x, in_splits, out_splits, group):
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
