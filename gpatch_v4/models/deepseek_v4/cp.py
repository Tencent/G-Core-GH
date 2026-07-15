# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""Context-parallelism primitives for DeepSeek-V4 (paper §3.4.3).

Sequence-partitioned CP: each rank holds ``s_local = S_total / cp_size`` tokens.
Four primitives used by :class:`DeepseekV4Attention` on the CP path:

* :class:`_SendLastAndPrepend` — isend last ``k`` tokens to next rank, irecv prefix
  from prev rank, return prepended tensor. Used for SWA ring and compressor stage-1.
* :class:`_AllGatherSeq` — all-gather along an arbitrary dim with reduce-scatter backward.
  Used for compressor stage-2.
* :func:`swa_ring_kv` / :func:`compressor_cp_ring` / :func:`compressor_cp_ag` — wrappers
  for the attention layer.
* :func:`build_cp_causal_mask` — manual ``[1, 1, s_local, s_local + swa_prefix_len]``
  causal + sliding-window mask with correct query offset per CP rank.

v1 single-hop only: SWA ring and compressor stage-1 send exactly one block to the
immediate neighbour. Multi-hop extensions are future work.
"""

# Suppress basedpyright false positives from incomplete torch.distributed stubs.
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from torch import distributed as dist
from torch.autograd import Function

from .thd import PackedSeqParams, cp_slice_layout

__all__ = [
    "cp_chunk_data",
    "swa_ring_kv",
    "compressor_cp_ring",
    "compressor_cp_ag",
    "build_cp_causal_mask",
]

# ---------------------------------------------------------------------------
# Autograd Functions
# ---------------------------------------------------------------------------


def _ring_all_to_all(
    *,
    send_tensor: torch.Tensor,
    send_peer: int,
    recv_peer: int,
    group,
    seq_dim: int,
) -> torch.Tensor:
    group_size = dist.get_world_size(group)
    send_flat = send_tensor.movedim(seq_dim, 0).contiguous()
    num_tokens = send_flat.shape[0]
    input_split_sizes = [0] * group_size
    output_split_sizes = [0] * group_size
    input_split_sizes[send_peer] = num_tokens
    output_split_sizes[recv_peer] = num_tokens
    recv_flat = torch.empty_like(send_flat)
    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return recv_flat.movedim(0, seq_dim).contiguous()


# todo zz: support forward and reverse fold-prefix exchange
class _SendLastAndPrepend(Function):
    r"""Send last ``k`` tokens along ``seq_dim`` to the next CP rank, recv
    the corresponding prefix from the previous CP rank, and **return the
    prepended tensor directly**.

    Output shape:
    * rank ``0``: ``x`` unchanged (no prefix to prepend).
    * rank ``r > 0``: ``cat([prefix_recv, x], dim=seq_dim)``.

    The Function owns the prepend so the caller never sees a
    "passthrough = x" output that aliases the input — that pattern was
    error-prone (e.g. accidentally using ``x`` instead of the passthrough
    breaks the autograd link). Here the output is always the right tensor
    to feed downstream, regardless of rank.

    Forward
    -------
    * Input: ``x`` shape ``[..., S, ...]`` along ``seq_dim``.
    * Side effects:
      - every rank sends its last ``k`` tokens to ``(cp_rank + 1) % cp_size``
        through an all-to-all permutation.
      - every rank receives a prefix from ``(cp_rank - 1) % cp_size``.
    * Output:
      - rank 0: ``x`` (the received last-rank prefix is intentionally dropped).
      - rank > 0: ``cat([prefix, x], dim=seq_dim)`` shape ``[..., k+S, ...]``.

    Backward
    --------
    Let ``grad_out`` be the upstream gradient on the (possibly prepended)
    output.
    * rank > 0: split ``grad_out`` along ``seq_dim`` into
      ``grad_prefix = grad_out[..., :k, ...]`` and
      ``grad_local = grad_out[..., k:, ...]``. ``grad_prefix`` is the
      gradient flowing into the prefix tokens we received from rank ``r-1``,
      so ``isend`` it back to rank ``r-1`` (where it lands at the last-k
      slice of *that rank's* input).
    * rank 0: no split — ``grad_local = grad_out``.
    * rank < cp_size-1: ``irecv`` the gradient from rank ``r+1`` (it computed
      the gradient on the slice we sent forward) and add it to the last-k
      slice of ``grad_local``.
    * rank 0 sends a dummy zero prefix-gradient to the last rank, which drops
      it; this keeps the ring communication topology symmetric without changing
      the math.

    The result is ``grad_x = grad_local`` with the cross-rank contribution
    folded into the last-k slice.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int, cp_group, seq_dim: int) -> torch.Tensor:
        cp_rank = dist.get_rank(cp_group)
        cp_size = dist.get_world_size(cp_group)
        ctx.cp_rank, ctx.cp_size, ctx.cp_group = cp_rank, cp_size, cp_group
        ctx.k, ctx.seq_dim = k, seq_dim

        send_buf = x.narrow(seq_dim, x.shape[seq_dim] - k, k).contiguous()
        prefix = _ring_all_to_all(
            send_tensor=send_buf,
            send_peer=(cp_rank + 1) % cp_size,
            recv_peer=(cp_rank - 1) % cp_size,
            group=cp_group,
            seq_dim=seq_dim,
        )

        if cp_rank == 0:
            # rank 0: no prefix → output is ``x`` itself. Returning ``x``
            # directly (rather than a clone) is fine: backward only uses
            # ``ctx.cp_rank`` etc.; no save_for_backward is needed.
            return x
        return torch.cat([prefix, x], dim=seq_dim)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        cp_rank, cp_size, cp_group = ctx.cp_rank, ctx.cp_size, ctx.cp_group
        k, seq_dim = ctx.k, ctx.seq_dim

        prefix_shape = list(grad_out.shape)
        prefix_shape[seq_dim] = k

        if cp_rank > 0:
            grad_prefix = grad_out.narrow(seq_dim, 0, k).contiguous()
            grad_local = grad_out.narrow(seq_dim, k, grad_out.shape[seq_dim] - k).contiguous()
        else:
            grad_prefix = grad_out.new_zeros(prefix_shape)
            grad_local = grad_out.contiguous()

        recv_grad = _ring_all_to_all(
            send_tensor=grad_prefix,
            send_peer=(cp_rank - 1) % cp_size,
            recv_peer=(cp_rank + 1) % cp_size,
            group=cp_group,
            seq_dim=seq_dim,
        )

        # local-path grad + cross-rank grad on the sent slice.
        # ``grad_local`` may be a narrowed view of ``grad_out``; clone so
        # the in-place add doesn't surprise upstream.
        grad_x = grad_local.clone()
        if cp_rank < cp_size - 1:
            grad_x.narrow(seq_dim, grad_x.shape[seq_dim] - k, k).add_(recv_grad)
        return grad_x, None, None, None


class _AllGatherSeq(Function):
    r"""All-gather a tensor along ``dim`` across the CP group, with a
    reduce-scatter backward.

    Forward
    -------
    * Input: ``x`` shape ``[..., s, ...]`` along ``dim``.
    * Output: ``concat([x_0, x_1, ..., x_{cp_size-1}], dim=dim)`` shape
      ``[..., cp_size * s, ...]``. Same on every rank.

    Backward
    --------
    Each rank's downstream loss may have a different gradient on the
    all-gathered output (different queries → different K/V grads). The
    correct gradient on rank ``r``'s ``x`` is ``sum_r' grad[r'][r-th chunk]``,
    which is exactly ``reduce_scatter`` along ``dim``.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, cp_group, dim: int) -> torch.Tensor:
        cp_size = dist.get_world_size(cp_group)
        ctx.cp_group = cp_group
        ctx.dim = dim
        ctx.cp_size = cp_size

        x_c = x.contiguous()
        gathered = [torch.empty_like(x_c) for _ in range(cp_size)]
        dist.all_gather(gathered, x_c, group=cp_group)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        cp_group, dim, cp_size = ctx.cp_group, ctx.dim, ctx.cp_size
        chunks = [c.contiguous() for c in grad.chunk(cp_size, dim=dim)]
        out = torch.empty_like(chunks[0])
        dist.reduce_scatter(out, chunks, group=cp_group)
        return out, None, None


# ---------------------------------------------------------------------------
# User-layer wrappers
# ---------------------------------------------------------------------------


def cp_chunk_data(
    cp_rank: int,
    cp_size: int,
    *,
    tokens: torch.Tensor,
    labels: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    packed_seq_params: PackedSeqParams | None = None,
) -> tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[PackedSeqParams],
]:
    """Per-rank CP chunking of training-data tensors and PSP layout.

    Slices ``tokens`` / ``labels`` / ``loss_mask`` to the local ``s_local``
    contiguous chunk for ``cp_rank`` and either slices the supplied
    ``position_ids`` (THD seg-local layout) or builds a fresh global
    ``arange`` (BSHD default).

    When ``packed_seq_params`` is provided, returns a NEW PSP whose
    ``layout`` is sliced via :func:`gpatch_v4.models.deepseek_v4.thd.cp_slice_layout`
    so per-token / per-window positional fields carry the per-rank
    "with CP all2all prefix" view that downstream compressors consume.
    Other PSP fields (cu_seqlens_q*, max_seqlen_q, total_seqlen) stay
    GLOBAL — attention's compressor stage-2 all-gathers back to the
    global window axis.

    Parameters
    ----------
    cp_rank : int
    cp_size : int
    tokens : torch.Tensor
        Shape ``(B, s_full)``; ``s_full`` MUST be divisible by ``cp_size``.
    labels : torch.Tensor, optional
        Shape ``(B, s_full)``.
    loss_mask : torch.Tensor, optional
        Shape ``(B, s_full)``.
    position_ids : torch.Tensor, optional
        Shape ``(B, s_full)``. When provided, sliced directly to the local
        chunk (used by THD callers to keep seg-local positions). Otherwise
        the local chunk gets fresh global positions
        ``cp_rank * s_local + arange(s_local)`` (BSHD default).
    packed_seq_params : PackedSeqParams, optional
        Carries the global layout; when provided the returned PSP holds a
        per-rank-with-prefix sliced layout. ``None`` is returned at the
        same tuple position when not supplied.

    Returns
    -------
    local_tokens : torch.Tensor
        Shape ``(B, s_full // cp_size)``.
    local_labels : torch.Tensor or None
    local_loss_mask : torch.Tensor or None
    local_position_ids : torch.Tensor
    local_psp : PackedSeqParams or None

    Raises
    ------
    AssertionError
        If ``tokens.shape[1] % cp_size != 0``, or if ``position_ids`` is
        provided and its shape does not match ``tokens``.
    """
    # todo zz: shard BSHD globally and THD independently per segment
    s_full = tokens.shape[1]
    assert s_full % cp_size == 0, (
        f"tokens seq_len ({s_full}) must be divisible by cp_size ({cp_size})"
    )
    s_local = s_full // cp_size
    start = cp_rank * s_local
    sl = slice(start, start + s_local)
    local_tokens = tokens[:, sl].contiguous()
    local_labels = labels[:, sl].contiguous() if labels is not None else None
    local_loss_mask = loss_mask[:, sl].contiguous() if loss_mask is not None else None
    if position_ids is None:
        local_position_ids = (
            torch.arange(s_local, device=tokens.device, dtype=torch.long) + start
        ).unsqueeze(0).expand(tokens.shape[0], -1).contiguous()
    else:
        assert position_ids.shape[1] == s_full, (
            f"position_ids seq_len ({position_ids.shape[1]}) must match "
            f"tokens seq_len ({s_full})"
        )
        local_position_ids = position_ids[:, sl].contiguous()

    if packed_seq_params is not None:
        # Slice the layout per CP rank; cu_seqlens_q* / max_seqlen_q /
        # total_seqlen stay global (compressor stage-2 all-gathers back).
        assert packed_seq_params.layout is not None, (
            "packed_seq_params.layout is None; pack_sequences() should "
            "have populated it"
        )
        sliced_layout = cp_slice_layout(
            packed_seq_params.layout,
            cp_rank,
            cp_size,
            packed_seq_params.total_seqlen,
        )
        local_psp = dataclasses.replace(packed_seq_params, layout=sliced_layout)
    else:
        local_psp = None

    return local_tokens, local_labels, local_loss_mask, local_position_ids, local_psp


def swa_ring_kv(
    kv_local: torch.Tensor,
    cp_group,
    sliding_window: int,
) -> torch.Tensor:
    """Prepend prior-rank KV tokens to ``kv_local`` for sliding-window attention.

    Sends ``min(sliding_window - 1, s_local)`` tokens to the next rank and
    prepends the received prefix. v1 single-hop — multi-hop is future work.

    Parameters
    ----------
    kv_local : torch.Tensor
        Shape ``[B, 1, s_local, head_dim]``.
    cp_group : dist.ProcessGroup
    sliding_window : int

    Returns
    -------
    torch.Tensor
        Shape ``[B, 1, s_local + N_ring, head_dim]`` on rank > 0,
        ``[B, 1, s_local, head_dim]`` on rank 0.
    """
    # todo zz: exchange both fold-half SWA prefixes per segment
    if cp_group is None:
        return kv_local
    cp_size = dist.get_world_size(cp_group)
    if cp_size == 1:
        return kv_local

    s_local = kv_local.shape[2]
    # v1 single-hop: cap at s_local; skip degenerate sliding_window==1
    k = min(sliding_window - 1, s_local)
    if k == 0:
        return kv_local
    return _SendLastAndPrepend.apply(kv_local, k, cp_group, 2)


def compressor_cp_ring(
    hidden_states: torch.Tensor,
    m: int,
    cp_group,
) -> tuple[torch.Tensor, int]:
    """Stage-1 compressor CP (paper §3.4.3): send last ``m`` tokens to next rank,
    recv from prev, prepend on rank > 0. Requires ``m <= s_local``.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape ``[B, s_local, hidden_dim]``.
    m : int
        Compressor compress_rate (number of boundary tokens to exchange).
    cp_group : dist.ProcessGroup

    Returns
    -------
    hs_with_prefix : torch.Tensor
        Shape ``[B, s_local + m, hidden_dim]`` on rank > 0, ``[B, s_local, ...]`` on rank 0.
    n_prefix : int
        ``m`` on rank > 0, ``0`` on rank 0. Use as ``start_position`` offset
        for the compressor so window absolute positions stay correct.
    """
    # todo zz: exchange per-segment folded compressor prefixes
    if cp_group is None:
        return hidden_states, 0
    cp_size = dist.get_world_size(cp_group)
    if cp_size == 1:
        return hidden_states, 0
    cp_rank = dist.get_rank(cp_group)

    s_local = hidden_states.shape[1]
    # Compressor cannot degrade to a smaller boundary unlike swa_ring_kv
    assert m <= s_local, (
        f"compressor stage-1 requires s_local ({s_local}) >= compress_rate "
        f"({m}); CP cannot split a sequence below one compressor window"
    )

    hs_out = _SendLastAndPrepend.apply(hidden_states, m, cp_group, 1)

    # Rank 0 has no prior rank; compressor runs in "first window" mode (Z_b=-inf).
    n_prefix = 0 if cp_rank == 0 else m
    return hs_out, n_prefix


def compressor_cp_ag(
    local_compressed: torch.Tensor,
    cp_group,
    n_prefix_windows: int,
    n_local_windows: int,
) -> torch.Tensor:
    """Stage-2 compressor CP: drop duplicate prefix windows, then all-gather.

    Rank 0: ``local_compressed`` has ``n_local_windows`` valid entries (no trim).
    Rank ``r > 0``: has ``n_prefix_windows + n_local_windows`` entries; drops
    the first ``n_prefix_windows`` (duplicates of rank ``r-1``'s tail).

    Parameters
    ----------
    local_compressed : torch.Tensor
        Shape ``[B, 1, n_local_windows, head_dim]`` on rank 0,
        ``[B, 1, n_prefix_windows + n_local_windows, head_dim]`` on rank > 0.
    cp_group : dist.ProcessGroup
    n_prefix_windows : int
        Windows to drop on rank > 0 (1 in v1 single-hop).
    n_local_windows : int
        Valid entries per rank after trim (= ``s_local / m``).

    Returns
    -------
    torch.Tensor
        Shape ``[B, 1, cp_size * n_local_windows, head_dim]``, same on all ranks.
    """
    # todo zz: trim folded prefixes and restore global window order
    if cp_group is None:
        return local_compressed
    cp_size = dist.get_world_size(cp_group)
    if cp_size == 1:
        return local_compressed
    cp_rank = dist.get_rank(cp_group)
    # Trim per-rank padding to uniform n_local_windows before all_gather
    if cp_rank == 0:
        trimmed = local_compressed.contiguous()
    else:
        assert local_compressed.shape[2] == n_prefix_windows + n_local_windows
        trimmed = local_compressed[:, :, n_prefix_windows:n_prefix_windows +
                                   n_local_windows, :].contiguous()
    assert trimmed.shape[2] == n_local_windows, (
        f"compressor_cp_post: rank {cp_rank} trimmed length {trimmed.shape[2]} "
        f"!= n_local_windows {n_local_windows}"
    )
    return _AllGatherSeq.apply(trimmed, cp_group, 2)


# ---------------------------------------------------------------------------
# Causal mask construction
# ---------------------------------------------------------------------------


def build_cp_causal_mask(
    s_local: int,
    cp_rank: int,
    swa_prefix_len: int,
    sliding_window: int,
    *,
    packed_seq_params: PackedSeqParams | None = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Causal + sliding-window mask aligned with the SWA-ringed KV layout.

    Query ``i`` is at absolute position ``cp_rank * s_local + i``.
    KV column ``j`` is at ``cp_rank * s_local - swa_prefix_len + j``.
    Visible iff causal (``q >= k``) and ``q - k < sliding_window``;
    when ``seg_id_full`` is given, additionally requires same-seg
    (THD pack-seq cross-seg gate).

    Manual implementation because ``create_sliding_window_causal_mask``
    cannot inject the per-rank query offset in training.

    Parameters
    ----------
    s_local : int
    cp_rank : int
    swa_prefix_len : int
        Prefix KV tokens prepended by :func:`swa_ring_kv`; zero on rank 0.
    sliding_window : int
    packed_seq_params : PackedSeqParams, optional
        When provided, ``layout.seg_id_per_token_full`` (shape ``[T] long``,
        global seg id per token) is used to gate q ``i`` and k ``j`` to
        ``-inf`` when their seg ids differ (THD pack-seq cross-seg gate).
        ``T`` must be ``>= (cp_rank + 1) * s_local``.

    Returns
    -------
    torch.Tensor
        Shape ``[1, 1, s_local, s_local + swa_prefix_len]``.
        ``0.0`` for visible positions, ``-inf`` otherwise.
    """
    # todo zz: build mask from folded q and KV position maps
    device = device or torch.device("cuda")
    q_pos = torch.arange(s_local, device=device) + cp_rank * s_local  # [s_local]
    k_pos = torch.arange(s_local + swa_prefix_len, device=device
                        ) + (  # [s_local + swa_prefix_len]
                            cp_rank * s_local - swa_prefix_len
                        )

    visible = (                                                            # [s_local, s_local + swa_prefix_len]
        (q_pos.unsqueeze(-1) >= k_pos.unsqueeze(0))
        & ((q_pos.unsqueeze(-1) - k_pos.unsqueeze(0)) < sliding_window)
    )

    if packed_seq_params is not None:
        seg_id_full = packed_seq_params.layout.seg_id_per_token_full
        seg_id_full = seg_id_full.to(device)
        seg_id_q = seg_id_full[cp_rank * s_local:(cp_rank + 1) * s_local]
        seg_id_k = seg_id_full[cp_rank * s_local - swa_prefix_len:(cp_rank + 1) * s_local]
        visible = visible & (seg_id_q.unsqueeze(-1) == seg_id_k.unsqueeze(0))

    base = torch.zeros(s_local, s_local + swa_prefix_len, dtype=dtype, device=device)
    base.masked_fill_(~visible, float("-inf"))
    return base.unsqueeze(0).unsqueeze(0)
