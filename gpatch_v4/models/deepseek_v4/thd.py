# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""THD pack-seq parameter container, derived-field factory, and packer.

Caller contract (read carefully)
--------------------------------
* :class:`PackedSeqParams` is ``frozen=True``: ``psp.cu_seqlens_q = ...``
  raises ``FrozenInstanceError``. ``frozen`` does **not** stop in-place
  tensor mutation: ``psp.cu_seqlens_q.add_(1)`` still succeeds and would
  silently corrupt downstream consumers.
* All tensors returned by :func:`make_packed_seq_layout` (including those
  inside the per-``m`` :class:`_PerMLayout` entries) are read-only views.
  Callers MUST NOT call ``add_`` / ``copy_`` / ``index_copy_`` /
  ``scatter_`` / ``fill_`` / any other in-place op on them.
* Violating the in-place contract under reentrant gradient checkpointing
  (``use_reentrant=True``) makes the recompute see post-mutation values
  and yields drifted gradients. This is enforced by code review and by
  ``test_recompute_mutate_negative``.
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )

__all__ = [
    "PackedSeqParams",
    "PackedSeqLayout",
    "make_packed_seq_layout",
    "pack_sequences",
    "cp_slice_layout",
]

# ---------------------------------------------------------------------------
# PackedSeqParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackedSeqParams:
    """V4-specific THD pack-seq metadata.

    Layout-only container; the data tensors (``input_ids`` / ``position_ids``
    / ``labels``) are returned alongside but kept outside this dataclass so
    they can be cp-sliced / device-moved by the caller without touching the
    layout fields.

    Running example used throughout this module's docstrings::

        2 sequences of effective lengths 10 and 20, packed with
        pad_to_multiple_of=8:

            seg 0: tokens 0..9   real, 10..15 pad   (s=10, s_padded=16)
            seg 1: tokens 0..19  real, 20..23 pad   (s=20, s_padded=24)

        T = 16 + 24 = 40
        cu_seqlens_q        = [0, 10, 30]
        cu_seqlens_q_padded = [0, 16, 40]
        max_seqlen_q        = 24
        total_seqlen        = 40

    Attributes
    ----------
    cu_seqlens_q : Tensor
        Shape ``[N+1]``, int64. Effective per-seg boundaries (excludes pad).

        Example::

            cu_seqlens_q = [0, 10, 30]
            # seg 0 has 10 effective tokens
            # seg 1 has 20 effective tokens (= 30 - 10)

    cu_seqlens_q_padded : Tensor
        Shape ``[N+1]``, int64, same device as ``cu_seqlens_q``. Layout
        boundaries (includes pad). ``cu_seqlens_q_padded[-1] == total_seqlen``.
        Each segment length must be a multiple of the coarsest compress rate
        ``m'`` (caller-validated; depends on a model config not visible
        here).

        Example::

            cu_seqlens_q_padded = [0, 16, 40]
            # seg 0 occupies layout positions 0..15
            # seg 1 occupies layout positions 16..39

    max_seqlen_q : int
        ``max(diff(cu_seqlens_q_padded))``. Carried as a Python ``int``
        (set by :func:`pack_sequences` on the host) so the hot path never
        needs a GPU→CPU sync.

        Example::

            max_seqlen_q = 24   # = max(16, 24), seg 1's padded length

    total_seqlen : int
        ``cu_seqlens_q_padded[-1]``. Carried as a Python ``int`` for the
        same hot-path reason as ``max_seqlen_q``; consumers (e.g.
        :func:`make_packed_seq_layout`) read it directly.

        Example::

            total_seqlen = 40

    qkv_format : str
        Always ``"thd"``.
    split_sizes_cont_to_zz : list of int, optional
        Per-destination token counts for contiguous-to-zigzag all-to-all.
        Populated by ``cp_chunk_data`` on the rank-local copy.
    split_sizes_zz_to_cont : list of int, optional
        Per-source token counts received by the rank's zigzag partition.
        Populated together with ``split_sizes_cont_to_zz``.
    order_cont_to_zz : Tensor, optional
        Rank-local permutation that groups contiguous tokens by zigzag
        destination rank.
    """

    cu_seqlens_q: torch.Tensor  # TODO unused: 仅 __post_init__/validate 自查 + make_packed_seq_layout 内部建 seg_id；外部未读
    cu_seqlens_q_padded: torch.Tensor  # used: smart_pad_helper.py（1 处）+ make_packed_seq_layout 内部
    max_seqlen_q: int  # TODO unused: 外部未读
    total_seqlen: int  # used: cp.py:cp_chunk_data
    qkv_format: str = "thd"  # TODO unused: 仅 __post_init__ assert，没人读字段值
    # Derived layout, computed by :func:`pack_sequences` (or
    # :func:`make_packed_seq_layout`) at construction. Default ``None``
    # exists only so test fixtures can build a bare PSP without the layout
    # factory; the production path always carries a non-None layout.
    # ``cp_chunk_data`` returns a new PSP with a CP-rank-sliced layout
    # (see :func:`cp_slice_layout`).
    layout: "_PackedSeqLayout | None" = None
    split_sizes_cont_to_zz: list[int] | None = None
    split_sizes_zz_to_cont: list[int] | None = None
    order_cont_to_zz: torch.Tensor | None = None

    def __post_init__(self) -> None:
        assert self.cu_seqlens_q.shape == self.cu_seqlens_q_padded.shape
        assert self.cu_seqlens_q.dtype == torch.int64
        assert self.cu_seqlens_q_padded.dtype == torch.int64
        assert self.cu_seqlens_q.device == self.cu_seqlens_q_padded.device
        assert self.qkv_format == "thd"
        assert isinstance(self.max_seqlen_q, int) and self.max_seqlen_q > 0
        assert isinstance(self.total_seqlen, int) and self.total_seqlen > 0
        assert (self.split_sizes_cont_to_zz is None) == (self.split_sizes_zz_to_cont is None)
        assert (self.split_sizes_cont_to_zz is None) == (self.order_cont_to_zz is None)
        if self.split_sizes_cont_to_zz is not None:
            assert self.split_sizes_zz_to_cont is not None
            assert self.order_cont_to_zz is not None
            cp_size = len(self.split_sizes_cont_to_zz)
            assert cp_size > 0
            assert len(self.split_sizes_zz_to_cont) == cp_size
            assert self.total_seqlen % (2 * cp_size) == 0
            assert all(n >= 0 for n in self.split_sizes_cont_to_zz)
            assert all(n >= 0 for n in self.split_sizes_zz_to_cont)
            s_local = self.total_seqlen // cp_size
            assert sum(self.split_sizes_cont_to_zz) == s_local
            assert sum(self.split_sizes_zz_to_cont) == s_local
            assert self.order_cont_to_zz.shape == (s_local, )
            assert self.order_cont_to_zz.dtype == torch.int64
            assert self.order_cont_to_zz.device == self.cu_seqlens_q_padded.device

    def validate(self) -> None:
        """Value/monotonicity/consistency checks; triggers GPU→CPU sync.

        Use only from tests or under a debug flag — calling on the trainer
        hot path defeats the design of carrying ``max_seqlen_q`` on host.
        """
        assert int(self.cu_seqlens_q[0]) == 0
        assert int(self.cu_seqlens_q_padded[0]) == 0
        assert (self.cu_seqlens_q[1:] >= self.cu_seqlens_q[:-1]).all().item()
        assert (self.cu_seqlens_q_padded[1:] >= self.cu_seqlens_q_padded[:-1]).all().item()
        assert (self.cu_seqlens_q <= self.cu_seqlens_q_padded).all().item()
        assert int(
            (self.cu_seqlens_q_padded[1:] - self.cu_seqlens_q_padded[:-1]).max()
        ) == self.max_seqlen_q
        assert int(self.cu_seqlens_q_padded[-1]) == self.total_seqlen


# ---------------------------------------------------------------------------
# PackedSeqLayout factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PerMLayout:
    """Derived per-``m`` fields shared by HCA / CSA / Indexer / topk_idxs builders.

    All tensors are read-only views; see module docstring for the in-place
    mutation contract.

    Attributes
    ----------
    causal_threshold_per_token : Tensor
        Shape ``[T]`` long. ``(t + 1) // m`` for global token index ``t``;
        after :func:`cp_slice_layout` the shape is ``[s_local]`` and the
        value at local index ``i`` is ``(cp_rank * s_local + i + 1) // m``.
        A compressed entry at global window index ``w`` is in the causal
        past of token ``t`` iff ``w < causal_threshold_per_token[t]``.

        Example (m=4)::

            token  5 → (5+1)//4  = 1   # sees entry 0
            token 16 → (16+1)//4 = 4   # sees entries 0..3 (cross-seg mask blocks them)
            token 35 → (35+1)//4 = 9   # sees entries 0..8 (cross-seg mask keeps only 4..8)

    seg_id_per_wnd : Tensor
        Shape ``[T // m]`` long. Per-window seg id. Each m-window sits
        inside exactly one seg (every seg's padded length is a multiple of
        the coarsest compress rate ``m'``, hence also of ``m`` since
        ``m | m'``). Used by HCA / CSA to mask cross-seg compressed
        entries (``layout_token_seg_id[t] != seg_id_per_wnd[w]`` ⇒ not
        visible).

        Example (m=4, padded_seqlens=[16, 24])::

            seg_id_per_wnd = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            #                 └─ seg 0 ─┘└──── seg 1 ────┘

    wnd_pos_ids : Tensor
        Shape ``[T // m]`` long (global); after :func:`cp_slice_layout`
        shape ``[s_local // m]``. Seg-local position id per window for HCA
        compressor RoPE — no CP all2all prefix. HCA does not use a ring
        all2all, so its compressor sees only the rank-local windows without
        any prefix from the previous rank.

    wnd_pos_ids_with_prefix : Tensor
        Shape ``[T // m]`` long (global); after :func:`cp_slice_layout`
        shape ``[(s_local + l_prefix) // m]``. Seg-local position id per
        window for CSA / Indexer compressor RoPE, in the "with CP all2all
        prefix" view — i.e. the per-rank window grid a compressor sees AFTER
        :func:`compressor_cp_ring` prepends ``m`` tokens (one window) from
        the previous CP rank (rank 0 has no prefix).

    pad_token_mask_with_prefix : Tensor
        Shape ``[T_view]`` bool. ``True`` at pad slots in the same
        "with CP all2all prefix" view. Used by CSA / Indexer to
        ``-inf``-gate Cb / Ca halves for pad tokens.

    first_of_seg_window_mask_with_prefix : Tensor
        Shape ``[n_wnd_view]`` bool. ``True`` at the first window of each
        non-leading seg in the "with CP all2all prefix" view. Used by CSA /
        Indexer to zero-kv / ``-inf``-gate the Ca half of seg-boundary
        windows.

    causal_threshold_per_token_zz : Tensor, optional
        Shape ``[s_local]`` long. Same values as
        ``causal_threshold_per_token`` but gathered into zigzag receive
        order for this CP rank. Populated by ``cp_chunk_data`` only;
        ``None`` on the global layout from :func:`pack_sequences`.
    """

    causal_threshold_per_token: torch.Tensor  # used: HCA / Indexer future-causal gate
    seg_id_per_wnd: torch.Tensor  # used: HCA/Indexer cross_seg_mask + Indexer top-k seg-membership check
    wnd_pos_ids: torch.Tensor  # used: HCA compressor RoPE positions (no CP prefix)
    wnd_pos_ids_with_prefix: torch.Tensor  # used: CSA/Indexer compressor RoPE positions (with CP prefix)
    pad_token_mask_with_prefix: torch.Tensor  # used: CSA/Indexer pad-token gate -inf
    first_of_seg_window_mask_with_prefix: torch.Tensor  # used: CSA/Indexer first-of-seg Ca slot zero+gate
    causal_threshold_per_token_zz: torch.Tensor | None = None  # used: Indexer fused zigzag mask/topk


@dataclass(frozen=True)
class _PackedSeqLayout:
    """Top-level THD-derived fields produced by :func:`make_packed_seq_layout`.

    All tensors are read-only views; see module docstring for the in-place
    mutation contract.

    Attributes
    ----------
    seg_id_per_token : Tensor
        Shape ``[T]`` long. Per-token seg id (0..N-1).

        Example (cu_seqlens_q_padded=[0,16,40])::

            seg_id_per_token = [0]*16 + [1]*24

    seg_id_per_token_full : Tensor
        Shape ``[T]`` long, **always global** (``cp_slice_layout`` does
        NOT slice this field). Same content as ``seg_id_per_token``
        before CP slicing. Prefer :attr:`seg_id_per_token_with_prefix`
        for SWA kv-side cross-seg gate when the layout was CP-sliced with
        ``sliding_window``.

    seg_id_per_token_with_prefix : Tensor
        Shape ``[T]`` long (global); after :func:`cp_slice_layout` shape
        ``[s_local + l_swa_prefix]`` where ``l_swa_prefix = 0`` on rank
        0 and ``sliding_window - 1`` on rank > 0. Seg id per token in
        the SWA kv buffer view (``swa_ring_kv`` prefix + local kv).

        Example (W=4, cp_size=2, s_local=4, rank=1)::

            seg_id_per_token_with_prefix = seg_id_full[0:7]
            # kv buffer slots [0..6] = prefix [0..2] ++ local [3..6]

    pad_token_mask : Tensor
        Shape ``[T]`` bool. ``True`` at pad slots, ``False`` at effective
        tokens — flag matches the *condition we want to act on* (mask-fill
        with -inf, drop from loss, …) so callers write
        ``x.masked_fill(pad_token_mask, value)`` without an extra ``~``.

        Example (cu_seqlens_q=[0,10,30], cu_seqlens_q_padded=[0,16,40])::

            pad_token_mask = [F]*10 + [T]*6 + [F]*20 + [T]*4
            #                └seg 0─┘└pad─┘ └─seg 1──┘└pad┘

    per_m : dict[int, _PerMLayout]
        One bundle per ``m`` in ``sorted(set(config.compress_rates.values()))``;
        see :class:`_PerMLayout`. Default V4 ``compress_rates = {4, 128}``
        ⇒ keys ``{4, 128}``.

    sliding_window : int
        From ``config.sliding_window``; used by :func:`cp_slice_layout` for
        ``seg_id_per_token_with_prefix`` SW prefix length.

    seg_id_per_token_zz : Tensor, optional
        Shape ``[s_local]`` long. Same values as ``seg_id_per_token_full``
        but gathered into zigzag receive order for this CP rank.
        Populated by ``cp_chunk_data`` only; ``None`` on the global layout
        from :func:`pack_sequences`.
    """

    seg_id_per_token: torch.Tensor  # used: modeling cross_seg_mask（HCA / Indexer 各 1 处）
    seg_id_per_token_full: torch.Tensor  # used: build_cp_causal_mask 等仍读全局 [T] 的路径
    seg_id_per_token_with_prefix: torch.Tensor  # used: fused SWA kv-side cross-seg gate
    pad_token_mask: torch.Tensor  # used: _per_m_layout 内部作 pad_token_mask_with_prefix 来源
    per_m: dict[int, _PerMLayout]
    sliding_window: int
    seg_id_per_token_zz: torch.Tensor | None = None  # used: Indexer fused zigzag mask/topk


# Public alias for cross-module type hints; the underscore prefix is kept on
# the class to mark the schema as still-evolving but consumers wire the alias.
PackedSeqLayout = _PackedSeqLayout


def _make_seg_layout(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_q_padded: torch.Tensor,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token seg-id and pad-mask lookup tables."""
    device = cu_seqlens_q_padded.device
    pos = torch.arange(T, device=device, dtype=torch.long)
    # right=True so a position equal to a boundary starts the next seg:
    # bucketize with `right=True` returns i for the half-open interval
    # [boundaries[i-1], boundaries[i]); right=False would put the boundary
    # itself in the previous bucket.
    seg_id = torch.bucketize(pos, cu_seqlens_q_padded[1:], right=True).to(torch.long)
    seg_id.clamp_(max=cu_seqlens_q_padded.shape[0] - 2)

    cu_q = cu_seqlens_q.to(torch.long)
    cu_pad = cu_seqlens_q_padded.to(torch.long)
    seg_start_per_token = cu_pad[:-1][seg_id]
    seg_real_len_per_token = (cu_q[1:] - cu_q[:-1])[seg_id]
    pad_token_mask = pos >= seg_start_per_token + seg_real_len_per_token
    return seg_id, pad_token_mask


def _per_m_layout(
    cu_seqlens_q_padded: torch.Tensor,
    seg_id_per_token: torch.Tensor,
    pad_token_mask: torch.Tensor,
    T: int,
    m: int,
) -> _PerMLayout:
    """Build the per-``m`` derived-field bundle.

    See :class:`_PerMLayout` for field semantics. Caller MUST ensure every
    seg's padded length is a multiple of ``m``.
    """
    device = cu_seqlens_q_padded.device

    causal_threshold_per_token = (torch.arange(T, device=device, dtype=torch.long) + 1) // m

    cu_n_wnd_padded = (cu_seqlens_q_padded // m).to(torch.long)
    n_windows = T // m
    first_of_seg_window_mask = torch.zeros(n_windows, device=device, dtype=torch.bool)
    if cu_n_wnd_padded.shape[0] > 2:
        # Skip seg 0's window-0 (its Ca half is already zero/-inf-gated by the
        # compressor's init layout) and the trailing boundary at index n_windows.
        first_of_seg_window_mask[cu_n_wnd_padded[1:-1]] = True

    # Per-window seg id: every m-window sits inside one seg (per-seg padded
    # len is a multiple of m), so a stride-m lookup over seg_id_per_token is
    # exact and one-shot.
    seg_id_per_wnd = seg_id_per_token[::m].contiguous()

    # Seg-local position ids for the global T tokens; cp_slice_layout carves
    # out the per-rank view including the m-token CP prefix.
    n_segs = cu_seqlens_q_padded.shape[0] - 1
    pos_ids = torch.empty(T, dtype=torch.long, device=device)
    for i in range(n_segs):
        seg_start = cu_seqlens_q_padded[i]
        seg_end = cu_seqlens_q_padded[i + 1]
        pos_ids[seg_start:seg_end] = torch.arange(
            seg_end - seg_start, dtype=torch.long, device=device
        )
    wnd_pos_ids = pos_ids[::m]
    wnd_pos_ids_with_prefix = pos_ids[::m]

    return _PerMLayout(
        causal_threshold_per_token=causal_threshold_per_token,
        seg_id_per_wnd=seg_id_per_wnd,
        wnd_pos_ids=wnd_pos_ids,
        wnd_pos_ids_with_prefix=wnd_pos_ids_with_prefix,
        pad_token_mask_with_prefix=pad_token_mask,
        first_of_seg_window_mask_with_prefix=first_of_seg_window_mask,
    )


def make_packed_seq_layout(
    psp: PackedSeqParams,
    config: "DeepseekV4Config",
) -> _PackedSeqLayout:
    """Compute every THD-derived field once.

    Parameters
    ----------
    psp : PackedSeqParams
        Provides ``cu_seqlens_q`` / ``cu_seqlens_q_padded`` /
        ``max_seqlen_q`` / ``total_seqlen``.

    config : DeepseekV4Config
        Used to read ``compress_rates`` and ``sliding_window``.

    Returns
    -------
    _PackedSeqLayout
        See :class:`_PackedSeqLayout` for field semantics. ``per_m`` keys
        come from ``sorted(set(config.compress_rates.values()))``;
        default V4 ``compress_rates = {"compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128}`` ⇒ ``{4, 128}``.

    Raises
    ------
    AssertionError
        If ``config.compress_rates`` has duplicate values (which would
        silently fold two attention types into one ``per_m`` entry).
    """
    T = psp.total_seqlen

    ms = sorted(set(config.compress_rates.values()))
    assert len(ms) == len(
        config.compress_rates
    ), (f"config.compress_rates values must be unique; got {dict(config.compress_rates)}")

    seg_id_per_token, pad_token_mask = _make_seg_layout(
        psp.cu_seqlens_q,
        psp.cu_seqlens_q_padded,
        T,
    )

    per_m: dict[int, _PerMLayout] = {
        m: _per_m_layout(psp.cu_seqlens_q_padded, seg_id_per_token, pad_token_mask, T, m)
        for m in ms
    }

    return _PackedSeqLayout(
        seg_id_per_token=seg_id_per_token,
        seg_id_per_token_full=seg_id_per_token,
        seg_id_per_token_with_prefix=seg_id_per_token,
        pad_token_mask=pad_token_mask,
        per_m=per_m,
        sliding_window=config.sliding_window,
    )


def cp_slice_layout(
    layout: _PackedSeqLayout,
    cp_rank: int,
    cp_size: int,
    total_seqlen: int,
) -> _PackedSeqLayout:
    """Slice a global :class:`_PackedSeqLayout` into a per-CP-rank view.

    Mirrors :func:`gpatch_v4.models.deepseek_v4.cp.compressor_cp_ring`'s
    output shape per ``m``:

    * rank 0: ``[start : start + s_local]`` (no prefix, ``l_prefix = 0``).
    * rank > 0: ``[start - m : start + s_local]`` (m-token prefix from
      the previous CP rank, ``l_prefix = m``).

    Returns a NEW frozen layout (does not mutate the input).

    Per-field slicing rule
    ----------------------
    ::

        ┌───────────────────────────────────────────────┬──────────────────────┬─────────────────────────────┐
        │                     field                     │    slice / keep      │      result shape           │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ seg_id_per_token                              │ ✅ slice             │ [s_local]                   │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ seg_id_per_token_full                         │ ❌ stay global       │ [T]                         │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ seg_id_per_token_with_prefix                  │ ✅ slice with SW     │ [s_local + l_swa_prefix]    │
        │                                               │    prefix            │                             │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ pad_token_mask                                │ ✅ slice             │ [s_local]                   │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ per_m.causal_threshold_per_token              │ ✅ slice             │ [s_local]                   │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ per_m.wnd_pos_ids                             │ ✅ slice             │ [s_local // m]              │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ per_m.wnd_pos_ids_with_prefix                 │ ✅ slice with prefix │ [(s_local + l_prefix) // m] │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ per_m.pad_token_mask_with_prefix              │ ✅ slice with prefix │ [s_local + l_prefix]        │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ per_m.first_of_seg_window_mask_with_prefix    │ ✅ slice with prefix │ [(s_local + l_prefix) // m] │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ per_m.seg_id_per_wnd                          │ ❌ stay global       │ [T // m]                    │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ seg_id_per_token_zz                           │ ✅ zigzag gather     │ [s_local]                   │
        ├───────────────────────────────────────────────┼──────────────────────┼─────────────────────────────┤
        │ per_m.causal_threshold_per_token_zz           │ ✅ zigzag gather     │ [s_local]                   │
        └───────────────────────────────────────────────┴──────────────────────┴─────────────────────────────┘

    q-axis fields are sliced to ``s_local`` so consumers index with the
    rank-local query offset. k-axis (per-window / per-seg) fields stay
    global because compressors all-gather to the global window grid in
    stage-2, and per-query causality references that global axis. The
    ``with_prefix`` per-token positional fields carry the m-token CP
    all2all prefix that :func:`compressor_cp_ring` prepends on rank > 0.

    ``*_zz`` fields are also rank-local ``[s_local]``, but ordered in
    zigzag receive order (not contiguous ``sl``). They are filled by
    :func:`gpatch_v4.models.deepseek_v4.cp.cp_chunk_data` after this
    slice — gathering from the global vectors with the same token index
    map that ``indexer_zigzag_all_to_all(..., cont_to_zz=True)`` would
    produce.

    Parameters
    ----------
    layout : _PackedSeqLayout
        Global layout from :func:`make_packed_seq_layout`.
    cp_rank : int
    cp_size : int
        ``total_seqlen`` MUST be divisible by ``cp_size``; per-``m`` ``m``
        MUST divide ``s_local`` (caller-validated when packing).
    total_seqlen : int
        Global ``T`` along which the layout was built.

    Returns
    -------
    _PackedSeqLayout
        Same per-``m`` keys; ``wnd_pos_ids`` and ``wnd_pos_ids_with_prefix``
        carry per-rank views (without and with CP prefix, respectively).
    """

    assert total_seqlen % cp_size == 0, (
        f"total_seqlen ({total_seqlen}) must be divisible by cp_size ({cp_size})"
    )
    s_local = total_seqlen // cp_size
    start = cp_rank * s_local
    sl = slice(start, start + s_local)
    sliding_window = layout.sliding_window
    l_swa_prefix = 0 if cp_rank == 0 else sliding_window - 1
    sl_swa_with_prefix = slice(start - l_swa_prefix, start + s_local)

    new_per_m: dict[int, _PerMLayout] = {}
    for m, per_m in layout.per_m.items():
        # rank 0: no prefix; rank > 0: m-token prefix from prev rank
        # (mirrors compressor_cp_ring). Caller MUST keep s_local >= m
        # (compressor cannot split a sequence below one window).
        l_prefix = 0 if cp_rank == 0 else m
        assert s_local >= m, (
            f"s_local ({s_local}) must be >= m ({m}); compressor stage-1 "
            f"requires at least one full window per rank"
        )
        sl_with_prefix = slice(start - l_prefix, start + s_local)
        sl_wnd = slice(start // m, (start + s_local) // m)
        sl_wnd_with_prefix = slice((start - l_prefix) // m, (start + s_local) // m)
        new_per_m[m] = dataclasses.replace(
            per_m,
            causal_threshold_per_token=per_m.causal_threshold_per_token[sl],
            wnd_pos_ids=per_m.wnd_pos_ids[sl_wnd],  # HCA RoPE
            wnd_pos_ids_with_prefix=per_m.
            wnd_pos_ids_with_prefix[sl_wnd_with_prefix],  # CSA/Indexer RoPE
            pad_token_mask_with_prefix=per_m.pad_token_mask_with_prefix[sl_with_prefix],
            first_of_seg_window_mask_with_prefix=per_m.
            first_of_seg_window_mask_with_prefix[sl_wnd_with_prefix],
        )

    return _PackedSeqLayout(
        seg_id_per_token=layout.seg_id_per_token[sl],
        seg_id_per_token_full=layout.seg_id_per_token_full,
        seg_id_per_token_with_prefix=layout.seg_id_per_token_with_prefix[sl_swa_with_prefix],
        pad_token_mask=layout.pad_token_mask[sl],
        per_m=new_per_m,
        sliding_window=layout.sliding_window,
    )


# ---------------------------------------------------------------------------
# pack_sequences
# ---------------------------------------------------------------------------


def pack_sequences(
    input_ids: list[torch.Tensor],
    labels: list[torch.Tensor] | None,
    *,
    config: "DeepseekV4Config",
    pad_to_multiple_of: int,
    cp_size: int = 1,
    pad_token_id: int = 0,
    label_ignore_index: int = -100,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, PackedSeqParams]:
    """Pack a list of variable-length 1D sequences into THD layout.

    Each segment of length ``s_i`` is right-padded with ``pad_token_id`` to
    the next multiple of ``pad_to_multiple_of`` (call it ``s_i_padded``).
    When ``cp_size > 1``, the last segment is further padded so that
    ``T_TOTAL`` is a multiple of ``cp_size * pad_to_multiple_of``.

    Parameters
    ----------
    input_ids : list[Tensor]
        Each element shape ``[s_i]`` long; ``s_i`` arbitrary positive int.
    labels : list[Tensor] | None
        If given, must match ``input_ids`` length and each element's shape
        must match the corresponding ``input_ids[i]``.
    config : DeepseekV4Config
        Used to build the per-``m`` layout (read ``compress_rates``).
    pad_to_multiple_of : int
        Each segment's padded length is the smallest multiple of this
        value that's ``>= s_i``.
    cp_size : int
        Context-parallelism world size. ``T_TOTAL`` is guaranteed to be
        a multiple of ``cp_size * pad_to_multiple_of``.
    pad_token_id : int
    label_ignore_index : int
    device : torch.device | None

    Returns
    -------
    input_ids_packed : Tensor
        Shape ``[1, T]`` long.
    position_ids_packed : Tensor
        Shape ``[1, T]`` long, seg-local positions.
    labels_packed : Tensor | None
        Shape ``[1, T]`` long, or ``None`` if ``labels`` is ``None``.
    psp : PackedSeqParams
    """
    if len(input_ids) == 0:
        raise ValueError("input_ids must be a non-empty list")
    if pad_to_multiple_of <= 0:
        raise ValueError(f"pad_to_multiple_of must be positive; got {pad_to_multiple_of}")
    if labels is not None and len(labels) != len(input_ids):
        raise ValueError(f"labels length {len(labels)} != input_ids length {len(input_ids)}")

    if device is None:
        device = input_ids[0].device

    total_align = cp_size * pad_to_multiple_of

    seqlens: list[int] = []
    padded_seqlens: list[int] = []
    for i, ids in enumerate(input_ids):
        if ids.dim() != 1:
            raise ValueError(f"input_ids[{i}] must be 1D; got shape {tuple(ids.shape)}")
        if ids.device != device:
            raise ValueError(
                f"input_ids[{i}] device {ids.device} != target device {device}; "
                "all segments must share one device"
            )
        s = int(ids.shape[0])
        if s == 0:
            raise ValueError(f"input_ids[{i}] is empty; zero-length segments are not allowed")
        if labels is not None:
            if labels[i].shape != ids.shape:
                raise ValueError(
                    f"labels[{i}] shape {tuple(labels[i].shape)} != "
                    f"input_ids[{i}] shape {tuple(ids.shape)}"
                )
            if labels[i].device != device:
                raise ValueError(f"labels[{i}] device {labels[i].device} != target device {device}")
        s_padded = ((s + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        seqlens.append(s)
        padded_seqlens.append(s_padded)

    T = sum(padded_seqlens)
    if T % total_align != 0:
        extra = total_align - (T % total_align)
        padded_seqlens[-1] += extra

    T = sum(padded_seqlens)
    n_segs = len(seqlens)

    cu_seqlens_q_host = [0]
    cu_seqlens_q_padded_host = [0]
    for s, s_padded in zip(seqlens, padded_seqlens):
        cu_seqlens_q_host.append(cu_seqlens_q_host[-1] + s)
        cu_seqlens_q_padded_host.append(cu_seqlens_q_padded_host[-1] + s_padded)

    cu_seqlens_q = torch.tensor(cu_seqlens_q_host, dtype=torch.int64, device=device)
    cu_seqlens_q_padded = torch.tensor(cu_seqlens_q_padded_host, dtype=torch.int64, device=device)

    input_ids_packed = torch.full((T, ), pad_token_id, dtype=torch.long, device=device)
    position_ids_flat = torch.empty(T, dtype=torch.long, device=device)
    labels_packed_flat = (
        torch.full((T, ), label_ignore_index, dtype=torch.long, device=device)
        if labels is not None else None
    )

    for i in range(n_segs):
        seg_start = cu_seqlens_q_padded_host[i]
        s = seqlens[i]
        s_padded = padded_seqlens[i]
        input_ids_packed[seg_start:seg_start + s] = input_ids[i].long()
        position_ids_flat[seg_start:seg_start + s_padded] = torch.arange(
            s_padded,
            dtype=torch.long,
            device=device,
        )
        if labels_packed_flat is not None:
            assert labels is not None  # narrows type for pyright
            labels_packed_flat[seg_start:seg_start + s] = labels[i].long()
            # mask the segment-final effective token; pad tail stays at ignore_index

    psp = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        max_seqlen_q=max(padded_seqlens),
        total_seqlen=T,
    )
    psp = dataclasses.replace(psp, layout=make_packed_seq_layout(psp, config))

    labels_packed = labels_packed_flat.unsqueeze(0) if labels_packed_flat is not None else None
    return (
        input_ids_packed.unsqueeze(0),
        position_ids_flat.unsqueeze(0),
        labels_packed,
        psp,
    )
