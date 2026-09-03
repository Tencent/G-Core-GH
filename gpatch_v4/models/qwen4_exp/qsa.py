# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen Sparse Attention (QSA) for training.

The upstream indexer (:class:`Qwen4ExpTextQSAIndexer`) is a **Python double loop over
batch × query position**. That is fine for a short generate() but hopeless for training:
a single 4096-token micro-batch would run 4096 iterations per sparse layer, times 12
layers, per step. This module replaces it with a batched selector and adds a
FlexAttention path so the selection never has to be materialized as a dense score matrix.

Two representations
-------------------
``select_qsa_membership`` returns a boolean membership table ``[B, S_q, S_kv]`` rather
than upstream's ordered ``selected_token_ids`` list. Both the dense oracle and
FlexAttention consume a *set* of keys, so ordering is irrelevant — and the reference
CUDA kernel does not define a stable order after top-k anyway.

Score ties are common because ``relu(q·k).sum(heads)`` often produces exact zeros.
Upstream does not define which tied blocks win, so parity requires the same optimal
score rather than the same block set when a tie crosses the top-k boundary.

Supported layouts
-----------------
Dense batches use right-tail padding. Packed THD rows additionally carry document
boundaries; their compression grid and causal prefix restart independently per document.
Left and interior padding without THD metadata are rejected loudly by
:func:`_visible_prefix_lengths`.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from typing_extensions import override

from .cp import Qwen4ExpCPContext, qwen4_exp_cp_all_gather
from .modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextQSAIndexer,
    apply_rotary_pos_emb,
)

__all__ = [
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAIndexer",
    "dense_sparse_gqa_attention",
    "flex_sparse_gqa_attention",
    "select_qsa_membership",
]

QSA_ATTN_BACKENDS = ("flex", "dense")


def _visible_prefix_lengths(visible: torch.Tensor) -> torch.Tensor:
    """Return visible-key counts while requiring every row to be a prefix."""
    prefix_len = visible.sum(dim=-1)
    positions = torch.arange(visible.shape[-1], device=visible.device)
    is_prefix = torch.all((positions < prefix_len.unsqueeze(-1)) == visible)
    error = (
        "QSA requires every query's visible key set to be a contiguous prefix [0, n) — "
        "i.e. a causal mask with right-tail padding only. Left padding, interior padding "
        "and non-causal masks are not supported."
    )
    if visible.device.type == "cuda":
        # A Python bool would synchronize every sparse-attention layer with the host.
        torch._assert_async(is_prefix, error)
    elif not bool(is_prefix):
        raise ValueError(error)
    return prefix_len


@torch.no_grad()
def select_qsa_membership(
    index_query_states: torch.Tensor,
    raw_key_states: torch.Tensor,
    visible: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    k_layernorm: nn.Module,
    compress_ratio: int,
    block_topk: int,
) -> torch.Tensor:
    """Batched equivalent of the upstream per-query QSA selection.

    Runs under ``no_grad``: the reference exposes discrete top-k ids with neither an
    auxiliary loss nor a straight-through estimator, so the indexer is frozen and
    nothing here needs to be differentiable.

    Parameters
    ----------
    index_query_states : torch.Tensor
        ``[B, S_q, H_idx, D_idx]``, already through ``q_layernorm`` and RoPE.
    raw_key_states : torch.Tensor
        ``[B, S_kv, D_idx]``, *before* pooling / norm / RoPE (pooling comes first).
    visible : torch.Tensor
        Bool ``[B, S_q, S_kv]``.
    key_cos, key_sin : torch.Tensor
        ``[B, S_kv, D_rot]`` at full key positions.
    compress_ratio : int
        Keys per index block.
    block_topk : int
        ``indexer_budget // compress_ratio``.

    Returns
    -------
    torch.Tensor
        Bool ``[B, S_q, S_kv]`` membership table.

    Notes
    -----
    Peak transient is the score tensor ``[B, S_q, H_idx, S_kv / compress_ratio]`` in
    fp32 (≈67 MB at B=1, S=4096, H_idx=4, ratio=4). Sequences far beyond the validated
    4096 would want query chunking here.
    """
    batch_size, q_len, _, head_dim = index_query_states.shape
    kv_len = raw_key_states.shape[1]
    prefix_len = _visible_prefix_lengths(visible)

    num_blocks = kv_len // compress_ratio
    if num_blocks == 0:
        # Shorter than one block: everything is tail, i.e. the plain causal mask.
        return visible.clone()

    # --- Pool keys into blocks. Query-independent, so this is computed once per block
    # instead of once per (query, block) as in the upstream loop.
    pooled = raw_key_states[:, :num_blocks * compress_ratio]
    pooled = pooled.view(batch_size, num_blocks, compress_ratio, head_dim)
    pooled = pooled.float().mean(dim=2).to(raw_key_states.dtype)
    pooled = k_layernorm(pooled)

    # RoPE at each block's first key position, matching upstream's `group_starts`.
    block_starts = torch.arange(num_blocks, device=raw_key_states.device) * compress_ratio
    pooled = apply_rotary_pos_emb(
        pooled.unsqueeze(2),
        cos=key_cos[:, block_starts, :],
        sin=key_sin[:, block_starts, :],
        unsqueeze_dim=2,
    ).squeeze(2)

    # --- Score every (query, block) pair: relu then sum over index heads.
    scores = torch.einsum("bqhd,bnd->bqhn", index_query_states.float(), pooled.float())
    scores = torch.relu(scores).sum(dim=2) / math.sqrt(head_dim)

    # A block is visible to a query iff the block is entirely within its prefix.
    visible_blocks = prefix_len // compress_ratio
    block_ids = torch.arange(num_blocks, device=scores.device)
    block_is_visible = block_ids < visible_blocks.unsqueeze(-1)
    scores = scores.masked_fill(~block_is_visible, float("-inf"))

    # `topk` needs a static k; ranks beyond a query's visible blocks are discarded below.
    k_max = min(block_topk, num_blocks)
    topk_blocks = torch.topk(scores, k_max, dim=-1, sorted=False).indices
    selected = block_is_visible.gather(-1, topk_blocks)

    block_member = scores.new_zeros((batch_size, q_len, num_blocks), dtype=torch.bool)
    block_member.scatter_(-1, topk_blocks, selected)
    membership = block_member.repeat_interleave(compress_ratio, dim=-1)
    if membership.shape[-1] < kv_len:
        membership = torch.nn.functional.pad(membership, (0, kv_len - membership.shape[-1]))

    # --- The incomplete causal tail is always attended: keys in
    # [visible_blocks * compress_ratio, prefix_len).
    key_ids = torch.arange(kv_len, device=scores.device)
    tail_start = (visible_blocks * compress_ratio).unsqueeze(-1)
    tail = (key_ids >= tail_start) & (key_ids < prefix_len.unsqueeze(-1))

    return membership | tail


class Qwen4ExpQSAIndexer(Qwen4ExpTextQSAIndexer):
    """Upstream indexer with the per-query Python loop replaced by a batched selector.

    Returns the membership table instead of upstream's 4-D additive/boolean mask; the
    caller decides whether to turn it into a dense mask or a FlexAttention ``BlockMask``.
    """
    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[object] = None,
        cp_context: Optional[Qwen4ExpCPContext] = None,
    ) -> torch.Tensor:
        if past_key_values is not None:
            raise NotImplementedError(
                "QSA training path does not support a KV cache. Use vLLM/SGLang for "
                "generation."
            )
        batch_size, q_len, _ = hidden_states.shape
        full_cos, full_sin = position_embeddings

        qk = self.index_qk_proj(hidden_states)
        query_states, raw_keys = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        hidden_shape = (batch_size, q_len, -1, self.index_head_dim)
        query_states = self.q_layernorm(query_states.reshape(*hidden_shape))
        query_states = apply_rotary_pos_emb(
            query_states,
            cos=full_cos[:, -q_len:, :],
            sin=full_sin[:, -q_len:, :],
            unsqueeze_dim=2,
        )
        raw_keys = raw_keys.reshape(*hidden_shape).squeeze(2)

        if cp_context is None:
            if attention_mask is None:
                raise ValueError("QSA requires an attention mask outside packed THD")
            visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
            visible = visible.squeeze(1)
        else:
            if q_len != cp_context.local_sequence_length:
                raise ValueError(
                    "QSA indexer local sequence length disagrees with its CP context, "
                    f"got {q_len} and {cp_context.local_sequence_length}"
                )
            raw_keys = qwen4_exp_cp_all_gather(
                raw_keys,
                cp_context,
                sequence_dim=1,
                differentiable=False,
            )
            full_cos = qwen4_exp_cp_all_gather(
                full_cos,
                cp_context,
                sequence_dim=1,
                differentiable=False,
            )
            full_sin = qwen4_exp_cp_all_gather(
                full_sin,
                cp_context,
                sequence_dim=1,
                differentiable=False,
            )
            if cp_context.global_cu_seqlens is not None:
                return self._packed_membership(
                    query_states,
                    raw_keys,
                    full_cos,
                    full_sin,
                    cp_context,
                )
            query_positions = cp_context.local_sequence_start + torch.arange(
                q_len, device=hidden_states.device
            )
            key_positions = torch.arange(
                cp_context.global_sequence_length, device=hidden_states.device
            )
            causal = key_positions.view(1, 1, -1) <= query_positions.view(1, -1, 1)
            valid_keys = key_positions.view(1, 1, -1) < (
                cp_context.global_sequence_lengths.to(hidden_states.device).view(-1, 1, 1)
            )
            visible = causal & valid_keys
        return select_qsa_membership(
            query_states,
            raw_keys,
            visible,
            full_cos,
            full_sin,
            self.k_layernorm,
            self.compress_ratio,
            self.block_topk,
        )

    def _packed_membership(
        self,
        query_states: torch.Tensor,
        raw_keys: torch.Tensor,
        key_cos: torch.Tensor,
        key_sin: torch.Tensor,
        cp_context: Qwen4ExpCPContext,
    ) -> torch.Tensor:
        """Select causal blocks on a document-relative compression grid."""
        if query_states.shape[0] != 1:
            raise ValueError("packed QSA requires a THD batch of one row")
        boundaries = cp_context.global_cu_seqlens
        if boundaries is None:
            raise RuntimeError("packed QSA context is missing document boundaries")
        if cp_context.size == 1:
            return self._packed_membership_single_rank(
                query_states,
                raw_keys,
                key_cos,
                key_sin,
                boundaries,
                cp_context.global_sequence_length,
            )

        membership = torch.zeros(
            1,
            query_states.shape[1],
            cp_context.global_sequence_length,
            dtype=torch.bool,
            device=query_states.device,
        )
        local_start = cp_context.local_sequence_start
        local_end = cp_context.local_sequence_end
        for document_start, document_end in zip(
            boundaries.tolist(), boundaries[1:].tolist()
        ):
            overlap_start = max(document_start, local_start)
            overlap_end = min(document_end, local_end)
            if overlap_start >= overlap_end:
                continue
            document_length = document_end - document_start
            query_slice = query_states[
                :, overlap_start - local_start : overlap_end - local_start
            ]
            query_positions = torch.arange(
                overlap_start - document_start,
                overlap_end - document_start,
                device=query_states.device,
            )
            key_positions = torch.arange(document_length, device=query_states.device)
            visible = key_positions.view(1, 1, -1) <= query_positions.view(1, -1, 1)
            document_membership = select_qsa_membership(
                query_slice,
                raw_keys[:, document_start:document_end],
                visible,
                key_cos[:, document_start:document_end],
                key_sin[:, document_start:document_end],
                self.k_layernorm,
                self.compress_ratio,
                self.block_topk,
            )
            membership[
                :,
                overlap_start - local_start : overlap_end - local_start,
                document_start:document_end,
            ] = document_membership
        return membership

    def _packed_membership_single_rank(
        self,
        query_states: torch.Tensor,
        raw_keys: torch.Tensor,
        key_cos: torch.Tensor,
        key_sin: torch.Tensor,
        boundaries: torch.Tensor,
        global_sequence_length: int,
    ) -> torch.Tensor:
        """Batch all documents together on the CP-size-one path."""
        document_starts = boundaries[:-1]
        document_lengths = boundaries.diff()
        max_length = int(document_lengths.max())
        local_positions = torch.arange(max_length, device=query_states.device)
        row_positions = document_starts[:, None] + local_positions[None, :]
        row_valid = local_positions[None, :] < document_lengths[:, None]
        safe_positions = row_positions.clamp(max=int(boundaries[-1]) - 1)

        query_rows = query_states[0, safe_positions]
        key_rows = raw_keys[0, safe_positions]
        cos_rows = key_cos[0, safe_positions]
        sin_rows = key_sin[0, safe_positions]
        visible = (
            local_positions.view(1, -1, 1) >= local_positions.view(1, 1, -1)
        ) & row_valid.unsqueeze(-1) & row_valid.unsqueeze(1)
        document_membership = select_qsa_membership(
            query_rows,
            key_rows,
            visible,
            cos_rows,
            sin_rows,
            self.k_layernorm,
            self.compress_ratio,
            self.block_topk,
        )

        membership = torch.zeros(
            1,
            global_sequence_length,
            global_sequence_length,
            dtype=torch.bool,
            device=query_states.device,
        )
        for row, (document_start, document_end) in enumerate(
            zip(boundaries.tolist(), boundaries[1:].tolist())
        ):
            document_length = document_end - document_start
            membership[
                0, document_start:document_end, document_start:document_end
            ] = document_membership[row, :document_length, :document_length]
        return membership


def dense_sparse_gqa_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    membership: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Reference sparse GQA via an explicit boolean mask and SDPA.

    Numerically the oracle for :func:`flex_sparse_gqa_attention`, and the only path on
    CPU. Materializes ``[B, H, S_q, S_kv]`` attention scores, so it is for tests and
    short sequences — not the training path.

    Parameters
    ----------
    query_states : torch.Tensor
        ``[B, H_q, S_q, D]``.
    key_states, value_states : torch.Tensor
        ``[B, H_kv, S_kv, D]``.
    membership : torch.Tensor
        Bool ``[B, S_q, S_kv]``.

    Returns
    -------
    torch.Tensor
        ``[B, H_q, S_q, D]``.
    """
    has_routes = membership.any(dim=-1)
    # A fully masked row makes softmax produce NaN; give it a dummy key and zero the
    # result afterwards so no NaN leaks into the backward pass.
    attn_mask = membership.clone()
    attn_mask[..., 0] |= ~has_routes

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask.unsqueeze(1),
        scale=scaling,
        enable_gqa=key_states.shape[1] != query_states.shape[1],
    )
    return attn_output * has_routes[:, None, :, None]


def flex_sparse_gqa_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    membership: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Sparse GQA through FlexAttention, driven by a ``BlockMask``.

    This is the training path: the ``BlockMask`` lets FlexAttention skip fully-masked
    key blocks, so the ``[B, H_q, S_q, S_kv]`` score matrix is never materialized.

    Parameters
    ----------
    membership : torch.Tensor
        Bool ``[B, S_q, S_kv]``.

    Returns
    -------
    torch.Tensor
        ``[B, H_q, S_q, D]``.
    """
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    has_routes = membership.any(dim=-1)
    routes = membership.clone()
    routes[..., 0] |= ~has_routes

    def mask_mod(batch_idx, _head_idx, q_idx, kv_idx):
        return routes[batch_idx, q_idx, kv_idx]

    block_mask = create_block_mask(
        mask_mod,
        B=routes.shape[0],
        H=None,
        Q_LEN=query_states.shape[2],
        KV_LEN=key_states.shape[2],
        device=query_states.device,
    )
    attn_output = _compiled_flex_attention()(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=key_states.shape[1] != query_states.shape[1],
    )
    return attn_output * has_routes[:, None, :, None]


_FLEX_ATTENTION_COMPILED = None


def _compiled_flex_attention():
    """Compile ``flex_attention`` once per process; recompiling per layer is very slow."""
    global _FLEX_ATTENTION_COMPILED
    if _FLEX_ATTENTION_COMPILED is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX_ATTENTION_COMPILED = torch.compile(flex_attention, dynamic=False)
    return _FLEX_ATTENTION_COMPILED


class Qwen4ExpQSAAttention(Qwen4ExpTextAttention):
    """Sparse-attention layer wired to the batched selector.

    Structurally identical to upstream (same parameters, same gate and ``o_proj``); the
    difference is that the indexer hands back a membership table which goes straight
    into FlexAttention, instead of an additive mask that would force a dense
    ``[B, H, S_q, S_kv]`` score matrix.
    """
    def __init__(self, config, layer_idx: int, attn_backend: str = "flex"):
        super().__init__(config, layer_idx)
        if attn_backend not in QSA_ATTN_BACKENDS:
            raise ValueError(
                f"attn_backend must be one of {QSA_ATTN_BACKENDS}, got {attn_backend!r}."
            )
        self.attn_backend = attn_backend
        self.indexer = Qwen4ExpQSAIndexer(config, layer_idx)
        # No auxiliary loss and no straight-through estimator exist for the discrete
        # top-k, so the indexer cannot be trained; freeze it explicitly.
        self.indexer.requires_grad_(False)

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[object] = None,
        cp_context: Optional[Qwen4ExpCPContext] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        membership = self.indexer(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=past_key_values,
            cp_context=cp_context,
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # The indexer needs cos/sin at full key positions, so slice to the current
        # query positions only here.
        cos, sin = (x[:, -hidden_states.shape[1]:, :] for x in position_embeddings)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if cp_context is not None:
            key_states = qwen4_exp_cp_all_gather(
                key_states,
                cp_context,
                sequence_dim=2,
                differentiable=True,
            )
            value_states = qwen4_exp_cp_all_gather(
                value_states,
                cp_context,
                sequence_dim=2,
                differentiable=True,
            )

        if self.attn_backend == "flex":
            if query_states.device.type != "cuda":
                raise RuntimeError(
                    "QSA attn_backend='flex' requires CUDA; use attn_backend='dense' on "
                    "CPU. Falling back silently would change the numerics under test."
                )
            attn_output = flex_sparse_gqa_attention(
                query_states, key_states, value_states, membership, self.scaling
            )
        else:
            attn_output = dense_sparse_gqa_attention(
                query_states, key_states, value_states, membership, self.scaling
            )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None
