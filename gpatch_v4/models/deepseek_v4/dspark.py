from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_layers import GradientCheckpointingLayer

from .cp import _ring_all_to_all
from .modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
    DeepseekV4SparseMoeBlock,
    apply_rotary_pos_emb,
    fp8_qat_linear,
    fp8_simulate_qat,
    sparse_attn_tilelang,
)
from .mtp import config_for_synthetic_layer
from .thd import PackedSeqParams


@dataclass
class DSparkBatch:
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    target_ids: torch.Tensor
    eval_mask: torch.Tensor
    prev_token_ids: torch.Tensor
    target_hidden_indices: torch.Tensor


@dataclass
class DSparkForwardOutput:
    draft_logits: torch.Tensor
    target_logits: torch.Tensor
    target_ids: torch.Tensor
    eval_mask: torch.Tensor
    block_keep_mask: torch.Tensor
    confidence_logits: torch.Tensor


def _expand_thd_layout(
    packed_seq_params: PackedSeqParams | None,
    bsz: int,
    seq_len: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if packed_seq_params is None:
        return None, None
    assert packed_seq_params.layout is not None
    seg_ids = packed_seq_params.layout.seg_id_per_token_full
    pad_mask = packed_seq_params.layout.pad_token_mask_full
    assert seg_ids.shape == pad_mask.shape == (seq_len, )
    return (
        seg_ids.view(1, -1).expand(bsz, -1),
        pad_mask.view(1, -1).expand(bsz, -1),
    )


def _sample_dspark_anchors_bshd(
    *,
    valid: torch.Tensor,
    indices: torch.Tensor,
    random_values: torch.Tensor,
    num_anchors: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = valid.shape[0]
    device = valid.device
    valid_counts = valid.sum(dim=1)
    masked_indices = torch.where(
        valid,
        indices,
        torch.full_like(indices, seq_len + 1),
    )
    random_values = torch.where(
        valid,
        random_values,
        torch.full_like(random_values, 2.0),
    )
    sorted_indices = random_values.argsort(dim=1)
    sampled = torch.gather(masked_indices, 1, sorted_indices)
    if seq_len < num_anchors:
        sampled = torch.cat(
            [
                sampled,
                torch.full(
                    (bsz, num_anchors - seq_len),
                    seq_len + 1,
                    dtype=sampled.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
    anchor_positions = sampled[:, :num_anchors].sort(dim=1).values
    block_keep_mask = torch.arange(
        num_anchors, device=device
    ).unsqueeze(0) < (valid_counts.clamp(max=num_anchors).unsqueeze(1))
    anchor_positions = torch.where(
        block_keep_mask,
        anchor_positions,
        torch.zeros_like(anchor_positions),
    )
    return anchor_positions, block_keep_mask


def _sample_dspark_anchors_thd(
    *,
    valid: torch.Tensor,
    seg_ids: torch.Tensor,
    random_values: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    num_anchors: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = valid.shape[0]
    device = valid.device
    cu_padded = packed_seq_params.cu_seqlens_q_padded
    num_segments = cu_padded.numel() - 1
    segmented_keys = (
        seg_ids.to(random_values.dtype) * 3.0 +
        torch.where(valid, random_values, torch.full_like(random_values, 2.0))
    )
    sorted_indices = segmented_keys.argsort(dim=1)
    seg_lengths = cu_padded[1:] - cu_padded[:-1]
    slot_offsets = torch.arange(
        num_anchors,
        device=device,
    ).unsqueeze(0).minimum(seg_lengths.unsqueeze(1) - 1)
    sorted_slots = (cu_padded[:-1].unsqueeze(1) + slot_offsets).reshape(-1)
    sampled = torch.gather(
        sorted_indices,
        1,
        sorted_slots.unsqueeze(0).expand(bsz, -1),
    ).reshape(bsz, num_segments, num_anchors)
    valid_counts = torch.zeros(
        (bsz, num_segments),
        dtype=torch.long,
        device=device,
    ).scatter_add(1, seg_ids, valid.long())
    block_keep_mask = (
        torch.arange(num_anchors, device=device).view(1, 1, -1)
        < valid_counts.clamp(max=num_anchors).unsqueeze(-1)
    )
    sampled = torch.where(
        block_keep_mask,
        sampled,
        torch.full_like(sampled, seq_len + 1),
    )
    anchor_positions = sampled.sort(dim=-1).values
    anchor_positions = torch.where(
        block_keep_mask,
        anchor_positions,
        torch.zeros_like(anchor_positions),
    ).reshape(bsz, -1)
    block_keep_mask = block_keep_mask.reshape(bsz, -1)
    return anchor_positions, block_keep_mask


def prepare_dspark_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    num_anchors: int,
    block_size: int,
    packed_seq_params: PackedSeqParams | None = None,
    rng: torch.Generator | None = None,
) -> DSparkBatch:
    """Sample anchors once on gcore's shifted next-token-label axis.

    Parameters
    ----------
    input_ids : torch.Tensor
        Shape ``[B, S]``; token at ``p`` is the anchor for ``labels[p]``.
    labels : torch.Tensor
        Shape ``[B, S]``; already shifted left by data preparation.
    loss_mask : torch.Tensor
        Shape ``[B, S]``; valid supervision entries are nonzero.
    num_anchors : int
        Anchor blocks sampled per sequence, or per packed segment under THD.
    block_size : int
    packed_seq_params : PackedSeqParams, optional
        THD segment layout; anchors, targets, and context remain inside one
        packed segment.
    rng : torch.Generator, optional
        Dedicated anchor RNG. CP ranks must use RNGs initialized with
        the same seed when preparing the same global sequence.

    Returns
    -------
    DSparkBatch
        Fixed anchor metadata reused by forward and step-global normalization.
    """
    assert input_ids.shape == labels.shape == loss_mask.shape
    bsz, seq_len = input_ids.shape
    assert seq_len > 1
    device = input_ids.device
    seg_ids, pad_mask = _expand_thd_layout(packed_seq_params, bsz, seq_len)

    valid = (loss_mask > 0.5) & (labels >= 0)
    # DeepSpec mask[p] & mask[p+1] becomes shifted_mask[p-1] & shifted_mask[p].
    valid[:, 1:] &= loss_mask[:, :-1] > 0.5
    if seg_ids is not None:
        assert pad_mask is not None
        valid &= ~pad_mask
        valid[:, 1:] &= seg_ids[:, 1:] == seg_ids[:, :-1]
    valid[:, 0] = False
    indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
    random_values = torch.rand(bsz, seq_len, device=device, generator=rng)

    if seg_ids is None:
        anchor_positions, block_keep_mask = _sample_dspark_anchors_bshd(
            valid=valid,
            indices=indices,
            random_values=random_values,
            num_anchors=num_anchors,
            seq_len=seq_len,
        )
    else:
        assert packed_seq_params is not None
        anchor_positions, block_keep_mask = _sample_dspark_anchors_thd(
            valid=valid,
            seg_ids=seg_ids,
            random_values=random_values,
            packed_seq_params=packed_seq_params,
            num_anchors=num_anchors,
            seq_len=seq_len,
        )

    num_anchor_slots = anchor_positions.shape[1]

    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    target_indices = anchor_positions.unsqueeze(-1) + offsets
    safe_indices = target_indices.clamp(max=seq_len - 1)
    expanded_labels = labels.unsqueeze(1).expand(-1, num_anchor_slots, -1)
    expanded_loss_mask = loss_mask.unsqueeze(1).expand(-1, num_anchor_slots, -1)
    gathered_labels = torch.gather(expanded_labels, 2, safe_indices)
    gathered_loss_mask = torch.gather(expanded_loss_mask, 2, safe_indices)
    eval_mask = (
        (target_indices < seq_len) & (gathered_labels >= 0) & (gathered_loss_mask > 0.5) &
        block_keep_mask.unsqueeze(-1)
    )
    if seg_ids is not None:
        assert pad_mask is not None
        expanded_seg_ids = seg_ids.unsqueeze(1).expand(-1, num_anchor_slots, -1)
        expanded_pad_mask = pad_mask.unsqueeze(1).expand(-1, num_anchor_slots, -1)
        target_seg_ids = torch.gather(expanded_seg_ids, 2, safe_indices)
        target_pad_mask = torch.gather(expanded_pad_mask, 2, safe_indices)
        anchor_seg_ids = torch.gather(seg_ids, 1, anchor_positions)
        eval_mask &= (target_seg_ids == anchor_seg_ids.unsqueeze(-1)) & ~target_pad_mask
    eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()
    target_ids = torch.where(eval_mask, gathered_labels.clamp_min(0), 0)

    anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
    prev_token_ids = torch.cat(
        [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
        dim=-1,
    )
    return DSparkBatch(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        target_ids=target_ids,
        eval_mask=eval_mask,
        prev_token_ids=prev_token_ids,
        target_hidden_indices=safe_indices,
    )


def shard_dspark_batch_for_contiguous_cp(
    batch: DSparkBatch,
    *,
    sequence_length: int,
    cp_rank: int,
    cp_size: int,
) -> DSparkBatch:
    """Assign every global anchor slot to its contiguous CP owner.

    All ranks retain the global slot dimension in this non-compacting version.
    Only the owner keeps an active block/evaluation mask; global coordinates
    remain unchanged for segment masking and later halo-buffer mapping.

    Parameters
    ----------
    batch : DSparkBatch
        Metadata prepared from the unsliced global sequence.
    sequence_length : int
        Global physical sequence length; divisible by ``cp_size``.
    cp_rank : int
    cp_size : int

    Returns
    -------
    DSparkBatch
        Owner-masked metadata with the same tensor shapes as ``batch``.
    """
    assert cp_size > 0
    assert 0 <= cp_rank < cp_size
    assert sequence_length % cp_size == 0
    shard_length = sequence_length // cp_size
    sequence_start = cp_rank * shard_length
    sequence_end = sequence_start + shard_length
    owned = (
        batch.block_keep_mask & (batch.anchor_positions >= sequence_start) &
        (batch.anchor_positions < sequence_end)
    )
    return DSparkBatch(
        anchor_positions=batch.anchor_positions,
        block_keep_mask=owned,
        target_ids=batch.target_ids,
        eval_mask=batch.eval_mask & owned.unsqueeze(-1),
        prev_token_ids=batch.prev_token_ids,
        target_hidden_indices=batch.target_hidden_indices,
    )


def _prepend_dspark_left_halo(
    tensor: torch.Tensor,
    halo_size: int,
    cp_group,
) -> tuple[torch.Tensor, int]:
    if cp_group is None or dist.get_world_size(cp_group) == 1 or halo_size == 0:
        return tensor, 0
    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)
    assert halo_size <= tensor.shape[1], (
        f"DSpark left halo ({halo_size}) exceeds local sequence length "
        f"({tensor.shape[1]})"
    )
    prefix = _ring_all_to_all(
        send_tensor=tensor[:, -halo_size:].contiguous(),
        send_peer=(cp_rank + 1) % cp_size,
        recv_peer=(cp_rank - 1) % cp_size,
        group=cp_group,
        seq_dim=1,
    )
    if cp_rank == 0:
        return tensor, 0
    return torch.cat([prefix, tensor], dim=1), halo_size


def _append_dspark_right_halo(
    tensor: torch.Tensor,
    halo_size: int,
    cp_group,
) -> torch.Tensor:
    if cp_group is None or dist.get_world_size(cp_group) == 1 or halo_size == 0:
        return tensor
    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)
    assert halo_size <= tensor.shape[1], (
        f"DSpark right halo ({halo_size}) exceeds local sequence length "
        f"({tensor.shape[1]})"
    )
    suffix = _ring_all_to_all(
        send_tensor=tensor[:, :halo_size].contiguous(),
        send_peer=(cp_rank - 1) % cp_size,
        recv_peer=(cp_rank + 1) % cp_size,
        group=cp_group,
        seq_dim=1,
    )
    if cp_rank == cp_size - 1:
        return tensor
    return torch.cat([tensor, suffix], dim=1)


def build_dspark_contiguous_cp_buffers(
    *,
    target_hidden_states: torch.Tensor,
    target_last_hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    sliding_window: int,
    block_size: int,
    cp_group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Exchange the boundary tensors required by CP-owned DSpark anchors.

    Parameters
    ----------
    target_hidden_states : torch.Tensor
        Detached local backbone targets shaped ``[B, s_local, D_target]``.
    target_last_hidden_states : torch.Tensor
        Detached local teacher states shaped ``[B, s_local, H]``.
    position_ids : torch.Tensor
        Local RoPE positions shaped ``[B or 1, s_local]``.
    sliding_window : int
        Number of preceding target states visible to each draft token.
    block_size : int
        Teacher block length; the right halo contains ``block_size - 1`` rows.
    cp_group : dist.ProcessGroup, optional

    Returns
    -------
    tuple
        Left-prefixed target states, matching positions, right-suffixed teacher
        states, and the target prefix length on this rank.
    """
    target_context, prefix_length = _prepend_dspark_left_halo(
        target_hidden_states,
        sliding_window,
        cp_group,
    )
    target_positions, position_prefix_length = _prepend_dspark_left_halo(
        position_ids,
        sliding_window,
        cp_group,
    )
    assert position_prefix_length == prefix_length
    teacher_hidden = _append_dspark_right_halo(
        target_last_hidden_states,
        block_size - 1,
        cp_group,
    )
    return target_context, target_positions, teacher_hidden, prefix_length


def build_dspark_sparse_topk(
    batch: DSparkBatch,
    *,
    seq_len: int,
    block_size: int,
    sliding_window: int,
    packed_seq_params: PackedSeqParams | None = None,
    sequence_start: int = 0,
    context_prefix_len: int = 0,
    target_context_len: int | None = None,
) -> torch.Tensor:
    """Build V4 visibility over preceding SWA context and the local draft block.

    Parameters
    ----------
    batch : DSparkBatch
    seq_len : int
    block_size : int
    sliding_window : int
    packed_seq_params : PackedSeqParams, optional
        THD segment layout used to mask cross-segment target context.
    sequence_start : int
        Global coordinate of the first local CP token.
    context_prefix_len : int
        Number of preceding tokens prepended to the local target buffer.
    target_context_len : int, optional
        Length of the target KV buffer. Defaults to ``seq_len`` for CP=1.

    Returns
    -------
    torch.Tensor
        Int32 indices shaped ``[B, num_anchor_slots * block_size, sliding_window +
        block_size]``; unavailable context slots are ``-1``.
    """
    # anchors: [B, A].
    # CP owner masking 后，只有属于当前 rank 的 slots 满足 block_keep_mask=True，但所有 rank 仍保留相同数量和顺序的 slots。
    anchors = batch.anchor_positions
    bsz, num_anchor_slots = anchors.shape
    device = anchors.device
    if target_context_len is None:
        target_context_len = seq_len
    assert 0 <= context_prefix_len <= target_context_len
    # 返回全局 [B, T], seg_ids：每个位置属于哪个 THD segment; pad_mask：每个位置是否为 padding。
    seg_ids, pad_mask = _expand_thd_layout(packed_seq_params, bsz, seq_len)
    context_offsets = torch.arange(
        -sliding_window,
        0,
        device=device,
    ).view(1, 1, -1)
    # context_indices: [B, A, W]
    context_indices = anchors.unsqueeze(-1) + context_offsets
    # context 后处理，越界设置 -1
    context_indices = context_indices.masked_fill(context_indices < 0, -1)
    if seg_ids is not None:
        assert pad_mask is not None
        safe_context_indices = context_indices.clamp(min=0)
        context_seg_ids = torch.gather(
            seg_ids.unsqueeze(1).expand(-1, num_anchor_slots, -1),
            2,
            safe_context_indices,
        )
        context_pad_mask = torch.gather(
            pad_mask.unsqueeze(1).expand(-1, num_anchor_slots, -1),
            2,
            safe_context_indices,
        )
        anchor_seg_ids = torch.gather(seg_ids, 1, anchors)
        # 跨 seg 段设置 -1
        context_indices = context_indices.masked_fill(
            (context_seg_ids != anchor_seg_ids.unsqueeze(-1)) | context_pad_mask,
            -1,
        )
    context_buffer_start = sequence_start - context_prefix_len
    context_indices = torch.where(
        context_indices >= 0,
        context_indices - context_buffer_start,
        context_indices,
    )
    context_indices = context_indices.masked_fill(
        (context_indices < 0) | (context_indices >= target_context_len),
        -1,
    )

    block_starts = (
        target_context_len +
        torch.arange(num_anchor_slots, device=device).view(1, -1, 1) * block_size
    )
    # W 个历史 target KV + K 个当前 draft KV
    draft_indices = block_starts + torch.arange(block_size, device=device).view(1, 1, -1)
    draft_indices = draft_indices.expand(bsz, -1, -1)
    context_indices = context_indices.masked_fill(
        ~batch.block_keep_mask.unsqueeze(-1),
        -1,
    )
    topk = torch.cat([context_indices, draft_indices], dim=-1)
    return (
        topk.unsqueeze(2).expand(-1, -1, block_size,
                                 -1).reshape(bsz, num_anchor_slots * block_size,
                                             -1).int().contiguous()
    )


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, prev_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.markov_w1(prev_token_ids)
        return self.markov_w2(embeddings), embeddings


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([hidden_states, markov_embeddings], dim=-1)
        return self.proj(features.float()).squeeze(-1)


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    def _project_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self.config.fp8_qat:
            kv = self.kv_norm(fp8_qat_linear(self.kv_proj, hidden_states,
                                             128)).view(*hidden_shape).transpose(1, 2)
        else:
            kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        kv = apply_rotary_pos_emb(kv, cos, sin)
        if self.config.fp8_qat:
            nope = self.config.head_dim - self.config.qk_rope_head_dim
            kv = torch.cat(
                [fp8_simulate_qat(kv[..., :nope], 64), kv[..., nope:]],
                dim=-1,
            )
        return kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        target_hidden_states: torch.Tensor,
        target_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        draft_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        sparse_topk: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        assert self.config.attn_backend == "fused"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self.config.fp8_qat:
            q_residual = self.q_a_norm(fp8_qat_linear(self.q_a_proj, hidden_states, 128))
            q = fp8_qat_linear(self.q_b_proj, q_residual, 128)
        else:
            q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(q_residual)
        q = q.view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        draft_cos, draft_sin = draft_position_embeddings
        q = apply_rotary_pos_emb(q, draft_cos, draft_sin)

        target_kv = self._project_kv(target_hidden_states, target_position_embeddings)
        draft_kv = self._project_kv(hidden_states, draft_position_embeddings)
        kv = torch.cat([target_kv, draft_kv], dim=2)
        sinks = self.sinks if self.config.amp_fp32 else self.sinks.float()
        attention_output = sparse_attn_tilelang(
            rearrange(q, "B H S D -> B S H D").contiguous(),
            rearrange(kv, "B 1 S D -> B S D").contiguous(),
            sinks,
            sparse_topk,
            sm_scale=self.scaling,
        )
        attention_output = apply_rotary_pos_emb(
            attention_output.transpose(1, 2),
            draft_cos,
            -draft_sin,
        ).transpose(1, 2)
        grouped = attention_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        if self.config.fp8_qat:
            output = fp8_qat_linear(self.o_b_proj, grouped, 128)
        else:
            output = self.o_b_proj(grouped)
        return output, None


class DeepseekV4DSparkBlock(GradientCheckpointingLayer):
    def __init__(
        self,
        config,
        *,
        layer_idx: int,
        stage_idx: int,
        num_stages: int,
    ) -> None:
        super().__init__()
        block_config = config_for_synthetic_layer(config, layer_idx)
        self.config = block_config
        self.stage_idx = stage_idx
        self.self_attn = DeepseekV4DSparkAttention(block_config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(block_config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = DeepseekV4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

        if stage_idx == 0:
            self.main_proj = nn.Linear(
                len(config.dspark_target_layer_ids) * config.hidden_size,
                config.hidden_size,
                bias=False,
            )
            self.main_norm = DeepseekV4RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        if stage_idx == num_stages - 1:
            self.hc_head = DeepseekV4HyperHead(config)
            self.norm = DeepseekV4RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.markov_head = DSparkMarkovHead(
                config.vocab_size,
                config.dspark_markov_rank,
            )
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
            )

    def project_target(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.stage_idx == 0
        if self.config.fp8_qat:
            return self.main_norm(fp8_qat_linear(self.main_proj, target_hidden_states, 128))
        return self.main_norm(self.main_proj(target_hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        draft_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        sparse_topk: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if self.stage_idx == 0:
            target_hidden_states = self.project_target(target_hidden_states)
        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        attention_output, _ = self.self_attn(
            self.input_layernorm(collapsed),
            target_hidden_states=target_hidden_states,
            target_position_embeddings=target_position_embeddings,
            draft_position_embeddings=draft_position_embeddings,
            sparse_topk=sparse_topk,
        )
        hidden_states = post.to(dtype).unsqueeze(-1
                                                ) * attention_output.unsqueeze(-2) + torch.matmul(
                                                    comb.to(dtype).transpose(-1, -2), hidden_states
                                                )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(
            self.post_attention_layernorm(collapsed),
            input_ids=input_ids,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        if not hasattr(self, "hc_head"):
            return hidden_states, target_hidden_states, None, None, None
        head_hidden = self.hc_head(hidden_states)
        normalized_hidden = self.norm(head_hidden)
        markov_bias, markov_embeddings = self.markov_head(prev_token_ids)
        confidence_logits = self.confidence_head(
            head_hidden.reshape(*prev_token_ids.shape, -1),
            markov_embeddings,
        )
        return (
            hidden_states,
            target_hidden_states,
            normalized_hidden,
            markov_bias,
            confidence_logits,
        )


class DeepseekV4DSparkModule(nn.Module):
    def __init__(self, config, rotary_emb) -> None:
        super().__init__()
        assert config.dspark_num_layers > 0
        assert config.dspark_block_size > 0
        assert config.dspark_noise_token_id is not None
        assert config.dspark_target_layer_ids
        assert config.dspark_num_anchors > 0
        assert config.dspark_target_layer_ids == sorted(set(config.dspark_target_layer_ids))
        assert config.dspark_target_layer_ids[-1] < config.num_hidden_layers
        base_layer_idx = config.num_hidden_layers
        num_stages = config.dspark_num_layers
        self.layers = nn.ModuleList(
            [
                DeepseekV4DSparkBlock(
                    config,
                    layer_idx=base_layer_idx + stage_idx,
                    stage_idx=stage_idx,
                    num_stages=num_stages,
                ) for stage_idx in range(num_stages)
            ]
        )
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        self.num_anchors = config.dspark_num_anchors
        self.sliding_window = config.sliding_window
        self.hidden_size = config.hidden_size
        object.__setattr__(self, "_rotary_emb", rotary_emb)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_last_hidden_states: torch.Tensor,
        batch: DSparkBatch,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        packed_seq_params: PackedSeqParams | None = None,
        cp_group=None,
    ) -> DSparkForwardOutput:
        bsz, seq_len = input_ids.shape
        cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
        cp_rank = 0 if cp_size == 1 else dist.get_rank(cp_group)
        global_seq_len = seq_len * cp_size
        sequence_start = cp_rank * seq_len
        num_anchor_slots = batch.anchor_positions.shape[1]
        local_anchor_positions = batch.anchor_positions - sequence_start
        local_anchor_positions = torch.where(
            batch.block_keep_mask,
            local_anchor_positions,
            torch.zeros_like(local_anchor_positions),
        )
        block_starts = (
            torch.arange(num_anchor_slots, device=input_ids.device).view(1, -1) * self.block_size
        )
        noise_ids = torch.full(
            (bsz, num_anchor_slots * self.block_size),
            self.noise_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        anchor_tokens = torch.gather(input_ids, 1, local_anchor_positions)
        batch_indices = torch.arange(bsz, device=input_ids.device).view(-1, 1)
        noise_ids[batch_indices, block_starts] = torch.where(
            batch.block_keep_mask,
            anchor_tokens,
            self.noise_token_id,
        )
        hidden_states = F.embedding(noise_ids, embed_weight.detach())
        hidden_states = hidden_states.unsqueeze(2).expand(
            -1,
            -1,
            self.layers[0].config.hc_mult,
            -1,
        ).contiguous()

        if position_ids.shape[0] == 1 and bsz > 1:
            position_ids = position_ids.expand(bsz, -1)
        anchor_position_ids = torch.gather(position_ids, 1, local_anchor_positions)
        offsets = torch.arange(self.block_size, device=input_ids.device).view(1, 1, -1)
        draft_position_ids = (anchor_position_ids.unsqueeze(-1) + offsets).reshape(bsz, -1)
        (
            target_hidden_states,
            target_position_ids,
            target_last_hidden_states,
            context_prefix_len,
        ) = build_dspark_contiguous_cp_buffers(
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=target_last_hidden_states,
            position_ids=position_ids,
            sliding_window=self.sliding_window,
            block_size=self.block_size,
            cp_group=cp_group,
        )
        target_position_embeddings = self._rotary_emb(
            target_hidden_states,
            position_ids=target_position_ids,
            layer_type="main",
        )
        draft_position_embeddings = self._rotary_emb(
            hidden_states[:, :, 0],
            position_ids=draft_position_ids,
            layer_type="main",
        )
        # sparse_topk： [B, A*K, W+K]
        sparse_topk = build_dspark_sparse_topk(
            batch,
            seq_len=global_seq_len,
            block_size=self.block_size,
            sliding_window=self.sliding_window,
            packed_seq_params=packed_seq_params,
            sequence_start=sequence_start,
            context_prefix_len=context_prefix_len,
            target_context_len=target_hidden_states.shape[1],
        )

        normalized_hidden = None
        markov_bias = None
        confidence_logits = None
        for layer in self.layers:
            (
                hidden_states,
                target_hidden_states,
                normalized_hidden,
                markov_bias,
                confidence_logits,
            ) = layer(
                hidden_states,
                input_ids=noise_ids,
                target_hidden_states=target_hidden_states,
                target_position_embeddings=target_position_embeddings,
                draft_position_embeddings=draft_position_embeddings,
                sparse_topk=sparse_topk,
                prev_token_ids=batch.prev_token_ids,
            )
        assert normalized_hidden is not None
        assert markov_bias is not None
        assert confidence_logits is not None

        gather_indices = batch.target_hidden_indices - sequence_start
        gather_indices = torch.where(
            batch.block_keep_mask.unsqueeze(-1),
            gather_indices,
            torch.zeros_like(gather_indices),
        ).reshape(bsz, -1)
        aligned_target_hidden = torch.gather(
            target_last_hidden_states,
            1,
            gather_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size),
        )
        head_inputs = torch.cat([normalized_hidden, aligned_target_hidden], dim=1)
        all_logits = F.linear(head_inputs, lm_head_weight.detach())
        draft_base_logits, target_logits = all_logits.split(
            normalized_hidden.shape[1],
            dim=1,
        )
        draft_logits = draft_base_logits + markov_bias.reshape_as(draft_base_logits)
        return DSparkForwardOutput(
            draft_logits=draft_logits.reshape(
                bsz,
                num_anchor_slots,
                self.block_size,
                -1,
            ),
            target_logits=target_logits.detach().reshape(
                bsz,
                num_anchor_slots,
                self.block_size,
                -1,
            ),
            target_ids=batch.target_ids,
            eval_mask=batch.eval_mask,
            block_keep_mask=batch.block_keep_mask,
            confidence_logits=confidence_logits,
        )
