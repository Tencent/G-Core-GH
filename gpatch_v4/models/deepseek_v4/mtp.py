from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer

from .cp import _ring_all_to_all
from .modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
    DeepseekV4SparseMoeBlock,
)


def mtp_roll_tensor(
    tensor: torch.Tensor,
    cp_group=None,
    packed_seq_params=None,
    trail: int = 1,
    fill_value: int | float = 0,
) -> torch.Tensor:
    """Left-shift ``tensor`` by 1 along the sequence dim and fill the
    last ``trail`` positions per segment with ``fill_value``.

    Three paths: THD (segment-aware via ``packed_seq_params``),
    CP (P2P boundary exchange via ``cp_group``), or plain roll.
    """
    cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
    cp_rank = 0 if cp_size <= 1 else dist.get_rank(cp_group)

    if packed_seq_params is not None:
        rolled = _roll_tensor_thd(tensor, packed_seq_params=packed_seq_params, cp_group=cp_group)
        cu = packed_seq_params.cu_seqlens_q_padded
        s_local = tensor.shape[1]
        global_start = cp_rank * s_local
        global_end = global_start + s_local
        for seg_i in range(cu.shape[0] - 1):
            seg_start_g = int(cu[seg_i].item())
            seg_end_g = int(cu[seg_i + 1].item())
            trail_start_g = max(seg_start_g, seg_end_g - trail)
            # Intersect [trail_start_g, seg_end_g) with local [global_start, global_end)
            lo = max(trail_start_g, global_start) - global_start
            hi = min(seg_end_g, global_end) - global_start
            if lo < hi:
                rolled[:, lo:hi] = fill_value
        return rolled
    elif cp_group is not None:
        rolled = _roll_tensor_cp(tensor, cp_group=cp_group)[0]
        local_seq = tensor.shape[1]
        s_full = local_seq * cp_size
        g0 = s_full - trail
        local_from = max(0, g0 - cp_rank * local_seq)
        if local_from < local_seq:
            rolled[:, local_from:] = fill_value
        return rolled
    else:
        rolled = torch.roll(tensor, shifts=-1, dims=1)
        idx = torch.arange(tensor.shape[1] - trail, tensor.shape[1], device=tensor.device)
        return rolled.index_fill(1, idx, fill_value)


def _roll_tensor_cp(
    tensor: torch.Tensor,
    cp_group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-shift by 1 along dim=1 with DSV4 contiguous-CP boundary exchange.

    ``cp_size == 1``: ``torch.roll(..., -1)`` + zero tail.

    ``cp_size > 1``: local roll left by one, replace the local tail token
    with the next rank's pre-roll first token via ``all_to_all_single``
    ring permutation; last CP rank zeros its tail.
    """
    if tensor.shape[1] == 0:
        return tensor, tensor.sum()

    if cp_group is None or dist.get_world_size(cp_group) == 1:
        rolled = torch.roll(tensor, shifts=-1, dims=1)
        rolled[:, -1] = 0
        return rolled, rolled.sum()

    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)

    rolled = torch.roll(tensor, shifts=-1, dims=1)

    send_buf = tensor[:, :1, ...].contiguous()
    recv_buf = _ring_all_to_all(
        send_tensor=send_buf,
        send_peer=(cp_rank - 1) % cp_size,
        recv_peer=(cp_rank + 1) % cp_size,
        group=cp_group,
        seq_dim=1,
    )
    if cp_rank < cp_size - 1:
        rolled[:, -1:] = recv_buf
    else:
        rolled[:, -1] = 0
    return rolled, rolled.sum()


def _roll_tensor_thd(
    tensor: torch.Tensor,
    packed_seq_params,
    cp_group=None,
) -> torch.Tensor:
    """Segment-aware left-shift by 1 along dim=1 for THD packed sequences.

    Each segment (defined by ``cu_seqlens_q_padded``, GLOBAL offsets)
    is rolled independently; the segment-final position is filled with
    zero. Tokens never cross segment boundaries.

    When ``cp_group`` has size > 1, ``tensor`` is the local CP chunk
    ``[B, s_local]`` while ``cu_seqlens_q_padded`` stays global.
    The local tail is exchanged with the next rank's first token
    via ``all_to_all_single`` ring permutation (same pattern as
    ``cp.py::_ring_all_to_all``); segment-final positions are then
    zeroed regardless of what the exchange put there.
    """

    cu = packed_seq_params.cu_seqlens_q_padded
    if cu.device != tensor.device:
        cu = cu.to(tensor.device)

    cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)

    if cp_size <= 1:
        rolled = tensor.clone()
        for seg_i in range(cu.shape[0] - 1):
            seg_start = int(cu[seg_i].item())
            seg_end = int(cu[seg_i + 1].item())
            if seg_end <= seg_start:
                continue
            seg_rolled = torch.roll(tensor[:, seg_start:seg_end], shifts=-1, dims=1)
            seg_rolled[:, -1] = 0
            rolled[:, seg_start:seg_end] = seg_rolled
        return rolled

    cp_rank = dist.get_rank(cp_group)
    s_local = tensor.shape[1]
    global_start = cp_rank * s_local

    # Step 1: local roll + ring exchange for CP boundary
    rolled = torch.roll(tensor, shifts=-1, dims=1)

    # Send first token to prev rank, recv from next rank into local tail.
    # Uses all_to_all_single ring pattern (avoids P2P hang issues).
    send_buf = tensor[:, :1, ...].contiguous()  # pre-roll first token
    recv_buf = _ring_all_to_all(
        send_tensor=send_buf,
        send_peer=(cp_rank - 1) % cp_size,
        recv_peer=(cp_rank + 1) % cp_size,
        group=cp_group,
        seq_dim=1,
    )
    if cp_rank < cp_size - 1:
        rolled[:, -1:] = recv_buf
    else:
        rolled[:, -1] = 0

    # Step 2: zero segment-final positions within local chunk.
    for seg_i in range(cu.shape[0] - 1):
        seg_end = int(cu[seg_i + 1].item())
        local_pos = seg_end - 1 - global_start
        if 0 <= local_pos < s_local:
            rolled[:, local_pos] = 0

    return rolled


@dataclass
class DeepseekV4MTPConfig:
    num_layers: int
    loss_scaling_factor: float = 0.1

    @property
    def enabled(self) -> bool:
        return self.num_layers > 0


class DeepseekV4MTPBlock(GradientCheckpointingLayer):
    """One MTP depth for DeepSeek-V4."""
    def __init__(self, config, layer_idx: int, rotary_emb):
        super().__init__()
        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.enorm = DeepseekV4RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepseekV4RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.e_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Build an attention config that can index layer_types/compress_ratios
        # at the synthetic MTP layer index.
        #
        # DSV4-Flash MTP blocks always use ``sliding_attention`` (no CSA/HCA
        # compressor) and ``moe`` (standard TopK router, not hash-based).
        # This is evident from the on-disk MTP keys: only wq_a/wq_b/wkv/
        # wo_a/wo_b/q_norm/kv_norm/attn_sink (no compressor.*) and
        # ffn.gate.weight (TopK, not hash). We must set these explicitly
        # rather than appending the backbone's last layer_type, because
        # truncated models may have a CSA/HCA layer at the end, which would
        # wrongly create compressor params that don't exist in the checkpoint.
        attn_cfg = config
        if layer_idx >= len(config.layer_types):
            attn_cfg = copy.deepcopy(config)
            attn_cfg.layer_types = list(config.layer_types)
            while len(attn_cfg.layer_types) <= layer_idx:
                attn_cfg.layer_types.append("sliding_attention")
            if getattr(attn_cfg, "mlp_layer_types", None) is not None:
                attn_cfg.mlp_layer_types = list(attn_cfg.mlp_layer_types)
                while len(attn_cfg.mlp_layer_types) <= layer_idx:
                    attn_cfg.mlp_layer_types.append("moe")
            if getattr(attn_cfg, "compress_ratios", None) is not None:
                attn_cfg.compress_ratios = list(attn_cfg.compress_ratios)
                while len(attn_cfg.compress_ratios) <= layer_idx:
                    attn_cfg.compress_ratios.append(0)

        self.config = attn_cfg
        self.self_attn = DeepseekV4Attention(attn_cfg, layer_idx=layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(attn_cfg, layer_idx=layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)
        self.hc_head = DeepseekV4HyperHead(config)
        self.norm = DeepseekV4RMSNorm(hidden_size, eps=config.rms_norm_eps)
        object.__setattr__(self, "_rotary_emb", rotary_emb)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.dim() != 4:
            raise ValueError(f"MTP expects HC hidden [B,S,hc,H], got {tuple(hidden_states.shape)}")

        e = self.e_proj(self.enorm(embed_input)).unsqueeze(2)
        h = self.h_proj(self.hnorm(hidden_states))
        hidden_states = e + h

        if position_ids is None:
            bsz, seq_len = embed_input.shape[:2]
            position_ids = torch.arange(seq_len, dtype=torch.long,
                                        device=embed_input.device).unsqueeze(0).expand(bsz, -1)
        position_embeddings = {
            "main":
                self._rotary_emb(embed_input, position_ids=position_ids, layer_type="main"),
            "compress":
                self._rotary_emb(embed_input, position_ids=position_ids, layer_type="compress"),
        }

        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(
            self.input_layernorm(collapsed),
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        hidden_states = post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        prediction_hidden = self.norm(self.hc_head(hidden_states))
        return hidden_states, prediction_hidden


class DeepseekV4MTPModule(nn.Module):
    def __init__(self, config, mtp_config: DeepseekV4MTPConfig, rotary_emb):
        super().__init__()
        if not mtp_config.enabled:
            raise ValueError("DeepseekV4MTPModule requires enabled mtp_config")
        self.mtp_config = mtp_config
        base_layer_idx = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                DeepseekV4MTPBlock(config, layer_idx=base_layer_idx + d, rotary_emb=rotary_emb)
                for d in range(mtp_config.num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        embed_fn,
        cp_group=None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        packed_seq_params=None,
        **kwargs,
    ) -> list[torch.Tensor]:
        per_depth_h: list[torch.Tensor] = []
        cur_input_ids = input_ids
        for block in self.layers:
            cur_input_ids = mtp_roll_tensor(
                cur_input_ids,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
            )
            embed_input = embed_fn(cur_input_ids)
            hidden_states, prediction_hidden = block(
                hidden_states,
                embed_input=embed_input,
                input_ids=cur_input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            per_depth_h.append(prediction_hidden)
        return per_depth_h
