from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer

from .modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
    DeepseekV4SparseMoeBlock,
)


def roll_tensor(
    tensor: torch.Tensor,
    shifts: int = -1,
    dim: int = -1,
    cp_group=None,
) -> torch.Tensor:
    """Roll ``tensor`` along ``dim`` by ``shifts`` and zero the wrapped slice."""
    if cp_group is not None:
        return mtp_roll_tensor_cp(
            tensor,
            shifts=shifts,
            dim=dim,
            cp_group=cp_group,
        )[0]
    rolled = torch.roll(tensor, shifts=shifts, dims=dim)
    if shifts == 0 or tensor.shape[dim] == 0:
        return rolled
    n = abs(shifts)
    if shifts < 0:
        idx = torch.arange(tensor.shape[dim] - n, tensor.shape[dim], device=tensor.device)
    else:
        idx = torch.arange(0, n, device=tensor.device)
    return rolled.index_fill(dim, idx, 0)


def mtp_roll_tensor_cp(
    tensor: torch.Tensor,
    *,
    shifts: int = -1,
    dim: int = -1,
    cp_group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll along sequence dim with DSV4 contiguous-CP boundary exchange.

    Designed for MTP's left-shift (`shifts=-1`) on tensors whose sequence axis
    is partitioned contiguously across CP ranks (`cp_chunk_data` contract).

    For ``cp_size == 1`` (or ``cp_group is None``), behavior matches
    ``torch.roll(..., -1)`` followed by zeroing the wrapped tail slot.

    For ``cp_size > 1`` and ``shifts == -1``:
    - local roll left by one token,
    - replace the local tail token with the *next rank's pre-roll first token*,
    - last CP rank fills the local tail token with zero.
    """
    if shifts == 0 or tensor.shape[dim] == 0:
        return tensor, tensor.sum()

    if cp_group is None or dist.get_world_size(cp_group) == 1:
        rolled = torch.roll(tensor, shifts=shifts, dims=dim)
        if shifts < 0:
            rolled.select(dim, tensor.shape[dim] - 1).zero_()
        else:
            rolled.select(dim, 0).zero_()
        return rolled, rolled.sum()

    assert shifts == -1, (
        f"mtp_roll_tensor_cp currently supports shifts=-1 only, got shifts={shifts}"
    )
    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)

    rolled = torch.roll(tensor, shifts=shifts, dims=dim)
    tail_view = rolled.select(dim, tensor.shape[dim] - 1)

    recv_from_next: torch.Tensor | None = None
    ops: list[dist.P2POp] = []
    if cp_rank > 0:
        send_to_prev = tensor.select(dim, 0).contiguous()
        ops.append(dist.P2POp(
            dist.isend,
            send_to_prev,
            group_peer=cp_rank - 1,
            group=cp_group,
        ))
    if cp_rank < cp_size - 1:
        recv_from_next = torch.empty_like(tail_view)
        ops.append(
            dist.P2POp(
                dist.irecv,
                recv_from_next,
                group_peer=cp_rank + 1,
                group=cp_group,
            )
        )
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    if recv_from_next is None:
        tail_view.zero_()
    else:
        tail_view.copy_(recv_from_next)
    return rolled, rolled.sum()


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
        **kwargs,
    ) -> list[torch.Tensor]:
        per_depth_h: list[torch.Tensor] = []
        cur_input_ids = input_ids
        for block in self.layers:
            cur_input_ids = roll_tensor(
                cur_input_ids,
                shifts=-1,
                dim=-1,
                cp_group=cp_group,
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
