# fmt: off
# yapf: disable
# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
# adapted from transformers 5.8.1

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations.moe import _grouped_linear
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs

try:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4GroupedLinear,
        DeepseekV4HashRouter,
        DeepseekV4HyperConnection,
        DeepseekV4HyperHead,
        DeepseekV4MLP,
        DeepseekV4RMSNorm,
        DeepseekV4RotaryEmbedding,
        DeepseekV4UnweightedRMSNorm,
        apply_rotary_pos_emb,
        eager_attention_forward,
        load_balancing_loss_func,
    )
except ImportError as exc:
    print(
        "无法导入 Deepseek-V4 依赖（transformers.models.deepseek_v4.*）。依赖 transformers==5.8.1, 不是使用 dsv4 的话可以忽略"
    )
    DeepseekV4Config = None
    DeepseekV4GroupedLinear = None
    DeepseekV4HashRouter = None
    DeepseekV4HyperConnection = None
    DeepseekV4HyperHead = None
    DeepseekV4MLP = None
    DeepseekV4RMSNorm = None
    DeepseekV4RotaryEmbedding = None
    DeepseekV4UnweightedRMSNorm = None


from einops import rearrange

from gpatch_v4.models.hp_module import HpModule

from .a2a import all_to_all_uneven
from .cp import build_cp_causal_mask, compressor_cp_ag, compressor_cp_ring, swa_ring_kv
from .deepep_a2a import fused_combine, fused_dispatch
from .kernel.tilelang_indexer_fwd import _make_causal_cu_seqlens, batched_indexer_fwd
from .kernel.tilelang_sparse_mla import sparse_attn_tilelang

try:
    from .qat import fp4_simulate_qat, fp8_qat_linear, fp8_simulate_qat
except ImportError:
    fp4_simulate_qat = None
    fp8_qat_linear = None
    fp8_simulate_qat = None
from .thd import PackedSeqParams


class _Fp32ParamHolder(nn.Module):
    """Single ``nn.Parameter`` wrapped in its own ``nn.Module`` so FSDP2 can
    apply a no-cast ``MixedPrecisionPolicy`` to it independently of the
    parent module's bf16 siblings.

    Two implementation notes:

    - Reads must go through ``__call__`` (e.g. via a ``@property`` on the
      parent module) so FSDP2's pre-forward hook fires and unshards before
      the value is consumed.
    - ``forward`` takes a positional arg even though it ignores it.
      Otherwise FSDP2's ``_root_pre_forward`` indexes an empty arg tuple
      and raises ``IndexError`` during grad-ckpt recompute.

    Parameters
    ----------
    shape : tuple of int
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape))

    def forward(self, _unused: torch.Tensor | None = None) -> torch.Tensor:
        return self.weight + 0.  # don't trust AI


class DeepseekV4HCACompressor(nn.Module):
    """
    Heavily Compressed Attention compressor (paper §2.3.2, eqs. 20–23). compresses
    every `compress_rate_hca` (m'=128) source tokens into a single compressed KV
    entry.

    Each closed window of m' tokens produces one compressed entry:
    `C^{Comp}_i = Σ_{j∈window} softmax(Z_j + B)_j ⊙ C_j`. RoPE on the trailing
    `rope_head_dim` slice is applied at the absolute position
    `i * compress_rate_hca + first_window_position` (``first_window_position`` is
    ``start_position`` on the CP path, ``0`` otherwise). Returns the compressed
    windows of the full sequence in one shot (shape `[B, 1, n_windows, head_dim]`).

    Training-only: ``past_key_values`` is ignored (stateless single-shot).
    Tokens past the last closed m'-window are discarded (no cross-call
    buffering).
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self._position_bias_holder = _Fp32ParamHolder(self.compress_rate, self.head_dim)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    @property
    def position_bias(self) -> torch.Tensor:
        return self._position_bias_holder(None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
        *,
        cp_group: "torch.distributed.ProcessGroup | None" = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        s_local = hidden_states.shape[1]
        if cp_group is None:
            cp_rank = 0
            first_window_position = 0
            n_local_windows = None
        else:
            cp_rank = torch.distributed.get_rank(cp_group)
            m = self.compress_rate
            # todo zz: validate chunks and derive folded HCA positions
            assert s_local % m == 0
            first_window_position = cp_rank * s_local
            n_local_windows = s_local // m

        batch, _, _ = hidden_states.shape
        cache_layer = None
        device = hidden_states.device

        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], first_window_position

        if chunk_kv.shape[1] > 0:  # there were at least self.compress_rate tokens
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(
                chunk_gate.dtype
            )
            compressed = self.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
            )

            # position ids compat with THD format
            if packed_seq_params is None:
                positions = torch.arange(n_windows, device=compressed.device)
                positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
            else:
                positions = packed_seq_params.layout.per_m[self.compress_rate].wnd_pos_ids
                assert positions.shape == (n_windows,)
                positions = positions.unsqueeze(0).expand(batch, -1)

            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
            if self.config.fp8_qat:
                # hca compressed torch.Size([1, 1, 512])
                nope = self.config.head_dim - self.config.qk_rope_head_dim
                compressed = torch.cat([fp8_simulate_qat(compressed[..., :nope], 64), compressed[..., nope:]], dim=-1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        compressed_kv = compressed.unsqueeze(1)

        # all-gather across CP so every rank holds the full compressed
        # sequence in absolute-position order.
        if cp_group is not None:
            assert n_local_windows is not None
            # todo zz: restore global HCA window order
            compressed_kv = compressor_cp_ag(
                compressed_kv, cp_group, 0, n_local_windows,
            )

        compressed_len = compressed_kv.shape[2]
        seq_len = position_ids.shape[1]
        assert compressed_len > 0, "compressed_len should be greater than 0"

        block_bias = None
        compressed_topk = None
        if self.config.attn_backend == 'eager':
            # query `t` may only see cache entries at pos `w` t > w * compress_rate (ex: t=7, w=2 t does not attend to it).
            if packed_seq_params is None:
                entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
                causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
                block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
                block_bias = block_bias.masked_fill(
                    entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
                    float("-inf"),
                )
            else:
                # [b=1, h=1, s_local, n_windows]
                wnd_idx = torch.arange(compressed_len, device=device)
                per_m = packed_seq_params.layout.per_m[self.compress_rate]
                future_mask = wnd_idx.view(1, 1, 1, -1) >= per_m.causal_threshold_per_token.view(1, 1, -1, 1)

                # [s_local, n_windows] bool mask, True if the token's seg is different from the window's seg
                cross_seg_mask = packed_seq_params.layout.seg_id_per_token.unsqueeze(-1) != per_m.seg_id_per_wnd.unsqueeze(0)
                future_mask = future_mask | cross_seg_mask.view(1, 1, s_local, compressed_len)

                block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
                block_bias = block_bias.masked_fill(future_mask, float("-inf"))
        else:
            entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
            if packed_seq_params is None:
                causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
                invalid = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
            else:
                per_m = packed_seq_params.layout.per_m[self.compress_rate]
                future_mask = entry_indices.view(1, 1, -1) >= per_m.causal_threshold_per_token.view(1, -1, 1)
                cross_seg_mask = (
                    packed_seq_params.layout.seg_id_per_token.unsqueeze(-1)
                    != per_m.seg_id_per_wnd.unsqueeze(0)
                ).view(1, s_local, compressed_len)
                invalid = future_mask | cross_seg_mask

            compressed_topk = entry_indices.view(1, 1, -1).expand(batch, seq_len, -1).int()
            compressed_topk = compressed_topk.masked_fill(invalid, -1)

        return compressed_kv, block_bias, compressed_topk


class DeepseekV4Indexer(nn.Module):
    r"""Lightning Indexer (paper §2.3.1, eqs. 13–17). Used by Compressed Sparse
    Attention (CSA) to pick the top-`k` compressed KV blocks per query, with
    `k = config.index_topk`. Each query then attends only to those `k` of the
    `seq_len / compress_rate_csa` compressed entries — reduction factor
    `(seq_len / compress_rate_csa) / index_topk` over full attention against
    the entire compressed sequence.

    The indexer runs its own scaled-down compressor at `index_head_dim` over
    the same windows as the outer CSA compressor, then scores queries against
    the compressed keys with `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)` and
    keeps the top `index_topk` indices.

    The indexer has its own rotary because it applies RoPE to two sets of
    tensors:

      * *compressed keys* at deterministic positions
        `i * compress_rate + first_window_position`,
      * *queries* at the model's current `position_ids` (variable per forward).

    Both must use the same theta as the outer compressor
    (`compress_rope_theta`) so query/key inner products are
    translation-invariant — if they used different thetas, `q · k` would carry
    a residual position-dependent skew. We can't precompute cos/sin once at
    init because the query positions vary per call, so the indexer owns its
    own rotary and calls it twice per forward (once for compressed keys, once
    for queries) with `layer_type=self.rope_layer_type` (always `"compress"`).
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.weights_scaling = self.num_heads**-0.5
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self._position_bias_holder = _Fp32ParamHolder(self.compress_rate, 2 * self.head_dim)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    @property
    def position_bias(self) -> torch.Tensor:
        return self._position_bias_holder(None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
        *,
        cp_group: "torch.distributed.ProcessGroup | None" = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> torch.LongTensor:
        s_local = position_ids.shape[1]
        # q_residual.shape=torch.Size([1, s_local, q_lora_rank=1024])
        assert q_residual.shape[1] == s_local
        if cp_group is None:
            cp_rank = 0
            first_window_position = 0
            n_local_windows = None
            n_prefix_windows = 0
            local_hidden_states = hidden_states
        else:
            cp_rank = torch.distributed.get_rank(cp_group)
            m = self.compress_rate
            # todo zz: validate every folded chunk is m-aligned
            assert s_local % m == 0
            l_prefix = 0 if cp_rank == 0 else m

            # todo zz: represent interleaved per-piece prefixes; one leading trim is invalid
            first_window_position = cp_rank * s_local - l_prefix
            n_local_windows = s_local // m
            n_prefix_windows = l_prefix // m
            if n_prefix_windows > 0:
                assert hidden_states.shape[1] == l_prefix + s_local
                local_hidden_states = hidden_states[:, l_prefix:, :]
            else:
                local_hidden_states = hidden_states

        batch, _, _ = hidden_states.shape
        cache_layer = None
        device = hidden_states.device

        # ---- per-rank compression (was compress_local) ----
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], first_window_position

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

            # Same Ca / Cb overlap layout as the outer CSA compressor, at index_head_dim.
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]

            # THD-only: gate out cross-seg Ca slots (segment boundary windows
            # would otherwise pull last-m tokens of the previous seg into the
            # current seg's compressed entry) and pad-token Ca/Cb slots
            # (pad logits would otherwise dilute the per-window softmax).
            if packed_seq_params is not None:
                per_m = packed_seq_params.layout.per_m[self.compress_rate]
                first_of_seg = per_m.first_of_seg_window_mask_with_prefix  # [n_windows]
                new_gate[:, first_of_seg, :ratio] = float("-inf")
                new_kv[:, first_of_seg, :ratio] = 0

            softmax_w = new_gate.softmax(dim=2, dtype=torch.float32)
            compressed = self.kv_norm(
                (new_kv * softmax_w.to(new_kv.dtype)).sum(dim=2)
            )

            # position ids compat with THD format
            if packed_seq_params is None:
                positions = torch.arange(n_windows, device=compressed.device)
                # todo zz: derive Indexer window positions from folded prefix layout
                positions = positions * self.compress_rate + first_window_position
            else:
                positions = packed_seq_params.layout.per_m[self.compress_rate].wnd_pos_ids_with_prefix
                assert positions.shape == (n_windows,)

            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
            if self.config.fp8_qat:
                # indexer compressed torch.Size([1, 32, 128])
                idx_nope = self.head_dim - self.config.qk_rope_head_dim
                compressed = torch.cat([fp8_simulate_qat(compressed[..., :idx_nope], 64), compressed[..., idx_nope:]], dim=-1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        # [B, n_windows, head_dim]
        compressed_kv = compressed

        # ---- CP stage-2: trim duplicate prefix windows + all-gather ----
        if cp_group is not None:
            assert n_local_windows is not None
            # compressor_cp_post operates on a [B, 1, T, D] layout
            # todo zz: trim all folded prefixes and restore global Indexer window order
            compressed_kv = compressor_cp_ag(
                compressed_kv.unsqueeze(1), cp_group, n_prefix_windows, n_local_windows,
            ).squeeze(1)

        # todo zz: keep query-local states separate from interleaved prefix states
        cos_q, sin_q = self.rotary_emb(local_hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, s_local, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)
        if self.config.fp8_qat:
            # indexer q torch.Size([1, 128, 64, 128])
            idx_nope = self.head_dim - self.config.qk_rope_head_dim
            q = torch.cat([fp8_simulate_qat(q[..., :idx_nope], 64), q[..., idx_nope:]], dim=-1)

        # ReLU(q·kᵀ) * weights, then top-k
        if self.config.indexer_backend == 'fused':
            weights = self.weights_proj(local_hidden_states).float() * self.weights_scaling
            q_sbhd = rearrange(q, 'b s h d -> s b h d').contiguous().to(torch.bfloat16)
            k_sbd = rearrange(compressed_kv, 'b t d -> t b d').contiguous().to(torch.bfloat16)
            w_sbh = rearrange(weights, 'b s h -> s b h').contiguous()
            if packed_seq_params is None:
                positions = position_ids[0].to(torch.int32)
                cu_ks, cu_ke = _make_causal_cu_seqlens(
                    s_local, compressed_kv.shape[1], self.compress_rate, device, positions=positions,
                )
            else:
                per_m = packed_seq_params.layout.per_m[self.compress_rate]
                seg_starts = (packed_seq_params.cu_seqlens_q_padded[:-1] // self.compress_rate).to(device)
                cu_ks = seg_starts[packed_seq_params.layout.seg_id_per_token].to(torch.int32)
                cu_ke = per_m.causal_threshold_per_token.to(device=device, dtype=torch.int32)
            index_scores = batched_indexer_fwd(q_sbhd, k_sbd, w_sbh, cu_ks, cu_ke)  # [B, S, T]
        else:
            scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
            scores = F.relu(scores) * self.softmax_scale
            weights = self.weights_proj(local_hidden_states).float() * self.weights_scaling  # [B, S, H]
            index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]

        compressed_len = compressed_kv.shape[1]
        top_k = min(self.index_topk, compressed_len)

        # not all queries can attend to the compressed entries. If a query's position
        # is small than the relative position of the key (say m=4, query 2 cannot attend
        # to compressed key at position 4, because it compressed info for states at position
        # 12 to 16. Thus we need to make sure that top_k does not land in that range.
        # Picks that still point past `causal_threshold` (early queries with too few ready
        # blocks) are replaced with a `-1` sentinel that the compresser treats as invalid.
        if compressed_len > 0:
            if packed_seq_params is None:
                causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
                entry_indices = torch.arange(compressed_len, device=index_scores.device)
                future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)  # [B, S, T]
                index_scores = index_scores.masked_fill(future_mask, float("-inf"))
                top_k_indices = index_scores.topk(top_k, dim=-1).indices  # [B, S, k]
                invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
            else:
                # [b=1, s_local, n_windows]
                wnd_idx = torch.arange(compressed_len, device=device)
                per_m = packed_seq_params.layout.per_m[self.compress_rate]
                future_mask = wnd_idx.view(1, 1, -1) >= per_m.causal_threshold_per_token.view(1, -1, 1)

                # [s_local, n_windows] bool mask, True if the token's seg is different from the window's seg
                cross_seg_mask = packed_seq_params.layout.seg_id_per_token.unsqueeze(-1) != per_m.seg_id_per_wnd.unsqueeze(0)
                future_mask = future_mask | cross_seg_mask.view(1, s_local, compressed_len)
                index_scores = index_scores.masked_fill(future_mask, float("-inf"))
                top_k_indices = index_scores.topk(top_k, dim=-1).indices  # [B, S, k]

                # invalid: top-k landed on a future window OR a window from a different seg.
                future_invalid = top_k_indices >= per_m.causal_threshold_per_token.view(1, -1, 1)
                # gather seg id at each picked window: [B=1, s_local, k]
                seg_id_at_topk = per_m.seg_id_per_wnd[top_k_indices]
                cross_seg_invalid = seg_id_at_topk != packed_seq_params.layout.seg_id_per_token.view(1, -1, 1)
                invalid = future_invalid | cross_seg_invalid

            return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)
        return index_scores.topk(top_k, dim=-1).indices


class DeepseekV4CSACompressor(nn.Module):
    """Compressed Sparse Attention compressor (paper §2.3.1, eqs. 9–17). Compresses
    every `compress_rate_csa` (m=4) source tokens and runs a Lightning Indexer on
    top of the compressed KV that scores queries with
    `∑_h w_{t,h} · ReLU(q_{t,h} · K^{IComp}_s)` to gather the top `index_topk`
    entries per query before they reach core attention.

    `kv_proj` / `gate_proj` / `position_bias` project to `2 * head_dim`: each
    token contributes two independent compressed series Ca and Cb stored in
    one tensor. Ca = `[..., :head_dim]` (its contribution to the *next*
    window's compressed entry), Cb = `[..., head_dim:]` (its contribution to
    the *current* window's compressed entry). Compressed entry `w` is the
    softmax-gated convex combination of window `w-1`'s Ca slice with window
    `w`'s Cb slice over `2 * compress_rate_csa` slots — width
    `2 * compress_rate_csa`, stride `compress_rate_csa`. Window 0 has no
    previous-window Ca slice, so its first half stays zero-kv / `-inf`-gate
    (softmax weight 0).

    Training-only: ``past_key_values`` is ignored (stateless single-shot,
    no cross-call overlap state).
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        # 对应原版推理的代码的 ape
        self._position_bias_holder = _Fp32ParamHolder(self.compress_rate, 2 * self.head_dim)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.indexer = DeepseekV4Indexer(config)

    @property
    def position_bias(self) -> torch.Tensor:
        return self._position_bias_holder(None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
        *,
        cp_group: "torch.distributed.ProcessGroup | None" = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        s_local = hidden_states.shape[1]
        if cp_group is None:
            first_window_position = 0
            n_local_windows = None
            n_prefix_windows = 0
        else:
            cp_rank = torch.distributed.get_rank(cp_group)
            m = self.compress_rate
            # todo zz: validate every folded chunk is m-aligned
            assert s_local % m == 0

            # Stage 1: send last m hidden_states to next rank, recv from prev.
            # NOTE: l_prefix is always 0 for rank 0, and m for rank > 0.
            # todo zz: build per-segment folded compressor prefix views
            hs_with_prefix, l_prefix = compressor_cp_ring(hidden_states, m, cp_group)
            hidden_states = hs_with_prefix

            # Absolute start position of `hs_with_prefix[:, 0, :]` in the
            # global sequence: rank 0 → 0; rank > 0 → cp_rank * s_local - l_prefix.
            first_window_position = cp_rank * s_local - l_prefix
            n_local_windows = s_local // m
            n_prefix_windows = l_prefix // m

        batch, _, _ = hidden_states.shape
        assert s_local == position_ids.shape[1]
        cache_layer = None
        device = hidden_states.device

        # ---- per-rank outer compression (was compress_local) ----
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], first_window_position

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

            # Lay out the two series in [B, n_win, 2*ratio, head_dim]: Cb
            # (`[..., head_dim:]`) goes in the second half (current window),
            # Ca of the previous window (`[..., :head_dim]`) goes in the
            # first half. Window 0 has no previous window, so its first half
            # stays zero-kv / -inf-gate (softmax weight 0).
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]

            # THD-only: gate out cross-seg Ca slots (segment boundary windows
            # would otherwise pull last-m tokens of the previous seg into the
            # current seg's compressed entry) and pad-token Ca/Cb slots
            # (pad logits would otherwise dilute the per-window softmax).
            if packed_seq_params is not None:
                per_m = packed_seq_params.layout.per_m[self.compress_rate]
                first_of_seg = per_m.first_of_seg_window_mask_with_prefix  # [n_windows]
                new_gate[:, first_of_seg, :ratio] = float("-inf")
                new_kv[:, first_of_seg, :ratio] = 0

            # Softmax in fp32 for stability (logits in bf16/fp16 can collapse pairs that
            # only differ by a small amount, especially with large window widths).
            softmax_w = new_gate.softmax(dim=2, dtype=torch.float32)
            compressed = self.kv_norm(
                (new_kv * softmax_w.to(new_kv.dtype)).sum(dim=2)
            )

            # position ids compat with THD format
            if packed_seq_params is None:
                positions = torch.arange(n_windows, device=compressed.device)
                # todo zz: derive CSA window positions from folded prefix layout
                positions = positions * self.compress_rate + first_window_position
            else:
                positions = packed_seq_params.layout.per_m[self.compress_rate].wnd_pos_ids_with_prefix
                assert positions.shape == (n_windows,)

            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
            if self.config.fp8_qat:
                # csa compressed torch.Size([1, 32, 512])
                nope = self.config.head_dim - self.config.qk_rope_head_dim
                compressed = torch.cat([fp8_simulate_qat(compressed[..., :nope], 64), compressed[..., nope:]], dim=-1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        # [B, 1, T_local, head_dim]
        compressed_kv = compressed.unsqueeze(1)

        # ---- CP stage-2: trim duplicate prefix windows + all-gather ----
        if cp_group is not None:
            assert n_local_windows is not None
            # todo zz: trim all folded prefixes and restore global CSA window order
            compressed_kv = compressor_cp_ag(
                compressed_kv, cp_group, n_prefix_windows, n_local_windows,
            )

        # ---- indexer top-k (CP and non-CP go through the same call) ----
        # Lightning Indexer: gather top-`index_topk` compressed entries per query.
        # in some cases, the output index can return top-k positions that should
        # not be attended to. Ex: for query at index 5, m=4, and `index_topk=1024`,
        # 1024 indices are returned but only 2 should be attended to. The indexer
        # marks those with `-1`; the per-query CSA block bias scatters `0` only
        # at the valid top-k entries and leaves `-inf` everywhere else (the
        # scatter sentinel `compressed_len` falls into a one-wider tail column
        # that we drop afterwards).
        # todo zz: pass unprefixed query states separately to Indexer
        top_k_indices = self.indexer(
            hidden_states, q_residual, position_ids, past_key_values, layer_idx,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
        )
        compressed_len = compressed_kv.shape[2]

        if self.config.attn_backend == 'eager':
            valid = top_k_indices >= 0  # [B, S, k]
            # Per-query block bias: query `t` may only see the cache entries that are <= `seq_len // m`
            # and in these, only the ones marked valid by the indexer. Everything else is `-inf`.
            # While the above negated the indexer, here we apply the "causal" masking.
            safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
            block_bias = compressed_kv.new_full((batch, 1, s_local, compressed_len + 1), float("-inf"))
            block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
            block_bias = block_bias[..., :compressed_len]
        else:
            block_bias = None
        return compressed_kv, block_bias, top_k_indices.int()


COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


class DeepseekV4Attention(nn.Module):
    r"""
    Diff with classic attentions:
    * Shared-KV Multi-Query Attention: `num_key_value_heads = 1`; `kv_proj` projects
      directly to that single KV head and the same tensor is read as both key and
      value.
    * Partial RoPE on the first `rope_head_dim` of each head ("Partial Rotary
      Positional Embedding"). RoPE is also applied with position `-i` to the
      attention output's rope slice, so the contribution of each KV entry stays a
      function of the *relative* distance to the query.
    * Per-head learnable attention sink like gpt OSS.
    * Grouped low-rank output projection for perfs.
    * 3 different attention variants, sliding, sliding+CSA, sliding+HCA.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        # Sliding-only layers use the "main" (plain θ=10000) rope; CSA/HCA layers
        # share the same yarn-scaled "compress" rope as their compressor.
        self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self._sink_holder = _Fp32ParamHolder(self.num_heads)
        self.compressor = (
            COMPRESSOR_CLASSES[self.layer_type](config) if self.layer_type != "sliding_attention" else None
        )
        # CP (context parallel) state. Default = non-CP; set by
        # gpatch_v4.models.deepseek_v4.hp.apply_hp(model, ..., cp_mesh=...).
        # When `cp_group` is None or `cp_size == 1`, attention runs identically
        # to upstream — no comm, no extra ops. See ``gpatch_v4/models/deepseek_v4/cp.py``.
        self.cp_group: Optional[dist.ProcessGroup] = None
        self.cp_size: int = 1
        self.cp_rank: int = 0

    @property
    def sinks(self) -> torch.Tensor:
        return self._sink_holder(None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        *,
        swa_topk: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cp_active = self.cp_group is not None and self.cp_size > 1
        if cp_active:
            assert past_key_values is None, (
                "CP path requires past_key_values=None (training only)"
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # position_embeddings is a {"main", "compress"} dict from the model; pick the
        # one that matches this layer's rope type (sliding → main, CSA/HCA → compress).
        cos, sin = position_embeddings[self.rope_layer_type]

        if self.config.fp8_qat:
            q_residual = self.q_a_norm(
                fp8_qat_linear(self.q_a_proj, hidden_states, 128)
            )
            q = fp8_qat_linear(self.q_b_proj, q_residual, 128).view(*hidden_shape).transpose(1, 2)
        else:
            q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        if self.config.fp8_qat:
            kv = self.kv_norm(
                fp8_qat_linear(self.kv_proj, hidden_states, 128)
            ).view(*hidden_shape).transpose(1, 2)
        else:
            kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)
        if self.config.fp8_qat:
            # attn kv torch.Size([1, 1, 128, 512])
            nope = self.config.head_dim - self.config.qk_rope_head_dim
            kv = torch.cat([fp8_simulate_qat(kv[..., :nope], 64), kv[..., nope:]], dim=-1)

        if past_key_values is not None:  # sliding where K==V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        block_bias = None
        compressed_topk = None
        assert position_ids is not None
        if not cp_active:
            # ----- non-CP path (upstream-equivalent) -----
            swa_kv_len = kv.shape[2]
            if self.compressor is not None:  # Compressed KV (CSA or HCA)
                compressed_kv, block_bias, compressed_topk = self.compressor(
                    hidden_states, q_residual, position_ids, past_key_values, self.layer_idx,
                    packed_seq_params=packed_seq_params,
                )
                kv = torch.cat([kv, compressed_kv], dim=2)
        else:
            # ----- CP path (paper §3.4.3) -----
            cp_group = self.cp_group

            # SWA ring: prepend prev-rank's last sliding_window-1 KVs.
            # todo zz: build per-segment folded SWA KV prefix views
            kv = swa_ring_kv(kv, cp_group, self.sliding_window)
            swa_kv_len = kv.shape[2]

            # Compressor stage-1 + stage-2 (CSA/HCA layers only).
            if self.compressor is not None:
                compressed_kv_full, block_bias, compressed_topk = self.compressor(
                    hidden_states,
                    q_residual,
                    position_ids,
                    past_key_values=None,
                    layer_idx=self.layer_idx,
                    cp_group=cp_group,
                    packed_seq_params=packed_seq_params,
                )
                kv = torch.cat([kv, compressed_kv_full], dim=2)

        if self.config.attn_backend == 'eager':
            # The compressor path concatenates extra entries onto the KV axis after the
            # standard sliding-window cache update, so a tensor `attention_mask` (built
            # for the pre-concat KV length) needs to be extended to cover them. The
            # compressor returns a `block_bias` carrying per-query causality + indexer
            # validity over those new slots — cat it in instead of zero-padding (which
            # would let every query see every compressed slot).
            if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
                if block_bias is not None:
                    attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
                else:
                    attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)
        else:
            # fused path: swa_topk holds absolute indices into the sliding-window
            # KV axis (length `swa_kv_len`); compressor entries were concatenated
            # right after it, so shift the compressor top-k slots by `swa_kv_len`
            # (NOT by swa_topk.shape[-1], which is the window size SW != KV length
            # whenever s_local != SW or a CP rank > 0 carries a ring prefix).
            assert attention_mask is None
            if compressed_topk is not None:
                assert swa_topk.shape[:2] == compressed_topk.shape[:2]
                topk_len = swa_topk.shape[-1]
                shifted_topk = torch.where(compressed_topk >= 0, compressed_topk + swa_kv_len, compressed_topk)
                swa_topk = torch.cat([swa_topk, shifted_topk], dim=-1)
                assert swa_topk.shape[-1] == topk_len + compressed_topk.shape[-1] and swa_topk.dtype == torch.int32

        # Backend dispatch — set by `apply_hp` from policy_config.attn_implementation.
        if self.config.attn_backend == "eager":
            # deepseek v4 的 sink 是 fp32，而 HF 的 eager 实现没有处理。我觉得应该算错误，不过 `torch.matmul`
            # 不支持 `out_dtype` 所以很难 simulate。
            # ```python
            # acc_s = T.alloc_fragment((h, block), FP32)
            # sum_exp = T.alloc_fragment(h, FP32)
            # for t in T.Pipelined(num_blocks, num_stages=num_stages):
            #     ...
            #     T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            #     ...
            # for i in T.Parallel(h):
            #     sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
            # ```
            attn_output, attn_weights = eager_attention_forward(
                self,
                q,
                kv,
                kv,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                s_aux=self.sinks,
                **kwargs,
            )
        elif self.config.attn_backend == "fused":
            _fsinks = self.sinks if self.config.amp_fp32 else self.sinks.float()
            attn_output = sparse_attn_tilelang(
                rearrange(q, 'B H S D -> B S H D').contiguous(),
                rearrange(kv, 'B 1 Skv D -> B Skv D').contiguous(),
                _fsinks,
                swa_topk,
                sm_scale=self.scaling,
            )
            attn_weights = None
        elif self.config.attn_backend == "eager-topk":
            from .kernel.eager_topk_attn import eager_topk_attention_forward
            attn_output = eager_topk_attention_forward(
                q, kv, self.sinks, swa_topk, sm_scale=self.scaling,
            )
            attn_weights = None
        else:
            raise ValueError(f"unknown attn_backend: {self.config.attn_backend!r}")

        # K=V in V4, so V picked up rope on its trailing rope slice. Apply the conjugate
        # rotation (`-sin`) at the query position to undo it on the rope slice of the
        # output before the grouped output projection mixes heads. The transpose pair is
        # just a layout fix-up: apply_rotary_pos_emb expects `[B, S, H, D]` (its
        # `unsqueeze_dim=1` adds a head-broadcast dim to cos/sin); attention gave us
        # `[B, H, S, D]`.
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        if self.config.fp8_qat:
            output = fp8_qat_linear(self.o_b_proj, grouped, 128)
        else:
            output = self.o_b_proj(grouped)
        return output, attn_weights


class DeepseekV4Experts(nn.Module):
    """Collection of expert weights stored as 3D tensors with EP support.

    Differences from upstream :class:`transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts`:

    * The ``@use_experts_implementation`` decorator is removed; this class is
      now the single, EP-aware implementation. Upstream's loop-over-experts /
      grouped-MM / DeepGEMM swap-in path is replaced by an explicit
      all-to-all + grouped-MM EP path (:meth:`fwd_gmm`).
    * ``forward`` is fail-fast: it requires ``ep_group is not None`` (set by
      :func:`gpatch_v4.models.deepseek_v4.hp.apply_hp`). Calling
      ``forward`` on a fresh, non-parallelized model is a programming bug.

    Weight layout (matches upstream): ``gate_up_proj`` has shape
    ``(num_experts, 2 * intermediate_dim, hidden_dim)``; ``down_proj`` has
    shape ``(num_experts, hidden_dim, intermediate_dim)``; no bias.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit

        # EP state. Default = non-EP (placeholders); apply_hp sets these
        # to the real ep_size / ep_rank / ep_group and shrinks num_local_experts.
        self.ep_size = 1
        self.ep_rank = 0
        self.ep_group: Optional[dist.ProcessGroup] = None
        self.num_local_experts = self.num_experts
        self.ep_backend = "eager"

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Expert Parallelism forward: all-to-all dispatch + local grouped-MM + combine.

        Inputs are flat (per-token) tensors after :class:`DeepseekV4SparseMoeBlock`
        reshapes ``[B, S, D]`` to ``[N, D]`` where ``N = B*S``.

        Parameters
        ----------
        hidden_states : Tensor, shape ``[N, H]``
            Flat token activations (post-norm).
        top_k_index : Tensor, shape ``[N, K]``, int64
            Expert id (in ``[0, num_experts)``) for each token's top-k routing slot.
        top_k_weights : Tensor, shape ``[N, K]``
            Routing weight for each top-k slot (already scaled by
            ``routed_scaling_factor`` on the router side).

        Returns
        -------
        Tensor, shape ``[N, H]``
            Sum of weighted expert outputs over the top-k slots, ready to be
            added to the shared-experts output by :class:`DeepseekV4SparseMoeBlock`.
        """
        assert self.ep_group is not None, (
            "DeepseekV4Experts.forward requires ep_group; "
            "did you forget to call apply_hp(model, ep_2d_mesh)?"
        )
        if self.ep_backend == "eager":
            return self._forward_eager(hidden_states, top_k_index, top_k_weights)
        if self.ep_backend == "deepep":
            return self._forward_deepep(hidden_states, top_k_index, top_k_weights)
        raise ValueError(f"unknown ep_backend: {self.ep_backend}")

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        N, H = hidden_states.shape
        num_top_k = top_k_index.size(-1)

        # 1) Flatten top-k: each (token, slot) becomes one independent dispatch unit.
        flat_expert = top_k_index.reshape(-1)            # [N*K]
        flat_weight = top_k_weights.reshape(-1)          # [N*K]
        flat_x = hidden_states.repeat_interleave(num_top_k, dim=0)  # [N*K, H]

        # 2) Permute by target rank so tokens for the same rank are contiguous.
        target_rank = flat_expert // self.num_local_experts  # [N*K]
        order = torch.argsort(target_rank)
        flat_x_sorted = flat_x[order]
        flat_expert_sorted = flat_expert[order]
        flat_weight_sorted = flat_weight[order]

        # 3) Exchange per-rank send counts via a small all_to_all_single.
        send_counts = torch.bincount(target_rank[order], minlength=self.ep_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        send_list = send_counts.tolist()
        recv_list = recv_counts.tolist()

        # 4) Dispatch: ship tokens to the rank that owns their expert.
        recv_x = all_to_all_uneven(flat_x_sorted, send_list, recv_list, self.ep_group)
        recv_expert = all_to_all_uneven(flat_expert_sorted, send_list, recv_list, self.ep_group)
        recv_weight = all_to_all_uneven(flat_weight_sorted, send_list, recv_list, self.ep_group)

        # 5) Local expert compute (grouped MM). Convert global expert id → local id.
        local_expert_id = recv_expert - self.ep_rank * self.num_local_experts
        local_out = self.fwd_gmm(recv_x, local_expert_id, recv_weight)

        # 6) Combine: ship results back to originating rank.
        back = all_to_all_uneven(local_out, recv_list, send_list, self.ep_group)

        # 7) Un-permute and sum across top-k slots.
        inv_order = torch.argsort(order)
        back = back[inv_order]                          # [N*K, H]
        return back.view(N, num_top_k, H).sum(dim=1)    # [N, H]

    def _forward_deepep(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        N, H = hidden_states.shape
        recv_x, recv_expert, recv_weight, _, handle = fused_dispatch(
            hidden_states,
            top_k_index,
            top_k_weights,
            self.num_experts,
            self.ep_group,
        )
        valid = recv_expert >= 0
        row_idx, slot_idx = valid.nonzero(as_tuple=True)
        recv_out = recv_x.new_zeros(recv_x.shape[0], H)
        if row_idx.numel() > 0:
            local_expert_id = recv_expert[row_idx, slot_idx]
            local_weight = recv_weight[row_idx, slot_idx]
            expanded_x = recv_x.index_select(0, row_idx)
            expanded_out = self.fwd_gmm(expanded_x, local_expert_id, local_weight)
            recv_out.index_add_(0, row_idx, expanded_out)
        return fused_combine(recv_out, self.ep_group, handle).view(N, H)

    def fwd_gmm(
        self,
        hidden_states: torch.Tensor,
        local_expert_id: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Local grouped-MM expert compute.

        Adapted from :func:`transformers.integrations.moe.grouped_mm_experts_forward`
        for the post-dispatch EP context where each input row is already
        assigned 1:1 to a single local expert (no top-k expansion / sentinel
        masking needed) and the weight layout is the upstream DSV4 default
        (``is_transposed=False``, no bias).

        Parameters
        ----------
        hidden_states : Tensor, shape ``[S, H]``
            Post-dispatch token activations received by this rank.
        local_expert_id : Tensor, shape ``[S]``, int64
            Local expert id (in ``[0, num_local_experts)``) for each row.
        top_k_weights : Tensor, shape ``[S]``
            Routing weight to apply to each row's output.

        Returns
        -------
        Tensor, shape ``[S, H]``
            Per-row weighted expert output, in the original row order.
        """
        device = hidden_states.device
        num_tokens, hidden_dim = hidden_states.shape

        # Sort by local expert id for grouped processing.
        expert_ids_g, perm = torch.sort(local_expert_id)
        x_g = hidden_states[perm]
        weights_g = top_k_weights[perm]

        histc_input = expert_ids_g.int()
        tokens_per_expert = torch.histc(
            histc_input,
            bins=self.num_local_experts,
            min=0,
            max=self.num_local_experts - 1,
        )
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

        gate_up_w = self.gate_up_proj
        down_w = self.down_proj
        if self.config.fp4_qat:
            gate_up_w = fp4_simulate_qat(gate_up_w)
            down_w = fp4_simulate_qat(down_w)
        elif self.config.fp8_qat:
            # vllm 和 sglang 的 moe 的 fp8 block (128, 128)
            gate_up_w = fp8_simulate_qat(gate_up_w, 128)
            down_w = fp8_simulate_qat(down_w, 128)

        # Up projection (gate||up packed): [S, 2I]
        if self.config.fp8:
            m_splits = tokens_per_expert.tolist()
            proj_out = self.fp8_grouped_linear(
                x_g.to(gate_up_w.dtype),
                gate_up_w,
                m_splits,
            )
        else:
            proj_out = _grouped_linear(
                x_g.to(gate_up_w.dtype),
                gate_up_w,
                offsets,
                bias=None,
                is_transposed=False,
            )
        # Apply swiglu_limit clamp + SiLU on gate, clamp on up, then gate*up.
        proj_out = self._apply_gate(proj_out)            # [S, I]

        # Down projection: [S, H]
        if self.config.fp8:
            proj_out = self.fp8_grouped_linear(
                proj_out.to(down_w.dtype),
                down_w,
                m_splits,
            )
        else:
            proj_out = _grouped_linear(
                proj_out.to(down_w.dtype),
                down_w,
                offsets,
                bias=None,
                is_transposed=False,
            )

        weighted = proj_out * weights_g.unsqueeze(-1)    # [S, H]

        # Restore original row order so the all-to-all combine sees the same
        # layout the dispatch path emitted.
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device)
        return weighted[inv_perm].view(num_tokens, hidden_dim).to(hidden_states.dtype)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        """SwiGLU with V4-Flash's symmetric clamp on up and one-sided clamp on gate.

        Kept verbatim from upstream :class:`DeepseekV4Experts._apply_gate`. The
        clamp values come from ``config.swiglu_limit`` and are essential to
        match the DSV4-Flash reference numerics; do not drop or simplify.
        """
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up


class DeepseekV4TopKRouter(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)
        # Set by enable_router_replay / router_replay_ctx; None means normal top-k.
        self.router_replay: "RouterReplay | None" = None  # noqa: F821

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        rr = self.router_replay
        if rr is None:
            indices = torch.topk(
                scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False
            ).indices
        else:
            # Replay: pin indices to the recorded baseline decisions. We discard
            # ``get_replay_topk``'s values (it gathers on ``scores+bias`` to
            # mirror the topk input, but the downstream ``weights`` gather uses
            # ``scores`` alone — matching the no-replay path exactly).
            _, indices = rr.get_replay_topk(scores + self.e_score_correction_bias)
            expected_shape = (scores.shape[0], self.top_k)
            assert indices.shape == expected_shape, (
                f"replay indices shape {tuple(indices.shape)} != expected {expected_shape}"
            )
            assert indices.dtype == torch.int64, (
                f"replay indices dtype {indices.dtype} != torch.int64"
            )
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        # Shared-experts weight QAT: simulate FP8 quantization noise on
        # shared expert weights (matching sglang FP8 weight quantization).
        # NOTE: deepseek v4 shared experts weights dtype is fp8, not `float4_e2m1fn`.
        se = self.shared_experts
        if self.experts.config.fp8_qat:
            gate_w = fp8_simulate_qat(se.gate_proj.weight, 128)
            up_w = fp8_simulate_qat(se.up_proj.weight, 128)
            down_w = fp8_simulate_qat(se.down_proj.weight, 128)
            intermediate = se.act_fn(F.linear(residual, gate_w, se.gate_proj.bias)) * F.linear(residual, up_w, se.up_proj.bias)
            se_out = F.linear(intermediate, down_w, se.down_proj.bias)
        else:
            se_out = se(residual)
        return routed + se_out


class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    r"""DeepSeek-V4 decoder block (paper §2). Differs from a classic residual block in
    two places:

    The residual is a stack of `hc_mult` parallel streams kept in shape
    `[B, S, hc_mult, D]` throughout the block, mixed in and out via two
    :class:`DeepseekV4HyperConnection` modules (Manifold-Constrained Hyper-
    Connections / mHC, paper §2.2; Xie et al., 2026). The mHC mappings constrain
    the residual transform to the manifold of doubly-stochastic matrices via the
    Sinkhorn-Knopp projection — making signal propagation non-expansive across
    deep stacks.

    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        *,
        packed_seq_params: PackedSeqParams | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if packed_seq_params is not None:
            assert hidden_states.shape[0] == 1

        # hidden_states throughout: [B, S, hc_mult, hidden].
        # `post` / `comb` come out of the HC modules in fp32 (Sinkhorn projection runs
        # in float); the .to(dtype) puts everything back to the input dtype before mixing
        # so both sites stay consistent with `hidden_states`'s entry dtype.
        # comb is consumed transposed: indexed as sum_j comb[j, k] * residual[j, d]
        # (sum over the FIRST hc axis), equivalent to comb.T @ residual. Sinkhorn
        # produces a doubly-stochastic but non-symmetric matrix, so the direction matters.
        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(
            self.input_layernorm(collapsed),
            packed_seq_params=packed_seq_params,
            **kwargs,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )


@auto_docstring
class DeepseekV4PreTrainedModel(PreTrainedModel):
    config: DeepseekV4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV4TopKRouter, index=0),
        "hidden_states": DeepseekV4DecoderLayer,
        "attentions": DeepseekV4Attention,
    }
    config_class = DeepseekV4Config
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "e_score_correction_bias",
        "q_a_norm",
        "kv_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ]
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    _is_stateful = True

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        if getattr(config, "fp8_qat", False):
            assert fp8_simulate_qat is not None, (
                "fp8_qat enabled but fp8_simulate_qat unavailable "
                "(tilelang/tile_kernels not installed?)"
            )
        if getattr(config, "fp4_qat", False):
            assert fp4_simulate_qat is not None, (
                "fp4_qat enabled but fp4_simulate_qat unavailable "
                "(tilelang/tile_kernels not installed?)"
            )

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
            init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, DeepseekV4TopKRouter):
                init.zeros_(module.e_score_correction_bias)  # buffer
            if isinstance(module, DeepseekV4HashRouter):
                init.zeros_(module.tid2eid)  # buffer; real values come from the checkpoint
        elif isinstance(module, DeepseekV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            init.zeros_(module._sink_holder.weight)
        elif isinstance(module, DeepseekV4HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)):
            init.zeros_(module._position_bias_holder.weight)
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


def _build_swa_topk(
    s_local: int,
    sliding_window: int,
    cp_rank: int,
    batch_size: int,
    packed_seq_params: PackedSeqParams | None,
    device: torch.device,
) -> torch.Tensor:
    """Build ``[B, s_local, W]`` int32 sliding-window KV indices for fused attention.

    Handles both BSHD (``packed_seq_params=None``) and THD
    (cross-seg positions masked to ``-1``).
    """
    # todo zz: index folded q rows against interleaved SWA KV slots
    assert sliding_window <= s_local
    ta = torch.arange(s_local, device=device).view(1, -1, 1)
    tb = torch.arange(sliding_window, device=device).view(1, 1, -1)
    if cp_rank == 0:
        tb = tb - (sliding_window - 1)
    swa_topk = (ta + tb).clamp(min=-1)

    if packed_seq_params is not None:
        seg_id_q = packed_seq_params.layout.seg_id_per_token
        seg_id_kv = packed_seq_params.layout.seg_id_per_token_with_prefix
        safe_idx = swa_topk.clamp(min=0).long()
        cross_seg = seg_id_q.view(1, -1, 1) != seg_id_kv[safe_idx.view(-1)].view(
            1, s_local, sliding_window
        )
        swa_topk = swa_topk.masked_fill(cross_seg, -1)

    swa_topk = swa_topk.expand(batch_size, -1, -1).int()
    return swa_topk


def _build_attn_mask_or_swa_topk(
    config,
    inputs_embeds: torch.Tensor,
    cp_active: bool,
    cp_rank: int,
    attention_mask=None,
    past_key_values=None,
    position_ids=None,
    packed_seq_params: PackedSeqParams | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build either a dense causal mask (eager) or SWA topk indices (fused).

    Shared by the backbone ``DeepseekV4Model.forward`` and the MTP path
    in ``DeepseekV4ForCausalLM.forward``.
    """
    causal_mask = None
    swa_topk = None
    if config.attn_backend == 'eager':
        if cp_active:
            s_local = inputs_embeds.shape[1]
            assert config.sliding_window <= s_local
            # todo zz: build eager mask from folded q and KV position maps
            swa_prefix_len = 0 if cp_rank == 0 else config.sliding_window - 1
            causal_mask = build_cp_causal_mask(
                s_local, cp_rank, swa_prefix_len, config.sliding_window,
                packed_seq_params=packed_seq_params,
                dtype=inputs_embeds.dtype, device=inputs_embeds.device,
            )
        elif isinstance(attention_mask, dict):
            assert NotImplementedError('not supported')
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if packed_seq_params is not None:
                seg_id = packed_seq_params.layout.seg_id_per_token
                cross_seg = seg_id.unsqueeze(0) != seg_id.unsqueeze(1)
                causal_mask = causal_mask.masked_fill(
                    cross_seg.view(1, 1, *cross_seg.shape), float("-inf"),
                )
    else:
        swa_topk = _build_swa_topk(
            s_local=inputs_embeds.shape[1],
            sliding_window=config.sliding_window,
            cp_rank=cp_rank,
            batch_size=inputs_embeds.shape[0],
            packed_seq_params=packed_seq_params,
            device=inputs_embeds.device,
        )
    return causal_mask, swa_topk


@auto_docstring
class DeepseekV4Model(DeepseekV4PreTrainedModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.hc_head = DeepseekV4HyperHead(config)

        # CP (context parallel) state. Default = non-CP; set by
        # gpatch_v4.models.deepseek_v4.hp.apply_hp(model, ..., cp_mesh=...).
        # When `cp_group` is None or `cp_size == 1`, the model runs identically to
        # upstream — no comm, no extra ops. When set, ``forward`` slices
        # ``input_ids`` along seq before embedding (each rank embeds its local
        # slice), constructs local position_ids and a local-rows × full-cols
        # SWA mask, and per-attention-layer the CP comm is internal to
        # :class:`DeepseekV4Attention._forward_cp`.
        self.cp_group: Optional[dist.ProcessGroup] = None
        self.cp_size: int = 1
        self.cp_rank: int = 0

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        *,
        packed_seq_params: PackedSeqParams | None = None,
        return_hc_hidden: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        packed_seq_params (`PackedSeqParams`, *optional*):
            THD-format packing metadata for variable-length sequences in a single batch row.
        return_hc_hidden (`bool`, *optional*, defaults to `False`):
            Whether to return the pre-head hyper-connection hidden states for MTP.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        cp_active = self.cp_group is not None and self.cp_size > 1
        if cp_active:
            assert input_ids is not None
            assert attention_mask is None
            assert position_ids is not None
            assert past_key_values is None
            use_cache = False

        return_cache = past_key_values if use_cache else None
        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config) if not cp_active else None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length()
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
            # `generate()` may pass a per-layer-type mask dict already built by
            # `create_masks_for_generate`; all V4 layer types use the same sliding-window
            # mask, so use the prebuilt one directly. Otherwise build it here.

        # todo zz: pass cp_size and folded metadata to mask/top-k builder
        causal_mask, swa_topk = _build_attn_mask_or_swa_topk(
            config=self.config,
            inputs_embeds=inputs_embeds,
            cp_active=cp_active,
            cp_rank=self.cp_rank,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                swa_topk=swa_topk,
                input_ids=input_ids,
                past_key_values=past_key_values,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )

        mtp_hc_hidden = hidden_states if return_hc_hidden else None
        hidden_states = self.norm(self.hc_head(hidden_states))
        outputs = MoeModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=return_cache
        )
        if mtp_hc_hidden is not None:
            outputs.mtp_hc_hidden = mtp_hc_hidden
        return outputs


@auto_docstring
class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin, HpModule):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        # Default for paths bypassing apply_hp; apply_hp overrides.
        super().__init__(config)
        # prevent circular import
        from .mtp import DeepseekV4MTPConfig, DeepseekV4MTPModule

        self.model = DeepseekV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.mtp_loss_scaling_factor = float(getattr(config, "mtp_loss_scaling_factor", 0.1))
        self.mtp_config = DeepseekV4MTPConfig(
            num_layers=config.num_nextn_predict_layers,
            loss_scaling_factor=self.mtp_loss_scaling_factor,
        )
        self.mtp = (
            DeepseekV4MTPModule(config, self.mtp_config, rotary_emb=self.model.rotary_emb)
            if self.mtp_config.enabled else None
        )
        if self.mtp is not None:
            # MTP-enabled training/load should not silently ignore mtp.* keys.
            self._keys_to_ignore_on_load_unexpected = []

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        *,
        packed_seq_params: PackedSeqParams | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        packed_seq_params (`PackedSeqParams`, *optional*):
            THD-format packing metadata for variable-length sequences in a single batch row.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV4ForCausalLM

        >>> model = DeepseekV4ForCausalLM.from_pretrained("mistralai/DeepseekV4-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/DeepseekV4-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # NOTE There is a naming issue here in transformers: this is not an attention mask [b, s, s],
        # but it's actually a padding mask [b, s].
        use_mtp = self.mtp is not None and self.training
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_hc_hidden=use_mtp,
            output_router_logits=output_router_logits,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        # not used, see `moe_balance_loss_coef` in `training_config.py` for more details.
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        mtp_per_depth_h = None
        if use_mtp:
            assert input_ids is not None, "MTP training path requires input_ids."
            mtp_hc_hidden = outputs.mtp_hc_hidden
            mtp_inputs_embeds = self.model.embed_tokens(input_ids)
            position_ids = position_ids
            if position_ids is None:
                position_ids = torch.arange(
                    mtp_inputs_embeds.shape[1], device=mtp_inputs_embeds.device
                ).unsqueeze(0)

            mtp_cp_active = self.model.cp_group is not None and self.model.cp_size > 1
            mtp_causal_mask, mtp_swa_topk = _build_attn_mask_or_swa_topk(
                config=self.config,
                inputs_embeds=mtp_inputs_embeds,
                cp_active=mtp_cp_active,
                cp_rank=self.model.cp_rank,
                attention_mask=attention_mask,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
            )

            mtp_per_depth_h = self.mtp(
                hidden_states=mtp_hc_hidden,
                input_ids=input_ids,
                embed_fn=self.model.embed_tokens,
                cp_group=self.model.cp_group,
                position_ids=position_ids,
                attention_mask=mtp_causal_mask,
                swa_topk=mtp_swa_topk,
                packed_seq_params=packed_seq_params,
            )

        result = MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
        result.mtp_per_depth_h = mtp_per_depth_h
        result.mtp_loss_scaling_factor = self.mtp_loss_scaling_factor
        return result


__all__ = ["DeepseekV4PreTrainedModel", "DeepseekV4Model", "DeepseekV4ForCausalLM"]
