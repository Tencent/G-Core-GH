# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Callable

import torch
from transformers import PretrainedConfig

from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.core.device import get_device_backend_name, get_device_module, is_cuda

_DEVICE_FLOPS = {
    "CPU": 448e9,
    "GB200": 2.5e15,
    "B200": 2.25e15,
    "MI300X": 1336e12,
    "H100": 989e12,
    "H800": 989e12,
    "H200": 989e12,
    "A100": 312e12,
    "A800": 312e12,
    "L40S": 362.05e12,
    "L40": 181.05e12,
    "A40": 149.7e12,
    "L20": 119.5e12,
    "H20": 148e12,
    "RTX 3070 Ti": 21.75e12,
}

try:
    from gpatch_v4.utils.flops_counter_priv import DEVICE_FLOPS_PRIV
    _DEVICE_FLOPS.update(DEVICE_FLOPS_PRIV)
except ImportError:
    pass


def get_torch_device():
    return get_device_module()


def get_device_flops(unit="T", device_name=None):
    """Get the theoretical FLOPS (Floating Point Operations Per Second) capacity of the current device.

    Args:
        unit (str): The unit to return the FLOPS in. Supported values are:
            "B" - Billion (1e9)
            "K" - Thousand (1e3)
            "M" - Million (1e6)
            "G" - Giga (1e9)
            "T" - Tera (1e12, default)
            "P" - Peta (1e15)

    Returns:
        float: The theoretical FLOPS capacity of the current device in the specified unit.
        Returns float('inf') for unknown GPU types.
    """
    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    # pass device_name is for testing purpose only
    if device_name is None:
        device = get_torch_device()
        if device == torch.cpu:
            device_name = "CPU"
        else:
            device_name = get_torch_device().get_device_name()

    flops = float("inf")  # INF flops for unkown gpu type

    for key, value in sorted(_DEVICE_FLOPS.items(), reverse=True):
        if key in device_name:
            flops = value
            break
    flops_unit = unit_convert(flops, unit)
    return flops_unit


def _estimate_qwen2_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_qwen3_vl_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    # qwen3_vl uses text_config and vision_config to distinguish configs of different parts.
    hidden_size = config.text_config.hidden_size
    vocab_size = config.text_config.vocab_size
    num_hidden_layers = config.text_config.num_hidden_layers
    num_key_value_heads = config.text_config.num_key_value_heads
    num_attention_heads = config.text_config.num_attention_heads
    intermediate_size = config.text_config.intermediate_size

    head_dim = hidden_size // num_attention_heads
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # qwen3_vl uses deepstack to merge visual embeds and text embeds, but it has no tensor operation.

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # vit flops
    images_seqlens = kargs.get("images_seqlens", None)
    if images_seqlens is not None:
        vit_flops = _estimate_qwen3_vit_flop(images_seqlens, config.vision_config)
    else:
        vit_flops = 0

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_qwen3_vl_moe_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    # qwen3_vl uses text_config and vision_config to distinguish configs of different parts.
    hidden_size = config.text_config.hidden_size
    vocab_size = config.text_config.vocab_size
    num_hidden_layers = config.text_config.num_hidden_layers
    num_key_value_heads = config.text_config.num_key_value_heads
    num_attention_heads = config.text_config.num_attention_heads
    moe_intermediate_size = config.text_config.moe_intermediate_size
    moe_num_expert = config.text_config.num_experts
    moe_topk = config.text_config.num_experts_per_tok

    head_dim = getattr(
        config.text_config, "head_dim",
        config.text_config.hidden_size // config.text_config.num_attention_heads
    )
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    moe_gata_N = hidden_size * moe_num_expert
    # moe has gate_proj, up_proj and down_proj using SwiGLU in ExpertMlp layer & shared experts
    moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk) * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    moe_N = (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers) + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * moe_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # vit flops
    images_seqlens = kargs.get("images_seqlens", None)
    if images_seqlens is not None:
        vit_flops = _estimate_qwen3_vit_flop(images_seqlens, config.vision_config)
    else:
        vit_flops = 0

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_qwen3_vit_flop(images_seqlens, config):
    """
    Estimate the FLOPS of the vision encoder for Qwen3-VL
    """

    if config is None:
        return 0
    tokens_sum = sum(images_seqlens)

    num_heads = config.num_heads
    depth = config.depth

    dim = config.hidden_size
    mlp_hidden_dim = config.intermediate_size
    out_hidden_size = config.out_hidden_size

    spatial_merge_size = config.spatial_merge_size

    head_dim = dim // num_heads

    # every vision token's patch_embed comes from a conv of (C, T, H, W) -> (dim,)
    patch_embed_N = dim * config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
    # Qwen3 VL vision mlp does not use GLU, thus 2.
    mlp_N = dim * mlp_hidden_dim * 2
    attn_linear_N = dim * (4 * dim)  # qkv and output proj
    merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (dim * (spatial_merge_size**2))

    # Qwen3 VL uses deep stack, one merger for every deepstack layer
    deepstack_merger_N = merger_N * len(config.deepstack_visual_indexes)
    # non-attn all_layer parm
    dense_N = patch_embed_N + (mlp_N + attn_linear_N) * depth + deepstack_merger_N + merger_N

    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # In Qwen3 VL, full attention is used in all vision layers.
    full_attn_layer_num = depth

    # full attn layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in images_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num

    vit_flops = dense_N_flops + attn_qkv_flops

    return vit_flops


def _estimate_deepseek_v3_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    moe_intermediate_size = config.moe_intermediate_size
    num_hidden_layers = config.num_hidden_layers
    first_k_dense_replace = config.first_k_dense_replace
    num_query_heads = config.num_attention_heads
    moe_num_expert = config.n_routed_experts

    moe_topk = config.num_experts_per_tok
    share_expert_num = config.n_shared_experts

    # non-attn per layer parm
    moe_gata_N = hidden_size * moe_num_expert
    # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
    moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
    # MLA attn
    attn_linear_N = 0
    q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if config.q_lora_rank is None:
        attn_linear_N += hidden_size * num_query_heads * q_head_dim
    else:
        attn_linear_N += hidden_size * config.q_lora_rank
        attn_linear_N += num_query_heads * q_head_dim * config.q_lora_rank

    attn_linear_N += hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)
    attn_linear_N += num_query_heads * (
        q_head_dim - config.qk_rope_head_dim + config.v_head_dim
    ) * config.kv_lora_rank
    attn_linear_N += num_query_heads * config.v_head_dim * hidden_size
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    moe_N = (
        (moe_gata_N + moe_expertmlp_N + attn_linear_N) *
        (num_hidden_layers - first_k_dense_replace) +
        (hidden_size * config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace +
        emd_and_lm_head_N
    )
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * moe_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen * num_hidden_layers

    # Core attention FLOPS for MLA with causal mask:
    # Q @ K^T: 3 * 2 * seq^2 * q_head_dim * num_heads / 2 (causal)
    # attn @ V: 3 * 2 * seq^2 * v_head_dim * num_heads / 2 (causal)
    attn_qkv_flops = 3 * seqlen_square_sum * (q_head_dim + config.v_head_dim) * num_query_heads
    # all_layer & all_token fwd & bwk flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12

    return flops_achieved


def _estimate_qwen2_moe_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    moe_intermediate_size = config.moe_intermediate_size
    moe_topk = config.num_experts_per_tok
    num_experts = config.num_experts

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    # gate + moe export
    moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_gemma3_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    # Gemma3 uses GeGLU (gelu_pytorch_tanh), having 3 matrices in MLP (inherited from Gemma2MLP)
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    # Gemma3 alternates between full and sliding window attention based on layer_types
    seqlen_square_sum = 0

    layer_types = getattr(config, "layer_types", None)
    sliding_window = getattr(config, "sliding_window", 1024)  # default 1024
    # default pattern: every 6th layer is full
    sliding_window_pattern = getattr(config, "sliding_window_pattern", 6)

    # If layer_types is not provided, generate it based on sliding_window_pattern
    if layer_types is None and sliding_window is not None and sliding_window_pattern is not None:
        layer_types = [
            "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
            for i in range(num_hidden_layers)
        ]

    if layer_types:
        # Calculate attention flops per layer based on attention type
        for layer_idx in range(num_hidden_layers):
            is_sliding = False
            if layer_types and layer_idx < len(layer_types):
                is_sliding = layer_types[layer_idx] == "sliding_attention"

            for seqlen in batch_seqlens:
                if is_sliding and sliding_window:
                    # Sliding window limits each token to attend to at most window_size tokens
                    effective_seqlen = min(seqlen, sliding_window)
                    seqlen_square_sum += seqlen * effective_seqlen
                else:
                    # Full attention
                    seqlen_square_sum += seqlen * seqlen
    else:
        # If no layer_types config, assume all layers use full attention
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        seqlen_square_sum *= num_hidden_layers

    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_apertus_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # Apertus MLP with XIELU activation uses only 2 linear layers (up_proj, down_proj)
    # No gate_proj for XIELU, unlike SwiGLU which has 3 layers
    mlp_N = hidden_size * intermediate_size * 2
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)

    # ApertusConfig has qk_norm defaulting to True.
    # This adds params for q_norm (on H) and k_norm (on num_kv_heads * head_dim)
    qk_norm_params_per_layer = hidden_size + num_key_value_heads * head_dim  # q_norm + k_norm

    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer params
    dense_N = (
        mlp_N + attn_linear_N + qk_norm_params_per_layer
    ) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_gpt_oss_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads

    # MoE params
    moe_intermediate_size = config.intermediate_size
    num_experts = config.num_local_experts
    num_experts_per_tok = config.num_experts_per_tok
    mlp_matrices = 3

    # Head dim
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # 1. Attention Block (GQA)
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    # 2. MLP / MoE Block
    # Gate network
    moe_gate_N = hidden_size * num_experts
    # Expert forward calculation, Active parameters: mlp_matrices * H * I * num_experts_per_tok
    moe_expert_N = hidden_size * moe_intermediate_size * mlp_matrices * num_experts_per_tok

    moe_mlp_N = moe_gate_N + moe_expert_N

    emd_and_lm_head_N = vocab_size * hidden_size * 2

    # Total non-attn params per layer * layers + embeddings
    # (moe_mlp_N + attn_linear_N) * layers
    dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N

    # FLOPs for dense part (fwd + bwd = 6 * N)
    dense_N_flops = 6 * dense_N * tokens_sum

    # 3. Attention Matrix FLOPs
    seqlen_square_sum = 0

    # Handle sliding window attention
    layer_types = getattr(config, "layer_types", None)
    sliding_window = getattr(config, "sliding_window", 128)

    if layer_types:
        for layer_type in layer_types:
            is_sliding = layer_type == "sliding_attention"

            for seqlen in batch_seqlens:
                if is_sliding and sliding_window:
                    # Sliding window limits each token to attend to at most window_size tokens
                    effective_seqlen = min(seqlen, sliding_window)
                    seqlen_square_sum += seqlen * effective_seqlen
                else:
                    # Full attention
                    seqlen_square_sum += seqlen * seqlen
    else:
        # Default to full attention for all layers
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        seqlen_square_sum *= num_hidden_layers

    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads

    # Total FLOPs
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_deepseek_v4_flops(config, tokens_sum, batch_seqlens, delta_time):
    """Estimate TFLOPs/s for DeepSeek-V4 training.

    Follows the same decomposition as Megatron-LM ``transformer_flops()`` for
    the ``dsv4_hybrid`` attention variant: all FLOPs are split into a
    *token-linear* part (scales with ``Σ L_i``) and a *core-attention L²* part
    (scales with ``Σ L_i²``).

    Parameters
    ----------
    config : DeepseekV4Config
    tokens_sum : int
    batch_seqlens : list[int]
    delta_time : float
        Seconds.
    """
    H = config.hidden_size
    V = config.vocab_size
    n_head = config.num_attention_heads
    d = config.head_dim
    r_q = config.q_lora_rank
    r_o = config.o_lora_rank
    g_o = config.o_groups
    W = config.sliding_window
    I_moe = getattr(config, "moe_intermediate_size", config.intermediate_size)
    I_shared = config.intermediate_size
    K = config.num_experts_per_tok
    num_layers = config.num_hidden_layers
    mtp_num_layers = getattr(config, "num_nextn_predict_layers", 0) or 0

    compress_rates = getattr(config, "compress_rates", {})
    r_csa = compress_rates.get("compressed_sparse_attention", 4)
    r_hca = compress_rates.get("heavily_compressed_attention", 128)

    layer_types = getattr(config, "layer_types", []) or []
    n_layers_r0 = 0
    n_layers_r4 = 0
    n_layers_r128 = 0
    for lt in layer_types[:num_layers]:
        if lt == "compressed_sparse_attention":
            n_layers_r4 += 1
        elif lt == "heavily_compressed_attention":
            n_layers_r128 += 1
        else:
            n_layers_r0 += 1

    # MTP: count as 1 extra MoE + attention layer.  If layer_types covers
    # the MTP layer use its type; otherwise default to window-only (r=0).
    num_total_layers = num_layers + mtp_num_layers
    for lt in layer_types[num_layers:num_layers + mtp_num_layers]:
        if lt == "compressed_sparse_attention":
            n_layers_r4 += 1
        elif lt == "heavily_compressed_attention":
            n_layers_r128 += 1
        else:
            n_layers_r0 += 1
    mtp_extra_layers = mtp_num_layers - len(layer_types[num_layers:num_layers + mtp_num_layers])
    n_layers_r0 += mtp_extra_layers

    # fwd + wgrad + dgrad = 3;  each GEMM m*n*k = 2mnk FLOPs
    FBE = 3
    FMA = 2

    # ---- 1. MLA projections (per layer, token-linear) ----
    q_term = r_q * (H + n_head * d + 1)
    kv_term = H * d + d
    o_term = n_head * d * r_o + g_o * r_o * H
    mla_proj_per_layer = FBE * FMA * (q_term + kv_term + o_term)

    # ---- 2. Sparse attention (replaces full core attention) ----
    # Use the first seqlen as representative for the effective topk
    # causal correction.  Megatron uses args.seq_length; we approximate
    # with the max seqlen in the batch.
    seq_len = max(batch_seqlens)
    w_cap = min(W, seq_len)
    w_eff = w_cap * (1 - w_cap / (2 * seq_len))

    # r=0: window-only, fixed per-token cost
    sparse_attn_r0 = n_layers_r0 * n_head * w_eff * d * 2

    # r=128 (HCA): window (token-linear) + all compressed KV (L²)
    sparse_attn_r128_window = n_layers_r128 * n_head * w_eff * d * 2
    sparse_attn_r128_core = n_layers_r128 * n_head * d / r_hca

    # r=4 (CSA): window + learned-topk compressed entries
    idx_n_heads = getattr(config, "index_n_heads", 64)
    idx_head_dim = getattr(config, "index_head_dim", 128)
    idx_topk = getattr(config, "index_topk", 512)

    if n_layers_r4 > 0:
        effective_topk = min(idx_topk, seq_len // r_csa)
        avg_comp_4 = effective_topk * (1 - effective_topk * r_csa / (2 * seq_len))
        sparse_attn_r4 = n_layers_r4 * n_head * (w_eff + avg_comp_4) * d * 2
    else:
        sparse_attn_r4 = 0

    sparse_attn_token_term = sparse_attn_r0 + sparse_attn_r4 + sparse_attn_r128_window

    # ---- 3. Main compressor projections (token-linear) ----
    # r=4 (CSA): kv_proj + gate_proj, each H → 2*d (overlapping windows)
    # r=128 (HCA): kv_proj + gate_proj, each H → d (non-overlapping)
    main_compressor_term = (n_layers_r4 * H * (2 * d) * 2 + n_layers_r128 * H * (1 * d) * 2)

    # ---- 4. Indexer (CSA layers only, token-linear + L²) ----
    if n_layers_r4 > 0:
        indexer_token_term = (
            n_layers_r4 * H * (2 * idx_head_dim) * 2 +
            n_layers_r4 * r_q * idx_n_heads * idx_head_dim + n_layers_r4 * H * idx_n_heads
        )
        indexer_scoring_core = n_layers_r4 * idx_n_heads * idx_head_dim / r_csa
    else:
        indexer_token_term = 0
        indexer_scoring_core = 0

    # ---- Assemble self-attention terms ----
    dsv4_extra_term = (
        FBE * FMA * (sparse_attn_token_term + main_compressor_term + indexer_token_term)
    )
    dsv4_core_term = FBE * FMA * (sparse_attn_r128_core + indexer_scoring_core)

    self_attn_term = mla_proj_per_layer * num_total_layers + dsv4_extra_term

    # ---- 5. MoE FFN (all layers are MoE in DSV4) ----
    # SwiGLU: gate + up + down = 3 matmuls
    ffn_expansion = 3
    moe_ffn_per_layer = FBE * FMA * H * (I_moe * K * ffn_expansion + I_shared * ffn_expansion)

    # ---- 6. MTP extra: norms + eh_proj ----
    mtp_extra_term = (FBE * FMA * mtp_num_layers * (3 * H + 2 * H * H))

    # ---- 7. Logit ----
    logit_term = FBE * FMA * H * V * (mtp_num_layers + 1)

    # ---- Total ----
    seqlen_sq_sum = sum(s * s for s in batch_seqlens)

    flops_total = (
        tokens_sum *
        (moe_ffn_per_layer * num_total_layers + self_attn_term + mtp_extra_term + logit_term) +
        seqlen_sq_sum * dsv4_core_term
    )

    return flops_total / delta_time / 1e12


def _estimate_welm_v4_flops(config, tokens_sum, batch_seqlens, delta_time):
    """Estimate WeLM-v4/v4.5 Megatron training TFLOPs/s."""
    total_flops = _welm_v4_total_flops(config, tokens_sum, batch_seqlens)
    return total_flops / delta_time / 1e12


def _welm_v4_total_flops(config, tokens_sum, batch_seqlens):
    """Raw WeLM-v4/v4.5 Megatron training FLOPs for the text backbone.

    This follows Megatron's ``transformer_flops()`` convention: count GEMMs in
    forward, weight-gradient, and data-gradient passes, while omitting
    embedding lookups, normalization, activation, routing, and softmax FLOPs.
    WeLM-specific OE projection, head-wise attention gate, GQA dimensions, MoE
    layout, shared experts, and per-layer sliding windows are included.

    MTP is intentionally excluded because ``Welm4MoeBridge`` currently builds
    the Megatron model with ``mtp_block_spec=None`` and ignores checkpoint MTP
    layers.
    """
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_query_groups = config.num_key_value_heads or num_attention_heads
    head_dim = config.head_dim or hidden_size // num_attention_heads

    query_projection_size = num_attention_heads * head_dim
    kv_projection_size = num_query_groups * head_dim

    # A GEMM costs 2mnk FLOPs and is run in forward, wgrad, and dgrad.
    fwd_bwd_gemm_factor = 6

    # QKV and output projections. WeLM's head-wise gate is H -> num_heads,
    # rather than the H -> query_projection_size gate used by standard MCore
    # gated attention.
    attention_linear_size = hidden_size * (
        query_projection_size + 2 * kv_projection_size + query_projection_size
    )
    if config.gated_self_attention_headwise:
        attention_linear_size += hidden_size * num_attention_heads
    attention_linear_flops = (fwd_bwd_gemm_factor * attention_linear_size * tokens_sum * num_layers)

    # A KV-mirror layer first computes its regular fused QKV projection, then
    # invokes the same fused QKV projection again on the mirrored hidden
    # states and keeps only K/V. The discarded Q is still computed.
    kv_mirror_imitated_layers = list(config.kv_mirror_imitated_layers or [])
    kv_mirror_layers = list(config.kv_mirror_layers or [])
    valid_kv_mirror_layers = {
        layer_idx
        for layer_idx in kv_mirror_layers[:len(kv_mirror_imitated_layers)]
        if 0 <= layer_idx < num_layers
    }
    mirrored_qkv_linear_size = (hidden_size * (query_projection_size + 2 * kv_projection_size))
    attention_linear_flops += (
        fwd_bwd_gemm_factor * mirrored_qkv_linear_size * tokens_sum * len(valid_kv_mirror_layers)
    )

    # WeLM-v4.5 uses a layerwise causal window. Values absent, non-positive, or
    # at least max_position_embeddings mean full attention in the MCore path.
    layerwise_windows = list(config.sliding_window_size_layerwise or [])
    max_positions = config.max_position_embeddings
    attention_pairs = 0.0
    seqlen_sq_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
    aggregate_only = not math.isclose(
        float(sum(batch_seqlens)),
        float(tokens_sum),
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    if aggregate_only and seqlen_sq_sum > 0:
        # ``estimate_flops_from_sums`` represents ΣS² as one synthetic
        # sqrt(ΣS²) sequence. Recover an effective sequence count so a
        # sliding-window layer still scales with ΣS rather than sqrt(ΣS²).
        effective_num_sequences = tokens_sum * tokens_sum / seqlen_sq_sum
        effective_seqlen = tokens_sum / effective_num_sequences
    else:
        effective_num_sequences = 0
        effective_seqlen = 0

    for layer_idx in range(num_layers):
        window = layerwise_windows[layer_idx] if layer_idx < len(layerwise_windows) else None
        is_full_attention = (
            window is None or window <= 0 or
            (max_positions is not None and window >= max_positions)
        )
        if aggregate_only:
            if is_full_attention or window >= effective_seqlen:
                attention_pairs += seqlen_sq_sum / 2
            else:
                attention_pairs += window * (tokens_sum - window * effective_num_sequences / 2)
            continue

        for seqlen in batch_seqlens:
            if is_full_attention or window >= seqlen:
                # Megatron approximates the causal triangle as S² / 2.
                attention_pairs += seqlen * seqlen / 2
            else:
                # Number of entries in a left-causal band, using the same
                # continuous approximation as the full-attention formula.
                attention_pairs += window * (seqlen - window / 2)

    # Per attended Q/K pair: QK^T and AV each cost 2 FLOPs per head dim;
    # both backward GEMMs add another 2x over the forward computation.
    core_attention_flops = (2 * fwd_bwd_gemm_factor * query_projection_size * attention_pairs)

    # HF WeLM chooses dense vs MoE per layer with decoder_sparse_step and
    # mlp_only_layers. The 80B-A3B checkpoint has MoE on every decoder layer,
    # but retaining the source rule makes the estimator work for reduced and
    # mixed checkpoints as well.
    num_experts = config.num_experts or 0
    sparse_step = config.decoder_sparse_step or 1
    mlp_only_layers = set(config.mlp_only_layers or [])
    num_moe_layers = sum(
        1 for layer_idx in range(num_layers)
        if num_experts > 0 and layer_idx not in mlp_only_layers and (layer_idx + 1) %
        sparse_step == 0
    )
    num_dense_layers = num_layers - num_moe_layers

    ffn_hidden_size = config.intermediate_size
    moe_ffn_hidden_size = config.moe_intermediate_size or ffn_hidden_size
    experts_per_token = config.num_experts_per_tok or 1
    shared_expert_hidden_size = (
        (config.shared_expert_intermediate_size or 0) * (config.num_shared_experts or 0)
    )

    # WeLM uses SwiGLU: gate/up/down are three matrix multiplications.
    ffn_linear_size = hidden_size * 3 * (
        ffn_hidden_size * num_dense_layers +
        (moe_ffn_hidden_size * experts_per_token + shared_expert_hidden_size) * num_moe_layers
    )
    ffn_flops = fwd_bwd_gemm_factor * ffn_linear_size * tokens_sum

    # OE embedding tables are lookups. Only the concatenated OE embedding
    # projection is a GEMM in the Megatron model.
    num_oe_embeddings = len(config.oe_vocab_sizes or [])
    oe_projection_size = (hidden_size * (config.oe_dim or 0) * num_oe_embeddings)
    oe_projection_flops = fwd_bwd_gemm_factor * oe_projection_size * tokens_sum

    # Input token embeddings are lookups; only the output vocabulary projection
    # is counted, matching Megatron's transformer_flops().
    logit_flops = fwd_bwd_gemm_factor * hidden_size * vocab_size * tokens_sum

    total_flops = (
        attention_linear_flops + core_attention_flops + ffn_flops + oe_projection_flops +
        logit_flops
    )
    return total_flops


def _audio_feat_extract_output_lengths(input_lengths, n_window):
    """Post-CNN output-token counts of the Qwen3-Omni audio encoder.

    Matches HF ``Qwen3OmniMoeAudioEncoder`` (and mbridge
    ``get_feat_extract_output_lengths``): every full chunk of
    ``2 * n_window`` mel frames yields 13 tokens, and the tail frames go
    through three stride-2 Conv2d layers.
    """
    chunk_len = n_window * 2
    output_lengths = []
    for length in input_lengths:
        leave = length % chunk_len
        feat_length = (leave - 1) // 2 + 1
        tail_tokens = ((feat_length - 1) // 2 + 1 - 1) // 2 + 1
        output_lengths.append(tail_tokens + (length // chunk_len) * 13)
    return output_lengths


def _estimate_welm_omni_audio_flop(audio_seqlens, config):
    """Estimate the FLOPs of the WeLM-Omni-V4.5 audio tower.

    The tower is Qwen3-Omni's Whisper-like audio encoder: three stride-2
    Conv2d layers (channels ``downsample_hidden_size``), a ``conv_out``
    projection to ``d_model``, ``encoder_layers`` bidirectional transformer
    layers (MHA + GELU MLP) with attention windowed to ``n_window_infer``
    mel frames, and a two-layer projector to ``output_dim``.

    Args:
        audio_seqlens: per-audio mel frame counts in the batch.
        config: ``Qwen3OmniMoeAudioEncoderConfig`` (``config.audio_config``).
    """
    if config is None:
        return 0

    dim = config.d_model
    num_heads = config.encoder_attention_heads
    depth = config.encoder_layers
    mlp_hidden_dim = config.encoder_ffn_dim
    downsample_dim = config.downsample_hidden_size
    n_window = config.n_window
    head_dim = dim // num_heads

    # fwd + wgrad + dgrad = 3; each GEMM m*n*k = 2mnk FLOPs.
    fwd_bwd_gemm_factor = 6

    # Three k=3 s=2 p=1 Conv2d layers halve mel freq 128 -> 64 -> 32 -> 16
    # and time T -> T/2 -> T/4 -> T/8. ``conv_N`` counts k^2 * Cin * Cout
    # multiply-accumulates over the output elements produced per mel frame.
    conv_N = 9 * (
        1 * downsample_dim * (64 // 2) + downsample_dim * downsample_dim *
        (32 // 4) + downsample_dim * downsample_dim * (16 // 8)
    )
    conv_flops = fwd_bwd_gemm_factor * conv_N * sum(audio_seqlens)

    audio_token_lens = _audio_feat_extract_output_lengths(audio_seqlens, n_window)
    tokens_sum = sum(audio_token_lens)
    if tokens_sum == 0:
        return 0

    # conv_out projects (downsample_dim * freq_out=16) -> d_model per token.
    conv_out_N = downsample_dim * 16 * dim
    # Encoder layer: QKV+O projections and a GELU MLP (fc1 + fc2, no GLU).
    attn_linear_N = dim * (4 * dim)
    mlp_N = dim * mlp_hidden_dim * 2
    # ln_post is not a GEMM; proj1 (d -> d) and proj2 (d -> output_dim) are.
    projector_N = dim * dim + dim * config.output_dim
    dense_N = conv_out_N + (attn_linear_N + mlp_N) * depth + projector_N
    dense_N_flops = fwd_bwd_gemm_factor * dense_N * tokens_sum

    # Attention is bidirectional inside independent windows of
    # ``n_window_infer`` mel frames, i.e. ``window_tokens`` encoder tokens.
    window_tokens = (config.n_window_infer // (n_window * 2)) * 13
    window_sq_sum = 0
    for token_len in audio_token_lens:
        n_full, remainder = divmod(token_len, window_tokens)
        window_sq_sum += n_full * window_tokens * window_tokens + remainder * remainder
    # Same convention as the ViT estimator: bidirectional QK^T and AV,
    # forward plus both backward GEMMs.
    attn_qkv_flops = 12 * window_sq_sum * head_dim * num_heads * depth

    return conv_flops + dense_N_flops + attn_qkv_flops


def _estimate_welm_omni_v4_5_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    """Estimate WeLM-Omni-V4.5 Megatron training TFLOPs/s.

    The composite HF config nests the WeLM4.5 text backbone in
    ``config.text_config`` (identical to the ``welmv4_moe`` estimator) and
    the Qwen3-Omni audio tower in ``config.audio_config``. Audio FLOPs are
    only counted when ``kargs["audio_seqlens"]`` (per-audio mel frame
    counts) is provided.
    """
    total_flops = _welm_v4_total_flops(config.text_config, tokens_sum, batch_seqlens)
    audio_seqlens = kargs.get("audio_seqlens", None)
    if audio_seqlens:
        total_flops += _estimate_welm_omni_audio_flop(audio_seqlens, config.audio_config)
    return total_flops / delta_time / 1e12


def _estimate_unknown_flops(config, tokens_sum, batch_seqlens, delta_time):
    return 0


ESTIMATE_FUNC = {
    MODEL_ARCH.QWEN2: _estimate_qwen2_flops,
    MODEL_ARCH.LLAMA: _estimate_qwen2_flops,
    MODEL_ARCH.QWEN2_MOE: _estimate_qwen2_moe_flops,
    MODEL_ARCH.QWEN2_VL: _estimate_qwen2_flops,
    MODEL_ARCH.QWEN2_5_VL: _estimate_qwen2_flops,
    MODEL_ARCH.QWEN3: _estimate_qwen2_flops,
    MODEL_ARCH.QWEN3_MOE: _estimate_qwen2_moe_flops,
    MODEL_ARCH.QWEN3_VL: _estimate_qwen3_vl_flops,
    MODEL_ARCH.QWEN3_VL_MOE: _estimate_qwen3_vl_moe_flops,
    MODEL_ARCH.DEEPSEEK_V3: _estimate_deepseek_v3_flops,
    MODEL_ARCH.DEEPSEEK_V4: _estimate_deepseek_v4_flops,
    MODEL_ARCH.WELMV4_MOE: _estimate_welm_v4_flops,
    MODEL_ARCH.WELM_OMNI_V4_5: _estimate_welm_omni_v4_5_flops,
    MODEL_ARCH.MISTRAL: _estimate_qwen2_flops,
    MODEL_ARCH.GEMMA3_TEXT: _estimate_gemma3_flops,
    MODEL_ARCH.SEED_OSS: _estimate_qwen2_flops,
    MODEL_ARCH.APERTUS: _estimate_apertus_flops,
    MODEL_ARCH.GLM4V: _estimate_qwen2_flops,
    MODEL_ARCH.GPT_OSS: _estimate_gpt_oss_flops,
    MODEL_ARCH.WEMM3_EMBEDDING: _estimate_qwen3_vl_flops,
}


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """
    def __init__(self, config: PretrainedConfig, model_arch):
        VALID_CONFIG_TYPE = ESTIMATE_FUNC.keys()
        if model_arch not in VALID_CONFIG_TYPE:
            print(
                f"Only support config type of {VALID_CONFIG_TYPE}, but got {model_arch}. MFU will always be "
                f"zero."
            )

        self.config = config
        self.model_arch = model_arch

    # TODO: actually we can make this a static method
    def estimate_flops(self, batch_seqlens, delta_time, **kargs):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the
                current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = ESTIMATE_FUNC.get(self.model_arch, _estimate_unknown_flops)
        sig = inspect.signature(func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            estimated_flops = func(self.config, tokens_sum, batch_seqlens, delta_time, **kargs)
        else:
            estimated_flops = func(self.config, tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops

    def estimate_flops_from_sums(self, tokens_sum, seqlen_sq_sum, delta_time, **kargs):
        """
        Estimate the FLOPS from the global-batch aggregates ``Σ S`` (``tokens_sum``)
        and ``Σ S²`` (``seqlen_sq_sum``). Dense FLOPs scale with ``Σ S`` and
        attention-core FLOPs scale with ``Σ S²``.

        Architectures with per-sample sequence-length branches (sliding-window
        attention in Gemma3 / GPT-OSS) are approximated as full attention.

        Args:
            tokens_sum (int | float): ``Σ S`` over the global batch.
            seqlen_sq_sum (int | float): ``Σ S²`` over the global batch.
            delta_time (float): Elapsed time for the batch, in seconds.

        Returns:
            Tuple[float, float]: ``(estimated_flops, promised_flops)``.
        """
        seqlen_sq_sum = max(float(seqlen_sq_sum), 0.0)
        batch_seqlens = [math.sqrt(seqlen_sq_sum)]
        func = ESTIMATE_FUNC.get(self.model_arch, _estimate_unknown_flops)
        sig = inspect.signature(func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            estimated_flops = func(self.config, tokens_sum, batch_seqlens, delta_time, **kargs)
        else:
            estimated_flops = func(self.config, tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
