import math
from types import SimpleNamespace

import pytest

from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.utils.flops_counter import (
    ESTIMATE_FUNC,
    _estimate_welm_omni_v4_5_flops,
    _estimate_welm_v4_flops,
)


def _welm_config(**overrides):
    values = {
        "hidden_size": 8,
        "vocab_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 12,
        "moe_intermediate_size": 6,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "shared_expert_intermediate_size": 5,
        "num_shared_experts": 1,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "gated_self_attention_headwise": True,
        "oe_dim": 3,
        "oe_vocab_sizes": [101, 103],
        "max_position_embeddings": 32,
        "sliding_window_size_layerwise": [32, 4],
        "kv_mirror_imitated_layers": [],
        "kv_mirror_layers": [],
        "num_nextn_predict_layers": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_welm_v4_estimator_is_registered():
    assert ESTIMATE_FUNC[MODEL_ARCH.WELMV4_MOE] is _estimate_welm_v4_flops


def test_welm_v4_estimator_counts_megatron_specific_paths():
    config = _welm_config()

    estimated = _estimate_welm_v4_flops(
        config,
        tokens_sum=8,
        batch_seqlens=[8],
        delta_time=2,
    )

    # Attention projections: 6 * 8 tokens * 2 layers *
    # (H*(Q+K+V+O) + H*num_heads).
    attention_linear_flops = 6 * 8 * 2 * (8 * (8 + 4 + 4 + 8) + 8 * 2)
    # Layer 0 is full: 8^2/2 pairs. Layer 1 has window 4:
    # 4*(8-4/2) pairs. Each pair has QK and AV, fwd+bwd.
    core_attention_flops = 2 * 6 * 8 * (32 + 24)
    # Two MoE layers, SwiGLU, top-2 routed experts and one shared expert.
    ffn_flops = 6 * 8 * (8 * 3 * ((6 * 2 + 5) * 2))
    oe_projection_flops = 6 * 8 * (8 * 3 * 2)
    logit_flops = 6 * 8 * 8 * 16
    expected = (
        attention_linear_flops
        + core_attention_flops
        + ffn_flops
        + oe_projection_flops
        + logit_flops
    ) / 2 / 1e12

    assert estimated == pytest.approx(expected)


def test_welm_v4_estimator_uses_hf_dense_moe_pattern_and_ignores_mtp():
    config = _welm_config(
        decoder_sparse_step=2,
        num_nextn_predict_layers=1,
        sliding_window_size_layerwise=[],
    )
    without_mtp = _estimate_welm_v4_flops(config, 8, [8], 1)

    config.num_nextn_predict_layers = 8
    with_mtp_metadata = _estimate_welm_v4_flops(config, 8, [8], 1)

    assert with_mtp_metadata == without_mtp


def test_welm_v4_estimator_counts_extra_fused_qkv_for_kv_mirror():
    tokens_sum = 8
    config = _welm_config(
        kv_mirror_imitated_layers=[0],
        kv_mirror_layers=[1],
    )
    with_kv_mirror = _estimate_welm_v4_flops(
        config, tokens_sum, [tokens_sum], 1
    )

    config.kv_mirror_imitated_layers = []
    config.kv_mirror_layers = []
    without_kv_mirror = _estimate_welm_v4_flops(
        config, tokens_sum, [tokens_sum], 1
    )

    query_projection_size = config.num_attention_heads * config.head_dim
    kv_projection_size = config.num_key_value_heads * config.head_dim
    expected_extra_qkv = (
        6
        * config.hidden_size
        * (query_projection_size + 2 * kv_projection_size)
        * tokens_sum
        / 1e12
    )
    assert with_kv_mirror - without_kv_mirror == pytest.approx(expected_extra_qkv)


def test_welm_v4_estimator_recovers_sliding_window_cost_from_sums():
    config = _welm_config()
    batch_seqlens = [8, 8]
    tokens_sum = sum(batch_seqlens)
    seqlen_sq_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)

    from_lengths = _estimate_welm_v4_flops(
        config,
        tokens_sum=tokens_sum,
        batch_seqlens=batch_seqlens,
        delta_time=1,
    )
    from_sums = _estimate_welm_v4_flops(
        config,
        tokens_sum=tokens_sum,
        batch_seqlens=[math.sqrt(seqlen_sq_sum)],
        delta_time=1,
    )

    assert from_sums == pytest.approx(from_lengths)


def _welm_omni_config(**audio_overrides):
    audio_values = {
        "num_mel_bins": 128,
        "d_model": 8,
        "encoder_layers": 2,
        "encoder_attention_heads": 2,
        "encoder_ffn_dim": 16,
        "downsample_hidden_size": 4,
        "n_window": 50,
        "n_window_infer": 800,
        "output_dim": 12,
    }
    audio_values.update(audio_overrides)
    return SimpleNamespace(
        text_config=_welm_config(),
        audio_config=SimpleNamespace(**audio_values),
    )


def test_welm_omni_estimator_is_registered():
    assert ESTIMATE_FUNC[MODEL_ARCH.WELM_OMNI_V4_5] is _estimate_welm_omni_v4_5_flops


def test_welm_omni_estimator_without_audio_matches_welm_v4():
    config = _welm_omni_config()

    omni = _estimate_welm_omni_v4_5_flops(
        config,
        tokens_sum=8,
        batch_seqlens=[8],
        delta_time=2,
    )
    text_only = _estimate_welm_v4_flops(
        config.text_config,
        tokens_sum=8,
        batch_seqlens=[8],
        delta_time=2,
    )

    assert omni == pytest.approx(text_only)


def test_welm_omni_estimator_counts_audio_tower():
    config = _welm_omni_config()
    # 100 mel frames -> one full chunk -> 13 tokens; 50 frames -> 7 tokens.
    audio_seqlens = [100, 50]
    audio_tokens = [13, 7]

    with_audio = _estimate_welm_omni_v4_5_flops(
        config,
        tokens_sum=8,
        batch_seqlens=[8],
        delta_time=2,
        audio_seqlens=audio_seqlens,
    )
    without_audio = _estimate_welm_omni_v4_5_flops(
        config,
        tokens_sum=8,
        batch_seqlens=[8],
        delta_time=2,
    )

    dim = config.audio_config.d_model
    conv_n = 9 * (1 * 4 * 32 + 4 * 4 * 8 + 4 * 4 * 2)
    conv_flops = 6 * conv_n * sum(audio_seqlens)
    dense_n = (
        4 * 16 * dim
        + (4 * dim * dim + 2 * dim * config.audio_config.encoder_ffn_dim) * 2
        + dim * dim
        + dim * config.audio_config.output_dim
    )
    dense_flops = 6 * dense_n * sum(audio_tokens)
    # Both audios fit one 104-token attention window; bidirectional core.
    window_sq_sum = 13 * 13 + 7 * 7
    attn_flops = 12 * window_sq_sum * (dim // 2) * 2 * 2
    expected_audio = (conv_flops + dense_flops + attn_flops) / 2 / 1e12

    assert with_audio - without_audio == pytest.approx(expected_audio)
