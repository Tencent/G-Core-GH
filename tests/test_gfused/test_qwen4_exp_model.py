"""Qwen3.8-Flash-Next (``qwen4_exp``) 语言模型骨架测试。

覆盖：tiny config 的层型/PLE 元数据、CUDA 前向反向、参数命名与真实
checkpoint 一致。真实 checkpoint 是 180B，这里只用 4 层 tiny 随机权重。
"""
import re

import pytest
import torch

from gpatch_v4.models.qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

# 与 tests/test_gfused/test_deepseek_v4_ep_cp.py 的 _truncate_config 思路一致：
# 保留真实架构的全部结构特征（3 linear + 1 sparse、PLE 落在 layers.1、MoE、
# 4 条 Gated-Residual 流），只把每个维度缩到最小可用值。
TINY_KWARGS = dict(
    vocab_size=512,
    hidden_size=128,
    num_hidden_layers=4,
    full_attention_interval=4,
    # QSA 主注意力：GQA 4/2，head_dim 32，partial rope 0.25 -> rotary_dim 8
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    # Gated DeltaNet
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    # MoE
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    # QSA indexer：budget 8 / ratio 4 -> 2 个 block
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=4,
    # Gated Residual（config 里叫 hc_*）
    hc_count=4,
    hc_lowrank=16,
    # Engram / n-gram（config 里叫 ple_*）；ple_layer_ids 是 1-based
    ple_layer_ids=[2],
    ple_embed_dim=128,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=1024,
    make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=4,
    eos_token_id=1,
    tie_word_embeddings=False,
    # SFT 不需要 cache；生成由独立推理引擎负责。
    use_cache=False,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")


def _tiny_config(**overrides) -> Qwen4ExpTextConfig:
    kwargs = {**TINY_KWARGS, **overrides}
    return Qwen4ExpTextConfig(**kwargs)


def _tiny_model(seed: int = 0, **overrides) -> Qwen4ExpForCausalLM:
    torch.manual_seed(seed)
    model = Qwen4ExpForCausalLM(_tiny_config(**overrides))
    model.eval()
    return model


def test_layer_types_follow_three_linear_one_sparse() -> None:
    config = _tiny_config()
    assert config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "qwen_sparse_attention",
    ]


def test_full_size_layer_pattern_matches_released_checkpoint() -> None:
    # 真实 checkpoint：36 个 linear_attn + 12 个 self_attn，稀疏层落在 3,7,...,47
    config = _tiny_config(num_hidden_layers=48, ple_layer_ids=[2])
    sparse = [i for i, t in enumerate(config.layer_types) if t == "qwen_sparse_attention"]
    linear = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    assert sparse == list(range(3, 48, 4))
    assert len(sparse) == 12 and len(linear) == 36


def test_ple_is_attached_to_second_decoder_layer_only() -> None:
    # ple_layer_ids=[2] 是 1-based，落到 0-based 的 layers.1
    model = _tiny_model()
    with_ple = [name for name, _ in model.named_modules() if re.fullmatch(r"model\.layers\.\d+\.ple", name)]
    assert with_ple == ["model.layers.1.ple"]


@requires_cuda
def test_hidden_stream_is_hc_count_wide() -> None:
    # 每个 decoder layer 的输入/输出都是 [B, S, hc_count * hidden_size]
    config = _tiny_config()
    model = _tiny_model().to(device="cuda", dtype=torch.bfloat16)
    captured = {}

    def hook(_module, args, _output):
        captured["shape"] = args[0].shape

    model.model.layers[2].register_forward_hook(hook, with_kwargs=False)
    model(input_ids=torch.randint(0, config.vocab_size, (1, 16), device="cuda"))

    assert captured["shape"][-1] == config.hc_count * config.hidden_size


@requires_cuda
def test_forward_and_backward_produce_finite_grads() -> None:
    config = _tiny_config()
    model = _tiny_model().to(device="cuda", dtype=torch.bfloat16)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 24), device="cuda")

    logits = model(input_ids=input_ids).logits
    assert logits.shape == (2, 24, config.vocab_size)
    assert torch.isfinite(logits).all()

    logits.float().pow(2).mean().backward()
    missing = [
        name for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    # QSA indexer 在参考实现里没有 aux loss / STE，梯度可以为 None；其余必须有梯度
    assert all(".indexer." in name for name in missing), missing
    non_finite = [
        name for name, param in model.named_parameters()
        if param.grad is not None and not torch.isfinite(param.grad).all()
    ]
    assert not non_finite, non_finite


@requires_cuda
def test_right_padding_does_not_change_unpadded_logits() -> None:
    # SFT 走右侧 padding；padding 不能污染 GDN 的循环状态或 Engram 的 hash
    config = _tiny_config()
    model = _tiny_model().to(device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(1)
    real = torch.randint(2, config.vocab_size, (1, 8), device="cuda")

    ref = model(input_ids=real, attention_mask=torch.ones_like(real)).logits
    padded_ids = torch.cat([real, torch.zeros(1, 16, dtype=torch.long, device="cuda")], dim=1)
    padded_mask = torch.cat(
        [
            torch.ones(1, 8, dtype=torch.long, device="cuda"),
            torch.zeros(1, 16, dtype=torch.long, device="cuda"),
        ],
        dim=1,
    )
    got = model(input_ids=padded_ids, attention_mask=padded_mask).logits[:, :8]

    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)


def test_parameter_names_match_released_checkpoint_scheme() -> None:
    """参数名必须和真实 checkpoint 对得上（去掉 `model.language_model.` 前缀后）。

    真实 key 形如 ``model.language_model.layers.1.linear_attn.in_proj_qkv.weight``；
    text-only 模型这里是 ``model.layers.1.linear_attn.in_proj_qkv.weight``。
    """
    model = _tiny_model()
    names = set(model.state_dict())

    expected_layer_1 = {
        "model.layers.1.linear_attn.in_proj_qkv.weight",
        "model.layers.1.linear_attn.in_proj_z.weight",
        "model.layers.1.linear_attn.in_proj_b.weight",
        "model.layers.1.linear_attn.in_proj_a.weight",
        "model.layers.1.linear_attn.conv1d.weight",
        "model.layers.1.linear_attn.A_log",
        "model.layers.1.linear_attn.dt_bias",
        "model.layers.1.linear_attn.norm.weight",
        "model.layers.1.linear_attn.out_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.experts.down_proj",
        "model.layers.1.mlp.shared_expert.gate_proj.weight",
        "model.layers.1.mlp.shared_expert.up_proj.weight",
        "model.layers.1.mlp.shared_expert.down_proj.weight",
        "model.layers.1.mlp.shared_expert_gate.weight",
        "model.layers.1.attn_hyper_connection.hc_norm.weight",
        "model.layers.1.attn_hyper_connection.input_mix_weight_down.weight",
        "model.layers.1.attn_hyper_connection.input_mix_weight_up.weight",
        "model.layers.1.attn_hyper_connection.block_inject_weight.weight",
        "model.layers.1.mlp_hyper_connection.hc_norm.weight",
        "model.layers.1.mlp_hyper_connection.input_mix_weight_down.weight",
        "model.layers.1.mlp_hyper_connection.input_mix_weight_up.weight",
        "model.layers.1.mlp_hyper_connection.block_inject_weight.weight",
        "model.layers.1.ple.key_proj.weight",
        "model.layers.1.ple.value_proj.weight",
        "model.layers.1.ple.norm_key.weight",
        "model.layers.1.ple.norm_query.weight",
        "model.layers.1.ple.norm_conv.weight",
        "model.layers.1.ple.conv1d.weight",
        "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
    }
    assert expected_layer_1 <= names, sorted(expected_layer_1 - names)

    expected_sparse_layer_3 = {
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
        "model.layers.3.self_attn.o_proj.weight",
        "model.layers.3.self_attn.q_norm.weight",
        "model.layers.3.self_attn.k_norm.weight",
        "model.layers.3.self_attn.indexer.index_qk_proj.weight",
        "model.layers.3.self_attn.indexer.q_layernorm.weight",
        "model.layers.3.self_attn.indexer.k_layernorm.weight",
    }
    assert expected_sparse_layer_3 <= names, sorted(expected_sparse_layer_3 - names)

    expected_root = {
        "model.embed_tokens.weight",
        "model.hyper_connection_mixer.hc_norm.weight",
        "model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.hyper_connection_mixer.input_mix_weight_up.weight",
        "lm_head.weight",
    }
    assert expected_root <= names, sorted(expected_root - names)

    # 最终 mixer 只做 read，没有 write gate；真实 checkpoint 同样没有这个 key
    assert "model.hyper_connection_mixer.block_inject_weight.weight" not in names
    # 这套架构没有 per-layer input/post layernorm，也没有最终 model.norm
    assert not [n for n in names if "input_layernorm" in n or "post_attention_layernorm" in n]
    assert "model.norm.weight" not in names
    assert not [n for n in names if n.startswith("mtp")]
    assert not [n for n in names if "visual" in n]


def test_meta_construction_matches_real_construction() -> None:
    """HpModule 路径在 meta device + fp32 默认 dtype 下建图，必须与实构建结构一致。"""
    real = _tiny_model()
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            meta = Qwen4ExpForCausalLM(_tiny_config())
    finally:
        torch.set_default_dtype(prev_dtype)

    real_sd = real.state_dict()
    meta_sd = meta.state_dict()
    assert set(meta_sd) == set(real_sd)
    mismatched = {k: (meta_sd[k].shape, real_sd[k].shape) for k in real_sd if meta_sd[k].shape != real_sd[k].shape}
    assert not mismatched, mismatched
    assert all(t.is_meta for t in meta_sd.values())
