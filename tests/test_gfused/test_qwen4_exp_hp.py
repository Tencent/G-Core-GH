"""`apply_hp` 的模块替换、expert 切片、激活重算策略和参数校验。

`fully_shard` 本身必须上卡；结构测试走 CPU/meta，数值对拍走 CUDA：
- QSA / EP / Engram 被正确换掉，CP 只 bind vendor layer；
- 换模块不改变数值（真实权重被搬过去）；
- meta 建图下 expert 被切成本 rank 的份额；
- 激活重算跳过 Engram 所在层；
- DSV4 专属的 flag 会明确报错而不是被静默忽略。
"""
import pathlib
from types import SimpleNamespace

import pytest
import torch

from gpatch_v4.models.qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpForCausalLM,
    Qwen4ExpTextConfig,
)
from gpatch_v4.models.qwen4_exp.engram import Qwen4ExpEngramEmbedding
from gpatch_v4.models.qwen4_exp.hp import (
    _bind_cp,
    set_activation_checkpointing,
    shard_experts_for_ep,
    swap_parallel_modules,
)
from gpatch_v4.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextPLELayer,
)
from gpatch_v4.models.qwen4_exp.moe import Qwen4ExpEPExperts
from gpatch_v4.models.qwen4_exp.qsa import Qwen4ExpQSAAttention
from gpatch_v4.training_backend.fsdp2_backend.mixin import Fsdp2EngineMixin

NUM_EXPERTS = 8

TINY_KWARGS = dict(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=8,
    full_attention_interval=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    num_experts=NUM_EXPERTS,
    num_experts_per_tok=2,
    moe_intermediate_size=16,
    shared_expert_intermediate_size=16,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=16,
    indexer_compress_ratio=4,
    hc_count=4,
    hc_lowrank=8,
    ple_layer_ids=[2],
    ple_embed_dim=64,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=512,
    make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=4,
    eos_token_id=1,
    tie_word_embeddings=False,
    use_cache=False,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
)


def _config(**overrides) -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(**{**TINY_KWARGS, **overrides})


def _model(seed: int = 0) -> Qwen4ExpForCausalLM:
    torch.manual_seed(seed)
    model = Qwen4ExpForCausalLM(_config())
    model.eval()
    return model


def test_swap_replaces_only_parallel_leaves_and_binds_cp() -> None:
    model = _model()
    config = model.config.get_text_config()
    before = set(model.state_dict())
    swap_parallel_modules(model, attn_backend="dense")

    assert set(model.state_dict()) == before
    assert all(type(layer) is Qwen4ExpTextDecoderLayer for layer in model.model.layers)
    assert all(layer.cp_size == 1 for layer in model.model.layers)
    sparse_indices = [i for i, t in enumerate(config.layer_types) if t == "qwen_sparse_attention"]
    assert sparse_indices == [3, 7]
    for layer_idx, layer in enumerate(model.model.layers):
        assert isinstance(layer.mlp.experts, Qwen4ExpEPExperts)
        if layer_idx in sparse_indices:
            assert isinstance(layer.self_attn, Qwen4ExpQSAAttention)
        else:
            assert not hasattr(layer, "self_attn")
            assert type(layer.linear_attn) is Qwen4ExpTextGatedDeltaNet
    # Engram 只在 layers.1（ple_layer_ids=[2] 是 1-based）
    assert type(model.model.layers[1].ple) is Qwen4ExpTextPLELayer
    assert isinstance(model.model.layers[1].ple.ple_embedding, Qwen4ExpEngramEmbedding)

    cp_group = object()
    cp_mesh = SimpleNamespace(
        get_group=lambda: cp_group,
        size=lambda: 2,
        get_local_rank=lambda: 1,
    )
    _bind_cp(model, cp_mesh)
    assert model._cp_group is cp_group
    assert model._cp_size == 2
    assert model._cp_rank == 1
    assert model._cp_mesh is cp_mesh
    assert all(layer.cp_size == 2 for layer in model.model.layers)
    with pytest.raises(RuntimeError, match="batch context is missing"):
        model(input_ids=torch.randint(0, config.vocab_size, (1, 20)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_swap_preserves_logits() -> None:
    """替换后数值必须不变——三个替换件都是等价实现（EP 未开启、QSA 用 dense oracle）。"""
    config = _config()
    reference = _model(seed=3).to("cuda")
    swapped = _model(seed=3).to("cuda")
    swap_parallel_modules(swapped, attn_backend="dense")

    input_ids = torch.randint(0, config.vocab_size, (2, 20), device="cuda")
    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        got = swapped(input_ids=input_ids).logits
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=2e-6)


def test_swap_on_meta_model_keeps_shapes() -> None:
    """HpModule 真实路径是 meta 建图 -> swap -> apply_hp -> load_checkpoint_hp。"""
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = Qwen4ExpForCausalLM(_config())
    finally:
        torch.set_default_dtype(prev_dtype)

    reference_shapes = {k: v.shape for k, v in model.state_dict().items()}
    swap_parallel_modules(model, attn_backend="flex")
    swapped = model.state_dict()
    assert set(swapped) == set(reference_shapes)
    assert all(t.is_meta for t in swapped.values())
    for key, shape in reference_shapes.items():
        assert swapped[key].shape == shape, key


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_expert_slicing_on_meta(ep_size: int) -> None:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = Qwen4ExpForCausalLM(_config())
    finally:
        torch.set_default_dtype(prev_dtype)
    swap_parallel_modules(model, attn_backend="flex")

    class _FakeMesh:
        def size(self):
            return ep_size

        def get_group(self):
            return None

    experts = model.model.layers[0].mlp.experts
    shard_experts_for_ep(experts, _FakeMesh())

    num_local = NUM_EXPERTS // ep_size
    assert experts.gate_up_proj.shape[0] == num_local
    assert experts.down_proj.shape[0] == num_local
    # get_group() 是 None，所以 EP 未启用；切片本身与 EP 状态是两回事
    assert experts.ep_size == 1

    # 幂等：重复调用不应再切一次
    shard_experts_for_ep(experts, _FakeMesh())
    assert experts.gate_up_proj.shape[0] == num_local


def test_expert_slicing_rejects_indivisible_ep_size() -> None:
    model = _model()
    swap_parallel_modules(model, attn_backend="dense")

    class _FakeMesh:
        def size(self):
            return 3

        def get_group(self):
            return None

    with pytest.raises(ValueError, match="not divisible"):
        shard_experts_for_ep(model.model.layers[0].mlp.experts, _FakeMesh())


def test_activation_checkpointing_skips_the_engram_layer() -> None:
    model = _model()
    set_activation_checkpointing(model, enabled=True)
    flags = [layer.gradient_checkpointing for layer in model.model.layers]
    # 8 层里只有 layers.1（Engram 所在层）保持 eager
    assert flags == [True, False, True, True, True, True, True, True]
    assert sum(flags) == len(flags) - 1

    set_activation_checkpointing(model, enabled=False)
    assert not any(layer.gradient_checkpointing for layer in model.model.layers)


def test_layer_types_length_mismatch_is_rejected() -> None:
    model = _model()
    model.config.get_text_config().layer_types = ["linear_attention"] * 3
    with pytest.raises(ValueError, match="layer_types"):
        swap_parallel_modules(model, attn_backend="dense")


def test_unknown_layer_type_is_rejected() -> None:
    model = _model()
    text_config = model.config.get_text_config()
    text_config.layer_types = list(text_config.layer_types)
    text_config.layer_types[0] = "sliding_attention"
    with pytest.raises(ValueError, match="unexpected layer_type"):
        swap_parallel_modules(model, attn_backend="dense")


@pytest.mark.parametrize(
    "flag,value",
    [
        ("indexer_backend", "fused"),
        ("fp8", True),
        ("fp8_qat", True),
        ("fp4_qat", True),
        ("fp4_qat_indexer", True),
        ("fsdp_fp8_gather", True),
        ("moe_router_force_load_balancing", True),
        ("dynamic_context_parallel", True),
    ],
)
def test_unsupported_policy_settings_are_rejected_not_ignored(
    flag: str, value: object
) -> None:
    policy_config = SimpleNamespace(
        indexer_backend="eager",
        fp8_qat=False,
        fp4_qat=False,
        fp4_qat_indexer=False,
        fp8=False,
        fsdp_fp8_gather=False,
        moe_router_force_load_balancing=False,
        dist_config=SimpleNamespace(dynamic_context_parallel=False),
    )
    if flag == "dynamic_context_parallel":
        setattr(policy_config.dist_config, flag, value)
    else:
        setattr(policy_config, flag, value)
    engine = SimpleNamespace(
        training_config=SimpleNamespace(enable_mtp=False, enable_dspark=False),
        policy_config=policy_config,
    )
    with pytest.raises(NotImplementedError, match=flag):
        Fsdp2EngineMixin._get_qwen4_exp_hp_model(engine, None, "", False)


RELEASED_CONFIG_DIR = pathlib.Path(
    "/mnt/ceph-hz1-csp/mm-base-plt2/user_edsonzhou/hf-hub/Qwen/Qwen3.8-Flash-Next"
)


@pytest.mark.skipif(
    not (RELEASED_CONFIG_DIR / "config.json").exists(), reason="需要挂载 CephFS 上的真实 checkpoint"
)
def test_released_config_parses_to_expected_architecture() -> None:
    """真实多模态 config 必须解析出发布 checkpoint 的文本架构。"""
    text_config = Qwen4ExpConfig.from_pretrained(RELEASED_CONFIG_DIR).get_text_config()
    assert text_config.hidden_size == 2560
    assert text_config.num_hidden_layers == 48
    assert text_config.vocab_size == 248320
    assert text_config.head_dim == 256
    assert (text_config.num_experts, text_config.num_experts_per_tok) == (512, 10)
    assert (text_config.hc_count, text_config.hc_lowrank) == (4, 320)
    assert text_config.ple_layer_ids == [2]
    assert text_config.ple_embed_dim == 2560
    assert (text_config.ngram_size, text_config.heads_per_ngram) == (3, 8)
    assert text_config.split_ngram_parts == 128
    assert (text_config.indexer_budget, text_config.indexer_compress_ratio) == (2048, 4)
    assert text_config.tie_word_embeddings is False
    sparse = [i for i, t in enumerate(text_config.layer_types) if t == "qwen_sparse_attention"]
    assert sparse == list(range(3, 48, 4))
