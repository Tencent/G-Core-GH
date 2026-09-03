"""引擎接线的测试：注册表、attention mask 约定、激活重算策略、task yaml 自洽性。

最关键的一条是 mask 约定：gcore 默认的 SFT data prep 产出的是 **Megatron 约定**的 4D
mask（True = 不能看），和 HF 正好相反，而 `create_causal_mask` 对已经是 4D 的 mask 会
原样返回。传错了不会崩，只会 loss 不对。
"""
import pathlib
from types import SimpleNamespace

import pytest
import torch
import yaml

from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.extended_model import (
    REGISTER_POST_INIT_MODEL,
    REGISTER_SFT_PREPARE_DATA_FORWARD,
)
from gpatch_v4.extended_model.llm import PrepareDataForwardLLM
from gpatch_v4.extended_model.qwen4_exp import (
    Qwen4ExpPostInitModel,
    Qwen4ExpPrepareDataForwardLLM,
)
from gpatch_v4.models.qwen4_exp import (
    Qwen4ExpHpForCausalLM,
    Qwen4ExpTextConfig,
)
from gpatch_v4.training_backend.fsdp2_backend.lr_scheduler import FSDPLRScheduler
from gpatch_v4.training_backend.fsdp2_backend.optimizer import setup_optimizer
from tasks.qwen3_8_next import finetune_dataset

TASK_DIR = pathlib.Path(__file__).resolve().parents[2] / "tasks" / "qwen3_8_next"

TINY_KWARGS = dict(
    vocab_size=128,
    hidden_size=32,
    num_hidden_layers=8,
    full_attention_interval=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=8,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=16,
    indexer_compress_ratio=4,
    hc_count=4,
    hc_lowrank=8,
    ple_layer_ids=[2],
    ple_embed_dim=32,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=256,
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


def _model() -> Qwen4ExpHpForCausalLM:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(0)
        model = Qwen4ExpHpForCausalLM(Qwen4ExpTextConfig(**TINY_KWARGS))
    finally:
        torch.set_default_dtype(prev)
    return model


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


def test_registered_in_the_sft_and_post_init_registries() -> None:
    assert REGISTER_SFT_PREPARE_DATA_FORWARD[MODEL_ARCH.QWEN4_EXP] is (
        Qwen4ExpPrepareDataForwardLLM
    )
    assert REGISTER_POST_INIT_MODEL[MODEL_ARCH.QWEN4_EXP] is Qwen4ExpPostInitModel
    assert MODEL_ARCH.QWEN4_EXP == "qwen4_exp"
    assert Qwen4ExpTextConfig.model_type == "qwen4_exp_text"


def test_transformers_registers_qwen4_exp_configs() -> None:
    """`AutoConfig` 必须认识模型的两个 model_type。

    引擎的 `Fsdp2EngineLm.__init__` 里有一句通用的
    `AutoConfig.from_pretrained(hf_model_path)`，它在任何 arch 分派之前执行。
    """
    from transformers import AutoConfig

    assert AutoConfig.for_model("qwen4_exp").model_type == "qwen4_exp"
    assert AutoConfig.for_model("qwen4_exp_text").model_type == "qwen4_exp_text"


# ---------------------------------------------------------------------------
# attention mask 约定
# ---------------------------------------------------------------------------


def test_padding_mask_is_hf_convention() -> None:
    """1 = 真实 token，0 = 右侧 padding；长度取自未 padding 的 tokens。"""
    batches = [
        {"tokens": torch.arange(5)},
        {"tokens": torch.arange(2)},
        {"tokens": torch.arange(20)},
    ]
    mask = Qwen4ExpPrepareDataForwardLLM._hf_padding_mask(batches, 8, torch.device("cpu"))
    assert mask.shape == (3, 8)
    assert mask[0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert mask[1].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert mask[2].tolist() == [1] * 8


def test_data_prep_does_not_build_the_generic_4d_mask(monkeypatch) -> None:
    from megatron.core import mpu

    seen = {}

    def base_sft_train(
        _self,
        batches,
        seq_len,
        _pad_token_id,
        comput_attn_mask=True,
        **_kwargs,
    ):
        seen["comput_attn_mask"] = comput_attn_mask
        return {}, {"input_ids": torch.zeros(len(batches), seq_len, dtype=torch.long)}

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(PrepareDataForwardLLM, "sft_train", base_sft_train)
    prep = Qwen4ExpPrepareDataForwardLLM.__new__(Qwen4ExpPrepareDataForwardLLM)
    prep.config = SimpleNamespace(policy=SimpleNamespace(ppo_pack_seq=False))
    batches = [{"tokens": torch.arange(3)}]

    _, fwd_kwargs = prep.sft_train(batches, 8, 0, comput_attn_mask=True)

    assert seen["comput_attn_mask"] is False
    assert fwd_kwargs["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0, 0, 0]]


def test_thd_data_prep_shifts_each_document_before_packing(monkeypatch) -> None:
    from megatron.core import mpu

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_context_parallel_group", lambda: None)

    first_tokens = torch.arange(10, 16)
    first_labels = first_tokens.clone()
    first_labels[:2] = -100
    second_tokens = torch.arange(30, 33)
    second_labels = second_tokens.clone()
    second_labels[0] = -100
    prep = Qwen4ExpPrepareDataForwardLLM.__new__(Qwen4ExpPrepareDataForwardLLM)
    prep._pad_each_doc_to_multi_of = 4
    batch, fwd_kwargs = prep._sft_train_thd(
        [
            {"tokens": first_tokens, "labels": first_labels},
            {"tokens": second_tokens, "labels": second_labels},
        ],
        seq_len=16,
        pad_token_id=0,
    )

    params = batch["full_packed_seq_params"]
    assert params.cu_seqlens_q.tolist() == [0, 5, 7]
    assert params.cu_seqlens_q_padded.tolist() == [0, 5, 8]
    assert fwd_kwargs["input_ids"].tolist() == [[10, 11, 12, 13, 14, 30, 31, 0]]
    assert fwd_kwargs["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 12, 13, 14, 15, 31, 32, -100]]
    assert fwd_kwargs["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1, 0]]
    assert fwd_kwargs["_qwen4_exp_cp_context"].global_cu_seqlens.tolist() == [0, 5, 7]


# ---------------------------------------------------------------------------
# 激活重算策略
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recompute", [True, False])
def test_post_init_hook_keeps_engram_layer_eager(recompute: bool) -> None:
    model = _model()
    # 模拟引擎：先全开，再由 hook 修正
    for layer in model.model.layers:
        layer.gradient_checkpointing = recompute

    config = SimpleNamespace(training=SimpleNamespace(recompute=recompute))
    Qwen4ExpPostInitModel(config)(model)

    flags = [layer.gradient_checkpointing for layer in model.model.layers]
    if recompute:
        # ple_layer_ids=[2] 是 1-based -> layers.1 必须保持 eager
        assert flags == [True, False] + [True] * 6
    else:
        assert not any(flags)


# ---------------------------------------------------------------------------
# task 配置自洽性
# ---------------------------------------------------------------------------


def test_task_yaml_is_self_consistent() -> None:
    config = yaml.safe_load((TASK_DIR / "yaml" / "sft_fsdp2.yaml").read_text())

    assert config["policy"]["model_arch"] == MODEL_ARCH.QWEN4_EXP
    assert config["training"]["training_backend"] == "fsdp2"
    # TP/PP 仍不支持；CP1 是默认基线，CP 实验通过 Hydra 覆盖。
    assert config["policy"]["dist_config"]["tensor_model_parallel_size"] == 1
    assert config["policy"]["dist_config"]["pipeline_model_parallel_size"] == 1
    assert config["policy"]["dist_config"]["context_parallel_size"] == 1
    assert config["policy"]["ep_backend"] == "eager"
    assert config["training"]["enable_mtp"] is False
    assert config["training"]["enable_dspark"] is False
    # 没有 ref model，否则会多出一份 352 GB
    assert config["policy"]["without_ref"] is True
    # Qwen hook 直接构造 2D padding mask，不应再创建通用的 4D Megatron mask
    assert config["training"]["comput_attn_mask"] is False
    # FSDP2 的 muon 在 gcore 里是禁用的
    assert config["optimizer"]["optimizer_type"] == "adamw"
    assert config["debug"]["disable_save_checkpoint"] is False
    assert "debug_truncate_num_hidden_layers" not in config["debug"]
    # EP 必须整除 512 个 expert
    assert 512 % config["policy"]["dist_config"]["expert_model_parallel_size"] == 0
    # 不能出现 DSV4 专属开关，apply_hp 会直接报错
    for forbidden in ("fp8", "fp8_qat", "fp4_qat", "fp4_qat_indexer", "indexer_backend"):
        assert forbidden not in config["policy"], forbidden
    # Ray actor 的 cwd 不固定，相对 HF 路径会被误解析成 Hub repo id。
    assert config["policy"]["hf_model_path"].startswith("/")
    assert config["policy"]["hf_tokenizer_path"].startswith("/")
    for path in config["data"]["data_pathes"]:
        assert path.startswith("/"), path


def test_task_script_sets_nccl_env_and_points_at_the_yaml() -> None:
    script = (TASK_DIR / "scripts" / "fsdp2_sft.sh").read_text()
    # 参考实现里 NCCL 的 network plugin 会破坏跨节点 Engram all-to-all
    assert "NCCL_NET_PLUGIN=none" in script
    assert "NCCL_NET=IB" in script
    assert "sft_fsdp2.yaml" in script
    assert "train_lm_finetune.py" in script


def test_task_dataset_requires_chat_template() -> None:
    tokenizer = SimpleNamespace(chat_template=None)
    with pytest.raises(ValueError, match="chat_template"):
        finetune_dataset.SimpleDataset(SimpleNamespace(), tokenizer)


def test_task_dataset_masks_from_the_full_tokenization() -> None:
    calls = []

    class Tokenizer:
        pad_token = "<pad>"

        def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
            calls.append(text)
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            return SimpleNamespace(
                input_ids=[20, 21, 22],
                offset_mapping=[(0, 3), (3, 6), (6, 8)],
            )

    input_ids, labels, seq_length, prompt_len = finetune_dataset.tokenize_text(
        Tokenizer(),
        prompt="abcd",
        full_text="abcdEFGH",
    )

    assert calls == ["abcdEFGH"]
    assert input_ids == [20, 21, 22]
    assert labels == [-100, 21, 22]
    assert seq_length == 3
    assert prompt_len == 1


def test_task_dataset_renders_prompt_and_target_in_the_same_thinking_mode() -> None:
    calls = []

    def apply_chat_template(messages, **kwargs):
        calls.append(kwargs)
        return "prompt" if kwargs["add_generation_prompt"] else "prompt answer"

    dataset = finetune_dataset.SimpleDataset.__new__(finetune_dataset.SimpleDataset)
    dataset.tokenizer = SimpleNamespace(apply_chat_template=apply_chat_template)
    dataset.system_prompt = "system"
    dataset.config = SimpleNamespace(
        training=SimpleNamespace(enable_thinking=False),
        data=SimpleNamespace(custom_add_eos=False),
    )

    assert dataset._apply_chat_template("question", "answer") == (
        "prompt",
        "prompt answer",
    )
    assert [call["enable_thinking"] for call in calls] == [False, False]


def test_task_dataset_loads_all_configured_directories(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_file = first / "b.jsonl"
    second_file = second / "a.jsonl"
    first_file.touch()
    second_file.touch()
    captured = {}

    def fake_load_dataset(format_name, *, data_files, split):
        captured.update(format_name=format_name, data_files=data_files, split=split)
        return []

    monkeypatch.setattr(finetune_dataset, "load_dataset", fake_load_dataset)
    config = SimpleNamespace(
        data=SimpleNamespace(data_pathes=[str(second), str(first)], system_prompt=None)
    )
    tokenizer = SimpleNamespace(chat_template="qwen template")

    finetune_dataset.SimpleDataset(config, tokenizer)

    assert captured == {
        "format_name": "json",
        "data_files": sorted([str(first_file), str(second_file)]),
        "split": "train",
    }


@pytest.mark.parametrize("optimizer_type", ["adam", "adamw"])
def test_engram_uses_reference_optimizer_multipliers(optimizer_type: str) -> None:
    model = torch.nn.Module()
    model.base = torch.nn.Parameter(torch.ones(()))
    model.engram = torch.nn.Parameter(torch.ones(()))
    model._engram_param_ids = {id(model.engram)}
    config = SimpleNamespace(
        policy=SimpleNamespace(model_arch=MODEL_ARCH.QWEN4_EXP),
        optimizer=SimpleNamespace(
            optimizer_type=optimizer_type,
            lr=1.0e-5,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=0.1,
            adam_epsilon=1.0e-8,
        ),
    )

    optimizer = setup_optimizer(config, model)
    assert len(optimizer.param_groups) == 2
    base_group, engram_group = optimizer.param_groups
    assert base_group["params"] == [model.base]
    assert engram_group["params"] == [model.engram]
    assert engram_group["weight_decay"] == 0.0
    assert engram_group["lr_mult"] == 5.0

    scheduler = FSDPLRScheduler(
        optimizer,
        init_lr=2.0e-6,
        max_lr=1.0e-5,
        min_lr=1.0e-6,
        lr_warmup_steps=10,
        lr_decay_steps=100,
        lr_decay_style="cosine",
    )
    scheduler.last_epoch = 5
    base_lr, engram_lr = scheduler.get_lr()
    assert engram_lr == pytest.approx(5.0 * base_lr)
