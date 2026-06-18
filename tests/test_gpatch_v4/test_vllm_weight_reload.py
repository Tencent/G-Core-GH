# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

import torch
import pytest
from torch import nn
from types import SimpleNamespace

from unittest.mock import MagicMock, patch

from gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload import (
    load_weights_for_update,
)
from gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4 import (
    GCORE_RELOAD_STATE_ATTR,
    _attach_fp8_reload_fallbacks,
    _create_param_from_subclass_attributes,
    _make_uint8_moe_param,
    _restore_mxfp4_fused_moe_module,
    candidate_param_names_for_reflection,
    finalize_weights_after_reload,
    get_module_from_param_name,
    load_checkpoint_weights_for_update,
    loaded_name_candidates,
    map_checkpoint_key_to_vllm_param_name,
    prepare_weights_for_reload,
    restore_quantized_moe_params_for_loading,
)


class DeepseekV4MegaMoEExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_local_experts = 2
        self.intermediate_size = 128
        self.hidden_size = 256
        self._transformed_l1_weights = (torch.zeros(1), torch.zeros(1))
        self._transformed_l2_weights = (torch.zeros(1), torch.zeros(1))
        self.w13_weight = None
        self.w2_weight = None
        self.w13_weight_scale = None
        self.w2_weight_scale = None

    def weight_loader(self, *args, **kwargs) -> bool:
        return True


class _FakeRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = DeepseekV4MegaMoEExperts()


def test_restore_mega_moe_clears_transformed_cache_and_recreates_params() -> None:
    model = _FakeRoot()
    assert restore_quantized_moe_params_for_loading(model) is True

    experts = model.experts
    assert experts._transformed_l1_weights is None
    assert experts._transformed_l2_weights is None
    assert experts.w13_weight is not None
    assert experts.w2_weight is not None
    assert experts.w13_weight_scale is not None
    assert experts.w2_weight_scale is not None
    assert experts.w13_weight.dtype == torch.uint8
    assert experts.w13_weight_scale.quant_method == "block"
    assert experts.w13_weight.weight_loader.__self__ is experts
    assert experts.w13_weight.weight_loader.__func__ is experts.weight_loader.__func__


def test_restore_mega_moe_is_idempotent_when_params_already_exist() -> None:
    model = _FakeRoot()
    restore_quantized_moe_params_for_loading(model)
    w13_ptr = model.experts.w13_weight.data_ptr()

    model.experts._transformed_l1_weights = (torch.zeros(1),)
    restore_quantized_moe_params_for_loading(model)

    assert model.experts._transformed_l1_weights is None
    assert model.experts.w13_weight.data_ptr() == w13_ptr


class _FakeMxfp4QuantMethod:
    def __init__(self) -> None:
        self.num_experts = 4
        self.intermediate_size = 2048
        self.hidden_size = 4096
        self.moe_kernel = object()
        self.moe_quant_config = object()
        self.w13_precision_config = object()
        self.w2_precision_config = object()


class _FakeMxfp4FusedMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant_method = _FakeMxfp4QuantMethod()
        self.w13_weight = nn.Parameter(
            torch.zeros(4, 4096, 1024, dtype=torch.uint8),
            requires_grad=False,
        )

    def weight_loader(self, *args, **kwargs) -> bool:
        return True


class _FakeMappedMxfp4FusedMoe(_FakeMxfp4FusedMoe):
    def __init__(self) -> None:
        super().__init__()
        self._expert_map = torch.tensor([-1, 0, -1, 1], dtype=torch.long)
        self.loader_calls = []

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ) -> bool:
        assert self._expert_map is None
        self.loader_calls.append((weight_name, shard_id, expert_id, return_success))
        return True


def test_restore_mxfp4_fused_moe_recreates_runtime_layout_params() -> None:
    module = _FakeMxfp4FusedMoe()

    _restore_mxfp4_fused_moe_module(module)

    assert tuple(module.w13_weight.shape) == (4, 4096, 2048)
    assert tuple(module.w2_weight.shape) == (4, 4096, 1024)
    assert tuple(module.w13_weight_scale.shape) == (4, 4096, 128)
    assert tuple(module.w2_weight_scale.shape) == (4, 4096, 64)
    assert module.w13_weight.weight_loader.__self__ is module
    assert module.quant_method.moe_kernel is None
    assert module.quant_method.moe_quant_config is None
    assert module.quant_method.w13_precision_config is None
    assert module.quant_method.w2_precision_config is None


def test_restore_mxfp4_fused_moe_uses_cpu_expert_map_for_reload() -> None:
    module = _FakeMappedMxfp4FusedMoe()

    _restore_mxfp4_fused_moe_module(module)
    success = module.w13_weight.weight_loader(
        module.w13_weight,
        torch.zeros(4096, 2048, dtype=torch.uint8),
        "layers.0.ffn.experts.w13_weight",
        shard_id="w1",
        expert_id=3,
        return_success=True,
    )
    skipped = module.w13_weight.weight_loader(
        module.w13_weight,
        torch.zeros(4096, 2048, dtype=torch.uint8),
        "layers.0.ffn.experts.w13_weight",
        shard_id="w1",
        expert_id=2,
        return_success=True,
    )

    assert success is True
    assert skipped is False
    assert module._expert_map.tolist() == [-1, 0, -1, 1]
    assert module.loader_calls == [
        ("layers.0.ffn.experts.w13_weight", "w1", 1, True),
    ]


class _FakeLoadModel(nn.Module):
    def __init__(self, loaded_names: set[str]) -> None:
        super().__init__()
        self.loaded_names = loaded_names
        self.seen_weights = None

    def load_weights(self, weights):
        self.seen_weights = list(weights)
        return self.loaded_names


class _FakeFp8Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.empty(128, 128, dtype=torch.float8_e4m3fn)


class _FakeMoe(nn.Module):
    pass


class _FakeSharedExpertReloadModel(_FakeLoadModel):
    def __init__(self, loaded_names: set[str]) -> None:
        super().__init__(loaded_names)
        self.model = nn.Module()
        layer = nn.Module()
        layer.ffn = nn.Module()
        layer.ffn.shared_experts = nn.Module()
        layer.ffn.shared_experts.gate_up_proj = _FakeFp8Linear()
        layer.ffn.shared_experts.down_proj = _FakeFp8Linear()
        self.model.layers = nn.ModuleList([layer])


def _fake_model_runner(model_type: str, *, expert_dtype: str = "fp4"):
    hf_config = SimpleNamespace(model_type=model_type, expert_dtype=expert_dtype)
    model_config = SimpleNamespace(hf_config=hf_config)
    vllm_config = SimpleNamespace(model_config=model_config)
    return SimpleNamespace(vllm_config=vllm_config)


def test_loaded_name_candidates_accept_checkpoint_scale_alias() -> None:
    candidates = loaded_name_candidates("layers.0.attn.wq_b.scale")

    assert "layers.0.attn.wq_b.weight_scale_inv" in candidates
    assert "model.layers.0.attn.wq_b.scale" in candidates
    assert "model.layers.0.attn.wq_b.weight_scale_inv" in candidates


def test_loaded_name_candidates_accept_global_dsv4_aliases() -> None:
    embed_candidates = loaded_name_candidates("embed.weight")
    head_candidates = loaded_name_candidates("head.weight")

    assert "model.embed_tokens.weight" in embed_candidates
    assert "model.lm_head.weight" in head_candidates


def test_loaded_name_candidates_accept_prefixed_checkpoint_key() -> None:
    candidates = loaded_name_candidates("model.layers.0.attn.wq_b.scale")

    assert "layers.0.attn.wq_b.weight_scale_inv" in candidates
    assert "model.layers.0.attn.wq_b.weight_scale_inv" in candidates
    assert "model.model.layers.0.attn.wq_b.scale" not in candidates


def test_loaded_name_candidates_accept_fused_wqa_wkv_alias() -> None:
    wq_a_candidates = loaded_name_candidates("layers.0.attn.wq_a.weight")
    wkv_candidates = loaded_name_candidates("layers.0.attn.wkv.weight")

    assert "model.layers.0.attn.fused_wqa_wkv.weight" in wq_a_candidates
    assert "model.layers.0.attn.fused_wqa_wkv.weight" in wkv_candidates


def test_loaded_name_candidates_accept_fused_wqa_wkv_scale_alias() -> None:
    wq_a_candidates = loaded_name_candidates("layers.0.attn.wq_a.scale")
    wkv_candidates = loaded_name_candidates("layers.0.attn.wkv.scale")

    expected = "model.layers.0.attn.fused_wqa_wkv.weight_scale_inv"
    assert expected in wq_a_candidates
    assert expected in wkv_candidates


def test_loaded_name_candidates_accept_compressor_aliases() -> None:
    wkv_candidates = loaded_name_candidates("layers.2.attn.compressor.wkv.weight")
    wgate_candidates = loaded_name_candidates("layers.2.attn.compressor.wgate.weight")
    norm_candidates = loaded_name_candidates("layers.2.attn.compressor.norm.weight")
    indexer_wkv_candidates = loaded_name_candidates(
        "layers.2.attn.indexer.compressor.wkv.weight"
    )
    indexer_wgate_candidates = loaded_name_candidates(
        "layers.2.attn.indexer.compressor.wgate.weight"
    )
    compressor_ape_candidates = loaded_name_candidates("layers.2.attn.compressor.ape")

    fused = "model.layers.2.attn.mla_attn.compressor.fused_wkv_wgate.weight"
    assert fused in wkv_candidates
    assert fused in wgate_candidates
    assert "model.layers.2.attn.mla_attn.compressor.norm.weight" in norm_candidates
    assert "model.layers.2.attn.mla_attn.compressor.ape" in compressor_ape_candidates

    indexer_fused = "model.layers.2.attn.indexer.compressor.fused_wkv_wgate.weight"
    assert indexer_fused in indexer_wkv_candidates
    assert indexer_fused in indexer_wgate_candidates


def test_reflection_candidates_split_expert_w13_and_w2_aliases() -> None:
    model = nn.Module()

    w1_candidates = candidate_param_names_for_reflection(
        model, "layers.0.ffn.experts.3.w1.weight"
    )
    w2_candidates = candidate_param_names_for_reflection(
        model, "layers.0.ffn.experts.3.w2.weight"
    )
    w3_candidates = candidate_param_names_for_reflection(
        model, "layers.0.ffn.experts.3.w3.weight"
    )

    assert "model.layers.0.ffn.experts.w13_weight" in w1_candidates
    assert "model.layers.0.ffn.experts.w2_weight" in w2_candidates
    assert "model.layers.0.ffn.experts.w13_weight" in w3_candidates


def test_reflection_candidates_map_shared_expert_gate_up_aliases() -> None:
    model = nn.Module()

    w1_candidates = candidate_param_names_for_reflection(
        model, "layers.0.ffn.shared_experts.w1.weight"
    )
    w3_candidates = candidate_param_names_for_reflection(
        model, "layers.0.ffn.shared_experts.w3.weight"
    )
    w2_candidates = candidate_param_names_for_reflection(
        model, "layers.0.ffn.shared_experts.w2.weight"
    )
    w1_scale_candidates = candidate_param_names_for_reflection(
        model, "layers.0.ffn.shared_experts.w1.scale"
    )

    assert "model.layers.0.ffn.shared_experts.gate_up_proj.weight" in w1_candidates
    assert "model.layers.0.ffn.shared_experts.gate_up_proj.weight" in w3_candidates
    assert "model.layers.0.ffn.shared_experts.down_proj.weight" in w2_candidates
    assert (
        "model.layers.0.ffn.shared_experts.gate_up_proj.weight_scale_inv"
        in w1_scale_candidates
    )


def test_loader_mapping_preserves_shared_expert_gate_up_shard_names() -> None:
    model = nn.Module()

    w1_mapped = map_checkpoint_key_to_vllm_param_name(
        model, "layers.0.ffn.shared_experts.w1.scale"
    )
    w3_mapped = map_checkpoint_key_to_vllm_param_name(
        model, "layers.0.ffn.shared_experts.w3.scale"
    )

    assert w1_mapped == "model.layers.0.ffn.shared_experts.w1.weight_scale_inv"
    assert w3_mapped == "model.layers.0.ffn.shared_experts.w3.weight_scale_inv"


def test_module_for_reload_key_resolves_shared_expert_gate_up_alias() -> None:
    model = _FakeSharedExpertReloadModel(set())

    module = get_module_from_param_name(model, "layers.0.ffn.shared_experts.w1.weight")

    assert module is model.model.layers[0].ffn.shared_experts.gate_up_proj


def test_load_checkpoint_weights_for_update_accepts_strict_attention_key() -> None:
    model = _FakeLoadModel({"layers.0.attn.wq_b.weight"})
    weights = [("layers.0.attn.wq_b.weight", torch.zeros(2, 2))]

    loaded = load_checkpoint_weights_for_update(model, weights)

    assert loaded == {"layers.0.attn.wq_b.weight"}
    assert model.seen_weights == weights


def test_load_weights_for_update_quantizes_deepseek_v4_fp8_linear_by_model(
    monkeypatch,
) -> None:
    model = _FakeLoadModel({
        "layers.0.attn.wq_b.weight",
        "layers.0.attn.wq_b.scale",
    })
    runner = _fake_model_runner("deepseek_v4", expert_dtype="fp4")
    weights = [("layers.0.attn.wq_b.weight", torch.zeros(128, 128, dtype=torch.bfloat16))]
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4.get_module_from_param_name",
        lambda _model, _name: _FakeFp8Linear(),
    )
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4._is_vllm_linear_module",
        lambda module: isinstance(module, _FakeFp8Linear),
    )

    loaded = load_weights_for_update(model, runner, weights)

    assert loaded == {"layers.0.attn.wq_b.weight", "layers.0.attn.wq_b.scale"}
    seen = dict(model.seen_weights)
    assert seen["layers.0.attn.wq_b.weight"].dtype == torch.float8_e4m3fn
    assert seen["layers.0.attn.wq_b.scale"].dtype == torch.float8_e8m0fnu


def test_load_weights_for_update_quantizes_shared_expert_gate_up_by_reflection(
    monkeypatch,
) -> None:
    model = _FakeSharedExpertReloadModel({
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.0.ffn.shared_experts.w1.scale",
        "layers.0.ffn.shared_experts.w3.weight",
        "layers.0.ffn.shared_experts.w3.scale",
    })
    runner = _fake_model_runner("deepseek_v4", expert_dtype="fp4")
    weights = [
        ("layers.0.ffn.shared_experts.w1.weight", torch.zeros(128, 128, dtype=torch.bfloat16)),
        ("layers.0.ffn.shared_experts.w3.weight", torch.zeros(128, 128, dtype=torch.bfloat16)),
    ]
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4._is_vllm_linear_module",
        lambda module: isinstance(module, _FakeFp8Linear),
    )

    loaded = load_weights_for_update(model, runner, weights)

    assert loaded == {
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.0.ffn.shared_experts.w1.scale",
        "layers.0.ffn.shared_experts.w3.weight",
        "layers.0.ffn.shared_experts.w3.scale",
    }
    seen = dict(model.seen_weights)
    assert seen["layers.0.ffn.shared_experts.w1.weight"].dtype == torch.float8_e4m3fn
    assert seen["layers.0.ffn.shared_experts.w1.scale"].dtype == torch.float8_e8m0fnu
    assert seen["layers.0.ffn.shared_experts.w3.weight"].dtype == torch.float8_e4m3fn
    assert seen["layers.0.ffn.shared_experts.w3.scale"].dtype == torch.float8_e8m0fnu


def test_load_weights_for_update_quantizes_deepseek_v4_moe_by_model(
    monkeypatch,
) -> None:
    model = _FakeLoadModel({
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w1.scale",
    })
    runner = _fake_model_runner("deepseek_v4", expert_dtype="fp4")
    weights = [
        ("layers.0.ffn.experts.0.w1.weight", torch.zeros(128, 64, dtype=torch.bfloat16))
    ]
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4.get_module_from_param_name",
        lambda _model, _name: _FakeMoe(),
    )
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4._is_moe_module_for_reload",
        lambda module: isinstance(module, _FakeMoe),
    )

    loaded = load_weights_for_update(model, runner, weights)

    assert loaded == {
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w1.scale",
    }
    seen = dict(model.seen_weights)
    assert seen["layers.0.ffn.experts.0.w1.weight"].dtype == torch.int8
    assert seen["layers.0.ffn.experts.0.w1.scale"].dtype == torch.float8_e8m0fnu


def test_load_weights_for_update_quantizes_deepseek_v4_expert_without_module(
    monkeypatch,
) -> None:
    model = _FakeLoadModel({
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w1.scale",
    })
    runner = _fake_model_runner("deepseek_v4", expert_dtype="fp4")
    weights = [
        ("layers.0.ffn.experts.0.w1.weight", torch.zeros(128, 64, dtype=torch.bfloat16))
    ]
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4.get_module_from_param_name",
        lambda _model, _name: None,
    )

    load_weights_for_update(model, runner, weights)

    seen = dict(model.seen_weights)
    assert seen["layers.0.ffn.experts.0.w1.weight"].dtype == torch.int8
    assert tuple(seen["layers.0.ffn.experts.0.w1.weight"].shape) == (128, 32)


def test_load_weights_for_update_detects_deepseek_v4_from_model(
    monkeypatch,
) -> None:
    class DeepseekV4ForCausalLM(_FakeLoadModel):
        pass

    model = DeepseekV4ForCausalLM({
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w1.scale",
    })
    runner = _fake_model_runner("unknown", expert_dtype="fp4")
    weights = [
        ("layers.0.ffn.experts.0.w1.weight", torch.zeros(128, 64, dtype=torch.bfloat16))
    ]
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4.get_module_from_param_name",
        lambda _model, _name: _FakeMoe(),
    )
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4._is_moe_module_for_reload",
        lambda module: isinstance(module, _FakeMoe),
    )

    load_weights_for_update(model, runner, weights)

    seen = dict(model.seen_weights)
    assert seen["layers.0.ffn.experts.0.w1.weight"].dtype == torch.int8


def test_load_weights_for_update_rejects_deepseek_v4_scale_input() -> None:
    model = _FakeLoadModel({"layers.0.attn.wq_b.scale"})
    runner = _fake_model_runner("deepseek_v4", expert_dtype="fp4")
    weights = [("layers.0.attn.wq_b.scale", torch.zeros(1, 1))]

    with pytest.raises(RuntimeError, match="expects bf16 weight keys"):
        load_weights_for_update(model, runner, weights)


def test_load_weights_for_update_keeps_non_deepseek_v4_path() -> None:
    model = _FakeLoadModel({"plain.weight"})
    runner = _fake_model_runner("qwen3")
    weights = [("plain.weight", torch.zeros(2, 2))]

    loaded = load_weights_for_update(model, runner, weights)

    assert loaded == {"plain.weight"}
    assert model.seen_weights == weights


def test_load_checkpoint_weights_for_update_accepts_scale_alias() -> None:
    model = _FakeLoadModel({"layers.0.attn.wq_b.weight_scale_inv"})
    weights = [("layers.0.attn.wq_b.scale", torch.zeros(1, 1))]

    loaded = load_checkpoint_weights_for_update(model, weights)

    assert loaded == {"layers.0.attn.wq_b.weight_scale_inv"}


def test_load_checkpoint_weights_for_update_accepts_mapped_scale_alias() -> None:
    model = _FakeLoadModel({"model.layers.0.attn.wq_b.weight_scale_inv"})
    weights = [("layers.0.attn.wq_b.scale", torch.zeros(1, 1))]

    loaded = load_checkpoint_weights_for_update(model, weights)

    assert loaded == {"model.layers.0.attn.wq_b.weight_scale_inv"}


def test_load_checkpoint_weights_for_update_accepts_fused_wqa_wkv_alias() -> None:
    model = _FakeLoadModel({"model.layers.0.attn.fused_wqa_wkv.weight"})
    weights = [
        ("layers.0.attn.wq_a.weight", torch.zeros(2, 2)),
        ("layers.0.attn.wkv.weight", torch.zeros(2, 2)),
    ]

    loaded = load_checkpoint_weights_for_update(model, weights)

    assert loaded == {"model.layers.0.attn.fused_wqa_wkv.weight"}


def test_load_checkpoint_weights_for_update_accepts_fused_wqa_wkv_scale_alias() -> None:
    model = _FakeLoadModel({"model.layers.0.attn.fused_wqa_wkv.weight_scale_inv"})
    weights = [
        ("layers.0.attn.wq_a.scale", torch.zeros(1, 1)),
        ("layers.0.attn.wkv.scale", torch.zeros(1, 1)),
    ]

    loaded = load_checkpoint_weights_for_update(model, weights)

    assert loaded == {"model.layers.0.attn.fused_wqa_wkv.weight_scale_inv"}


def test_load_checkpoint_weights_for_update_accepts_compressor_aliases() -> None:
    loaded_names = {
        "model.layers.2.attn.mla_attn.compressor.fused_wkv_wgate.weight",
        "model.layers.2.attn.mla_attn.compressor.norm.weight",
        "model.layers.2.attn.indexer.compressor.fused_wkv_wgate.weight",
    }
    model = _FakeLoadModel(loaded_names)
    weights = [
        ("layers.2.attn.compressor.wkv.weight", torch.zeros(2, 2)),
        ("layers.2.attn.compressor.wgate.weight", torch.zeros(2, 2)),
        ("layers.2.attn.compressor.norm.weight", torch.zeros(2)),
        ("layers.2.attn.indexer.compressor.wkv.weight", torch.zeros(2, 2)),
        ("layers.2.attn.indexer.compressor.wgate.weight", torch.zeros(2, 2)),
    ]

    loaded = load_checkpoint_weights_for_update(model, weights)

    assert loaded == loaded_names


def test_load_checkpoint_weights_for_update_ignores_mtp_keys() -> None:
    model = _FakeLoadModel({"layers.0.attn.wkv.weight"})
    weights = [("mtp.0.attn.wq_b.weight", torch.zeros(2, 2))]

    loaded = load_checkpoint_weights_for_update(model, weights)

    assert loaded == {"layers.0.attn.wkv.weight"}


def test_prepare_weights_for_reload_skips_layerwise(monkeypatch) -> None:
    model = nn.Module()
    device = torch.device("cpu")
    layerwise_init = MagicMock()
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4.restore_quantized_moe_params_for_loading",
        lambda _model: False,
    )
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4.patch_vllm_moe_model_weight_loader",
        lambda _model: None,
    )
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4._detect_mxfp4_moe_present",
        lambda _model: False,
    )
    ensure_reloadable = MagicMock()
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4._ensure_model_params_reloadable",
        ensure_reloadable,
    )
    with patch(
        "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
        layerwise_init,
    ):
        reload_state = prepare_weights_for_reload(model, device)

    layerwise_init.assert_not_called()
    ensure_reloadable.assert_called_once_with(model)
    assert reload_state == {"is_mxfp4_moe": False}
    assert getattr(model, GCORE_RELOAD_STATE_ATTR) == reload_state


def test_attach_fp8_reload_fallbacks_adds_merged_loader_for_plain_parameter() -> None:
    class _FakeSubclass:
        pass

    param = nn.Parameter(torch.zeros(2, 4), requires_grad=False)
    param.subclass_type = _FakeSubclass
    _attach_fp8_reload_fallbacks(param)

    loaded = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    param.load_merged_column_weight(loaded)

    assert torch.equal(param.data, loaded)


def test_create_param_from_subclass_attributes_keeps_quant_method() -> None:
    class _CustomParam(nn.Parameter):
        pass

    custom = _CustomParam(torch.zeros(2, 2), requires_grad=False)
    custom.quant_method = "block"
    custom.custom_flag = 7
    source = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    source.tp_rank = 1

    rebuilt = _create_param_from_subclass_attributes(custom, source)

    assert rebuilt.subclass_type is _CustomParam
    assert getattr(rebuilt, "quant_method", None) == "block"
    assert getattr(rebuilt, "custom_flag", None) == 7
    assert getattr(rebuilt, "tp_rank", None) == 1


def test_make_uint8_moe_param_keeps_source_attrs_and_overrides_quant_method() -> None:
    source = nn.Parameter(torch.zeros(1, dtype=torch.uint8), requires_grad=False)
    source.quant_method = "legacy"
    source.tp_rank = 3
    source.custom_attr = "ok"

    param = _make_uint8_moe_param(
        (2, 2),
        torch.device("cpu"),
        lambda *args, **kwargs: True,
        source_param=source,
        quant_method="block",
    )

    assert param.dtype == torch.uint8
    assert getattr(param, "custom_attr", None) == "ok"
    assert getattr(param, "tp_rank", None) == 3
    assert getattr(param, "quant_method", None) == "block"


def test_make_uint8_moe_param_does_not_overwrite_existing_weight_loader_attr() -> None:
    original_loader = lambda *args, **kwargs: False
    source = nn.Parameter(torch.zeros(1, dtype=torch.uint8), requires_grad=False)
    source.weight_loader = original_loader
    source.custom_attr = "keep"
    new_loader = lambda *args, **kwargs: True

    param = _make_uint8_moe_param(
        (1, 1),
        torch.device("cpu"),
        new_loader,
        source_param=source,
        quant_method=None,
    )

    assert getattr(param, "custom_attr", None) == "keep"
    assert getattr(param, "weight_loader", None) is new_loader


def test_finalize_weights_after_reload_verl_style_not_layerwise(
    monkeypatch,
) -> None:
    model = nn.Module()
    model_config = MagicMock()
    device = torch.device("cpu")
    setattr(model, GCORE_RELOAD_STATE_ATTR, {"is_mxfp4_moe": False})

    calls: list[str] = []
    layerwise_finalize = MagicMock()
    vllm_process = MagicMock()
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4._finalize_mega_moe_weights",
        lambda _model: calls.append("mega"),
    )
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4.reload_cached_deepseek_v4_dense_fp8_scales",
        lambda _model: calls.append("scales"),
    )
    monkeypatch.setattr(
        "gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4._validate_fp8_attention_modules_after_reload",
        lambda _model: calls.append("validate"),
    )

    with patch(
        "vllm.model_executor.model_loader.reload.finalize_layerwise_reload",
        layerwise_finalize,
    ), patch(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        vllm_process,
    ):
        finalize_weights_after_reload(model, model_config, device)

    layerwise_finalize.assert_not_called()
    vllm_process.assert_not_called()
    assert calls == ["mega", "scales", "validate"]
    assert not hasattr(model, GCORE_RELOAD_STATE_ATTR)


def test_load_checkpoint_weights_for_update_defers_mega_moe_finalize() -> None:
    class _DeferModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kwargs = None

        def load_weights(self, weights, *, defer_mega_moe_finalize=False):
            self.kwargs = {"defer_mega_moe_finalize": defer_mega_moe_finalize}
            return {name for name, _ in weights}

    model = _DeferModel()
    weights = [("layers.0.attn.wkv.weight", torch.zeros(2, 2))]

    load_checkpoint_weights_for_update(model, weights)

    assert model.kwargs == {"defer_mega_moe_finalize": True}


def test_deepseek_v4_load_weights_defers_mega_moe_finalize(monkeypatch) -> None:
    from vllm.model_executor.models.deepseek_v4 import DeepseekV4ForCausalLM

    class _FakeLoader:
        def __init__(self, model, **kwargs) -> None:
            self.model = model
            self.kwargs = kwargs

        def load_weights(self, weights, mapper=None):
            self.model.loader_calls.append((list(weights), mapper, self.kwargs))
            return {"loaded.weight"}

    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_v4.AutoWeightsLoader",
        _FakeLoader,
    )
    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    model.model = SimpleNamespace(finalize_mega_moe_weights=MagicMock())
    model.hf_to_vllm_mapper = object()
    model.loader_calls = []
    weights = [("loaded.weight", torch.zeros(1))]

    loaded = model.load_weights(weights, defer_mega_moe_finalize=True)

    assert loaded == {"loaded.weight"}
    model.model.finalize_mega_moe_weights.assert_not_called()
    assert model.loader_calls == [(weights, model.hf_to_vllm_mapper, {"skip_substrs": ["mtp."]})]

    model.load_weights(weights)

    model.model.finalize_mega_moe_weights.assert_called_once_with()


def test_deepseek_v4_mtp_load_weights_accepts_defer_flag() -> None:
    from vllm.model_executor.models.deepseek_v4_mtp import DeepSeekV4MTP

    param = inspect.signature(DeepSeekV4MTP.load_weights).parameters[
        "defer_mega_moe_finalize"
    ]

    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False


def test_load_checkpoint_weights_for_update_rejects_missing_strict_key() -> None:
    model = _FakeLoadModel({"layers.0.attn.wkv.weight"})
    weights = [("layers.0.attn.wq_b.weight", torch.zeros(2, 2))]

    with pytest.raises(RuntimeError, match="strict GCore reload keys"):
        load_checkpoint_weights_for_update(model, weights)
