# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

_WORKSPACE_VLLM = Path(__file__).resolve().parents[3] / "vllm"
if _WORKSPACE_VLLM.exists():
    sys.path.insert(0, str(_WORKSPACE_VLLM))


def test_deepseek_v4_loads_finalized_shared_expert_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _WORKSPACE_VLLM.exists():
        import vllm
        import vllm.model_executor
        import vllm.model_executor.layers
        import vllm.model_executor.models

        vllm.__path__.insert(0, str(_WORKSPACE_VLLM / "vllm"))
        vllm.model_executor.__path__.insert(
            0, str(_WORKSPACE_VLLM / "vllm" / "model_executor")
        )
        vllm.model_executor.layers.__path__.insert(
            0, str(_WORKSPACE_VLLM / "vllm" / "model_executor" / "layers")
        )
        vllm.model_executor.models.__path__.insert(
            0, str(_WORKSPACE_VLLM / "vllm" / "model_executor" / "models")
        )
        sys.modules.pop("vllm.model_executor.models.deepseek_v4", None)

    fake_attention = ModuleType("vllm.model_executor.layers.deepseek_v4_attention")
    fake_attention.DeepseekV4Indexer = nn.Module
    fake_attention.DeepseekV4MLAModules = nn.Module
    fake_attention.DeepseekV4MultiHeadLatentAttentionWrapper = nn.Module
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.deepseek_v4_attention", fake_attention
    )
    fake_fp8 = ModuleType("vllm.model_executor.layers.quantization.fp8")
    fake_fp8.Fp8Config = object
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.quantization.fp8", fake_fp8)
    fake_mxfp4 = ModuleType("vllm.model_executor.layers.quantization.mxfp4")
    fake_mxfp4.Mxfp4MoEMethod = object
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.quantization.mxfp4", fake_mxfp4
    )
    fake_weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")

    def _default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    fake_weight_utils.default_weight_loader = _default_weight_loader
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.model_loader.weight_utils", fake_weight_utils
    )
    fake_reload = ModuleType("vllm.model_executor.model_loader.reload")

    def _identity_decorator(fn):
        return fn

    fake_reload.support_quantized_model_reload_from_hp_weights = _identity_decorator
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.reload", fake_reload)

    try:
        import vllm.model_executor.models.deepseek_v4 as deepseek_v4
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear
        from vllm.model_executor.models.deepseek_v4 import DeepseekV4Model
    except ImportError as exc:
        pytest.skip(f"DeepSeek-V4 dependencies are unavailable: {exc}")

    monkeypatch.setattr(
        deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_rank", lambda: 0)

    class _FakeMergedColumnParallelLinear(MergedColumnParallelLinear):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.output_sizes = [256, 256]
            self.tp_size = 2
            self.tp_rank = 1
            self.weight_block_size = (128, 128)

            scale = nn.Parameter(torch.zeros(2, 1), requires_grad=False)
            scale.weight_loader = self._unexpected_weight_loader
            self.register_parameter("weight_scale_inv", scale)

        @staticmethod
        def _unexpected_weight_loader(*args, **kwargs) -> None:
            raise AssertionError("plain scale parameter should use the reload fallback")

    class _FakeSharedExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = _FakeMergedColumnParallelLinear()

    class _FakeFfn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared_experts = _FakeSharedExperts()

    class _FakeLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ffn = _FakeFfn()

    class _FakeDeepseekV4Model(DeepseekV4Model):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.config = SimpleNamespace(num_attention_heads=1)
            self.layers = nn.ModuleList([_FakeLayer()])

        def get_expert_mapping(self):
            return []

    model = _FakeDeepseekV4Model()

    loaded_params = model.load_weights(
        [
            (
                "layers.0.ffn.shared_experts.w1.weight_scale_inv",
                torch.tensor([[10.0], [11.0]]),
            ),
            (
                "layers.0.ffn.shared_experts.w3.weight_scale_inv",
                torch.tensor([[20.0], [21.0]]),
            ),
        ]
    )

    scale = model.layers[0].ffn.shared_experts.gate_up_proj.weight_scale_inv
    expected = torch.tensor([[11.0], [21.0]])
    assert torch.equal(scale, expected)
    assert loaded_params == {"layers.0.ffn.shared_experts.gate_up_proj.weight_scale_inv"}
    print(f"pass test1: send together")

    # test2
    loaded_params = model.load_weights(
        [
            (
                "layers.0.ffn.shared_experts.w1.weight_scale_inv",
                torch.tensor([[13.0], [15.0]]),
            ),
        ]
    )
    loaded_params = model.load_weights(
        [
            (
                "layers.0.ffn.shared_experts.w3.weight_scale_inv",
                torch.tensor([[23.0], [27.0]]),
            ),
        ]
    )

    scale = model.layers[0].ffn.shared_experts.gate_up_proj.weight_scale_inv
    expected = torch.tensor([[15.0], [27.0]])
    assert torch.equal(scale, expected)
    assert loaded_params == {"layers.0.ffn.shared_experts.gate_up_proj.weight_scale_inv"}
    print(f"pass test2: send separately")
