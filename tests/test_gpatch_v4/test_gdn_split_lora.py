"""Unit tests for Qwen3.5 GDN fused-in_proj canonical split-LoRA."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from mbridge.peft.canonical_lora import (
    CanonicalLoRA,
    LoRALinearSplitGDNInProj,
    ModuleDict,
)
from mbridge.peft.lora import (
    LoRAMerge,
    deinterleave_gdn_qkv_lora_b,
    interleave_gdn_qkv_lora_b,
    mcore_adapter_name_to_hf,
)


def _gdn_config():
    return SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=3,
    )


class _BaseInProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _gdn_config()
        self.weight = nn.Parameter(torch.randn(24, 8))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class _TinyAdapter(nn.Module):
    def __init__(self, out_features, value):
        super().__init__()
        self.dim = 4
        self.alpha = 8
        self.linear_in = nn.Linear(8, self.dim, bias=False)
        self.linear_out = nn.Linear(self.dim, out_features, bias=False)
        with torch.no_grad():
            self.linear_in.weight.fill_(value)
            self.linear_out.weight.fill_(value + 1)

    def forward(self, x):
        return (
            torch.nn.functional.linear(
                torch.nn.functional.linear(x, self.linear_in.weight),
                self.linear_out.weight,
            )
            * (self.alpha / self.dim)
        )


def _build_wrapper():
    adapters = ModuleDict(
        {
            "adapter_qkv": _TinyAdapter(14, 1),
            "adapter_z": _TinyAdapter(6, 2),
            "adapter_b": _TinyAdapter(2, 3),
            "adapter_a": _TinyAdapter(2, 4),
        }
    )
    return LoRALinearSplitGDNInProj(_BaseInProj(), adapters)


def test_gdn_targets_map_to_one_fused_in_proj():
    peft = CanonicalLoRA(
        target_modules=[
            "*.in_proj_qkv",
            "*.in_proj_z",
            "*.in_proj_b",
            "*.in_proj_a",
        ]
    )
    assert peft.canonical_mapping["*.in_proj"] == {
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
    }


def test_gdn_qkv_lora_b_tp_layout_roundtrip():
    full = torch.arange(14 * 3).reshape(14, 3)
    rank_major = interleave_gdn_qkv_lora_b(full, (4, 4, 6), tp_size=2)
    restored = deinterleave_gdn_qkv_lora_b(
        rank_major,
        (4, 4, 6),
        tp_size=2,
    )
    torch.testing.assert_close(restored, full, rtol=0, atol=0)


def test_gdn_split_forward_is_four_independent_adapter_outputs():
    wrapper = _build_wrapper()
    x = torch.randn(3, 1, 8)
    base_output, _ = wrapper.to_wrap(x)
    output, bias = wrapper(x)
    expected_delta = torch.cat(
        [adapter(x) for adapter in wrapper.adapter.values()],
        dim=-1,
    )
    assert bias is None
    torch.testing.assert_close(output, base_output + expected_delta)
    assert len(
        {
            id(adapter.linear_in.weight)
            for adapter in wrapper.adapter.values()
        }
    ) == 4


def test_gdn_split_forward_supports_partial_targets():
    wrapper = _build_wrapper()
    wrapper.adapter["adapter_z"] = None
    wrapper.adapter["adapter_b"] = None
    wrapper.adapter["adapter_a"] = None
    x = torch.randn(2, 1, 8)
    base_output, _ = wrapper.to_wrap(x)
    output, _ = wrapper(x)
    expected_delta = torch.cat(
        [
            wrapper.adapter.adapter_qkv(x),
            torch.zeros_like(base_output[..., 14:]),
        ],
        dim=-1,
    )
    torch.testing.assert_close(output, base_output + expected_delta)


def test_gdn_split_merge_matches_four_adapter_deltas(monkeypatch):
    import mbridge.peft.lora as lora_module

    monkeypatch.setattr(
        lora_module.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        lora_module.parallel_state,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )

    wrapper = _build_wrapper()
    before = wrapper.to_wrap.weight.detach().clone()
    expected_delta = torch.cat(
        [
            (adapter.alpha / adapter.dim)
            * (adapter.linear_out.weight @ adapter.linear_in.weight)
            for adapter in wrapper.adapter.values()
        ],
        dim=0,
    )
    LoRAMerge().transform(wrapper)
    torch.testing.assert_close(wrapper.to_wrap.weight, before + expected_delta)


def test_gdn_subadapter_names_map_to_hf_modules():
    class _Bridge:
        @staticmethod
        def _weight_name_mapping_mcore_to_hf(_):
            return [
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                "model.layers.0.linear_attn.in_proj_z.weight",
                "model.layers.0.linear_attn.in_proj_b.weight",
                "model.layers.0.linear_attn.in_proj_a.weight",
            ]

    for adapter_name, hf_name in (
        ("adapter_qkv", "in_proj_qkv"),
        ("adapter_z", "in_proj_z"),
        ("adapter_b", "in_proj_b"),
        ("adapter_a", "in_proj_a"),
    ):
        key = mcore_adapter_name_to_hf(
            "decoder.layers.0.self_attention.in_proj."
            f"adapter.{adapter_name}.linear_in.weight",
            bridge=_Bridge(),
        )
        assert key.endswith(f".linear_attn.{hf_name}.lora_A.weight")
