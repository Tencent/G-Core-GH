"""Unit tests for official Qwen3.5/3.6 FP8 skip list generation."""

from __future__ import annotations

import os
import sys
import unittest

from qwen3_official_fp8_skips import (
    build_qwen3_official_modules_to_not_convert,
    load_modules_to_not_convert_from_reference,
)


def _qwen36_moe_vl_config() -> dict:
    """Minimal structural clone of Qwen3.6-35B-A3B (40 layers, 30 linear / 10 full)."""
    layer_types = []
    for i in range(40):
        layer_types.append("linear_attention" if bool((i + 1) % 4) else "full_attention")
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config":
            {
                "model_type": "qwen3_5_moe_text",
                "num_hidden_layers": 40,
                "num_experts": 256,
                "layer_types": layer_types,
            },
        "vision_config": {
            "depth": 27,
            "deepstack_visual_indexes": []
        },
    }


class Qwen3OfficialFp8SkipsTest(unittest.TestCase):
    def test_qwen36_moe_vl_module_count(self):
        modules = build_qwen3_official_modules_to_not_convert(_qwen36_moe_vl_config())
        self.assertEqual(len(modules), 648)

    def test_linear_attn_layers_have_seven_skips_each(self):
        modules = build_qwen3_official_modules_to_not_convert(_qwen36_moe_vl_config())
        linear_layers = [
            i for i, t in enumerate(_qwen36_moe_vl_config()["text_config"]["layer_types"])
            if t == "linear_attention"
        ]
        self.assertEqual(len(linear_layers), 30)
        for i in linear_layers:
            prefix = f"model.language_model.layers.{i}.linear_attn."
            hits = [m for m in modules if m.startswith(prefix)]
            self.assertEqual(len(hits), 7, f"layer {i}: {hits}")
            self.assertFalse(any("in_proj_qkv" in m for m in hits))
            self.assertFalse(any("in_proj_z" in m for m in hits))
            self.assertFalse(any("out_proj" in m for m in hits))

    def test_full_attn_layers_have_qk_norm_skips(self):
        modules = build_qwen3_official_modules_to_not_convert(_qwen36_moe_vl_config())
        full_layers = [
            i for i, t in enumerate(_qwen36_moe_vl_config()["text_config"]["layer_types"])
            if t == "full_attention"
        ]
        self.assertEqual(len(full_layers), 10)
        for i in full_layers:
            self.assertIn(f"model.language_model.layers.{i}.self_attn.q_norm", modules)
            self.assertIn(f"model.language_model.layers.{i}.self_attn.k_norm", modules)

    def test_text_only_uses_model_prefix(self):
        cfg = {
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type":
                "qwen3_5_text",
            "num_hidden_layers":
                4,
            "layer_types":
                [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
        }
        modules = build_qwen3_official_modules_to_not_convert(
            cfg, include_mtp=False, include_vision=False
        )
        self.assertTrue(any(m.startswith("model.layers.0.linear_attn.") for m in modules))
        self.assertFalse(any("language_model" in m for m in modules))

    def test_reference_loader_roundtrip(self):
        ref = {
            "quantization_config":
                {
                    "modules_to_not_convert": ["lm_head", "model.layers.0.mlp.gate"],
                }
        }
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ref, f)
            path = f.name
        loaded = load_modules_to_not_convert_from_reference(path)
        self.assertEqual(loaded, ["lm_head", "model.layers.0.mlp.gate"])


if __name__ == "__main__":
    unittest.main()
