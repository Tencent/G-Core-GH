"""Unit tests for Qwen3 MTP FP8 checkpoint post-processing."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import torch
from qwen3_mtp_fp8_postprocess import append_mtp_weights_to_fp8_checkpoint
from safetensors import safe_open
from safetensors.torch import save_file


class Qwen3MtpFp8PostprocessTest(unittest.TestCase):
    @staticmethod
    def _write_moe_mtp_config(input_dir: str) -> None:
        with open(os.path.join(input_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "text_config":
                        {
                            "model_type": "qwen3_5_moe_text",
                            "num_experts": 2,
                            "mtp_num_hidden_layers": 1,
                        }
                },
                f,
            )

    @staticmethod
    def _mtp_tensors() -> dict[str, torch.Tensor]:
        return {
            "mtp.fc.weight":
                torch.ones(4, 8, dtype=torch.bfloat16),
            "mtp.norm.weight":
                torch.ones(4, dtype=torch.bfloat16),
            "mtp.pre_fc_norm_embedding.weight":
                torch.ones(4, dtype=torch.bfloat16),
            "mtp.pre_fc_norm_hidden.weight":
                torch.ones(4, dtype=torch.bfloat16),
            "mtp.layers.0.input_layernorm.weight":
                torch.ones(4, dtype=torch.bfloat16),
            "mtp.layers.0.post_attention_layernorm.weight":
                torch.ones(4, dtype=torch.bfloat16),
            "mtp.layers.0.self_attn.q_proj.weight":
                torch.arange(4 * 4, dtype=torch.bfloat16).reshape(4, 4),
            "mtp.layers.0.self_attn.k_proj.weight":
                torch.ones(2, 4, dtype=torch.bfloat16),
            "mtp.layers.0.self_attn.v_proj.weight":
                torch.ones(2, 4, dtype=torch.bfloat16),
            "mtp.layers.0.self_attn.o_proj.weight":
                torch.ones(4, 4, dtype=torch.bfloat16),
            "mtp.layers.0.self_attn.q_norm.weight":
                torch.ones(4, dtype=torch.bfloat16),
            "mtp.layers.0.self_attn.k_norm.weight":
                torch.ones(4, dtype=torch.bfloat16),
            "mtp.layers.0.mlp.experts.gate_up_proj":
                torch.arange(2 * 4 * 4, dtype=torch.bfloat16).reshape(2, 4, 4),
            "mtp.layers.0.mlp.experts.down_proj":
                torch.arange(2 * 4 * 2, dtype=torch.bfloat16).reshape(2, 4, 2),
            "mtp.layers.0.mlp.gate.weight":
                torch.ones(2, 4, dtype=torch.bfloat16),
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight":
                torch.ones(2, 4, dtype=torch.bfloat16),
            "mtp.layers.0.mlp.shared_expert.up_proj.weight":
                torch.ones(2, 4, dtype=torch.bfloat16),
            "mtp.layers.0.mlp.shared_expert.down_proj.weight":
                torch.ones(4, 2, dtype=torch.bfloat16),
            "mtp.layers.0.mlp.shared_expert_gate.weight":
                torch.ones(1, 4, dtype=torch.bfloat16),
        }

    def test_append_mtp_expands_grouped_experts_and_updates_index(self):
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory(
        ) as output_dir:
            self._write_moe_mtp_config(input_dir)
            save_file(self._mtp_tensors(), os.path.join(input_dir, "input.safetensors"))
            save_file(
                {
                    "model.language_model.embed_tokens.weight":
                        torch.ones(2, 2, dtype=torch.bfloat16)
                },
                os.path.join(output_dir, "model.safetensors"),
            )
            with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "format": "pt",
                            "total_size": 123,
                        },
                        "weight_map":
                            {
                                "model.language_model.embed_tokens.weight": "model.safetensors",
                                "mtp.stale.weight": "mtp.safetensors",
                            },
                    },
                    f,
                )

            written = append_mtp_weights_to_fp8_checkpoint(
                input_dir,
                output_dir,
                block_size=(2, 2),
                modules_to_not_convert=[
                    "mtp.fc",
                    "mtp.norm",
                    "mtp.pre_fc_norm_embedding",
                    "mtp.pre_fc_norm_hidden",
                    "mtp.layers.0.input_layernorm",
                    "mtp.layers.0.post_attention_layernorm",
                    "mtp.layers.0.self_attn.q_norm",
                    "mtp.layers.0.self_attn.k_norm",
                    "mtp.layers.0.mlp.gate",
                    "mtp.layers.0.mlp.shared_expert_gate",
                ],
                device="cpu",
            )

            self.assertGreater(written, 0)
            mtp_path = os.path.join(output_dir, "mtp.safetensors")
            self.assertTrue(os.path.exists(mtp_path))

            with safe_open(mtp_path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                self.assertNotIn("mtp.layers.0.mlp.experts.gate_up_proj", keys)
                self.assertIn("mtp.layers.0.mlp.experts.0.gate_proj.weight", keys)
                self.assertIn("mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv", keys)
                self.assertIn("mtp.layers.0.mlp.experts.1.up_proj.weight", keys)
                self.assertIn("mtp.layers.0.mlp.experts.1.down_proj.weight", keys)

                gate_proj = f.get_tensor("mtp.layers.0.mlp.experts.0.gate_proj.weight")
                gate_scale = f.get_tensor("mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv")
                self.assertEqual(gate_proj.dtype, torch.float8_e4m3fn)
                self.assertEqual(gate_scale.dtype, torch.bfloat16)
                self.assertEqual(tuple(gate_scale.shape), (1, 2))

                q_proj = f.get_tensor("mtp.layers.0.self_attn.q_proj.weight")
                q_norm = f.get_tensor("mtp.layers.0.self_attn.q_norm.weight")
                gate = f.get_tensor("mtp.layers.0.mlp.gate.weight")
                fc = f.get_tensor("mtp.fc.weight")
                self.assertEqual(q_proj.dtype, torch.float8_e4m3fn)
                self.assertEqual(q_norm.dtype, torch.bfloat16)
                self.assertEqual(gate.dtype, torch.bfloat16)
                self.assertEqual(fc.dtype, torch.bfloat16)
                self.assertNotIn("mtp.layers.0.self_attn.q_norm.weight_scale_inv", keys)
                self.assertNotIn("mtp.fc.weight_scale_inv", keys)

            with open(os.path.join(output_dir, "model.safetensors.index.json")) as f:
                index = json.load(f)
            self.assertEqual(index["metadata"]["format"], "pt")
            self.assertGreater(index["metadata"]["total_size"], 123)
            weight_map = index["weight_map"]
            self.assertEqual(
                weight_map["model.language_model.embed_tokens.weight"],
                "model.safetensors",
            )
            self.assertNotIn("mtp.stale.weight", weight_map)
            self.assertEqual(
                weight_map["mtp.layers.0.mlp.experts.0.gate_proj.weight"],
                "mtp.safetensors",
            )
            self.assertEqual(
                weight_map["mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"],
                "mtp.safetensors",
            )

    def test_missing_required_mtp_tensors_raises(self):
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory(
        ) as output_dir:
            self._write_moe_mtp_config(input_dir)
            save_file(
                {"mtp.fc.weight": torch.ones(4, 8, dtype=torch.bfloat16)},
                os.path.join(input_dir, "bad.safetensors")
            )
            save_file(
                {"model.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
                os.path.join(output_dir, "model.safetensors")
            )

            with self.assertRaises(KeyError):
                append_mtp_weights_to_fp8_checkpoint(
                    input_dir,
                    output_dir,
                    block_size=(2, 2),
                    modules_to_not_convert=[],
                    device="cpu",
                )

    def test_mtp_enabled_without_tensors_raises(self):
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory(
        ) as output_dir:
            self._write_moe_mtp_config(input_dir)
            save_file(
                {"model.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
                os.path.join(output_dir, "model.safetensors")
            )

            with self.assertRaises(ValueError):
                append_mtp_weights_to_fp8_checkpoint(
                    input_dir,
                    output_dir,
                    block_size=(2, 2),
                    modules_to_not_convert=[],
                    device="cpu",
                )

    def test_multiple_mtp_layers_raise_not_implemented(self):
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory(
        ) as output_dir:
            with open(os.path.join(input_dir, "config.json"), "w") as f:
                json.dump(
                    {
                        "text_config":
                            {
                                "model_type": "qwen3_5_moe_text",
                                "num_experts": 2,
                                "mtp_num_hidden_layers": 2,
                            }
                    },
                    f,
                )
            save_file(self._mtp_tensors(), os.path.join(input_dir, "input.safetensors"))
            save_file(
                {"model.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
                os.path.join(output_dir, "model.safetensors")
            )

            with self.assertRaises(NotImplementedError):
                append_mtp_weights_to_fp8_checkpoint(
                    input_dir,
                    output_dir,
                    block_size=(2, 2),
                    modules_to_not_convert=[],
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
