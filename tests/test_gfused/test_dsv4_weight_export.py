# coding=utf-8
"""DSV4 weight export helpers unit tests (CPU-only)."""

import unittest

import torch

from gpatch_v4.models.deepseek_v4.weight_export import (
    iter_disk_checkpoint_tensors,
)


class TestDsv4WeightExport(unittest.TestCase):
    def test_split_and_quantize_fp4_experts(self):
        gate_up = torch.randn(2, 256, 128, dtype=torch.float32)
        exported = dict(
            iter_disk_checkpoint_tensors(
                "layers.0.mlp.experts.gate_up_proj",
                gate_up,
                dtype_format="quantized",
                expert_dtype="fp4",
            )
        )
        expected_weight = "layers.0.ffn.experts.0.w1.weight"
        expected_scale = "layers.0.ffn.experts.0.w1.scale"
        self.assertIn(expected_weight, exported)
        self.assertIn(expected_scale, exported)
        self.assertEqual(exported[expected_weight].dtype, torch.int8)
        self.assertEqual(exported[expected_scale].dtype, torch.float8_e8m0fnu)

    def test_fp8_expert_override(self):
        down = torch.randn(2, 128, 128, dtype=torch.float32)
        exported = dict(
            iter_disk_checkpoint_tensors(
                "layers.0.mlp.experts.down_proj",
                down,
                dtype_format="quantized",
                expert_dtype="fp8",
            )
        )
        expected_weight = "layers.0.ffn.experts.0.w2.weight"
        expected_scale = "layers.0.ffn.experts.0.w2.scale"
        self.assertIn(expected_weight, exported)
        self.assertIn(expected_scale, exported)
        self.assertEqual(exported[expected_weight].dtype, torch.float8_e4m3fn)
        self.assertEqual(exported[expected_scale].dtype, torch.float8_e8m0fnu)

    def test_bf16_fold_path(self):
        dense = torch.randn(128, 128, dtype=torch.float32)
        exported = dict(
            iter_disk_checkpoint_tensors(
                "layers.0.self_attn.q_a_proj.weight",
                dense,
                dtype_format="bf16",
            )
        )
        self.assertIn("layers.0.attn.wq_a.weight", exported)
        self.assertEqual(exported["layers.0.attn.wq_a.weight"].dtype, torch.bfloat16)
        self.assertNotIn("layers.0.attn.wq_a.scale", exported)

    def test_skip_mtp(self):
        dense = torch.randn(128, 128, dtype=torch.float32)
        exported = list(
            iter_disk_checkpoint_tensors(
                "mtp.layers.0.self_attn.q_a_proj.weight",
                dense,
                include_mtp=False,
            )
        )
        self.assertEqual(exported, [])


if __name__ == "__main__":
    unittest.main()
