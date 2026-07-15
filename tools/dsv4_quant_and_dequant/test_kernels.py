# coding=utf-8
"""Smoke tests for DSV4 official <-> SGL format kernels (no full ckpt needed)."""

from __future__ import annotations

import unittest

import torch

from tools.dsv4_quant_and_dequant.kernels import (
    cast_e2m1fn_to_e4m3fn,
    dequant_fp8_block,
    official_dense_to_sgl,
    official_expert_to_sgl,
    official_wo_a_to_sgl,
    sgl_dense_to_official,
    sgl_expert_to_official,
    sgl_wo_a_to_official,
)


def _expand_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return dequant_fp8_block(weight, scale).float()


class TestKernels(unittest.TestCase):
    def test_dense_scale_roundtrip(self):
        w = torch.randn(256, 256).to(torch.float8_e4m3fn)
        # Exact powers of two so E8M0 roundtrip is bit-exact.
        s = torch.tensor([[1.0, 2.0], [0.5, 4.0]], dtype=torch.float32).to(
            torch.float8_e8m0fnu
        )
        w2, s2 = official_dense_to_sgl(w, s)
        self.assertEqual(w2.dtype, torch.float8_e4m3fn)
        self.assertEqual(s2.dtype, torch.float32)
        self.assertTrue(torch.equal(w.view(torch.uint8), w2.view(torch.uint8)))
        self.assertTrue(torch.equal(s.float(), s2))

        w3, s3 = sgl_dense_to_official(w2, s2)
        self.assertEqual(s3.dtype, torch.float8_e8m0fnu)
        self.assertTrue(torch.equal(s.view(torch.uint8), s3.view(torch.uint8)))

    def test_wo_a_roundtrip_close(self):
        # Random bf16 -> official fp8 -> sgl bf16 should be near identity after
        # one quantize (second hop is dequant of that fp8).
        torch.manual_seed(0)
        bf = torch.randn(128, 128, dtype=torch.bfloat16)
        q, s = sgl_wo_a_to_official(bf)
        back = official_wo_a_to_sgl(q, s)
        mad = (bf.float() - back.float()).abs().max().item()
        self.assertLess(mad, 0.5)

    def test_expert_official_to_sgl_dequant_match(self):
        # Build a tiny valid FP4 packed expert + E8M0 (1x32) scale.
        out_dim, in_dim = 128, 128
        torch.manual_seed(1)
        packed = torch.randint(-128, 128, (out_dim, in_dim // 2), dtype=torch.int8)
        exps = torch.randint(-2, 3, (out_dim, in_dim // 32))
        scale = torch.pow(2.0, exps.float()).to(torch.float8_e8m0fnu)

        q_off, s_off = cast_e2m1fn_to_e4m3fn(packed, scale)
        q_sgl, s_sgl = official_expert_to_sgl(packed, scale)
        self.assertEqual(s_sgl.dtype, torch.float32)
        self.assertTrue(torch.equal(s_off.float(), s_sgl))
        self.assertTrue(torch.equal(q_off.view(torch.uint8), q_sgl.view(torch.uint8)))

        # Roundtrip SGL -> official FP4 -> dequant should stay near SGL dequant.
        packed2, scale2 = sgl_expert_to_official(q_sgl, s_sgl)
        self.assertEqual(packed2.dtype, torch.int8)
        self.assertEqual(scale2.dtype, torch.float8_e8m0fnu)
        ref = _expand_fp8(q_sgl, s_sgl)
        # Re-expand via official convert again for a fair compare.
        q3, s3 = official_expert_to_sgl(packed2, scale2)
        got = _expand_fp8(q3, s3)
        mad = (ref - got).abs().max().item()
        self.assertLess(mad, 1.0)


if __name__ == "__main__":
    unittest.main()
