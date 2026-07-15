# coding=utf-8
"""CPU vs CUDA bit-exact parity for DSV4 quant/dequant kernels.

Why this exists
---------------
The GPU convert path runs ``float8_e8m0fnu`` casts (``f32 -> e8m0`` on quant,
``e8m0 -> f32`` on dequant) on CUDA. If a torch/CUDA build only *half*-supports
that dtype it may **silently return wrong bytes instead of raising**. We can't
detect that from "it ran without error" alone.

CPU is the trusted reference (its e8m0 rounding is well-exercised). Every kernel
used by ``convert.py`` is run on CPU and CUDA with identical inputs; the CUDA
result is copied back and compared **byte-for-byte** (``view(uint8)`` for
float8/int8, exact equal for float32). Any silent mismatch fails loudly here.

Run
---
    PYTHONPATH=$PWD python3 tools/dsv4_quant_and_dequant/test_cuda_parity.py

Skips automatically if no CUDA device is visible.
"""

from __future__ import annotations

import sys
import unittest

import torch

from tools.dsv4_quant_and_dequant.kernels import (
    cast_e2m1fn_to_e4m3fn,
    official_dense_to_sgl,
    official_expert_to_sgl,
    official_wo_a_to_sgl,
    sgl_dense_to_official,
    sgl_expert_to_official,
    sgl_wo_a_to_official,
)

_HAS_CUDA = torch.cuda.is_available()


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-exact compare, treating float8 as raw bytes (NaN-safe, no ==)."""
    a = a.cpu().contiguous()
    b = b.cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
    a8 = a.cpu().contiguous().view(torch.uint8)
    b8 = b.cpu().contiguous().view(torch.uint8)
    diff = (a8.int() != b8.int())
    n = int(diff.sum())
    idx = diff.nonzero()[:5].flatten().tolist() if n else []
    return f"{name}: {n}/{a8.numel()} bytes differ; first flat idx={idx}"


def _make_e8m0_scale(shape, seed):
    g = torch.Generator().manual_seed(seed)
    exps = torch.randint(-8, 9, shape, generator=g)
    # include some zero-blocks (scale pinned to 1.0) implicitly via powers
    return torch.pow(2.0, exps.float()).to(torch.float8_e8m0fnu)


class TestCudaParity(unittest.TestCase):
    @unittest.skipUnless(_HAS_CUDA, "no CUDA device")
    def test_raw_e8m0_cast_f32_to_e8m0(self):
        # Edge cases: zero, subnormal, exact powers, non-powers, huge/tiny, neg.
        vals = torch.tensor(
            [
                0.0, 1.0, 0.5, 2.0, 6.0, 448.0, 3.7, 1e-30, 1e30,
                2.0 ** -60, 2.0 ** 60, -3.0, -0.25, 1234.5,
            ],
            dtype=torch.float32,
        )
        cpu = vals.to(torch.float8_e8m0fnu)
        cuda = vals.cuda().to(torch.float8_e8m0fnu)
        self.assertTrue(_bytes_equal(cpu, cuda), _report("f32->e8m0", cpu, cuda))

    @unittest.skipUnless(_HAS_CUDA, "no CUDA device")
    def test_raw_e8m0_cast_e8m0_to_f32(self):
        e = _make_e8m0_scale((64,), seed=7)
        cpu = e.float()
        cuda = e.cuda().float()
        self.assertTrue(torch.equal(cpu, cuda.cpu()), "e8m0->f32 mismatch")

    @unittest.skipUnless(_HAS_CUDA, "no CUDA device")
    def test_cast_e2m1fn_to_e4m3fn(self):
        out_dim, in_dim = 256, 256
        g = torch.Generator().manual_seed(1)
        packed = torch.randint(-128, 128, (out_dim, in_dim // 2), dtype=torch.int8, generator=g)
        scale = _make_e8m0_scale((out_dim, in_dim // 32), seed=2)
        q_c, s_c = cast_e2m1fn_to_e4m3fn(packed, scale)
        q_g, s_g = cast_e2m1fn_to_e4m3fn(packed.cuda(), scale.cuda())
        self.assertTrue(_bytes_equal(q_c, q_g), _report("cast q", q_c, q_g))
        self.assertTrue(_bytes_equal(s_c, s_g), _report("cast s", s_c, s_g))

    @unittest.skipUnless(_HAS_CUDA, "no CUDA device")
    def test_official_expert_to_sgl(self):
        g = torch.Generator().manual_seed(3)
        packed = torch.randint(-128, 128, (256, 128), dtype=torch.int8, generator=g)
        scale = _make_e8m0_scale((256, 256 // 32), seed=4)
        q_c, s_c = official_expert_to_sgl(packed, scale)
        q_g, s_g = official_expert_to_sgl(packed.cuda(), scale.cuda())
        self.assertTrue(_bytes_equal(q_c, q_g), _report("expert->sgl q", q_c, q_g))
        self.assertTrue(torch.equal(s_c, s_g.cpu()), "expert->sgl scale f32 mismatch")

    @unittest.skipUnless(_HAS_CUDA, "no CUDA device")
    def test_sgl_expert_to_official(self):
        g = torch.Generator().manual_seed(5)
        w = (torch.randn(256, 256, generator=g) * 4).to(torch.float8_e4m3fn)
        s = (torch.randn(2, 2, generator=g).abs() + 0.1).to(torch.float32)
        q_c, sc_c = sgl_expert_to_official(w, s)
        q_g, sc_g = sgl_expert_to_official(w.cuda(), s.cuda())
        self.assertTrue(_bytes_equal(q_c, q_g), _report("sgl->off expert q", q_c, q_g))
        self.assertTrue(_bytes_equal(sc_c, sc_g), _report("sgl->off expert scale", sc_c, sc_g))

    @unittest.skipUnless(_HAS_CUDA, "no CUDA device")
    def test_official_dense_to_sgl(self):
        w = torch.randint(0, 255, (256, 256), dtype=torch.uint8).view(torch.float8_e4m3fn)
        s = _make_e8m0_scale((2, 2), seed=8)
        w_c, s_c = official_dense_to_sgl(w, s)
        w_g, s_g = official_dense_to_sgl(w.cuda(), s.cuda())
        self.assertTrue(_bytes_equal(w_c, w_g), _report("dense->sgl w", w_c, w_g))
        self.assertTrue(torch.equal(s_c, s_g.cpu()), "dense->sgl scale f32 mismatch")

    @unittest.skipUnless(_HAS_CUDA, "no CUDA device")
    def test_sgl_dense_to_official(self):
        w = torch.randint(0, 255, (256, 256), dtype=torch.uint8).view(torch.float8_e4m3fn)
        s = (torch.rand(2, 2) * 8).to(torch.float32)
        w_c, s_c = sgl_dense_to_official(w, s)
        w_g, s_g = sgl_dense_to_official(w.cuda(), s.cuda())
        self.assertTrue(_bytes_equal(w_c, w_g), _report("sgl->off dense w", w_c, w_g))
        self.assertTrue(_bytes_equal(s_c, s_g), _report("sgl->off dense scale", s_c, s_g))

    @unittest.skipUnless(_HAS_CUDA, "no CUDA device")
    def test_official_wo_a_to_sgl(self):
        # wo_a is the only fp8 path that does *arithmetic* on the weight
        # (dequant), so feed realistic non-NaN fp8. Random fp8 byte patterns
        # include e4m3fn NaN encodings (0x7F/0xFF); NaN*x propagates a NaN whose
        # payload bytes legitimately differ CPU vs CUDA — a test artifact, not a
        # kernel bug (real wo_a weights are never NaN).
        g = torch.Generator().manual_seed(11)
        w = (torch.randn(256, 256, generator=g) * 3).to(torch.float8_e4m3fn)
        s = _make_e8m0_scale((2, 2), seed=9)
        b_c = official_wo_a_to_sgl(w, s)
        b_g = official_wo_a_to_sgl(w.cuda(), s.cuda())
        self.assertTrue(_bytes_equal(b_c, b_g), _report("wo_a->sgl bf16", b_c, b_g))

    @unittest.skipUnless(_HAS_CUDA, "no CUDA device")
    def test_sgl_wo_a_to_official(self):
        g = torch.Generator().manual_seed(10)
        bf = torch.randn(256, 256, dtype=torch.bfloat16, generator=g)
        q_c, s_c = sgl_wo_a_to_official(bf)
        q_g, s_g = sgl_wo_a_to_official(bf.cuda())
        self.assertTrue(_bytes_equal(q_c, q_g), _report("wo_a->off q", q_c, q_g))
        self.assertTrue(_bytes_equal(s_c, s_g), _report("wo_a->off scale", s_c, s_g))


if __name__ == "__main__":
    if not _HAS_CUDA:
        print("[parity] no CUDA device visible; nothing to verify", flush=True)
        sys.exit(0)
    print(f"[parity] torch={torch.__version__} device={torch.cuda.get_device_name(0)}", flush=True)
    unittest.main(verbosity=2)
