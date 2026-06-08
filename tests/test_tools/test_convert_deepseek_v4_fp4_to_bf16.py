# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""Tests for tools/convert_deepseek_v4_fp4_to_bf16.py.

Two layers of verification:

1. **Microtest (manual ground truth)**: build tiny int8/fp8 inputs whose every
   nibble / byte we can hand-decode, then assert exact bf16 output. Catches
   nibble order, FP4 table layout, and scale-broadcast direction bugs.

2. **Cross-check vs official**: run our ``cast_e2m1fn_to_bf16`` against the
   official ``inference/convert.py::cast_e2m1fn_to_e4m3fn`` (FP4 -> FP8 path).
   Round-trip the official output back to fp32 and check that both paths
   represent the same FP4 values to within FP4 quantization precision.

Pure CPU, no real checkpoint required; runs in <1s.
"""

import importlib.util
import os
import sys
import unittest

import torch

# ---------------------------------------------------------------------------
# Locate the module under test (gcore-dev/tools is not a package by default)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_CONV = _load_module(
    os.path.join(_TOOLS_DIR, "convert_deepseek_v4_fp4_to_bf16.py"),
    "convert_deepseek_v4_fp4_to_bf16",
)

# Official inference/convert.py lives under the checkpoint dir. Default to the
# project layout; allow override via env var for portability.
_OFFICIAL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash/inference/convert.py"


def _try_load_official():
    return _load_module(_OFFICIAL_PATH, "_dsv4_official_convert")


# ---------------------------------------------------------------------------
# 1. Microtests against hand-computed ground truth
# ---------------------------------------------------------------------------


class TestFp4DequantMicro(unittest.TestCase):
    """Hand-built int8 weight + scale; expected bf16 output is hand-derived."""

    def test_nibble_order_and_table_lookup(self):
        """Single block (out=1, in_packed=16 -> in=32), scale=1.

        Build 32 logical FP4 positions where position k selects FP4 index k%16
        (i.e. cycle through the table once forward, then once again). This both
        (a) covers every FP4 table entry, and (b) lets us check nibble order:
        the low nibble of byte i must land at logical position 2i (taking value
        FP4_TABLE[2i % 16]), and the high nibble at 2i+1.
        """
        in_packed = 16
        bytes_ = torch.tensor(
            [(((2 * i + 1) % 16) << 4) | ((2 * i) % 16) for i in range(in_packed)],
            dtype=torch.uint8,
        ).view(torch.int8).reshape(1, in_packed)
        scale = torch.ones(1, 1, dtype=torch.float32).to(torch.float8_e8m0fnu)

        out = _CONV.cast_e2m1fn_to_bf16(bytes_, scale)
        # Logical position k -> FP4_TABLE[k % 16]; in=32 covers each entry twice.
        expected = _CONV.FP4_TABLE[torch.arange(32) % 16].bfloat16().reshape(1, 32)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (1, 32))
        self.assertTrue(torch.equal(out, expected),
                        f"\nout      = {out.flatten().tolist()}\n"
                        f"expected = {expected.flatten().tolist()}")

    def test_scale_broadcast_direction(self):
        """Two columns of FP4 blocks, two rows; per-(row, block) scale.

        scale[r, b] = 2^(r + b) so that swapping row/col of scale is detectable.
        Weight pattern per block: low=1 (val 0.5), high=2 (val 1.0).
        Expected: row r, block b dequantizes to (alternating 0.5, 1.0) * 2^(r+b).
        """
        out_dim = 2
        in_dim = 64                     # two blocks of 32 along input
        in_packed = in_dim // 2
        # All bytes have low=1, high=2 -> nibble pair (0x21).
        weight = torch.full((out_dim, in_packed), 0x21, dtype=torch.uint8) \
                      .view(torch.int8)
        # E8M0 stores pure exponent; values 1, 2, 2, 4 are clean powers of two.
        scale_f32 = torch.tensor([[1.0, 2.0],
                                  [2.0, 4.0]], dtype=torch.float32)
        scale = scale_f32.to(torch.float8_e8m0fnu)

        out = _CONV.cast_e2m1fn_to_bf16(weight, scale).float()
        # Construct expected manually.
        block_pat = torch.tensor([0.5, 1.0]).repeat(16)        # length 32
        expected = torch.empty(out_dim, in_dim)
        for r in range(out_dim):
            for b in range(2):
                expected[r, b * 32:(b + 1) * 32] = block_pat * scale_f32[r, b]
        self.assertTrue(torch.equal(out, expected),
                        f"\nout      = {out}\nexpected = {expected}")

    def test_negative_nibbles(self):
        """Indices 8..15 are negatives; verify symmetry of the table."""
        # Byte 0xF0: low=0 (0.0), high=15 (-6.0). Then 0x9A: low=10 (-1.0), high=9 (-0.5).
        in_packed = 16
        b = bytearray([0xF0, 0x9A] + [0x00] * (in_packed - 2))
        weight = torch.tensor(list(b), dtype=torch.uint8).view(torch.int8) \
                      .reshape(1, in_packed)
        scale = torch.ones(1, 1, dtype=torch.float32).to(torch.float8_e8m0fnu)

        out = _CONV.cast_e2m1fn_to_bf16(weight, scale).float().flatten()
        # First four logical positions:
        #  pos 0 (byte 0 low) = 0.0; pos 1 (byte 0 high) = -6.0
        #  pos 2 (byte 1 low) = -1.0; pos 3 (byte 1 high) = -0.5
        self.assertEqual(out[0].item(), 0.0)
        self.assertEqual(out[1].item(), -6.0)
        self.assertEqual(out[2].item(), -1.0)
        self.assertEqual(out[3].item(), -0.5)
        # Remaining positions are byte 0x00 (low=0, high=0) -> all 0.
        self.assertTrue(torch.all(out[4:] == 0.0))


class TestFp8DequantMicro(unittest.TestCase):
    """Hand-built fp8 weight + 2D block scale."""

    def test_2d_scale_broadcast(self):
        """2x2 grid of 128x128 fp8 blocks, scale[i, j] = 2^(i + 2*j).

        Each block is filled with constant value 1.0 in fp8. Output = scale.
        Distinguishes the row-vs-col axis of the 2D scale.
        """
        out_dim = 2 * 128
        in_dim = 2 * 128
        weight = torch.ones(out_dim, in_dim).to(torch.float8_e4m3fn)
        scale_f32 = torch.tensor([[1.0, 4.0],
                                  [2.0, 8.0]], dtype=torch.float32)
        scale = scale_f32.to(torch.float8_e8m0fnu)

        out = _CONV.cast_e4m3fn_to_bf16(weight, scale).float()
        # Each [128,128] block should be constant = scale_f32[i, j].
        for i in range(2):
            for j in range(2):
                blk = out[i*128:(i+1)*128, j*128:(j+1)*128]
                self.assertTrue(
                    torch.all(blk == scale_f32[i, j]),
                    f"block ({i},{j}): expected {scale_f32[i,j].item()}, "
                    f"got min={blk.min().item()} max={blk.max().item()}",
                )


# ---------------------------------------------------------------------------
# 2. Cross-check vs official inference/convert.py::cast_e2m1fn_to_e4m3fn
# ---------------------------------------------------------------------------


class TestFp4VsOfficial(unittest.TestCase):
    """Compare our fp4->bf16 path against the official fp4->fp8 path.

    The official ``cast_e2m1fn_to_e4m3fn`` rebases the per-32 e8m0 scale into
    per-128 e8m0 (``scale_max_offset_bits``) and stores the per-32 residual
    inside the fp8 mantissa. So we can recover the *true* dequantized value as::

        true = fp8_w.float() * scale_max_offset_bits.float() (broadcasted per 128)

    Our path should match exactly because both decode the same FP4 nibbles
    against the same e8m0 scale.
    """

    def setUp(self):
        self.official = _try_load_official()

    @staticmethod
    def _expand_official(fp8_w: torch.Tensor, scale_offset_bits: torch.Tensor) -> torch.Tensor:
        """Recover dequantized fp32 from the official fp8 path.

        ``fp8_w`` is [out, in] fp8_e4m3; ``scale_offset_bits`` is
        [out//128, in//128] e8m0 (the per-128 outer scale).
        """
        out_dim, in_dim = fp8_w.shape
        s = (scale_offset_bits.float()
                              .repeat_interleave(128, dim=0)
                              .repeat_interleave(128, dim=1))
        return fp8_w.float() * s

    def _make_random_packed(self, out_dim: int, in_dim: int, seed: int = 0):
        """Return random (int8 weight [out, in/2], e8m0 scale [out, in/32])."""
        torch.manual_seed(seed)
        # Random bytes -> int8 view; uniform over all 16x16 nibble combos.
        weight = torch.randint(-128, 128, (out_dim, in_dim // 2), dtype=torch.int8)
        # E8M0 only stores exponents; pick small random exponents in [-3, 3].
        # 2^k for k in {-3..3} are exactly representable.
        exps = torch.randint(-3, 4, (out_dim, in_dim // 32))
        scale_f32 = torch.pow(2.0, exps.float())
        scale = scale_f32.to(torch.float8_e8m0fnu)
        return weight, scale

    def test_match_official_small(self):
        """Smallest size that satisfies official's fp8_block_size=128 constraint."""
        out_dim, in_dim = 128, 128
        weight, scale = self._make_random_packed(out_dim, in_dim, seed=42)

        ours = _CONV.cast_e2m1fn_to_bf16(weight, scale).float()
        fp8_w, scale_offset_bits = self.official.cast_e2m1fn_to_e4m3fn(weight, scale)
        theirs = self._expand_official(fp8_w, scale_offset_bits)

        # Both paths decode the same FP4 nibbles against the same scale, so
        # values should match exactly modulo bf16 rounding (~1 ULP). FP4 magnitudes
        # are <= 6, scaled by 2^[-3,3] -> <= 48, so ~3e-2 absolute is plenty.
        max_abs_diff = (ours - theirs).abs().max().item()
        max_rel_diff = ((ours - theirs).abs() / (theirs.abs() + 1e-6)).max().item()
        self.assertLess(max_abs_diff, 1e-1,
                        f"ours vs official: max_abs_diff={max_abs_diff}")
        # Restrict relative comparison to non-tiny entries.
        big = theirs.abs() > 0.1
        if big.any():
            rel = ((ours[big] - theirs[big]).abs() / theirs[big].abs()).max().item()
            self.assertLess(rel, 5e-2,
                            f"ours vs official (|x|>0.1): max_rel_diff={rel}")

    def test_match_official_two_blocks(self):
        """Multiple 128x128 blocks along both dims to also exercise the per-128 outer scale."""
        out_dim, in_dim = 256, 256
        weight, scale = self._make_random_packed(out_dim, in_dim, seed=7)

        ours = _CONV.cast_e2m1fn_to_bf16(weight, scale).float()
        fp8_w, scale_offset_bits = self.official.cast_e2m1fn_to_e4m3fn(weight, scale)
        theirs = self._expand_official(fp8_w, scale_offset_bits)

        # Element-wise match on shape.
        self.assertEqual(tuple(ours.shape), tuple(theirs.shape))
        max_abs_diff = (ours - theirs).abs().max().item()
        self.assertLess(max_abs_diff, 1e-1,
                        f"ours vs official: max_abs_diff={max_abs_diff}")


if __name__ == "__main__":
    unittest.main()
