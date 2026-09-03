# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""tile_kernels.quant FP8 block-wise quantization 正确性测试。

覆盖 ``per_token_cast``（1D row-wise）、``per_block_cast``（2D）和
``per_token_cast_back``（dequant），对比纯 torch reference 实现。

不依赖 TE，不依赖 CUDA 12.9。只需 tile_kernels + CUDA。

Usage::

    cd /work/wepsdl/gcore-dev
    PYTHONPATH=. pytest -v -s tests/test_gfused/test_deepseek_tilekernels_quant.py
"""

import unittest

import torch


def _reference_1d_quant(x: torch.Tensor, block_size: int, *, round_sf: bool = False):
    """Pure-torch 1D (per-token/row-wise) block quant reference.

    Returns ``(q_fp8 [M, N], scale [M, N//block_size])``。
    ``round_sf=True`` → scale = 2^ceil(log2(amax/448))（pow2 rounding）。
    ``round_sf=False`` → scale = amax / 448（fp32 scale）。
    """
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    M, N = x.shape
    blocks = x.float().reshape(M, N // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    if round_sf:
        exponent = torch.ceil(torch.log2(amax / FP8_MAX))
        scale = torch.pow(2.0, exponent)
    else:
        scale = amax / FP8_MAX
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(M, N).contiguous(), scale.squeeze(-1)


def _reference_2d_quant(x: torch.Tensor, block_size: tuple[int, int], *, round_sf: bool = False):
    """Pure-torch 2D block quant reference.

    Returns ``(q_fp8 [M, N], scale [M//bm, N//bn])``。
    """
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    M, N = x.shape
    bm, bn = block_size
    blocks = x.float().reshape(M // bm, bm, N // bn, bn)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    if round_sf:
        exponent = torch.ceil(torch.log2(amax / FP8_MAX))
        scale = torch.pow(2.0, exponent)
    else:
        scale = amax / FP8_MAX
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(M, N).contiguous(), scale.squeeze(1).squeeze(-1)


class TestTileKernelsPerTokenCast(unittest.TestCase):
    """tile_kernels.quant.per_token_cast (1D row-wise block quant) 正确性。"""

    device = "cuda"
    dtype = torch.bfloat16

    def test_output_dtype_and_shape(self):
        from tile_kernels.quant import per_token_cast

        x = torch.randn(128, 512, device=self.device, dtype=self.dtype)
        y, s = per_token_cast(x, 'e4m3', 128)
        self.assertEqual(y.dtype, torch.float8_e4m3fn)
        self.assertEqual(y.shape, (128, 512))
        self.assertEqual(s.shape, (128, 512 // 128))
        print(f"  y.dtype={y.dtype}, s.dtype={s.dtype}, s.shape={s.shape}")

    def test_vs_reference_fp32_scale(self):
        from tile_kernels.quant import per_token_cast

        x = torch.randn(64, 256, device=self.device, dtype=self.dtype)
        y_kern, s_kern = per_token_cast(x, 'e4m3', 128, round_sf=False)
        y_ref, s_ref = _reference_1d_quant(x, 128, round_sf=False)

        scale_rel = (s_kern - s_ref).abs().max() / s_ref.abs().max()
        mismatch = (y_kern.view(torch.int8) != y_ref.view(torch.int8)).float().mean()
        print(f"  fp32 scale: scale_rel_diff={scale_rel.item():.4e}, fp8_mismatch_rate={mismatch.item():.4e}")
        self.assertLess(scale_rel.item(), 1e-5)
        self.assertLess(mismatch.item(), 0.02)

    def test_vs_reference_pow2_scale(self):
        from tile_kernels.quant import per_token_cast

        x = torch.randn(64, 256, device=self.device, dtype=self.dtype)
        y_kern, s_kern = per_token_cast(x, 'e4m3', 128, round_sf=True)
        y_ref, s_ref = _reference_1d_quant(x, 128, round_sf=True)

        scale_diff = (s_kern - s_ref).abs().max().item()
        mismatch = (y_kern.view(torch.int8) != y_ref.view(torch.int8)).float().mean()
        print(f"  pow2 scale: scale_abs_diff={scale_diff:.4e}, fp8_mismatch_rate={mismatch.item():.4e}")
        self.assertLess(scale_diff, 1e-6)
        self.assertLess(mismatch.item(), 0.02)

    def test_roundtrip(self):
        from tile_kernels.quant import per_token_cast, per_token_cast_back

        x = torch.randn(128, 512, device=self.device, dtype=self.dtype)
        y, s = per_token_cast(x, 'e4m3', 128, round_sf=False)
        x_recon = per_token_cast_back((y, s), 'bf16', 128)

        rel_diff = (x_recon.float() - x.float()).norm() / x.float().norm()
        print(f"  per_token roundtrip rel_diff = {rel_diff.item():.4e}")
        self.assertLess(rel_diff.item(), 0.05)

    def test_roundtrip_pow2(self):
        from tile_kernels.quant import per_token_cast, per_token_cast_back

        x = torch.randn(128, 512, device=self.device, dtype=self.dtype)
        y, s = per_token_cast(x, 'e4m3', 128, round_sf=True)
        x_recon = per_token_cast_back((y, s), 'bf16', 128)

        rel_diff = (x_recon.float() - x.float()).norm() / x.float().norm()
        print(f"  per_token pow2 roundtrip rel_diff = {rel_diff.item():.4e}")
        self.assertLess(rel_diff.item(), 0.05)

    def test_various_shapes(self):
        from tile_kernels.quant import per_token_cast, per_token_cast_back

        shapes = [(16, 128), (32, 256), (256, 1024), (64, 7168), (1, 128)]
        for m, n in shapes:
            with self.subTest(M=m, N=n):
                x = torch.randn(m, n, device=self.device, dtype=self.dtype)
                y, s = per_token_cast(x, 'e4m3', 128)
                x_recon = per_token_cast_back((y, s), 'bf16', 128)
                rel = (x_recon.float() - x.float()).norm() / x.float().norm()
                print(f"  shape ({m}, {n}): roundtrip rel_diff = {rel.item():.4e}")
                self.assertLess(rel.item(), 0.1)


class TestTileKernelsPerBlockCast(unittest.TestCase):
    """tile_kernels.quant.per_block_cast (2D block quant) 正确性。"""

    device = "cuda"
    dtype = torch.bfloat16

    def test_output_dtype_and_shape(self):
        from tile_kernels.quant import per_block_cast

        x = torch.randn(256, 512, device=self.device, dtype=self.dtype)
        y, s = per_block_cast(x, 'e4m3', (128, 128))
        self.assertEqual(y.dtype, torch.float8_e4m3fn)
        self.assertEqual(y.shape, (256, 512))
        self.assertEqual(s.shape, (256 // 128, 512 // 128))
        print(f"  y.dtype={y.dtype}, s.dtype={s.dtype}, s.shape={s.shape}")

    def test_vs_reference_fp32_scale(self):
        from tile_kernels.quant import per_block_cast

        x = torch.randn(256, 512, device=self.device, dtype=self.dtype)
        y_kern, s_kern = per_block_cast(x, 'e4m3', (128, 128), round_sf=False)
        y_ref, s_ref = _reference_2d_quant(x, (128, 128), round_sf=False)

        scale_rel = (s_kern - s_ref).abs().max() / s_ref.abs().max()
        mismatch = (y_kern.view(torch.int8) != y_ref.view(torch.int8)).float().mean()
        print(f"  2D fp32 scale: scale_rel_diff={scale_rel.item():.4e}, fp8_mismatch_rate={mismatch.item():.4e}")
        self.assertLess(scale_rel.item(), 1e-5)
        self.assertLess(mismatch.item(), 0.02)

    def test_vs_reference_pow2_scale(self):
        from tile_kernels.quant import per_block_cast

        x = torch.randn(256, 512, device=self.device, dtype=self.dtype)
        y_kern, s_kern = per_block_cast(x, 'e4m3', (128, 128), round_sf=True)
        y_ref, s_ref = _reference_2d_quant(x, (128, 128), round_sf=True)

        scale_diff = (s_kern - s_ref).abs().max().item()
        mismatch = (y_kern.view(torch.int8) != y_ref.view(torch.int8)).float().mean()
        print(f"  2D pow2 scale: scale_abs_diff={scale_diff:.4e}, fp8_mismatch_rate={mismatch.item():.4e}")
        self.assertLess(scale_diff, 1e-6)
        self.assertLess(mismatch.item(), 0.02)

    def test_2d_roundtrip_via_per_token_dequant(self):
        """2D quant → reshape scale to 1D → per_token_cast_back dequant。

        per_block_cast (128,128) 的 scale shape 是 [M//128, N//128]，
        需要 broadcast 回 [M, N//128] 才能喂 per_token_cast_back。
        """
        from tile_kernels.quant import per_block_cast, per_token_cast_back

        M, N, bm, bn = 256, 512, 128, 128
        x = torch.randn(M, N, device=self.device, dtype=self.dtype)
        y, s_2d = per_block_cast(x, 'e4m3', (bm, bn), round_sf=False)

        s_1d = s_2d.repeat_interleave(bm, dim=0)
        x_recon = per_token_cast_back((y, s_1d), 'bf16', bn)

        rel_diff = (x_recon.float() - x.float()).norm() / x.float().norm()
        print(f"  2D roundtrip rel_diff = {rel_diff.item():.4e}")
        self.assertLess(rel_diff.item(), 0.05)

    def test_various_shapes(self):
        from tile_kernels.quant import per_block_cast

        shapes = [(128, 128), (256, 256), (256, 1024), (128, 7168)]
        for m, n in shapes:
            with self.subTest(M=m, N=n):
                x = torch.randn(m, n, device=self.device, dtype=self.dtype)
                y, s = per_block_cast(x, 'e4m3', (128, 128))
                self.assertEqual(y.shape, (m, n))
                self.assertEqual(s.shape, (m // 128, n // 128))
                self.assertTrue(torch.isfinite(s).all())
                print(f"  shape ({m}, {n}): PASS")


class TestTileKernelsVsEagerFp8Block(unittest.TestCase):
    """tile_kernels.per_block_cast(round_sf=True) vs quant_fp8_e4m3_scale_e8m0。"""

    device = "cuda"

    def test_pow2_block_vs_eager_e8m0(self):
        from tile_kernels.quant import per_block_cast

        from gpatch_v4.kernel.quantize.eager_quant_kernels import (
            quant_fp8_e4m3_scale_e8m0,
        )

        torch.manual_seed(0)
        x = torch.randn(256, 512, device=self.device, dtype=torch.float32)
        q_tk, s_tk = per_block_cast(x.contiguous(), "e4m3", (128, 128), round_sf=True)
        q_eager, s_eager = quant_fp8_e4m3_scale_e8m0(x, block_size=(128, 128))

        scale_rel = (s_tk - s_eager.float()).abs().max() / s_eager.float().abs().max()
        mismatch = (q_tk.view(torch.int8) != q_eager.view(torch.int8)).float().mean()
        # tile vs eager: scale_rel=0.0000e+00, fp8_mismatch=0.0000e+00, s_tk.dtype=torch.float32, s_eager.dtype=torch.float8_e8m0fnu
        print(
            f"  tile vs eager: scale_rel={scale_rel.item():.4e}, "
            f"fp8_mismatch={mismatch.item():.4e}, "
            f"s_tk.dtype={s_tk.dtype}, s_eager.dtype={s_eager.dtype}"
        )
        self.assertLess(scale_rel.item(), 1e-6)
        self.assertLess(mismatch.item(), 0.02)

    def test_zero_block_scale_differs(self):
        """文档化已知差异：全零 block 上 tile_kernels clamp_min=1e-4 ≠ eager scale=1。"""
        from tile_kernels.quant import per_block_cast

        from gpatch_v4.kernel.quantize.eager_quant_kernels import (
            quant_fp8_e4m3_scale_e8m0,
        )

        x = torch.zeros(128, 128, device=self.device, dtype=torch.float32)
        _, s_tk = per_block_cast(x.contiguous(), "e4m3", (128, 128), round_sf=True)
        _, s_eager = quant_fp8_e4m3_scale_e8m0(x, block_size=(128, 128))
        # tests/test_gfused/test_deepseek_tilekernels_quant.py::TestTileKernelsVsEagerFp8Block::test_zero_block_scale_differs   zero-block: s_tk=2.38419e-07, s_eager=1
        print(
            f"  zero-block: s_tk={s_tk.item():.6g}, "
            f"s_eager={s_eager.float().item():.6g}"
        )
        self.assertNotEqual(s_tk.item(), s_eager.float().item())


if __name__ == "__main__":
    unittest.main()
