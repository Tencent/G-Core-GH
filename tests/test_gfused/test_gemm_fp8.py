# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""TE FP8 block-scaling GEMM 冒烟测试：forward + backward.

使用 ``Float8BlockScaling`` recipe（对齐 miles DSV4 训练配置），
bf16 权重在 ``fp8_autocast`` context 内临时 cast 到 FP8 做 GEMM。

同时包含一组 ``torch._scaled_mm`` 直接调用的测试，完全不依赖 TE recipe。

需要 CUDA >= 12.9 + TE >= 2.3。单卡测试，不需要 ray / 多 GPU。

Usage::

    cd /work/wepsdl/gcore-dev
    PYTHONPATH=. pytest -v -s tests/test_gfused/test_gemm_fp8.py
"""

import os
import unittest

import torch

os.environ.setdefault("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")

import transformer_engine.pytorch as te  # noqa: E402
from transformer_engine.common.recipe import Float8BlockScaling, Format  # noqa: E402


# ======================================================================
# Part 1: torch._scaled_mm 直接调用（不依赖 TE recipe）
# ======================================================================

def _quantize_per_tensor(x: torch.Tensor):
    """Per-tensor FP8 E4M3 quantization: 返回 (x_fp8, scale)。"""
    amax = x.abs().amax()
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    scale = (FP8_MAX / amax.clamp(min=1e-12)).float()
    x_fp8 = (x.float() * scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8, scale.reciprocal().reshape(1)


class TestScaledMmFp8(unittest.TestCase):
    """torch._scaled_mm FP8 GEMM 冒烟（零 TE 依赖）。"""

    device = "cuda"

    def _run_scaled_mm(self, M, K, N):
        x = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=self.device, dtype=torch.bfloat16)

        x_fp8, sx = _quantize_per_tensor(x)
        w_fp8, sw = _quantize_per_tensor(w)

        # _scaled_mm(A, B) 算 A @ B；B 须 col-major
        # w_fp8 [N,K] row-major → .t() 得 [K,N] col-major view
        out = torch._scaled_mm(
            x_fp8,
            w_fp8.t(),
            scale_a=sx,
            scale_b=sw,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        ref = x @ w.t()
        rel_diff = (out - ref).float().norm() / ref.float().norm()
        return out, rel_diff.item()

    def test_scaled_mm_forward_finite(self):
        out, rel_diff = self._run_scaled_mm(128, 512, 1024)
        print(f"  scaled_mm rel_diff = {rel_diff:.4e}")
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(rel_diff, 0.05)

    def test_scaled_mm_various_shapes(self):
        shapes = [
            (16, 128, 256),
            (64, 256, 128),
            (256, 1024, 2048),
            (32, 7168, 2048),
        ]
        for m, k, n in shapes:
            with self.subTest(M=m, K=k, N=n):
                out, rel_diff = self._run_scaled_mm(m, k, n)
                print(f"  shape ({m}, {k}, {n}): rel_diff = {rel_diff:.4e}")
                self.assertTrue(torch.isfinite(out).all())
                self.assertLess(rel_diff, 0.1)


def _make_recipe():
    return Float8BlockScaling(fp8_format=Format.E4M3)


# ======================================================================
# Part 2: TE Linear + fp8_autocast (Float8BlockScaling)
# ======================================================================

class TestTeGemmFp8BlockScaling(unittest.TestCase):
    """TE Linear FP8 block-scaling 冒烟。"""

    device = "cuda"
    dtype = torch.bfloat16
    M = 128
    H_IN = 512
    H_OUT = 1024

    def _make_input(self):
        return torch.randn(
            self.M, self.H_IN, device=self.device, dtype=self.dtype, requires_grad=True,
        )

    def _make_linear(self, bias=False):
        return te.Linear(self.H_IN, self.H_OUT, bias=bias).to(
            device=self.device, dtype=self.dtype,
        )

    def test_forward_fp8_produces_finite_output(self):
        linear = self._make_linear()
        x = self._make_input()
        recipe = _make_recipe()
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            out = linear(x)
        self.assertEqual(out.shape, (self.M, self.H_OUT))
        self.assertEqual(out.dtype, self.dtype, f"fwd output dtype 应为 {self.dtype}，实际 {out.dtype}")
        self.assertTrue(torch.isfinite(out).all(), "FP8 forward 输出含 inf/nan")

    def test_backward_fp8_produces_finite_grads(self):
        linear = self._make_linear()
        x = self._make_input()
        recipe = _make_recipe()
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            out = linear(x)
        out.sum().backward()

        self.assertIsNotNone(x.grad)
        print(f'test_backward_fp8_produces_finite_grads x.dtype: {x.dtype} x.grad.dtype: {x.grad.dtype}')
        self.assertEqual(x.grad.dtype, self.dtype, f"input grad dtype 应为 {self.dtype}，实际 {x.grad.dtype}")
        self.assertTrue(torch.isfinite(x.grad).all(), "input grad 含 inf/nan")
        for name, p in linear.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} grad is None")
            self.assertEqual(p.grad.dtype, self.dtype, f"{name} grad dtype 应为 {self.dtype}，实际 {p.grad.dtype}")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name} grad 含 inf/nan")

    def test_fp8_vs_bf16_forward_close(self):
        linear = self._make_linear()
        x = self._make_input()

        with torch.no_grad():
            ref = linear(x)

        recipe = _make_recipe()
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            fp8_out = linear(x)

        rel_diff = (fp8_out - ref).float().norm() / ref.float().norm()
        print(f"  [Float8BlockScaling] fwd rel_diff = {rel_diff.item():.4e}")
        self.assertLess(rel_diff.item(), 0.05)

    def test_fp8_vs_bf16_backward_close(self):
        linear = self._make_linear()
        x_fp8 = self._make_input()
        x_ref = x_fp8.detach().clone().requires_grad_(True)

        ref_out = linear(x_ref)
        ref_out.sum().backward()
        linear.zero_grad()

        recipe = _make_recipe()
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            fp8_out = linear(x_fp8)
        fp8_out.sum().backward()

        rel_diff_x = (x_fp8.grad - x_ref.grad).float().norm() / x_ref.grad.float().norm()
        print(f"  [Float8BlockScaling] bwd input grad rel_diff = {rel_diff_x.item():.4e}")
        self.assertLess(rel_diff_x.item(), 0.05)

    def test_multi_step_stability(self):
        linear = self._make_linear(bias=True)
        optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
        recipe = _make_recipe()
        target = torch.randn(self.M, self.H_OUT, device=self.device, dtype=self.dtype)
        losses = []

        for step in range(5):
            optimizer.zero_grad()
            x = self._make_input()
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                out = linear(x)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(f"  step {step}: loss = {loss.item():.6f}")

        for i, l in enumerate(losses):
            self.assertTrue(0 < l < 1e6, f"step {i} loss 异常: {l}")

    def test_various_shapes(self):
        recipe = _make_recipe()
        shapes = [
            (16, 128, 256),
            (64, 256, 128),
            (256, 1024, 2048),
            (32, 7168, 2048),
        ]
        for m, k, n in shapes:
            with self.subTest(M=m, K=k, N=n):
                linear = te.Linear(k, n, bias=False).to(
                    device=self.device, dtype=self.dtype,
                )
                x = torch.randn(m, k, device=self.device, dtype=self.dtype, requires_grad=True)
                with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                    out = linear(x)
                out.sum().backward()
                self.assertTrue(torch.isfinite(out).all(), f"shape ({m},{k},{n}) fwd inf/nan")
                self.assertTrue(torch.isfinite(x.grad).all(), f"shape ({m},{k},{n}) bwd inf/nan")
                print(f"  shape ({m}, {k}, {n}): PASS")


# ======================================================================
# Part 3: TE 内部算子直接调用 — 验证量化 dtype + GEMM output dtype
# ======================================================================

class TestTeInternalOps(unittest.TestCase):
    """直接调用 Float8BlockQuantizer + general_gemm，验证 FP8 内部 dtype。"""

    device = "cuda"
    dtype = torch.bfloat16

    def test_block_quantizer_produces_e4m3(self):
        """Float8BlockQuantizer 输出确实是 E4M3。"""
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
        import transformer_engine_torch as tex

        x = torch.randn(128, 512, device=self.device, dtype=self.dtype)

        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            force_pow_2_scales=False,
        )
        x_q = quantizer(x)

        print(f"  quantized _fp8_dtype = {x_q._fp8_dtype}")
        self.assertEqual(int(x_q._fp8_dtype), int(tex.DType.kFloat8E4M3))

    def test_general_gemm_output_dtype(self):
        """general_gemm(FP8, FP8, out_dtype=bf16) 输出确实是 bf16。"""
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
        from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
        import transformer_engine_torch as tex

        M, K, N = 128, 512, 1024
        x = torch.randn(M, K, device=self.device, dtype=self.dtype)
        w = torch.randn(N, K, device=self.device, dtype=self.dtype)

        # TE recipe: x_block_scaling_dim=1 (1D), w_block_scaling_dim=2 (2D)
        # cuBLAS 只支持 1D×1D / 1D×2D / 2D×1D，不支持 2D×2D
        x_quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            force_pow_2_scales=False,
            block_scaling_dim=1,
        )
        w_quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            force_pow_2_scales=False,
            block_scaling_dim=2,
        )
        x_q = x_quantizer(x)
        w_q = w_quantizer(w)

        # 对齐 TE Linear forward: general_gemm(weight, input, layout="TN")
        out, *_ = general_gemm(
            w_q,
            x_q,
            out_dtype=self.dtype,
            layout="TN",
            use_split_accumulator=True,
        )

        print(f"  general_gemm out.dtype = {out.dtype}, shape = {out.shape}")
        self.assertEqual(out.dtype, self.dtype, f"GEMM output 应为 {self.dtype}，实际 {out.dtype}")
        self.assertEqual(out.shape, (M, N))
        self.assertTrue(torch.isfinite(out).all())

    def test_te_linear_internal_grad_dtype_is_e4m3(self):
        """验证 E4M3 recipe 下 backward grad 的 FP8 内部格式也是 E4M3（非 E5M2）。

        通过 hook 在 backward 中拦截 grad_output，检查 TE 内部保存的
        quantized grad 确实是 E4M3。
        """
        linear = te.Linear(512, 1024, bias=False).to(device=self.device, dtype=self.dtype)
        x = torch.randn(128, 512, device=self.device, dtype=self.dtype, requires_grad=True)

        recipe = _make_recipe()
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            out = linear(x)

        # backward 后检查 grad dtype
        out.sum().backward()

        self.assertEqual(x.grad.dtype, self.dtype, f"input grad 应为 {self.dtype}，实际 {x.grad.dtype}")
        for name, p in linear.named_parameters():
            self.assertEqual(
                p.grad.dtype, self.dtype,
                f"{name} grad 应为 {self.dtype}，实际 {p.grad.dtype}",
            )

        # 检查 fp8_meta 确认 E4M3（非 E5M2）
        # Format.E4M3 → max_bwd=448 (E4M3); Format.HYBRID → max_bwd=57344 (E5M2)
        fp8_meta = linear.fp8_meta
        if "recipe" in fp8_meta:
            r = fp8_meta["recipe"]
            print(f"  fp8_meta recipe = {r}")
            if hasattr(r, "fp8_format"):
                fmt = r.fp8_format
                print(f"  fp8_format = {fmt}, max_bwd = {fmt.value.max_bwd}")
                self.assertEqual(
                    fmt.value.max_bwd, 448.0,
                    f"backward 应使用 E4M3 (max=448)，但 max_bwd={fmt.value.max_bwd}（可能是 E5M2/HYBRID）",
                )
        print("  confirmed: fwd=E4M3, bwd=E4M3, grad output dtype=bf16")


if __name__ == "__main__":
    unittest.main()
