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

测试 docker image：mirrors.tencent.com/wepsdl/rl-sglang:v2.25.55.55.7-cuda-12.9-cudnn-9-py-3.10-torch-2.11.0-fa-2.8.3-te-2.10-mlm-wxdev-hf-5.8.1-sglang-wx0.5.14
"""

import os
import unittest

import torch
import pytest

os.environ.setdefault("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")

_HAS_TE = False
try:
    import transformer_engine.pytorch as te  # noqa: E402
    from transformer_engine.common.recipe import Float8BlockScaling, Format  # noqa: E402
    _HAS_TE = True
except ImportError:
    te = None  # type: ignore[assignment]
    Float8BlockScaling = None  # type: ignore[assignment,misc]
    Format = None  # type: ignore[assignment,misc]

_SKIP_TE = not _HAS_TE
_SKIP_TE_REASON = "transformer_engine not installed"

def _gpu_cc() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()

def _cuda_version() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    v = torch.version.cuda
    if v is None:
        return (0, 0)
    parts = v.split(".")
    return (int(parts[0]), int(parts[1]))

_SKIP_CC = _gpu_cc() < (9, 0) or _cuda_version() < (12, 9)
_SKIP_CC_REASON = "requires compute capability >= 9.0 (Hopper) and CUDA >= 12.9"


# torch 有 API for scaled grouped linear，但 dW 的 backward 似乎未完成。
# https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/functional.py


def _quantize_per_tensor(x: torch.Tensor):
    """Per-tensor FP8 E4M3 quantization: 返回 (x_fp8, scale)。"""
    amax = x.abs().amax()
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    scale = (FP8_MAX / amax.clamp(min=1e-12)).float()
    x_fp8 = (x.float() * scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8, scale.reciprocal().reshape(1)


_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def _to_col_major(t: torch.Tensor) -> torch.Tensor:
    """2D tensor → column-major [R, C] (stride [1, R])."""
    return t.contiguous().t().contiguous().t()


def _quant_1d(x: torch.Tensor, block_size: int = 128):
    """1D row-wise block quant (activation). Returns (fp8 [M,K], scale [M, K//bs])."""
    M, K = x.shape
    nb = K // block_size
    blocks = x.float().reshape(M, nb, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / _FP8_MAX
    q = (blocks / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(M, K).contiguous(), scale.squeeze(-1)


def _quant_2d(w: torch.Tensor, block_size: tuple[int, int] = (128, 128)):
    """2D block quant (weight). Returns (fp8 [N,K], scale [N//bm, K//bn])."""
    N, K = w.shape
    bm, bn = block_size
    blocks = w.float().reshape(N // bm, bm, K // bn, bn)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / _FP8_MAX
    q = (blocks / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(N, K).contiguous(), scale.squeeze(1).squeeze(-1)


# ======================================================================
# Part 2: TE Linear + fp8_autocast (Float8BlockScaling)
# ======================================================================

def _make_recipe():
    return Float8BlockScaling(fp8_format=Format.E4M3)


@pytest.mark.skipif(_SKIP_TE, reason=_SKIP_TE_REASON)
@pytest.mark.skipif(_SKIP_CC, reason=_SKIP_CC_REASON)
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
# Part 2b: TE GroupedLinear + fp8_autocast (Float8BlockScaling)
# ======================================================================

@pytest.mark.skipif(_SKIP_TE, reason=_SKIP_TE_REASON)
@pytest.mark.skipif(_SKIP_CC, reason=_SKIP_CC_REASON)
class TestTeGroupedLinearFp8(unittest.TestCase):
    """TE GroupedLinear FP8 block-scaling 冒烟（模拟 MoE expert GEMMs）。"""

    device = "cuda"
    dtype = torch.bfloat16
    NUM_EXPERTS = 4
    H_IN = 256
    H_OUT = 512

    def _make_grouped_linear(self, bias=False):
        return te.GroupedLinear(
            num_gemms=self.NUM_EXPERTS,
            in_features=self.H_IN,
            out_features=self.H_OUT,
            bias=bias,
        ).to(device=self.device, dtype=self.dtype)

    def _make_input_and_splits(self):
        # TE FP8 path calls tex.split_quantize(..., split_sections: list[int], ...);
        # a torch.Tensor here raises TypeError.
        tokens_per_expert = [32, 16, 2048, 256]
        total = sum(tokens_per_expert)
        x = torch.randn(total, self.H_IN, device=self.device, dtype=self.dtype, requires_grad=True)
        return x, tokens_per_expert

    def test_grouped_forward_fp8_finite(self):
        gl = self._make_grouped_linear()
        x, m_splits = self._make_input_and_splits()
        recipe = _make_recipe()
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            out = gl(x, m_splits)
        total = sum(m_splits)
        self.assertEqual(out.shape, (total, self.H_OUT))
        self.assertEqual(out.dtype, self.dtype)
        self.assertTrue(torch.isfinite(out).all(), "GroupedLinear FP8 fwd inf/nan")

    def test_grouped_backward_fp8_finite(self):
        gl = self._make_grouped_linear()
        x, m_splits = self._make_input_and_splits()
        recipe = _make_recipe()
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            out = gl(x, m_splits)
        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.dtype, self.dtype)
        self.assertTrue(torch.isfinite(x.grad).all(), "input grad inf/nan")
        for name, p in gl.named_parameters():
            if p.grad is not None:
                self.assertEqual(p.grad.dtype, self.dtype, f"{name} grad dtype 应为 {self.dtype}")
                self.assertTrue(torch.isfinite(p.grad).all(), f"{name} grad inf/nan")

    def test_grouped_fp8_vs_bf16_fwd_bwd_close(self):
        gl = self._make_grouped_linear()
        x_fp8, m_splits = self._make_input_and_splits()
        x_ref = x_fp8.detach().clone().requires_grad_(True)

        ref_out = gl(x_ref, m_splits)
        ref_out.sum().backward()
        ref_w_grads = {
            n: p.grad.detach().clone()
            for n, p in gl.named_parameters()
            if p.grad is not None
        }
        gl.zero_grad()

        recipe = _make_recipe()
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            fp8_out = gl(x_fp8, m_splits)
        fp8_out.sum().backward()

        rel_diff = (fp8_out - ref_out).float().norm() / ref_out.float().norm()
        print(f"  [GroupedLinear] fwd rel_diff = {rel_diff.item():.4e}")
        self.assertLess(rel_diff.item(), 0.05)

        rel_diff_x = (x_fp8.grad - x_ref.grad).float().norm() / x_ref.grad.float().norm()
        print(f"  [GroupedLinear] bwd input grad rel_diff = {rel_diff_x.item():.4e}")
        self.assertLess(rel_diff_x.item(), 0.05)

        for name, ref_g in ref_w_grads.items():
            fp8_g = dict(gl.named_parameters())[name].grad
            self.assertIsNotNone(fp8_g, f"{name} fp8 grad is None")
            rel_diff_w = (fp8_g - ref_g).float().norm() / ref_g.float().norm()
            print(f"  [GroupedLinear] bwd {name} grad rel_diff = {rel_diff_w.item():.4e}")
            self.assertLess(rel_diff_w.item(), 0.05, f"{name} grad rel_diff too large")

    def test_grouped_bf16_vs_bf16_fwd_bwd_close(self):
        """同一 bf16 GroupedLinear 跑两遍，fwd/bwd 应对齐（无 FP8）。"""
        gl = self._make_grouped_linear()
        x1, m_splits = self._make_input_and_splits()
        x2 = x1.detach().clone().requires_grad_(True)

        out1 = gl(x1, m_splits)
        out1.sum().backward()
        w_grads1 = {
            n: p.grad.detach().clone()
            for n, p in gl.named_parameters()
            if p.grad is not None
        }
        gl.zero_grad()

        out2 = gl(x2, m_splits)
        out2.sum().backward()

        rel_diff = (out2 - out1).float().norm() / out1.float().norm().clamp(min=1e-12)
        print(f"  [GroupedLinear bf16] fwd rel_diff = {rel_diff.item():.4e}")
        self.assertLess(rel_diff.item(), 1e-6)

        rel_diff_x = (x2.grad - x1.grad).float().norm() / x1.grad.float().norm().clamp(min=1e-12)
        print(f"  [GroupedLinear bf16] bwd input grad rel_diff = {rel_diff_x.item():.4e}")
        self.assertLess(rel_diff_x.item(), 1e-6)

        for name, g1 in w_grads1.items():
            g2 = dict(gl.named_parameters())[name].grad
            self.assertIsNotNone(g2, f"{name} grad is None")
            rel_diff_w = (g2 - g1).float().norm() / g1.float().norm().clamp(min=1e-12)
            print(f"  [GroupedLinear bf16] bwd {name} grad rel_diff = {rel_diff_w.item():.4e}")
            self.assertLess(rel_diff_w.item(), 1e-6, f"{name} grad rel_diff too large")

    def test_grouped_multi_step_stability(self):
        gl = self._make_grouped_linear(bias=True)
        optimizer = torch.optim.Adam(gl.parameters(), lr=1e-3)
        recipe = _make_recipe()
        losses = []

        for step in range(5):
            optimizer.zero_grad()
            x, m_splits = self._make_input_and_splits()
            target = torch.randn(sum(m_splits), self.H_OUT, device=self.device, dtype=self.dtype)
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                out = gl(x, m_splits)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(f"  step {step}: loss = {loss.item():.6f}")

        for i, l in enumerate(losses):
            self.assertTrue(0 < l < 1e6, f"step {i} loss 异常: {l}")

    def test_grouped_uneven_splits(self):
        """不均匀 split（含 0 token expert）。"""
        gl = self._make_grouped_linear()
        splits_list = [
            [64, 0, 32, 32],
            [0, 0, 128, 0],
            [16, 16, 16, 16],
        ]
        recipe = _make_recipe()
        for splits in splits_list:
            total = sum(splits)
            if total == 0:
                continue
            with self.subTest(splits=splits):
                x = torch.randn(total, self.H_IN, device=self.device, dtype=self.dtype, requires_grad=True)
                with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                    out = gl(x, splits)
                out.sum().backward()
                self.assertTrue(torch.isfinite(out).all(), f"splits={splits} fwd inf/nan")
                self.assertTrue(torch.isfinite(x.grad).all(), f"splits={splits} bwd inf/nan")
                print(f"  splits={splits}: PASS")


# ======================================================================
# Part 2c: MyGroupedLinearFp8（实现见 gpatch_v4.models.deepseek_v4.fp8）
# ======================================================================

try:
    from gpatch_v4.models.deepseek_v4.fp8 import (  # noqa: E402
        MyGroupedLinearFp8,
        _pad_m_splits,
    )
except ImportError:
    MyGroupedLinearFp8 = None  # type: ignore[misc,assignment]
    _pad_m_splits = None  # type: ignore[misc,assignment]


def _bench_cuda_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@pytest.mark.skipif(_SKIP_TE, reason=_SKIP_TE_REASON)
@pytest.mark.skipif(_SKIP_CC, reason=_SKIP_CC_REASON)
@pytest.mark.skipif(MyGroupedLinearFp8 is None, reason="MyGroupedLinearFp8 import failed")
class TestTeGroupedGemmFp8CustomOp(unittest.TestCase):
    """``MyGroupedLinearFp8`` vs ``te.GroupedLinear`` FP8 对齐 + perf。"""

    device = "cuda"
    dtype = torch.bfloat16
    NUM_EXPERTS = 4
    H_IN = 256
    H_OUT = 512
    M_SPLITS = [8192, 16, 2048, 256]
    # 含非 16 对齐 split；custom op 应自动 pad，te.GroupedLinear 裸调会挂。
    M_SPLITS_UNEVEN = [32, 1, 2000, 323]

    def _make_inputs(self, m_splits=None):
        m_splits = self.M_SPLITS if m_splits is None else m_splits
        total = sum(m_splits)
        x = torch.randn(
            total, self.H_IN, device=self.device, dtype=self.dtype, requires_grad=True,
        )
        weight = torch.randn(
            self.NUM_EXPERTS,
            self.H_OUT,
            self.H_IN,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        return x, weight

    def test_custom_op_vs_te_grouped_linear_fp8_fwd_bwd(self):
        x, weight = self._make_inputs()
        x_te = x.detach().clone().requires_grad_(True)

        gl = te.GroupedLinear(
            num_gemms=self.NUM_EXPERTS,
            in_features=self.H_IN,
            out_features=self.H_OUT,
            bias=False,
        ).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            for i in range(self.NUM_EXPERTS):
                getattr(gl, f"weight{i}").copy_(weight[i])

        custom = MyGroupedLinearFp8(num_gemms=self.NUM_EXPERTS)

        recipe = _make_recipe()
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            te_out = gl(x_te, self.M_SPLITS)
        te_out.sum().backward()

        custom_out = custom(x, weight, self.M_SPLITS)
        custom_out.sum().backward()

        rel_fwd = (custom_out - te_out).float().norm() / te_out.float().norm()
        print(f"  [custom vs TE] fwd rel_diff = {rel_fwd.item():.4e}")
        self.assertEqual(rel_fwd.item(), 0.)

        rel_x = (x.grad - x_te.grad).float().norm() / x_te.grad.float().norm()
        print(f"  [custom vs TE] bwd input grad rel_diff = {rel_x.item():.4e}")
        self.assertEqual(rel_x.item(), 0.)

        for i in range(self.NUM_EXPERTS):
            te_wg = getattr(gl, f"weight{i}").grad
            custom_wg = weight.grad[i]
            rel_w = (custom_wg - te_wg).float().norm() / te_wg.float().norm()
            print(f"  [custom vs TE] bwd weight{i} grad rel_diff = {rel_w.item():.4e}")
            self.assertEqual(rel_w.item(), 0., f"weight{i} grad rel_diff too large")

        self.assertTrue(torch.isfinite(custom_out).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())

        # ---- perf: Module 复用 quantizer，不应明显慢于 te.GroupedLinear ----
        x_te_b = x.detach().clone().requires_grad_(True)
        x_cu_b = x.detach().clone().requires_grad_(True)
        w_cu_b = weight.detach().clone().requires_grad_(True)
        with torch.no_grad():
            for i in range(self.NUM_EXPERTS):
                getattr(gl, f"weight{i}").copy_(w_cu_b[i])
        gl.zero_grad()
        dO = torch.randn_like(te_out)

        def run_te():
            gl.zero_grad()
            x_te_b.grad = None
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                out = gl(x_te_b, self.M_SPLITS)
            out.backward(dO)

        def run_custom():
            x_cu_b.grad = None
            w_cu_b.grad = None
            out = custom(x_cu_b, w_cu_b, self.M_SPLITS)
            out.backward(dO)

        te_ms = _bench_cuda_ms(run_te)
        custom_ms = _bench_cuda_ms(run_custom)
        ratio = custom_ms / te_ms
        print(
            f"  [custom vs TE] fwd+bwd te={te_ms:.3f} ms  custom={custom_ms:.3f} ms  "
            f"ratio={ratio:.3f}x"
        )
        self.assertLess(ratio, 1.1, f"custom op too slow vs TE GroupedLinear: {ratio:.3f}x")

    def test_custom_op_uneven_splits_fp8_fwd_bwd(self):
        """非 16 对齐 m_splits：Module 自动 pad，fwd/bwd 有限且接近 bf16。"""
        m_splits = self.M_SPLITS_UNEVEN
        self.assertNotEqual(m_splits, _pad_m_splits(m_splits))

        x, weight = self._make_inputs(m_splits)
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = weight.detach().clone().requires_grad_(True)

        # bf16 reference：逐 expert matmul（无 pad）
        outs = []
        offset = 0
        for i, m in enumerate(m_splits):
            xi = x_ref[offset : offset + m]
            outs.append(xi @ w_ref[i].T)
            offset += m
        ref_out = torch.cat(outs, dim=0)
        ref_out.sum().backward()

        custom = MyGroupedLinearFp8(num_gemms=self.NUM_EXPERTS)
        custom_out = custom(x, weight, m_splits)
        custom_out.sum().backward()

        self.assertEqual(custom_out.shape, ref_out.shape)
        rel_fwd = (custom_out - ref_out).float().norm() / ref_out.float().norm()
        print(f"  [custom uneven] fwd rel_diff = {rel_fwd.item():.4e}")
        self.assertLess(rel_fwd.item(), 0.05)

        rel_x = (x.grad - x_ref.grad).float().norm() / x_ref.grad.float().norm()
        print(f"  [custom uneven] bwd input grad rel_diff = {rel_x.item():.4e}")
        self.assertLess(rel_x.item(), 0.05)

        rel_w = (weight.grad - w_ref.grad).float().norm() / w_ref.grad.float().norm()
        print(f"  [custom uneven] bwd weight grad rel_diff = {rel_w.item():.4e}")
        self.assertLess(rel_w.item(), 0.05)

        self.assertTrue(torch.isfinite(custom_out).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())
