"""
Tests for gfused.flash_attn_triton_og — Triton Flash Attention forward & backward.

Requires at least 1 GPU with CUDA support and the `triton` package installed.
"""

import math
import unittest

import pytest
import torch


def _ref_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """Reference causal attention implementation using PyTorch ops.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Shape ``(Z, H, N_CTX, D_HEAD)``.
    sm_scale : float
        Softmax scaling factor (typically ``1 / sqrt(D_HEAD)``).

    Returns
    -------
    torch.Tensor
        Attention output, same shape as *q*.
    """
    # (Z, H, N_CTX, N_CTX)
    attn = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    # causal mask
    N_CTX = q.shape[2]
    causal_mask = torch.tril(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool))
    attn = attn.masked_fill(~causal_mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def _make_qkv(
    Z: int,
    H: int,
    N_CTX: int,
    D_HEAD: int,
    dtype: torch.dtype = torch.float16,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create random Q, K, V tensors and the softmax scale factor.

    Returns
    -------
    tuple
        ``(q, k, v, sm_scale)``
    """
    torch.manual_seed(42)
    q = torch.randn(Z, H, N_CTX, D_HEAD, device="cuda", dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(Z, H, N_CTX, D_HEAD, device="cuda", dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(Z, H, N_CTX, D_HEAD, device="cuda", dtype=dtype, requires_grad=requires_grad)
    sm_scale = 1.0 / math.sqrt(D_HEAD)
    return q, k, v, sm_scale


class FlashAttnForwardTest(unittest.TestCase):
    """Forward-pass correctness tests for Triton Flash Attention."""
    @classmethod
    def setUpClass(cls) -> None:
        # Lazy import so the module-level skip can take effect first.
        from gfused.flash_attn_triton_og import attention
        cls.attention = staticmethod(attention)

    # ------------------------------------------------------------------
    # Parameterised over (Z, H, N_CTX, D_HEAD)
    # ------------------------------------------------------------------

    def _check_forward(
        self,
        Z: int,
        H: int,
        N_CTX: int,
        D_HEAD: int,
        *,
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ) -> None:
        q, k, v, sm_scale = _make_qkv(Z, H, N_CTX, D_HEAD, dtype=torch.float16)
        tri_out = self.attention(q, k, v, sm_scale)

        ref_out = _ref_attention(q.float(), k.float(), v.float(), sm_scale).to(torch.float16)

        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=rtol)

    def test_fwd_d16(self) -> None:
        self._check_forward(Z=2, H=4, N_CTX=128, D_HEAD=16)

    def test_fwd_d32(self) -> None:
        self._check_forward(Z=2, H=4, N_CTX=128, D_HEAD=32)

    def test_fwd_d64(self) -> None:
        self._check_forward(Z=2, H=4, N_CTX=128, D_HEAD=64)

    def test_fwd_d128(self) -> None:
        self._check_forward(Z=2, H=4, N_CTX=128, D_HEAD=128)

    def test_fwd_longer_seq(self) -> None:
        """Sequence length > one tile (BLOCK=128)."""
        self._check_forward(Z=1, H=2, N_CTX=512, D_HEAD=64)

    def test_fwd_single_batch_head(self) -> None:
        self._check_forward(Z=1, H=1, N_CTX=256, D_HEAD=64)


class FlashAttnBackwardTest(unittest.TestCase):
    """Backward-pass (gradient) correctness tests for Triton Flash Attention."""
    @classmethod
    def setUpClass(cls) -> None:
        from gfused.flash_attn_triton_og import attention
        cls.attention = staticmethod(attention)

    def _check_backward(
        self,
        Z: int,
        H: int,
        N_CTX: int,
        D_HEAD: int,
        *,
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ) -> None:
        # Triton path
        q, k, v, sm_scale = _make_qkv(Z, H, N_CTX, D_HEAD, dtype=torch.float16, requires_grad=True)
        tri_out = self.attention(q, k, v, sm_scale)
        dout = torch.randn_like(tri_out)
        tri_out.backward(dout)
        tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

        # Reference path (float32 for numerical stability)
        q2, k2, v2, _ = _make_qkv(Z, H, N_CTX, D_HEAD, dtype=torch.float16, requires_grad=True)
        ref_out = _ref_attention(q2.float(), k2.float(), v2.float(), sm_scale)
        ref_out.backward(dout.float())
        ref_dq = q2.grad.to(torch.float16)
        ref_dk = k2.grad.to(torch.float16)
        ref_dv = v2.grad.to(torch.float16)

        torch.testing.assert_close(tri_dq, ref_dq, atol=atol, rtol=rtol)
        torch.testing.assert_close(tri_dk, ref_dk, atol=atol, rtol=rtol)
        torch.testing.assert_close(tri_dv, ref_dv, atol=atol, rtol=rtol)

    def test_bwd_d64(self) -> None:
        self._check_backward(Z=2, H=4, N_CTX=128, D_HEAD=64)

    def test_bwd_d128(self) -> None:
        self._check_backward(Z=2, H=4, N_CTX=128, D_HEAD=128)

    def test_bwd_longer_seq(self) -> None:
        self._check_backward(Z=1, H=2, N_CTX=256, D_HEAD=64)


class FlashAttnOutputShapeTest(unittest.TestCase):
    """Smoke tests: verify output shapes and dtypes."""
    @classmethod
    def setUpClass(cls) -> None:
        from gfused.flash_attn_triton_og import attention
        cls.attention = staticmethod(attention)

    def test_output_shape_matches_query(self) -> None:
        Z, H, N_CTX, D_HEAD = 2, 4, 256, 64
        q, k, v, sm_scale = _make_qkv(Z, H, N_CTX, D_HEAD, dtype=torch.float16)
        out = self.attention(q, k, v, sm_scale)
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_output_dtype_matches_query(self) -> None:
        Z, H, N_CTX, D_HEAD = 1, 1, 128, 64
        q, k, v, sm_scale = _make_qkv(Z, H, N_CTX, D_HEAD, dtype=torch.float16)
        out = self.attention(q, k, v, sm_scale)
        assert out.dtype == q.dtype, f"Expected {q.dtype}, got {out.dtype}"

    def test_output_is_contiguous(self) -> None:
        Z, H, N_CTX, D_HEAD = 1, 2, 128, 64
        q, k, v, sm_scale = _make_qkv(Z, H, N_CTX, D_HEAD, dtype=torch.float16)
        out = self.attention(q, k, v, sm_scale)
        assert out.is_contiguous()

    def test_output_no_nan(self) -> None:
        Z, H, N_CTX, D_HEAD = 2, 4, 128, 64
        q, k, v, sm_scale = _make_qkv(Z, H, N_CTX, D_HEAD, dtype=torch.float16)
        out = self.attention(q, k, v, sm_scale)
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_output_no_inf(self) -> None:
        Z, H, N_CTX, D_HEAD = 2, 4, 128, 64
        q, k, v, sm_scale = _make_qkv(Z, H, N_CTX, D_HEAD, dtype=torch.float16)
        out = self.attention(q, k, v, sm_scale)
        assert not torch.isinf(out).any(), "Output contains Inf values"
