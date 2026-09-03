# coding=utf-8
"""Verify deep_gemm grouped GEMM (fp8 -> bf16) correctness.

Mirrors ``mmq``'s ``DeepSeekFP8GroupedGemmFunction``: fp8 block-quantized
activation (per-128 along k) and weight (per-128x128 block), fed to
``deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous``.

Reference = dequantize the fp8 tensors back to fp32, then do the per-expert
GEMM exactly. The kernel output (bf16) is compared against the reference.

Usage::
    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=600 tests/test_gpatch_v4/test_deep_gemm_grouped_gemm_fp8.py
"""

import pytest
import torch
import torch.nn.functional as F

deep_gemm = pytest.importorskip("deep_gemm")


FP8_MAX = 448.0  # e4m3 largest finite value
GROUP = 128


def _quant_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-128-along-k blockwise e4m3 quant of activation.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(m, k)`` bf16.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(x_fp8, x_scale)`` with ``x_fp8`` shape ``(m, k)`` fp8 and
        ``x_scale`` shape ``(m, k // 128)`` fp32.
    """
    m, k = x.shape
    x_r = x.reshape(m, k // GROUP, GROUP).to(torch.float32)
    amax = x_r.abs().amax(dim=-1)
    scale = amax / FP8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    x_fp8 = (x_r / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8.reshape(m, k), scale.to(torch.float32)


def _quant_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-128x128 blockwise e4m3 quant of weight.

    Parameters
    ----------
    w : torch.Tensor
        Shape ``(num_experts, n, k)`` bf16.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(w_fp8, w_scale)`` with ``w_fp8`` shape ``(num_experts, n, k)``
        fp8 and ``w_scale`` shape ``(num_experts, n // 128, k // 128)`` fp32.
    """
    e, n, k = w.shape
    w_r = w.reshape(e, n // GROUP, GROUP, k // GROUP, GROUP).to(torch.float32)
    amax = w_r.abs().amax(dim=(2, 4))
    scale = amax / FP8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    w_fp8 = (w_r / scale.unsqueeze(2).unsqueeze(4)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w_fp8.reshape(e, n, k), scale.to(torch.float32)


def _reference(x_fp8, x_scale, w_fp8, w_scale, m_indices, n):
    """Dequantize fp8 -> fp32 and do the per-expert GEMM exactly."""
    m, k = x_fp8.shape
    e = w_fp8.shape[0]
    x_deq = (x_fp8.float().reshape(m, k // GROUP, GROUP) * x_scale.unsqueeze(-1)).reshape(m, k)
    w_deq = (
        w_fp8.float().reshape(e, n // GROUP, GROUP, k // GROUP, GROUP)
        * w_scale.unsqueeze(2).unsqueeze(4)
    ).reshape(e, n, k)

    ref = torch.zeros(m, n, dtype=torch.float32, device=x_fp8.device)
    for expert in range(e):
        rows = m_indices == expert
        ref[rows] = x_deq[rows] @ w_deq[expert].T
    return ref


def _run_once(num_experts, m_per_expert, k, n):
    device = "cuda"
    m_total = num_experts * m_per_expert

    torch.manual_seed(0)
    x = torch.randn(m_total, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(num_experts, n, k, device=device, dtype=torch.bfloat16)

    # contiguous / sorted m_indices: expert 0 rows first, expert 1 next, ...
    m_indices = torch.arange(num_experts, device=device).repeat_interleave(m_per_expert).to(torch.int)

    x_fp8, x_scale = _quant_act(x)
    w_fp8, w_scale = _quant_weight(w)

    out = torch.empty(m_total, n, device=device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
        (x_fp8, x_scale), (w_fp8, w_scale), out, m_indices
    )

    ref = _reference(x_fp8, x_scale, w_fp8, w_scale, m_indices, n)

    out_f = out.float()
    ref_f = ref.float()
    max_err = (out_f - ref_f).abs().max().item()
    cos = F.cosine_similarity(out_f.flatten(), ref_f.flatten(), dim=0).item()

    assert cos > 0.999, f"cos={cos} too low (num_experts={num_experts}, k={k}, n={n})"
    assert max_err < 1.0, f"max_err={max_err} too high (num_experts={num_experts}, k={k}, n={n})"
    return max_err, cos


def test_deep_gemm_grouped_gemm_fp8_correctness():
    if not hasattr(deep_gemm, "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous"):
        pytest.skip(
            "deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous not in this image"
        )
    for num_experts, m_per_expert, k, n in [
        (8, 256, 512, 256),
        (16, 128, 1024, 512),
        (4, 256, 7168, 2048),
    ]:
        max_err, cos = _run_once(num_experts, m_per_expert, k, n)
        print(f"[deep_gemm grouped gemm] E={num_experts} m/E={m_per_expert} k={k} n={n} "
              f"max_err={max_err:.4f} cos={cos:.6f}")
