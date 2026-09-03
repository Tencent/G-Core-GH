"""Basic DeepGEMM FP8 GEMM vs QAT (dequant + bf16 matmul) checks.

Mirrors ``tests/test_gfused/test_my_gemm_fp8.py`` correctness style:
same block-quant layout (act 1x128, weight 128x128), compare kernel
output against QAT with ``calc_diff`` as the primary metric.
"""

from __future__ import annotations

import pytest
import torch

deep_gemm = pytest.importorskip("deep_gemm")


GRP = 128

# Subset of DSV4-ish shapes from test_my_gemm_fp8 (keep light for test_libs).
SHAPES = [
    # short seq
    (128, 4096, 2048),  # MoE down_proj: hidden -> moe_intermediate
    (512, 4096, 2048),
    (2048, 4096, 2048),  # down_proj, long sequence
    (128, 4096, 4096),   # MoE gate_up / attn o_proj: hidden -> hidden
    (1024, 4096, 4096),
    (4096, 4096, 4096),  # gate_up, full-sequence
    (128, 1024, 4096),   # q_lora_b: hidden -> q_lora_rank
    (512, 1024, 4096),
    (128, 4096, 1024),   # o_proj: q_lora_rank -> hidden
    (1024, 4096, 1024),
    (128, 129280, 4096), # lm_head: hidden -> vocab
    (256, 129280, 4096),

    # long seq
    (8096, 4096, 2048),  # MoE down_proj: hidden -> moe_intermediate
    (8096, 4096, 2048),
    (8096, 4096, 2048),  # down_proj, long sequence
    (8096, 4096, 4096),   # MoE gate_up / attn o_proj: hidden -> hidden
    (8096, 4096, 4096),
    (8096, 4096, 4096),  # gate_up, full-sequence
    (8096, 1024, 4096),   # q_lora_b: hidden -> q_lora_rank
    (8096, 1024, 4096),
    (8096, 4096, 1024),   # o_proj: q_lora_rank -> hidden
    (8096, 4096, 1024),
    (8096, 129280, 4096), # lm_head: hidden -> vocab
    (8096, 129280, 4096),
]


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    # https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py#L134-L138
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return float(1 - sim)


def report_diff(tag: str, shape, my_f: torch.Tensor, ref_f: torch.Tensor):
    abs_diff = (my_f - ref_f).abs()
    avg_abs = abs_diff.mean().item()
    avg_rel = (abs_diff / ref_f.abs().clamp_min(1e-8)).mean().item()
    diff = calc_diff(my_f, ref_f)
    m, n, k = shape
    print(
        f"  [{tag}] shape=({m},{n},{k}) "
        f"avg_abs={avg_abs:.4e} avg_rel={avg_rel:.4e} calc_diff={diff:.4e}"
    )
    return avg_abs, avg_rel, diff


def _use_ue8m0() -> bool:
    # SM90: FP32 SF; SM100+: UE8M0 (matches DeepGEMM tests/generators.py).
    return torch.cuda.get_device_capability()[0] >= 10


def _quant_fp8(x: torch.Tensor, w: torch.Tensor):
    use_ue8m0 = _use_ue8m0()
    a = deep_gemm.per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0)
    b = deep_gemm.per_block_cast_to_fp8(w, use_ue8m0=use_ue8m0)
    return a, b, use_ue8m0


def _dequant_act(x_fp8: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
    m, k = x_fp8.shape
    assert k % GRP == 0
    return (
        x_fp8.float().view(m, k // GRP, GRP) * x_s.float().unsqueeze(-1)
    ).reshape(m, k).to(torch.bfloat16)


def _dequant_weight(w_fp8: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
    n, k = w_fp8.shape
    assert n % GRP == 0 and k % GRP == 0
    return (
        w_fp8.float().view(n // GRP, GRP, k // GRP, GRP)
        * w_s.float()[:, None, :, None]
    ).reshape(n, k).to(torch.bfloat16)


def _qat_gemm(a, b) -> torch.Tensor:
    """QAT: fp8 --dequant--> bf16 -> gemm."""
    x = _dequant_act(a[0], a[1])
    w = _dequant_weight(b[0], b[1])
    return x @ w.T


def _deep_gemm_nt(a, b, out: torch.Tensor, *, use_ue8m0: bool) -> torch.Tensor:
    deep_gemm.fp8_gemm_nt(a, b, out, disable_ue8m0_cast=not use_ue8m0)
    return out


def _require_cuda_sm90_plus():
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("DeepGEMM requires SM90+")


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8_gemm_nt_vs_qat(shape):
    _require_cuda_sm90_plus()
    torch.manual_seed(0)
    m, n, k = shape
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    a, b, use_ue8m0 = _quant_fp8(x, w)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    _deep_gemm_nt(a, b, out, use_ue8m0=use_ue8m0)
    qat = _qat_gemm(a, b)

    assert out.shape == (m, n)
    assert torch.isfinite(out.float()).all()

    # deep_gemm may round per K-tile; prefer calc_diff over avg_abs (see mygemm_fp8 notes).
    avg_abs, avg_rel, diff = report_diff("qat", shape, out.float(), qat.float())
    assert diff < 1e-3, f"calc_diff={diff}"
    assert avg_abs < 5e-1, f"avg_abs={avg_abs}"
    assert avg_rel < 5e-1, f"avg_rel={avg_rel}"


def test_m_grouped_fp8_gemm_nt_contiguous_vs_qat():
    _require_cuda_sm90_plus()
    if not hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous"):
        pytest.skip("m_grouped_fp8_gemm_nt_contiguous unavailable")

    torch.manual_seed(0)
    num_experts, m_per_expert, n, k = 4, 128, 256, 512
    m = num_experts * m_per_expert
    use_ue8m0 = _use_ue8m0()

    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
    m_indices = (
        torch.arange(num_experts, device="cuda")
        .repeat_interleave(m_per_expert)
        .to(torch.int32)
    )

    a = deep_gemm.per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0)
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    w_s = torch.empty(
        (num_experts, (n + GRP - 1) // GRP, (k + GRP - 1) // GRP),
        device="cuda",
        dtype=torch.float32,
    )
    for i in range(num_experts):
        wi_fp8, wi_s = deep_gemm.per_block_cast_to_fp8(w[i], use_ue8m0=use_ue8m0)
        w_fp8[i] = wi_fp8
        w_s[i] = wi_s.float()
    b = (w_fp8, w_s)

    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a, b, out, m_indices, disable_ue8m0_cast=not use_ue8m0
    )

    # QAT per-expert
    x_bf16 = _dequant_act(a[0], a[1])
    ref = torch.empty_like(out, dtype=torch.float32)
    for i in range(num_experts):
        rows = m_indices == i
        w_i = _dequant_weight(w_fp8[i], w_s[i])
        ref[rows] = (x_bf16[rows].float() @ w_i.float().T)

    avg_abs, avg_rel, diff = report_diff(
        "grouped_qat", (m, n, k), out.float(), ref
    )
    assert diff < 1e-3, f"calc_diff={diff}"
    assert avg_abs < 5e-1, f"avg_abs={avg_abs}"
    assert avg_rel < 5e-1, f"avg_rel={avg_rel}"


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


def _print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def test_perf_fp8_gemm_nt_vs_qat():
    """Latency: bf16 matmul vs QAT vs deep_gemm (prequant outside)."""
    '''
DeepGEMM fp8_gemm_nt (ms): bf16 vs qat vs deep_gemm (prequant outside)
  *_vs_* = baseline / deep_gemm  (>1 means deep_gemm faster)
shape             bf16    qat     deep_gemm  dg_vs_bf16  dg_vs_qat
----------------  ------  ------  ---------  ----------  ---------
128x4096x2048     0.021   0.141   0.021      0.98x       6.74x    
512x4096x2048     0.069   0.196   0.041      1.70x       4.82x    
2048x4096x2048    0.255   0.410   0.143      1.79x       2.87x    
128x4096x4096     0.044   0.271   0.023      1.90x       11.61x   
1024x4096x4096    0.256   0.515   0.140      1.83x       3.68x    
4096x4096x4096    1.042   1.453   0.536      1.95x       2.71x    
128x1024x4096     0.018   0.086   0.021      0.87x       4.03x    
512x1024x4096     0.042   0.124   0.026      1.61x       4.80x    
128x4096x1024     0.014   0.075   0.021      0.67x       3.65x    
1024x4096x1024    0.069   0.141   0.042      1.64x       3.37x    
128x129280x4096   0.981   7.293   0.521      1.88x       14.01x   
256x129280x4096   1.940   8.256   0.990      1.96x       8.34x    
8096x4096x2048    1.024   1.329   0.544      1.88x       2.44x    
8096x4096x2048    1.024   1.329   0.544      1.88x       2.44x    
8096x4096x2048    1.021   1.326   0.543      1.88x       2.44x    
8096x4096x4096    2.032   2.630   1.059      1.92x       2.48x    
8096x4096x4096    2.033   2.631   1.059      1.92x       2.48x    
8096x4096x4096    2.032   2.630   1.059      1.92x       2.48x    
8096x1024x4096    0.507   0.947   0.310      1.64x       3.06x    
8096x1024x4096    0.507   0.947   0.309      1.64x       3.06x    
8096x4096x1024    0.515   0.670   0.282      1.83x       2.38x    
8096x4096x1024    0.516   0.670   0.281      1.83x       2.38x    
8096x129280x4096  61.123  67.687  31.097     1.97x       2.18x    
8096x129280x4096  61.045  67.700  31.095     1.96x       2.18x 
    '''
    _require_cuda_sm90_plus()
    rows = []
    for m, n, k in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        a, b, use_ue8m0 = _quant_fp8(x, w)
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

        def run_bf16():
            return x @ w.T

        def run_qat():
            return _qat_gemm(a, b)

        def run_dg():
            return _deep_gemm_nt(a, b, out, use_ue8m0=use_ue8m0)

        run_dg()
        torch.cuda.synchronize()
        ms_bf16 = _bench_cuda_ms(run_bf16)
        ms_qat = _bench_cuda_ms(run_qat)
        ms_dg = _bench_cuda_ms(run_dg)
        rows.append((
            f"{m}x{n}x{k}",
            f"{ms_bf16:.3f}",
            f"{ms_qat:.3f}",
            f"{ms_dg:.3f}",
            f"{ms_bf16 / ms_dg:.2f}x",
            f"{ms_qat / ms_dg:.2f}x",
        ))

    print("\nDeepGEMM fp8_gemm_nt (ms): bf16 vs qat vs deep_gemm (prequant outside)")
    print("  *_vs_* = baseline / deep_gemm  (>1 means deep_gemm faster)")
    _print_table(
        ["shape", "bf16", "qat", "deep_gemm", "dg_vs_bf16", "dg_vs_qat"],
        rows,
    )
