import pytest
import tilelang
import tilelang.language as T
import torch
import torch.autograd as autograd

try:
    from tilelang.language.fp8 import determine_fp8_type
except ModuleNotFoundError:
    from tilelang.utils import determine_fp8_type

from tile_kernels.quant import (
    per_block_cast,
    per_block_cast_back,
    per_token_cast,
    per_token_cast_back,
)

from gfused.mygemm_fp8 import gemm_impl, gemm_fp8_block_scaling_impl


def calc_diff(x, y):
    # https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py#L134-L138
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


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


class ExactMatmulFunction(autograd.Function):
    """Straight-through ``C = A @ B^T``; reference for FP8 STE backward checks."""
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return A @ B.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_tensors
        return grad_C @ B, grad_C.transpose(-1, -2) @ A


def test_backward_gradcheck():
    # A (M, K), B (N, K), C = A @ B^T (M, N)
    torch.manual_seed(0)
    M, N, K = 32, 128, 64
    A = torch.randn(M, K, dtype=torch.float64, device="cuda", requires_grad=True)
    B = torch.randn(N, K, dtype=torch.float64, device="cuda", requires_grad=True)
    grad_C = torch.randn(M, N, dtype=torch.float64, device="cuda")

    C = ExactMatmulFunction.apply(A, B)
    C.backward(grad_C)

    # ground truth: torch 原生 A @ B^T
    A_ref = A.detach().clone().requires_grad_(True)
    B_ref = B.detach().clone().requires_grad_(True)
    (A_ref @ B_ref.T).backward(grad_C)
    assert torch.allclose(A.grad, A_ref.grad)
    assert torch.allclose(B.grad, B_ref.grad)

    # numerical vs analytical
    assert autograd.gradcheck(
        ExactMatmulFunction.apply,
        (A.detach().requires_grad_(True), B.detach().requires_grad_(True)),
        atol=1e-3,
        rtol=1e-3,
    )


# Shapes derived from DeepSeek-V4-Flash config.json:
#   hidden_size=4096, moe_intermediate_size=2048, q_lora_rank=1024,
#   o_lora_rank=1024, vocab_size=129280.
# M is a multiple of block_M=128 to avoid out-of-bounds writes.
MODEL_SHAPES = [
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


# 基本的 GEMM
@pytest.mark.parametrize("dtype", [determine_fp8_type(), determine_fp8_type("e4m3")])
@pytest.mark.parametrize("shape", MODEL_SHAPES, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_gemm_fp8(shape, dtype):
    M, N, K = shape
    torch_dtype = T.dtype(dtype).as_torch()

    a = torch.randn(M, K, dtype=torch.float16, device="cuda").to(dtype=torch_dtype)
    b = torch.randn(N, K, dtype=torch.float16, device="cuda").to(dtype=torch_dtype)

    c = gemm_impl(a, b, 128, 128, 64, dtype=torch_dtype, out_dtype=torch_dtype)
    ref_c = (a.half() @ b.half().T).to(dtype=torch_dtype)

    diff = calc_diff(c, ref_c)
    assert diff < 1e-3


def test_perf_fp8_vs_bf16():
    """Same tilelang GEMM: FP8 e4m3 should beat BF16 on latency."""
    # tests/test_gfused/test_my_gemm_fp8.py::test_perf_fp8_vs_bf16   GEMM 4096x4096x4096: fp8=0.575 ms, bf16=1.087 ms, speedup=1.89x
    M, N, K = 4096, 4096, 4096
    block_M, block_N, block_K = 128, 128, 64

    kernel_fp8 = gemm_impl.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype=torch.float8_e4m3fn,
        out_dtype=torch.float8_e4m3fn,
    )
    latency_fp8 = kernel_fp8.get_profiler(tilelang.TensorSupplyType.Integer).do_bench()

    kernel_bf16 = gemm_impl.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    latency_bf16 = kernel_bf16.get_profiler(tilelang.TensorSupplyType.Integer).do_bench()

    speedup = latency_bf16 / latency_fp8
    print(
        f"  GEMM {M}x{N}x{K}: fp8={latency_fp8:.3f} ms, "
        f"bf16={latency_bf16:.3f} ms, speedup={speedup:.2f}x"
    )
    assert speedup > 1.2, (
        f"expected fp8 faster than bf16, got speedup={speedup:.2f}x "
        f"(fp8={latency_fp8:.3f} ms, bf16={latency_bf16:.3f} ms)"
    )


def _ref_gemm_fp8_blk_quant_qat(
    aq: torch.Tensor,
    a_s: torch.Tensor,
    bq: torch.Tensor,
    b_s: torch.Tensor,
    grp_sz: int = 128,
) -> torch.Tensor:
    """QAT: fp8 --cast_back--> bf16 -> gemm (scales cast to fp32 for tile_kernels)."""
    a = per_token_cast_back((aq, a_s.float()), 'bf16', grp_sz)
    b = per_block_cast_back((bq, b_s.float()), 'bf16', (grp_sz, grp_sz))
    return a @ b.T


def ceildiv(a, b):
    return (a + b - 1) // b


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


def _quant_e8m0(x: torch.Tensor, w: torch.Tensor, grp_sz: int):
    w_q, w_s = per_block_cast(w, 'e4m3', (grp_sz, grp_sz), round_sf=True)
    w_s = w_s.to(dtype=torch.float8_e8m0fnu)
    x_q, x_s = per_token_cast(x, 'e4m3', grp_sz, round_sf=True)
    x_s = x_s.to(dtype=torch.float8_e8m0fnu)
    return x_q, x_s, w_q, w_s


# fp8 + block quant GEMM
# TODO grad 是如何处理 quant block 的？
@pytest.mark.parametrize("shape", MODEL_SHAPES, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_gemm_fp8_blk_quant(shape):
    torch.manual_seed(0)
    m, n, k = shape
    grp_sz = 128
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    # e8m0 路径：quant 时 round_sf=True，再存 e8m0（勿先 float scale 再事后 cast）
    w_q, w_s = per_block_cast(w, 'e4m3', (grp_sz, grp_sz), round_sf=True)
    w_s = w_s.to(dtype=torch.float8_e8m0fnu)
    x_q, x_s = per_token_cast(x, 'e4m3', grp_sz, round_sf=True)
    x_s = x_s.to(dtype=torch.float8_e8m0fnu)

    my_c = gemm_fp8_block_scaling_impl(x_q, x_s, w_q, w_s)
    assert my_c.shape == (m, n)
    assert my_c.dtype == torch.bfloat16
    assert torch.isfinite(my_c.float()).all()
    my_f = my_c.float()

    # baseline 1: QAT cast_back + bf16 gemm（整体误差）
    qat_c = _ref_gemm_fp8_blk_quant_qat(x_q, x_s, w_q, w_s, grp_sz)
    avg_abs, avg_rel, diff = report_diff("qat", shape, my_f, qat_c.float())
    assert avg_abs < 1e-2, f"qat avg_abs={avg_abs}"
    assert avg_rel < 1e-2, f"qat avg_rel={avg_rel}"
    assert diff < 1e-5, f"qat calc_diff={diff}"
    torch.testing.assert_close(my_f, qat_c.float(), atol=1e-1, rtol=1e-2)


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


def test_perf_gemm_fp8_blk_quant():
    # bf16 matmul vs QAT(quant+dequant+bf16) vs quant+fp8 blk gemm
    '''
blk_quant GEMM (ms): bf16 vs qat(quant+dequant+bf16) vs quant+fp8
shape             bf16    qat     fp8     fp8_vs_bf16  fp8_vs_qat
----------------  ------  ------  ------  -----------  ----------
128x4096x2048     0.021   0.158   0.134   0.16x        1.18x
512x4096x2048     0.069   0.168   0.137   0.50x        1.22x
2048x4096x2048    0.256   0.316   0.211   1.21x        1.50x
128x4096x4096     0.045   0.159   0.134   0.33x        1.18x
1024x4096x4096    0.256   0.333   0.206   1.24x        1.62x
4096x4096x4096    1.039   1.142   0.675   1.54x        1.69x
128x1024x4096     0.019   0.175   0.135   0.14x        1.30x
512x1024x4096     0.042   0.162   0.136   0.31x        1.19x
128x4096x1024     0.015   0.166   0.132   0.11x        1.26x
1024x4096x1024    0.069   0.168   0.139   0.49x        1.21x
128x129280x4096   0.976   2.213   1.195   0.82x        1.85x
256x129280x4096   1.933   3.182   1.795   1.08x        1.77x
8096x4096x2048    1.021   1.106   0.716   1.43x        1.54x
8096x4096x4096    2.033   2.173   1.299   1.57x        1.67x
8096x1024x4096    0.507   0.619   0.381   1.33x        1.63x
8096x4096x1024    0.515   0.574   0.424   1.21x        1.35x
8096x129280x4096  61.115  62.325  38.959  1.57x        1.60x
    '''

    grp_sz = 128
    rows = []
    for m, n, k in dict.fromkeys(MODEL_SHAPES):
        torch.manual_seed(0)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

        def run_bf16():
            return x @ w.T

        def run_qat():
            x_q, x_s, w_q, w_s = _quant_e8m0(x, w, grp_sz)
            return _ref_gemm_fp8_blk_quant_qat(x_q, x_s, w_q, w_s, grp_sz)

        def run_fp8():
            x_q, x_s, w_q, w_s = _quant_e8m0(x, w, grp_sz)
            return gemm_fp8_block_scaling_impl(x_q, x_s, w_q, w_s)

        ms_bf16 = _bench_cuda_ms(run_bf16)
        ms_qat = _bench_cuda_ms(run_qat)
        ms_fp8 = _bench_cuda_ms(run_fp8)
        assert ms_fp8 < ms_qat, (
            f"shape=({m},{n},{k}): expected quant+fp8 faster than QAT, "
            f"got fp8={ms_fp8:.3f} ms, qat={ms_qat:.3f} ms"
        )
        rows.append((
            f"{m}x{n}x{k}",
            f"{ms_bf16:.3f}",
            f"{ms_qat:.3f}",
            f"{ms_fp8:.3f}",
            f"{ms_bf16 / ms_fp8:.2f}x",
            f"{ms_qat / ms_fp8:.2f}x",
        ))

    print("\nblk_quant GEMM (ms): bf16 vs qat(quant+dequant+bf16) vs quant+fp8")
    _print_table(
        ["shape", "bf16", "qat", "fp8", "fp8_vs_bf16", "fp8_vs_qat"],
        rows,
    )


def test_perf_quant_dequant_vs_gemm():
    # quant / dequant 相对 gemm 的耗时占比
    grp_sz = 128
    rows = []
    for m, n, k in dict.fromkeys(MODEL_SHAPES):
        torch.manual_seed(0)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        x_q, x_s, w_q, w_s = _quant_e8m0(x, w, grp_sz)

        def run_quant_x():
            qx, sx = per_token_cast(x, 'e4m3', grp_sz, round_sf=True)
            return qx, sx.to(dtype=torch.float8_e8m0fnu)

        def run_quant_w():
            qw, sw = per_block_cast(w, 'e4m3', (grp_sz, grp_sz), round_sf=True)
            return qw, sw.to(dtype=torch.float8_e8m0fnu)

        def run_dequant_x():
            return per_token_cast_back((x_q, x_s.float()), 'bf16', grp_sz)

        def run_dequant_w():
            return per_block_cast_back((w_q, w_s.float()), 'bf16', (grp_sz, grp_sz))

        def run_bf16():
            return x @ w.T

        def run_fp8():
            return gemm_fp8_block_scaling_impl(x_q, x_s, w_q, w_s)

        ms_qx = _bench_cuda_ms(run_quant_x)
        ms_qw = _bench_cuda_ms(run_quant_w)
        ms_dqx = _bench_cuda_ms(run_dequant_x)
        ms_dqw = _bench_cuda_ms(run_dequant_w)
        ms_bf16 = _bench_cuda_ms(run_bf16)
        ms_fp8 = _bench_cuda_ms(run_fp8)

        ms_quant = ms_qx + ms_qw
        ms_dequant = ms_dqx + ms_dqw
        qat_total = ms_quant + ms_dequant + ms_bf16
        fp8_total = ms_quant + ms_fp8
        rows.append((
            f"{m}x{n}x{k}",
            f"{ms_qx:.3f}",
            f"{ms_qw:.3f}",
            f"{ms_dqx:.3f}",
            f"{ms_dqw:.3f}",
            f"{ms_bf16:.3f}",
            f"{ms_fp8:.3f}",
            f"{100 * ms_quant / qat_total:.1f}%",
            f"{100 * ms_dequant / qat_total:.1f}%",
            f"{100 * ms_bf16 / qat_total:.1f}%",
            f"{100 * ms_quant / fp8_total:.1f}%",
            f"{100 * ms_fp8 / fp8_total:.1f}%",
        ))

    print("\nquant/dequant vs gemm breakdown (ms / % of path)")
    _print_table(
        [
            "shape",
            "quant_x",
            "quant_w",
            "dequant_x",
            "dequant_w",
            "bf16",
            "fp8",
            "qat_quant%",
            "qat_dequant%",
            "qat_gemm%",
            "fp8_quant%",
            "fp8_gemm%",
        ],
        rows,
    )


def test_perf_my_vs_te():
    # bf16 / my(quant+gemm) / TE MyTeGroupedLinearFp8(E=1) / my_gemm_only / TE prequant weight
    '''
blk_quant GEMM vs TE (ms): my=tilelang quant+gemm, te=MyTeGroupedLinearFp8(E=1)
  my_gemm = gemm-only (prequant outside); te_pq = TE prequant weight (still quant act)
  *_vs_* = TE / my  (>1 means my faster)
shape             bf16    my      te      my_vs_te  my_pq   te_pq   my_pq_vs_te_pq
----------------  ------  ------  ------  --------  ------  ------  --------------
128x4096x2048     0.021   0.135   0.200   1.48x     0.041   0.189   4.58x
512x4096x2048     0.069   0.134   0.197   1.47x     0.055   0.185   3.37x
2048x4096x2048    0.256   0.210   0.207   0.99x     0.180   0.198   1.10x
128x4096x4096     0.045   0.131   0.183   1.40x     0.044   0.217   4.97x
1024x4096x4096    0.256   0.205   0.203   0.99x     0.166   0.210   1.27x
4096x4096x4096    1.043   0.676   0.588   0.87x     0.625   0.571   0.91x
128x1024x4096     0.017   0.135   0.195   1.44x     0.042   0.196   4.69x
512x1024x4096     0.042   0.133   0.190   1.43x     0.041   0.191   4.67x
128x4096x1024     0.014   0.133   0.188   1.41x     0.040   0.210   5.21x
1024x4096x1024    0.068   0.134   0.189   1.41x     0.056   0.208   3.72x
128x129280x4096   0.977   1.196   1.062   0.89x     0.615   0.514   0.84x
256x129280x4096   1.936   1.799   1.548   0.86x     1.221   0.999   0.82x
8096x4096x2048    1.023   0.718   0.587   0.82x     0.678   0.578   0.85x
8096x4096x4096    2.033   1.299   1.134   0.87x     1.240   1.117   0.90x
8096x1024x4096    0.509   0.383   0.383   1.00x     0.331   0.377   1.14x
8096x4096x1024    0.517   0.424   0.312   0.74x     0.396   0.308   0.78x
8096x129280x4096  61.126  38.932  31.274  0.80x     38.334  30.667  0.80x
    '''
    try:
        from gpatch_v4.kernel.quantize.eager_quant_kernels import (
            quant_fp8_e4m3_scale_e8m0,
        )
        from gpatch_v4.models.deepseek_v4.fp8 import MyTeGroupedLinearFp8
        from gpatch_v4.models.deepseek_v4.fp8_tensor import Fp8TensorTrain
    except ImportError as e:
        pytest.skip(f"TE / MyTeGroupedLinearFp8 unavailable: {e}")

    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    cc = torch.cuda.get_device_capability()
    cuda_ver = torch.version.cuda
    if cc < (9, 0) or cuda_ver is None or tuple(int(x) for x in cuda_ver.split(".")[:2]) < (12, 9):
        pytest.skip("requires Hopper + CUDA >= 12.9 for TE Float8BlockScaling")

    grp_sz = 128
    te_mod = MyTeGroupedLinearFp8(num_gemms=1)
    rows = []
    for m, n, k in dict.fromkeys(MODEL_SHAPES):
        torch.manual_seed(0)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        w_te = w.unsqueeze(0)  # [1, N, K]
        x_q, x_s, w_q, w_s = _quant_e8m0(x, w, grp_sz)
        w_data, w_scale = quant_fp8_e4m3_scale_e8m0(w_te)
        w_fp8 = Fp8TensorTrain(
            w_data.view(torch.uint8),
            w_scale,
            torch.bfloat16,
        )

        def run_bf16():
            return x @ w.T

        def run_my():
            xq, xs, wq, ws = _quant_e8m0(x, w, grp_sz)
            return gemm_fp8_block_scaling_impl(xq, xs, wq, ws)

        def run_te():
            with torch.no_grad():
                return te_mod(x, w_te, [m])

        def run_my_pq():
            return gemm_fp8_block_scaling_impl(x_q, x_s, w_q, w_s)

        def run_te_pq():
            with torch.no_grad():
                return te_mod(x, w_fp8, [m])

        # warmup compile / quantizer paths once per shape
        run_my()
        run_te()
        run_my_pq()
        run_te_pq()
        torch.cuda.synchronize()

        ms_bf16 = _bench_cuda_ms(run_bf16)
        ms_my = _bench_cuda_ms(run_my)
        ms_te = _bench_cuda_ms(run_te)
        ms_my_pq = _bench_cuda_ms(run_my_pq)
        ms_te_pq = _bench_cuda_ms(run_te_pq)
        rows.append((
            f"{m}x{n}x{k}",
            f"{ms_bf16:.3f}",
            f"{ms_my:.3f}",
            f"{ms_te:.3f}",
            f"{ms_te / ms_my:.2f}x",
            f"{ms_my_pq:.3f}",
            f"{ms_te_pq:.3f}",
            f"{ms_te_pq / ms_my_pq:.2f}x",
        ))

    print("\nblk_quant GEMM vs TE (ms): my=tilelang quant+gemm, te=MyTeGroupedLinearFp8(E=1)")
    print("  my_gemm = gemm-only (prequant outside); te_pq = TE prequant weight (still quant act)")
    print("  *_vs_* = TE / my  (>1 means my faster)")
    _print_table(
        [
            "shape",
            "bf16",
            "my",
            "te",
            "my_vs_te",
            "my_pq",
            "te_pq",
            "my_pq_vs_te_pq",
        ],
        rows,
    )
