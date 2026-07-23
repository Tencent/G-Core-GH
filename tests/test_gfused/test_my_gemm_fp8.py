import pytest
import torch
import tilelang
import tilelang.language as T

pytest.importorskip("tilelang.language.fp8")
from tilelang.language.fp8 import determine_fp8_type


# pip3 install pytest tilelang==0.1.12 tile_kernels==1.0.0


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((N, K), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_N, block_K), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


# Shapes derived from DeepSeek-V4-Flash config.json:
#   hidden_size=4096, moe_intermediate_size=2048, q_lora_rank=1024,
#   o_lora_rank=1024, vocab_size=129280.
# M is a multiple of block_M=128 to avoid out-of-bounds writes.
MODEL_SHAPES = [
    (128, 4096, 2048),   # MoE down_proj: hidden -> moe_intermediate
    (512, 4096, 2048),
    (2048, 4096, 2048),  # down_proj, long sequence
    # (128, 4096, 4096),   # MoE gate_up / attn o_proj: hidden -> hidden
    # (1024, 4096, 4096),
    # (4096, 4096, 4096),  # gate_up, full-sequence
    # (128, 1024, 4096),   # q_lora_b: hidden -> q_lora_rank
    # (512, 1024, 4096),
    # (128, 4096, 1024),   # o_proj: q_lora_rank -> hidden
    # (1024, 4096, 1024),
    # (128, 129280, 4096), # lm_head: hidden -> vocab
    # (256, 129280, 4096),
]


@pytest.mark.parametrize("dtype", [determine_fp8_type(), determine_fp8_type("e4m3")])
@pytest.mark.parametrize("shape", MODEL_SHAPES, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_gemm_fp8(shape, dtype):
    M, N, K = shape
    torch_dtype = T.dtype(dtype).as_torch()

    a = torch.randn(M, K, dtype=torch.float16, device="cuda").to(dtype=torch_dtype)
    b = torch.randn(N, K, dtype=torch.float16, device="cuda").to(dtype=torch_dtype)

    c = matmul(a, b, 128, 128, 64, dtype)

    ref_c = (a.half() @ b.half().T).to(dtype=torch_dtype)

    diff = calc_diff(c, ref_c)
    assert diff < 1e-3


def run_regression_perf():
    M, N, K = 4096, 4096, 4096
    dtype = determine_fp8_type()
    kernel_e4m3 = matmul.compile(M=M, N=N, K=K, block_M=128, block_N=128, block_K=64, dtype=dtype)
    profiler_e4m3 = kernel_e4m3.get_profiler(tilelang.TensorSupplyType.Integer)
    if torch.version.hip is None:
        latency_e4m3 = profiler_e4m3.do_bench(backend="cupti")
        dtype = determine_fp8_type("e5m2")
        kernel_e5m2 = matmul.compile(M=M, N=N, K=K, block_M=128, block_N=128, block_K=64, dtype=dtype)
        profiler_e5m2 = kernel_e5m2.get_profiler(tilelang.TensorSupplyType.Integer)
        latency_e5m2 = profiler_e5m2.do_bench(backend="cupti")
        return (latency_e4m3 + latency_e5m2) / 2
    latency_e4m3 = profiler_e4m3.do_bench()
    return latency_e4m3


if __name__ == "__main__":
    for shape in MODEL_SHAPES:
        test_gemm_fp8(shape, determine_fp8_type())
        test_gemm_fp8(shape, determine_fp8_type("e5m2"))


import torch.autograd as autograd

_FP8_MAX = 448.0  # e4m3 largest finite value


def _block_quant_fp8(x: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Per-128-along-k blockwise e4m3 quant, matching the fprop GEMM input.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(m, k)`` in a float dtype.

    Returns
    -------
    torch.Tensor
        Shape ``(m, k)`` ``float8_e4m3fn``.
    """
    m, k = x.shape
    x_r = x.reshape(m, k // group, group).float()
    scale = x_r.abs().amax(dim=-1) / _FP8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    x_fp8 = (x_r / scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8.reshape(m, k)


class ExactMatmulFunction(autograd.Function):
    """Straight-through ``C = A @ B^T``; used to validate the backward via gradcheck."""

    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return A @ B.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_tensors
        return grad_C @ B, grad_C.transpose(-1, -2) @ A


class FP8MatmulFunction(autograd.Function):
    """FP8 GEMM (tilelang kernel) wrapped as a differentiable op.

    Forward quantizes inputs to e4m3 and runs the raw fp8 ``matmul`` kernel,
    returning the fp8 output promoted to bf16. Backward is the exact
    straight-through gradient of ``C = Aq @ Bq^T`` w.r.t. the quantized inputs.
    """

    @staticmethod
    def forward(ctx, A, B):
        Aq = _block_quant_fp8(A)
        Bq = _block_quant_fp8(B)
        ctx.save_for_backward(Aq, Bq)
        C_fp8 = matmul(Aq, Bq, 128, 128, 64, determine_fp8_type())
        return C_fp8.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad_C):
        Aq, Bq = ctx.saved_tensors
        Aq_bf = Aq.to(torch.bfloat16)
        Bq_bf = Bq.to(torch.bfloat16)
        return grad_C @ Bq_bf, grad_C.transpose(-1, -2) @ Aq_bf


def test_backward_gradcheck():
    torch.manual_seed(0)
    A = torch.randn(32, 32, dtype=torch.float64, device="cuda", requires_grad=True)
    B = torch.randn(32, 32, dtype=torch.float64, device="cuda", requires_grad=True)
    autograd.gradcheck(ExactMatmulFunction.apply, (A, B), atol=1e-3, rtol=1e-3)


def test_backward_fp8_matches_exact():
    torch.manual_seed(0)
    A = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    Aq_bf = _block_quant_fp8(A).to(torch.bfloat16)
    Bq_bf = _block_quant_fp8(B).to(torch.bfloat16)

    # Forward exercises the fp8 tilelang kernel (output saturates to fp8 range).
    fp8_out = FP8MatmulFunction.apply(A, B)
    assert fp8_out.shape == (256, 256)
    assert torch.isfinite(fp8_out.float()).all()

    # Backward must be the straight-through gradient of ``C = Aq @ Bq^T`` using
    # the quantized values actually multiplied by the kernel.
    A_grad = A.clone().requires_grad_(True)
    B_grad = B.clone().requires_grad_(True)
    FP8MatmulFunction.apply(A_grad, B_grad).sum().backward()
    Aq_grad = Aq_bf.clone().requires_grad_(True)
    Bq_grad = Bq_bf.clone().requires_grad_(True)
    ExactMatmulFunction.apply(Aq_grad, Bq_grad).sum().backward()

    assert torch.allclose(A_grad.grad, Aq_grad.grad, atol=0.05, rtol=0.05)
    assert torch.allclose(B_grad.grad, Bq_grad.grad, atol=0.05, rtol=0.05)