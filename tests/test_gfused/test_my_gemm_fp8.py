import pytest
import torch
import tilelang
import tilelang.language as T
from tilelang.utils import determine_fp8_type


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
    (128, 4096, 4096),   # MoE gate_up / attn o_proj: hidden -> hidden
    (1024, 4096, 4096),
    (4096, 4096, 4096),  # gate_up, full-sequence
    (128, 1024, 4096),   # q_lora_b: hidden -> q_lora_rank
    (512, 1024, 4096),
    (128, 4096, 1024),   # o_proj: q_lora_rank -> hidden
    (1024, 4096, 1024),
    (128, 129280, 4096), # lm_head: hidden -> vocab
    (256, 129280, 4096),
]


@pytest.mark.parametrize("dtype", [determine_fp8_type(), determine_fp8_type("e5m2")])
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