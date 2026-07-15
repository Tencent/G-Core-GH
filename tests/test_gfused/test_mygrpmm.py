import torch
import math
import pytest
import tilelang
import tilelang.language as T

# 1. 加 bwd
# 2. perf
# 3. fp8
# 4. dyn shape
# 5. report and PR


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def grouped_gemm_fwd(
    A,
    B,
    batch_sizes,
    batch_offsets,
    batch_padded_offsets,
    block_M,
    block_N,
    block_K,
    num_stages=2,
    threads=128,
    dtype=T.float16
):
    """
    args:
        a (torch.Tensor): Input tensor of shape (M, K).
        b (torch.Tensor): Input tensor of shape (G, K, N).
    """
    batch_sum, batch_count, K, N = T.const("batch_sum, batch_count, K, N")
    accum_dtype = T.float32

    A: T.Tensor([batch_sum, K], dtype)
    B: T.Tensor([batch_count, K, N], dtype)
    C = T.empty([batch_sum, N], dtype)
    batch_sizes: T.Tensor([batch_count], T.int32)
    batch_offsets: T.Tensor([batch_count], T.int32)
    batch_padded_offsets: T.Tensor([batch_count], T.int32)

    with T.Kernel(
        T.ceildiv(batch_sum, block_M) + batch_count, T.ceildiv(N, block_N), threads=threads
    ) as (bx, by):
        A_shared = T.alloc_shared([block_M, block_K], dtype)
        B_shared = T.alloc_shared([block_K, block_N], dtype)
        C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
        cur_batch_idx = T.alloc_var(dtype=T.int32)
        cur_batch_size = T.alloc_var(dtype=T.int32)

        m_start_padded = bx * block_M

        for i in range(batch_count):
            in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
            cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

        cur_batch_size = batch_sizes[cur_batch_idx]
        m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
        actual_rows = T.max(
            0,
            T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded)
        )

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[m_start:m_start + block_M, k * block_K:(k + 1) * block_K], A_shared)
            T.copy(
                B[cur_batch_idx, k * block_K:(k + 1) * block_K, by * block_N:(by + 1) * block_N],
                B_shared
            )
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            if i < actual_rows:
                C[m_start + i, by * block_N + j] = C_local[i, j]

    return C


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def grouped_gemm_bwd(
    A,
    B,
    batch_sizes,
    batch_offsets,
    block_M,
    block_N,
    block_K,
    num_stages=2,
    threads=128,
    dtype=T.float16
):
    """
    args:
        a (torch.Tensor): Input tensor of shape (M, K).
        b (torch.Tensor): Input tensor of shape (M, N). dO.
    """
    batch_sum, batch_count, K, N = T.const("batch_sum, batch_count, K, N")
    accum_dtype = T.float32

    A: T.Tensor([batch_sum, K], dtype)
    B: T.Tensor([batch_sum, N], dtype)
    batch_sizes: T.Tensor([batch_count], T.int32)
    batch_offsets: T.Tensor([batch_count], T.int32)
    C = T.empty([batch_count, K, N], dtype)

    with T.Kernel(T.ceildiv(K, block_M), T.ceildiv(N, block_N), batch_count,
                  threads=threads) as (bx, by, bz):
        A_shared = T.alloc_shared([block_K, block_M], dtype)
        B_shared = T.alloc_shared([block_K, block_N], dtype)
        C_local = T.alloc_fragment([block_M, block_N], accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(batch_sizes[bz], block_K), num_stages=num_stages):
            for i, j in T.Parallel(block_K, block_M):
                in_cur_batch = k * block_K + i < batch_sizes[bz]
                A_shared[i, j] = T.if_then_else(
                    in_cur_batch, A[batch_offsets[bz] + k * block_K + i, bx * block_M + j], 0
                )
            for i, j in T.Parallel(block_K, block_N):
                in_cur_batch = k * block_K + i < batch_sizes[bz]
                B_shared[i, j] = T.if_then_else(
                    in_cur_batch, B[batch_offsets[bz] + k * block_K + i, by * block_N + j], 0
                )
            T.gemm(A_shared, B_shared, C_local, transpose_A=True)

        T.copy(C_local, C[bz, bx * block_M, by * block_N])

    return C


class Mygrpmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, batch_sizes, block_M, block_N, block_K, num_stages, threads):
        padding_M = block_M
        batch_sum = a.shape[0]  # [M, K]
        batch_count = b.shape[0]  # [G, K, N]
        K = a.shape[1]

        assert a.shape[1] == b.shape[1]  # K
        assert batch_sizes.shape[0] == batch_count  # G
        assert batch_sizes.sum() == batch_sum  # M

        batch_offsets_list = [0]
        batch_padded_offsets_list = [0]
        for i in range(batch_count - 1):
            batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes[i])
        for i in range(batch_count - 1):
            batch_padded_offsets_list.append(
                batch_padded_offsets_list[-1] +
                math.ceil((batch_sizes[i] + 1) / padding_M) * padding_M
            )
        batch_offsets = torch.tensor(batch_offsets_list, device=a.device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(
            batch_padded_offsets_list, device=a.device, dtype=torch.int32
        )

        o = grouped_gemm_fwd(
            a, b, batch_sizes, batch_offsets, batch_padded_offsets, block_M, block_N, block_K,
            num_stages, threads
        )
        ctx.save_for_backward(a, b, batch_sizes, batch_offsets, batch_padded_offsets)
        ctx.batch_sum = batch_sum
        ctx.batch_count = batch_count
        ctx.K = K
        ctx.block_M = block_M
        ctx.block_N = block_N
        ctx.block_K = block_K
        ctx.num_stages = num_stages
        ctx.threads = threads
        return o

    @staticmethod
    def backward(ctx, grad_output):
        block_M = ctx.block_M
        block_N = ctx.block_N
        block_K = ctx.block_K
        num_stages = ctx.num_stages
        threads = ctx.threads

        A, B, batch_sizes, batch_offsets, batch_padded_offsets = ctx.saved_tensors

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        A, B, batch_sizes, grad_output = [
            maybe_contiguous(x) for x in (A, B, batch_sizes, grad_output)
        ]

        dA = grouped_gemm_fwd(
            grad_output,
            B.transpose(-1, -2).contiguous(),
            batch_sizes,
            batch_offsets,
            batch_padded_offsets,
            block_M,
            block_K,
            block_K,
            num_stages,
            threads,
        )
        dB = grouped_gemm_bwd(
            A, grad_output, batch_sizes, batch_offsets, block_M, block_N, block_K, num_stages,
            threads
        )
        return dA, dB, None, None, None, None, None, None


def ref_program(a, b, batch_sizes):
    assert a.shape[0] == sum(batch_sizes)
    assert b.shape[0] == len(batch_sizes)

    output = torch.empty((sum(batch_sizes), b.shape[2]), device=a.device, dtype=a.dtype)

    start = 0
    a_list = []
    b_list = []
    for i, size in enumerate(batch_sizes):
        end = start + size
        part_a = a[start:end]
        part_b = b[i]
        output[start:end] = torch.mm(part_a, part_b)

        a_list.append(part_a)
        b_list.append(part_b)
        start = end

    return output


def construct_inputs(batch_sizes_list, K, N, trans_b, padding_M, device, dtype):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count - 1):
        batch_padded_offsets_list.append(
            batch_padded_offsets_list[-1] +
            math.ceil((batch_sizes_list[i] + 1) / padding_M) * padding_M
        )
    A = torch.randn(batch_sum, K, device=device, dtype=dtype)
    B = torch.randn(batch_count, K, N, device=device, dtype=dtype)
    C = torch.empty(batch_sum, N, device=device, dtype=dtype)
    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)
    return A, B, C, batch_sizes, batch_offsets, batch_padded_offsets


def _torch_grouped_mm(input, weight, offs):
    if hasattr(torch.nn.functional, "grouped_mm"):
        return torch.nn.functional.grouped_mm(input.to(weight.dtype), weight, offs=offs)
    if hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm(input.to(weight.dtype), weight, offs=offs)
    pytest.skip("torch grouped_mm is not available")


def _bench_cuda_ms(fn, warmup=5, iters=20):
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


def _run_tilelang_grouped_gemm(
    batch_sizes_list,
    K,
    N,
    block_M,
    block_N,
    block_K,
    trans_b=False,
    num_stages=2,
    threads=128,
    profile=False,
):
    if trans_b:
        raise NotImplementedError("grouped GEMM backward example does not support transposed B")

    padding_M = block_M
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(0)
    A, B, _, batch_sizes, _, _ = construct_inputs(
        batch_sizes_list, K, N, False, padding_M, device, dtype
    )

    A.requires_grad_(True)
    B.requires_grad_(True)
    O_ref = ref_program(A, B, batch_sizes_list)
    dO = torch.randn_like(O_ref)

    O_ref.backward(dO, retain_graph=True)
    dA_ref, A.grad = A.grad.clone(), None
    dB_ref, B.grad = B.grad.clone(), None

    GroupedGEMM = Mygrpmm.apply
    O = GroupedGEMM(A, B, batch_sizes, block_M, block_N, block_K, num_stages, threads)
    O.backward(dO, retain_graph=True)
    dA, A.grad = A.grad.clone(), None
    dB, B.grad = B.grad.clone(), None

    assert torch.allclose(O, O_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dA, dA_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dB, dB_ref, rtol=1e-2, atol=1e-2)

    if profile:
        from tilelang.profiler import do_bench

        latency = do_bench(
            lambda: GroupedGEMM(A, B, batch_sizes, block_M, block_N, block_K, num_stages, threads)
        )
        print(f"Latency: {latency} ms")


GROUPED_GEMM_CASES = [
    pytest.param([64], 64, 128, id="single_aligned_minimal"),
    pytest.param([17], 128, 128, id="single_small_m"),
    pytest.param([65], 256, 128, id="single_m_tail"),
    pytest.param([64, 128], 256, 256, id="official_aligned_two_experts"),
    pytest.param([63, 1], 128, 256, id="split_around_block_m"),
    pytest.param([65, 30, 100], 256, 384, id="multi_ragged_m"),
    pytest.param([1, 63, 64, 65, 127], 128, 256, id="batch_size_boundaries"),
    pytest.param([33, 64, 95], 512, 512, id="large_k_n_ragged_m"),
    pytest.param([128, 7, 129], 384, 128, id="mixed_large_and_tiny_m"),
    pytest.param([1024, 2048, 4096], 512, 512, id="long_sequence_large_segments"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("batch_sizes_list", "K", "N"),
    GROUPED_GEMM_CASES,
)
def test_tilelang_grouped_gemm(batch_sizes_list, K, N):
    _run_tilelang_grouped_gemm(
        batch_sizes_list,
        K,
        N,
        block_M=64,
        block_N=128,
        block_K=64,
        trans_b=False,
        num_stages=2,
        threads=256,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tilelang_grouped_gemm_perf_vs_torch():
    batch_sizes_list = [1024, 1, 9192, 4096]
    K = 2048
    N = 2048
    block_M = 64
    block_K = 64
    block_N = 128
    num_stages = 2
    threads = 256
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(0)
    A, B, _, batch_sizes, _, _ = construct_inputs(
        batch_sizes_list, K, N, False, block_M, device, dtype
    )
    offs = batch_sizes.cumsum(0).to(torch.int32)
    dO = torch.randn(sum(batch_sizes_list), N, device=device, dtype=dtype)
    GroupedGEMM = Mygrpmm.apply

    A_tl = A.detach().clone().requires_grad_(True)
    B_tl = B.detach().clone().requires_grad_(True)
    A_torch = A.detach().clone().requires_grad_(True)
    B_torch = B.detach().clone().requires_grad_(True)

    O_torch = _torch_grouped_mm(A_torch, B_torch, offs)
    O_torch.backward(dO)

    O_tl = GroupedGEMM(A_tl, B_tl, batch_sizes, block_M, block_N, block_K, num_stages, threads)
    O_tl.backward(dO)

    assert torch.allclose(O_tl, O_torch, rtol=1e-2, atol=1e-2)
    assert torch.allclose(A_tl.grad, A_torch.grad, rtol=1e-2, atol=1e-2)
    assert torch.allclose(B_tl.grad, B_torch.grad, rtol=1e-2, atol=1e-2)

    A_tl_bench = A.detach().clone().requires_grad_(True)
    B_tl_bench = B.detach().clone().requires_grad_(True)
    A_torch_bench = A.detach().clone().requires_grad_(True)
    B_torch_bench = B.detach().clone().requires_grad_(True)

    def run_tilelang():
        A_tl_bench.grad = None
        B_tl_bench.grad = None
        O = GroupedGEMM(
            A_tl_bench, B_tl_bench, batch_sizes, block_M, block_N, block_K, num_stages, threads
        )
        O.backward(dO)

    def run_torch():
        A_torch_bench.grad = None
        B_torch_bench.grad = None
        O = _torch_grouped_mm(A_torch_bench, B_torch_bench, offs)
        O.backward(dO)

    tilelang_ms = _bench_cuda_ms(run_tilelang)
    torch_ms = _bench_cuda_ms(run_torch)
    print(f"TileLang grouped GEMM fwd+bwd: {tilelang_ms:.3f} ms")
    print(f"Torch grouped_mm fwd+bwd: {torch_ms:.3f} ms")
    print(f"TileLang / Torch latency: {tilelang_ms / torch_ms:.3f}x")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_matmul_backward_perf_ratio():
    M = 4096
    K = 2048
    N = 2048
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(0)
    A = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=dtype, requires_grad=True)
    dO = torch.randn(M, N, device=device, dtype=dtype)

    def run_forward():
        A @ B

    O = A @ B

    def run_backward():
        A.grad = None
        B.grad = None
        O.backward(dO, retain_graph=True)

    fwd_ms = _bench_cuda_ms(run_forward)
    bwd_ms = _bench_cuda_ms(run_backward)
    print(f"Torch matmul forward: {fwd_ms:.3f} ms")
    print(f"Torch matmul backward: {bwd_ms:.3f} ms")
    print(f"Torch matmul backward / forward latency: {bwd_ms / fwd_ms:.3f}x")
