"""
myfa_mask forward / backward correctness tests.

Example usage::

    cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/wepsdl/gcore-dev
    pytest -s tests/test_gfused/test_myfa_mask.py -v
    pytest -s tests/test_gfused/test_myfa_mask.py::test_forward -v
    pytest -s tests/test_gfused/test_myfa_mask.py::test_backward -v
"""

import math

import einops
import pytest
import tilelang as tl
import tilelang.language as T
import torch
from flash_attn import flash_attn_func


def ref_attn(q, k, v, mask, scaling):
    b, h, sq, d = q.shape
    s = torch.matmul(q, einops.rearrange(k, 'b h s d -> b h d s')) * scaling
    s = s + mask
    p = torch.nn.functional.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return o


def ref_bwd_preprocess(o, dO):
    o = o.float()
    dO = dO.float()
    b, h, sq, d = o.shape
    delta = (o * dO).sum(dim=-1)
    return delta


@tl.jit(
    out_idx=[-2, -1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_mask_fwd(
    H,
    D,
    scaling=1.0,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    num_stages=3,
    BLOCK_Q=64,
    BLOCK_K=64,
):
    assert D == tl.math.next_power_of_2(D)
    B = T.dynamic('B', torch.int32)
    LQ = T.dynamic('LQ', torch.int32)
    LK = T.dynamic('LK', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, H, LQ, D), dtype),
        K: T.Tensor((B, H, LK, D), dtype),
        V: T.Tensor((B, H, LK, D), dtype),
        M: T.Tensor((B, 1, LQ, LK), dtype),
        O: T.Tensor((B, H, LQ, D), dtype),
        LSE: T.Tensor((B, H, LQ), acc_dtype),
    ):
        with T.Kernel(B, H, T.ceildiv(LQ, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
            q = T.alloc_shared((BLOCK_Q, D), dtype)
            k = T.alloc_shared((BLOCK_K, D), dtype)
            v = T.alloc_shared((BLOCK_K, D), dtype)
            mask = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            se = T.alloc_fragment((BLOCK_Q, ), acc_dtype)

            acc_s = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            acc_s_cast = T.alloc_fragment((BLOCK_Q, BLOCK_K), dtype)
            m = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            m_prev = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            alpha = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            tmp_sum = T.alloc_fragment((BLOCK_Q, ), acc_dtype)
            acc_o = T.alloc_fragment((BLOCK_Q, D), acc_dtype)

            T.fill(m, float('-inf'))
            T.fill(se, 0)
            T.fill(acc_o, 0)

            q_beg = q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(Q[b_i, h_i, q_beg:q_end, :], q)

            for k_idx in T.Pipelined(T.ceildiv(LK, BLOCK_K), num_stages=num_stages):
                k_beg = k_idx * BLOCK_K
                k_end = k_beg + BLOCK_K
                T.copy(K[b_i, h_i, k_beg:k_end, :], k)
                T.copy(M[b_i, 0, q_beg:q_end, k_beg:k_end], mask)

                # s = qk^T
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    # mask
                    acc_s[i, j] = T.if_then_else(
                        k_beg + j < LK,
                        mask[i, j],
                        -T.infinity(acc_s.dtype),
                    )
                T.gemm(q, k, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # scaling
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[i, j] = acc_s[i, j] * scaling

                # fix m
                T.copy(m, m_prev)
                T.reduce_max(acc_s, m, dim=1, clear=True)
                for i in T.Parallel(BLOCK_Q):
                    m[i] = T.max(m[i], m_prev[i])

                # alpha
                for i in T.Parallel(BLOCK_Q):
                    alpha[i] = T.exp((m_prev[i] - m[i]))

                # exp(s)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    acc_s[i, j] = T.exp(acc_s[i, j] - m[i])
                T.copy(acc_s, acc_s_cast)

                # l(SE) and fix l
                T.reduce_sum(acc_s, tmp_sum, dim=1, clear=True)  # sum exp(s)
                for i in T.Parallel(BLOCK_Q):
                    se[i] = se[i] * alpha[i] + tmp_sum[i]

                # a = pv
                for i, j in T.Parallel(BLOCK_Q, D):
                    acc_o[i, j] = alpha[i] * acc_o[i, j]
                T.copy(V[b_i, h_i, k_beg:k_end, :], v)
                T.gemm(acc_s_cast, v, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # o = a / l
            for i, j in T.Parallel(BLOCK_Q, D):
                acc_o[i, j] = acc_o[i, j] / se[i]
            T.copy(acc_o, O[b_i, h_i, q_beg:q_end, :])

            # lse
            for i in T.Parallel(BLOCK_Q):
                se[i] = m[i] + T.log(se[i])
            T.copy(se, LSE[b_i, h_i, q_beg:q_end])

    return kernel


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def myfa_mask_bwd_pre(
    H,
    D,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    BLOCK_Q=64,
):
    assert D == tl.math.next_power_of_2(D)
    B = T.dynamic('B', torch.int32)
    LQ = T.dynamic('LQ', torch.int32)

    @T.prim_func
    def kernel(
        O: T.Tensor((B, H, LQ, D), dtype),
        dO: T.Tensor((B, H, LQ, D), dtype),
        DELTA: T.Tensor((B, H, LQ), acc_dtype),
    ):
        with T.Kernel(B, H, T.ceildiv(LQ, BLOCK_Q), threads=threads) as (b_i, h_i, q_idx):
            o = T.alloc_fragment((BLOCK_Q, D), dtype)
            do = T.alloc_fragment((BLOCK_Q, D), dtype)
            acc = T.alloc_fragment((BLOCK_Q, D), acc_dtype)
            delta = T.alloc_fragment((BLOCK_Q, ), acc_dtype)

            # delta = sum_d(o * dO)
            q_beg = q_idx * BLOCK_Q
            q_end = q_beg + BLOCK_Q
            T.copy(O[b_i, h_i, q_beg:q_end, :], o)
            T.copy(dO[b_i, h_i, q_beg:q_end, :], do)
            for i, j in T.Parallel(BLOCK_Q, D):
                acc[i, j] = o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, dim=1, clear=True)
            T.copy(delta, DELTA[b_i, h_i, q_beg:q_end])

    return kernel


@tl.jit(pass_configs={
    tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, )
def myfa_mask_bwd(
    H,
    D,
    scaling=1.0,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    num_stages=3,
    BLOCK_Q=64,
    BLOCK_K=64,
):
    assert D == tl.math.next_power_of_2(D)
    B = T.dynamic('B', torch.int32)
    LQ = T.dynamic('LQ', torch.int32)
    LK = T.dynamic('LK', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, H, LQ, D), dtype),
        K: T.Tensor((B, H, LK, D), dtype),
        V: T.Tensor((B, H, LK, D), dtype),
        M: T.Tensor((B, 1, LQ, LK), dtype),
        LSE: T.Tensor((B, H, LQ), acc_dtype),
        dO: T.Tensor((B, H, LQ, D), dtype),
        DELTA: T.Tensor((B, H, LQ), acc_dtype),
        dQ: T.Tensor((B, H, LQ, D), acc_dtype),
        dK: T.Tensor((B, H, LK, D), dtype),
        dV: T.Tensor((B, H, LK, D), dtype),
    ):
        with T.Kernel(B, H, T.ceildiv(LK, BLOCK_K), threads=threads) as (b_i, h_i, k_idx):
            q = T.alloc_shared((BLOCK_Q, D), dtype)
            k = T.alloc_shared((BLOCK_K, D), dtype)
            v = T.alloc_shared((BLOCK_K, D), dtype)
            mask = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            lse = T.alloc_shared((BLOCK_Q, ), acc_dtype)
            do = T.alloc_shared((BLOCK_Q, D), dtype)
            delta = T.alloc_shared((BLOCK_Q, ), acc_dtype)

            p = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            p_cast = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            ds = T.alloc_fragment((BLOCK_Q, BLOCK_K), acc_dtype)
            ds_cast = T.alloc_shared((BLOCK_Q, BLOCK_K), dtype)
            dq = T.alloc_fragment((BLOCK_Q, D), acc_dtype)
            dk = T.alloc_fragment((BLOCK_K, D), acc_dtype)
            dv = T.alloc_fragment((BLOCK_K, D), acc_dtype)

            k_beg = k_idx * BLOCK_K
            k_end = k_beg + BLOCK_K
            T.copy(K[b_i, h_i, k_beg:k_end, :], k)
            T.copy(V[b_i, h_i, k_beg:k_end, :], v)
            T.clear(dk)
            T.clear(dv)

            for q_idx in T.Pipelined(T.ceildiv(LQ, BLOCK_Q), num_stages=num_stages):
                q_beg = q_idx * BLOCK_Q
                q_end = q_beg + BLOCK_Q
                T.copy(Q[b_i, h_i, q_beg:q_end, :], q)
                T.copy(M[b_i, 0, q_beg:q_end, k_beg:k_end], mask)

                # S = QK^T * scaling
                T.clear(p)
                T.gemm(q, k, p, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = p[i, j] * scaling

                # P = softmax(S)
                T.copy(LSE[b_i, h_i, q_beg:q_end], lse)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = T.exp(p[i, j] - lse[i])

                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    p[i, j] = T.if_then_else(
                        k_beg + j >= LK or mask[i, j] == float('-inf'),
                        0.,
                        p[i, j],
                    )

                # dP = dO @ V^T
                T.copy(dO[b_i, h_i, q_beg:q_end, :], do)
                T.clear(ds)
                T.gemm(do, v, ds, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # dV = P^T @ dO
                T.copy(p, p_cast)
                T.gemm(p_cast, do, dv, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                # dS = P * (dO - delta), shape [LQ, LK]
                # delta = sum_d(dO * O)
                T.copy(DELTA[b_i, h_i, q_beg:q_end], delta)
                for i, j in T.Parallel(BLOCK_Q, BLOCK_K):
                    ds_cast[i, j] = p[i, j] * (ds[i, j] - delta[i])
                    ds_cast[i, j] *= scaling

                # dK = (scaling * dS)^T @ Q
                T.gemm(ds_cast, q, dk, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                # dQ = (scaling * dS) @ K^T)
                T.clear(dq)
                T.gemm(ds_cast, k, dq, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BLOCK_Q, D):
                    T.atomic_add(dQ[b_i, h_i, q_idx * BLOCK_Q + i, j], dq[i, j])

            T.copy(dk, dK[b_i, h_i, k_beg:k_end, :])
            T.copy(dv, dV[b_i, h_i, k_beg:k_end, :])

    return kernel


class MyfaMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask):
        _, h, _, d = q.shape
        scaling = 1.0 / math.sqrt(d)
        fwd_kernel = myfa_mask_fwd.compile(H=h, D=d, scaling=scaling)
        o, lse = fwd_kernel(q, k, v, mask)
        ctx.save_for_backward(q, k, v, mask, o, lse)
        ctx.scaling = scaling
        return o

    @staticmethod
    def backward(ctx, dO):
        q, k, v, mask, o, lse = ctx.saved_tensors
        _, h, _, d = q.shape
        scaling = ctx.scaling

        bwd_pre_kernel = myfa_mask_bwd_pre.compile(H=h, D=d)
        delta = bwd_pre_kernel(o, dO)

        bwd_kernel = myfa_mask_bwd.compile(H=h, D=d, scaling=scaling)
        dQ = torch.zeros_like(q, dtype=torch.float32)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)
        bwd_kernel(q, k, v, mask, lse, dO, delta, dQ, dK, dV)
        dQ = dQ.to(q.dtype)

        return dQ, dK, dV, None


def myfa_mask(q, k, v, mask):
    return MyfaMask.apply(q, k, v, mask)


def _make_inputs(causal=True):
    b, h, s, d = 2, 4, 32 * 1024, 128
    scaling = 1.0 / math.sqrt(d)
    if causal:
        mask = torch.triu(
            torch.full((b, 1, s, s), float("-inf"), device='cuda', dtype=torch.bfloat16),
            diagonal=1,
        )
    else:
        mask = torch.zeros(b, 1, s, s, device='cuda', dtype=torch.bfloat16)
    rng = torch.Generator(device='cuda').manual_seed(1919810)
    q = torch.randn(b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16)
    return q, k, v, mask, h, d, scaling


def _check(tag, a, b, atol_avg=0.01, atol_max=0.05):
    avg = (a - b).abs().mean().item()
    mx = (a - b).abs().max().item()
    print(f'{tag} | avg {avg:.6e} | max {mx:.6e}')
    assert avg < atol_avg, f'{tag} avg {avg} >= {atol_avg}'
    assert mx < atol_max, f'{tag} max {mx} >= {atol_max}'


@pytest.mark.parametrize("causal", [True, False])
@torch.no_grad()
def test_forward(causal):
    q, k, v, mask, h, d, scaling = _make_inputs(causal)

    ref_o = ref_attn(q, k, v, mask, scaling)

    myfa_fwd_kernel = myfa_mask_fwd.compile(H=h, D=d, scaling=scaling)
    my_o, _ = myfa_fwd_kernel(q, k, v, mask)

    fa2_o = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=causal,
        softmax_scale=scaling,
    ).transpose(1, 2)

    tag = 'causal' if causal else 'full'
    _check(f'[{tag}] my  vs ref', my_o, ref_o)
    _check(f'[{tag}] fa2 vs ref', fa2_o, ref_o)
    _check(f'[{tag}] my  vs fa2', my_o, fa2_o)


@pytest.mark.parametrize("causal", [True, False])
def test_backward(causal):
    q, k, v, mask, h, d, scaling = _make_inputs(causal)

    myfa_fwd_kernel = myfa_mask_fwd.compile(H=h, D=d, scaling=scaling)
    with torch.no_grad():
        my_o, my_lse = myfa_fwd_kernel(q, k, v, mask)

    dO = torch.rand_like(my_o)

    # bwd preprocess (delta)
    ref_delta = ref_bwd_preprocess(my_o, dO)
    myfa_bwd_preprocess_kernel = myfa_mask_bwd_pre.compile(H=h, D=d)
    my_delta = myfa_bwd_preprocess_kernel(my_o, dO)
    _check('delta', my_delta, ref_delta, atol_avg=1e-3, atol_max=0.1)

    # myfa backward
    myfa_bwd_kernel = myfa_mask_bwd.compile(H=h, D=d, scaling=scaling)
    my_dQ = torch.zeros_like(q, device='cuda', dtype=torch.float32)
    my_dK = torch.empty_like(k, device='cuda', dtype=torch.bfloat16)
    my_dV = torch.empty_like(v, device='cuda', dtype=torch.bfloat16)
    myfa_bwd_kernel(q, k, v, mask, my_lse, dO, my_delta, my_dQ, my_dK, my_dV)
    my_dQ = my_dQ.to(torch.bfloat16)

    # eager backward
    with torch.enable_grad():
        q_ag = q.detach().requires_grad_(True)
        k_ag = k.detach().requires_grad_(True)
        v_ag = v.detach().requires_grad_(True)
        eager_o = ref_attn(q_ag, k_ag, v_ag, mask, scaling)
        eager_o.backward(dO)
    ref_dQ, ref_dK, ref_dV = q_ag.grad, k_ag.grad, v_ag.grad

    # fa2 backward
    with torch.enable_grad():
        q_fa = q.detach().transpose(1, 2).contiguous().requires_grad_(True)
        k_fa = k.detach().transpose(1, 2).contiguous().requires_grad_(True)
        v_fa = v.detach().transpose(1, 2).contiguous().requires_grad_(True)
        fa2_o2 = flash_attn_func(q_fa, k_fa, v_fa, causal=causal, softmax_scale=scaling)
        fa2_o2.backward(dO.transpose(1, 2))
    fa2_dQ = q_fa.grad.transpose(1, 2)
    fa2_dK = k_fa.grad.transpose(1, 2)
    fa2_dV = v_fa.grad.transpose(1, 2)

    tag = 'causal' if causal else 'full'
    for name, my_g, ref_g, fa2_g in [
        ('dQ', my_dQ, ref_dQ, fa2_dQ),
        ('dK', my_dK, ref_dK, fa2_dK),
        ('dV', my_dV, ref_dV, fa2_dV),
    ]:
        _check(f'[{tag}] {name} my  vs ref', my_g, ref_g, atol_max=1.0)
        _check(f'[{tag}] {name} fa2 vs ref', fa2_g, ref_g, atol_max=1.0)
        _check(f'[{tag}] {name} my  vs fa2', my_g, fa2_g, atol_max=1.0)


@pytest.mark.parametrize("causal", [True, False])
def test_autofunc(causal):
    q, k, v, mask, h, d, scaling = _make_inputs(causal)

    # myfa_mask autograd
    q_my = q.detach().requires_grad_(True)
    k_my = k.detach().requires_grad_(True)
    v_my = v.detach().requires_grad_(True)
    my_o = myfa_mask(q_my, k_my, v_my, mask)

    # eager ref
    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)
    ref_o = ref_attn(q_ref, k_ref, v_ref, mask, scaling)

    tag = 'causal' if causal else 'full'
    _check(f'[{tag}] fwd my vs ref', my_o, ref_o)

    dO = torch.rand_like(my_o)
    my_o.backward(dO)
    ref_o.backward(dO)

    _check(f'[{tag}] dQ my vs ref', q_my.grad, q_ref.grad, atol_max=1.0)
    _check(f'[{tag}] dK my vs ref', k_my.grad, k_ref.grad, atol_max=1.0)
    _check(f'[{tag}] dV my vs ref', v_my.grad, v_ref.grad, atol_max=1.0)


@pytest.mark.parametrize(
    "s,d",
    [
        (256, 64),
        (512, 64),
        (1024, 64),
        (4096, 64),
        (256, 128),
        (512, 128),
        (1024, 128),
        (4096, 128),
        # 256 会导致 shared memory 溢出，有时间再 tune 下。
        # (256, 256),
        # (512, 256),
        # (1024, 256),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@torch.no_grad()
def test_shapes(s, d, causal):
    b, h = 2, 4
    scaling = 1.0 / math.sqrt(d)
    if causal:
        mask = torch.triu(
            torch.full((b, 1, s, s), float("-inf"), device='cuda', dtype=torch.bfloat16),
            diagonal=1,
        )
    else:
        mask = torch.zeros(b, 1, s, s, device='cuda', dtype=torch.bfloat16)

    rng = torch.Generator(device='cuda').manual_seed(1919810)
    q = torch.randn(b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16)

    ref_o = ref_attn(q, k, v, mask, scaling)

    myfa_fwd_kernel = myfa_mask_fwd.compile(H=h, D=d, scaling=scaling)
    my_o, my_lse = myfa_fwd_kernel(q, k, v, mask)

    tag = f'[s={s},d={d},{"causal" if causal else "full"}]'
    _check(f'{tag} fwd my vs ref', my_o, ref_o)

    dO = torch.rand_like(my_o)

    # myfa backward
    myfa_bwd_preprocess_kernel = myfa_mask_bwd_pre.compile(H=h, D=d)
    my_delta = myfa_bwd_preprocess_kernel(my_o, dO)
    myfa_bwd_kernel = myfa_mask_bwd.compile(H=h, D=d, scaling=scaling)
    my_dQ = torch.zeros_like(q, device='cuda', dtype=torch.float32)
    my_dK = torch.empty_like(k, device='cuda', dtype=torch.bfloat16)
    my_dV = torch.empty_like(v, device='cuda', dtype=torch.bfloat16)
    myfa_bwd_kernel(q, k, v, mask, my_lse, dO, my_delta, my_dQ, my_dK, my_dV)
    my_dQ = my_dQ.to(torch.bfloat16)

    # eager backward
    with torch.enable_grad():
        q_ag = q.detach().requires_grad_(True)
        k_ag = k.detach().requires_grad_(True)
        v_ag = v.detach().requires_grad_(True)
        eager_o = ref_attn(q_ag, k_ag, v_ag, mask, scaling)
        eager_o.backward(dO)
    ref_dQ, ref_dK, ref_dV = q_ag.grad, k_ag.grad, v_ag.grad

    _check(f'{tag} dQ my vs ref', my_dQ, ref_dQ, atol_max=1.0)
    _check(f'{tag} dK my vs ref', my_dK, ref_dK, atol_max=1.0)
    _check(f'{tag} dV my vs ref', my_dV, ref_dV, atol_max=1.0)


def _bench(fn, n_warmup=10, n_iter=8):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iter):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / n_iter


@pytest.mark.parametrize("causal", [True, False])
@torch.no_grad()
def test_throughput_forward(causal):
    q, k, v, mask, h, d, scaling = _make_inputs(causal)
    q_bshd = q.transpose(1, 2)
    k_bshd = k.transpose(1, 2)
    v_bshd = v.transpose(1, 2)

    myfa_fwd_kernel = myfa_mask_fwd.compile(H=h, D=d, scaling=scaling)

    eager_ms = _bench(lambda: ref_attn(q, k, v, mask, scaling))
    my_ms = _bench(lambda: myfa_fwd_kernel(q, k, v, mask))
    fa2_ms = _bench(
        lambda: flash_attn_func(q_bshd, k_bshd, v_bshd, causal=causal, softmax_scale=scaling)
    )

    tag = 'causal' if causal else 'full'
    print(f'[{tag}] fwd eager: {eager_ms:.4f} ms')
    print(f'[{tag}] fwd myfa : {my_ms:.4f} ms')
    print(f'[{tag}] fwd fa2  : {fa2_ms:.4f} ms')


@pytest.mark.parametrize("causal", [True, False])
def test_throughput_backward(causal):
    q, k, v, mask, h, d, scaling = _make_inputs(causal)

    myfa_fwd_kernel = myfa_mask_fwd.compile(H=h, D=d, scaling=scaling)
    myfa_bwd_preprocess_kernel = myfa_mask_bwd_pre.compile(H=h, D=d)
    myfa_bwd_kernel = myfa_mask_bwd.compile(H=h, D=d, scaling=scaling)

    with torch.no_grad():
        my_o, my_lse = myfa_fwd_kernel(q, k, v, mask)
    dO = torch.rand_like(my_o)

    def myfa_bwd():
        my_delta = myfa_bwd_preprocess_kernel(my_o, dO)
        my_dQ = torch.zeros_like(q, device='cuda', dtype=torch.float32)
        my_dK = torch.empty_like(k, device='cuda', dtype=torch.bfloat16)
        my_dV = torch.empty_like(v, device='cuda', dtype=torch.bfloat16)
        myfa_bwd_kernel(q, k, v, mask, my_lse, dO, my_delta, my_dQ, my_dK, my_dV)

    def eager_bwd():
        with torch.enable_grad():
            q_ag = q.detach().requires_grad_(True)
            k_ag = k.detach().requires_grad_(True)
            v_ag = v.detach().requires_grad_(True)
            eager_o = ref_attn(q_ag, k_ag, v_ag, mask, scaling)
            eager_o.backward(dO)

    def fa2_bwd():
        with torch.enable_grad():
            q_fa = q.detach().transpose(1, 2).contiguous().requires_grad_(True)
            k_fa = k.detach().transpose(1, 2).contiguous().requires_grad_(True)
            v_fa = v.detach().transpose(1, 2).contiguous().requires_grad_(True)
            fa2_o2 = flash_attn_func(q_fa, k_fa, v_fa, causal=causal, softmax_scale=scaling)
            fa2_o2.backward(dO.transpose(1, 2))

    eager_ms = _bench(eager_bwd)
    my_ms = _bench(myfa_bwd)
    fa2_ms = _bench(fa2_bwd)

    tag = 'causal' if causal else 'full'
    print(f'[{tag}] bwd eager: {eager_ms:.4f} ms')
    print(f'[{tag}] bwd myfa : {my_ms:.4f} ms')
    print(f'[{tag}] bwd fa2  : {fa2_ms:.4f} ms')
