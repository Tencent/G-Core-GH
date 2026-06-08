"""
A emtpy template for tilelang fused op.

Example usage:

    cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/wepsdl/gcore-dev
    python3 tests/test_gfused/test_mytmp.py
"""

import math
import unittest

from tilelang import language as T
import einops
import pytest
import tilelang as tl
import torch


@tl.jit(out_idx=[-2, -1])
def mytmp_fwd(
    H,
    D,
    scaling=1.0,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    num_stages=3,
    BLOCK_S=64,
):
    assert D == tl.math.next_power_of_2(D)
    B = T.dynamic('B', torch.int32)
    S = T.dynamic('S', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, H, S, D), dtype),
        K: T.Tensor((B, H, S, D), dtype),
        V: T.Tensor((B, H, S, D), dtype),
        M: T.Tensor((1, 1, S, S), dtype),
        O: T.Tensor((B, H, S, D), dtype),
        LSE: T.Tensor((B, H, S), dtype),
    ):
        with T.Kernel(B, H, threads=threads) as (b_i, h_i):
            Q_i = T.alloc_shared((BLOCK_S, D), dtype)
            K_i = T.alloc_shared((BLOCK_S, D), dtype)
            V_i = T.alloc_shared((BLOCK_S, D), dtype)
            O_i = T.alloc_fragment((BLOCK_S, D), dtype)
            LSE_i = T.alloc_fragment((BLOCK_S,), dtype)

            for i in T.Pipelined(T.ceildiv(S, BLOCK_S), num_stages=num_stages):
                s_beg = i * BLOCK_S
                s_end = s_beg + BLOCK_S
                T.copy(Q[b_i, h_i, s_beg:s_end, :], Q_i)
                T.copy(K[b_i, h_i, s_beg:s_end, :], K_i)
                T.copy(V[b_i, h_i, s_beg:s_end, :], V_i)

                T.clear(O_i)
                for j, k in T.Parallel(BLOCK_S, D):
                    O_i[j, k] = Q_i[j, k] + K_i[j, k] + V_i[j, k]
                T.clear(LSE_i)

                T.copy(O_i, O[b_i, h_i, s_beg:s_end, :])
                T.copy(LSE_i, LSE[b_i, h_i, s_beg:s_end])

    return kernel


def ref_eager_attn(q, k, v, mask, scaling):
    b, h, s, d = q.shape
    p = torch.matmul(q, einops.rearrange(k, 'b h s d -> b h d s')) * scaling
    p = p + mask
    s = torch.nn.functional.softmax(p, dim=-1)
    o = torch.matmul(s, v)
    return o


def main():
    b = 2
    h = 4
    s = 1024
    d = 128
    scaling = 1.0 / math.sqrt(d)

    causal_mask = torch.triu(
        torch.full(
            (1, 1, s, s), float("-inf"), device='cuda', dtype=torch.bfloat16
        ),
        diagonal=1
    )

    rng = torch.Generator(device='cuda').manual_seed(1919810)
    q = torch.randn(
        b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        b, h, s, d, generator=rng, device="cuda", dtype=torch.bfloat16
    )

    # ref_o = ref_eager_attn(q, k, v, causal_mask, scaling)
    # print(ref_o.shape)

    myfa_fwd_kernel = mytmp_fwd.compile(H=h, D=d, scaling=scaling, dtype=torch.bfloat16)
    o, lse = myfa_fwd_kernel(q, k, v, causal_mask)

    torch.testing.assert_close(o, q + k + v)
    torch.testing.assert_close(lse, torch.zeros_like(lse))


main()