"""
A emtpy template for tilelang fused op.

Example usage:

    cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/wepsdl/gcore-dev
    python3 tests/test_gfused/test_myfa.py
"""

import pytest

pytest.skip("legacy myfa template; not a collected pytest case", allow_module_level=True)

import math

from tilelang import language as T
import einops
import tilelang as tl
import torch


@tl.jit(out_idx=[-1])
def myfa_fwd(
    H=1,
    D=128,
    scaling=1.0,
    dtype=torch.bfloat16,
    acc_dtype=torch.float32,
    threads=128,
    BLOCK_S=128
):
    assert D == tl.math.next_power_of_2(D)
    B = T.dynamic('B', torch.int32)
    S = T.dynamic('S', torch.int32)

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, S, H, D), dtype),
        K: T.Tensor((B, S, H, D), dtype),
        V: T.Tensor((B, S, H, D), dtype),
        M: T.Tensor((1, 1, S, S), dtype),
        O: T.Tensor((B, S, H, D), dtype),
        # LSE: T.Tensor((B, S, H), dtype),
    ):
        with T.Kernel(B, H, threads=threads) as (bx, by):
            b_i = bx
            h_i = by
            q_i = T.alloc_shared((BLOCK_S, D), dtype)
            o_i = T.alloc_fragment((BLOCK_S, D), dtype)

            for s_idx in T.Pipelined(T.ceildiv(S, BLOCK_S), num_stages=3):
                T.copy(
                    Q[b_i, s_idx * BLOCK_S:(s_idx + 1) * BLOCK_S, h_i, :], q_i
                )
                T.copy(q_i, o_i)
                T.copy(
                    o_i, O[b_i, s_idx * BLOCK_S:(s_idx + 1) * BLOCK_S, h_i, :]
                )

    return kernel


def ref_eager_attn(q, k, v, mask, scaling):
    # megatron 也是 bshd -> bhsd (dot_product_attention.py)，方便 astra code。
    b, s, h, d = q.shape
    q = einops.rearrange(q, 'b s h d -> b h s d')
    k = einops.rearrange(k, 'b s h d -> b h s d')
    v = einops.rearrange(v, 'b s h d -> b h s d')

    p = torch.matmul(q, einops.rearrange(k, 'b h s d -> b h d s')) * scaling
    p + p + mask
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
            (1, 1, s, s), float("-inf"), device='cuda', dtype=torch.float16
        ),
        diagonal=1
    )

    rng = torch.Generator(device='cuda').manual_seed(1919810)
    q = torch.randn(
        b, s, h, d, generator=rng, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        b, s, h, d, generator=rng, device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        b, s, h, d, generator=rng, device="cuda", dtype=torch.float16
    )

    # ref_o = ref_eager_attn(q, k, v, causal_mask, scaling)
    # print(ref_o.shape)

    myfa_fwd_kernel = myfa_fwd.compile(H=h, D=d, scaling=scaling)
    o, lse = myfa_fwd_kernel(q, k, v, causal_mask)
    print(o.shape, lse.shape)


if __name__ == "__main__":
    main()
