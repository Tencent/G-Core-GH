"""
myfa_varlen_sw_sinks forward / backward correctness tests.

Varlen GQA attention kernel with sliding-window causal masking
and learnable sinks.

Example usage::

    cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/wepsdl/gcore-dev
    pytest -s tests/test_gfused/test_myfa_varlen_sw_sinks.py -v
    pytest -s tests/test_gfused/test_myfa_varlen_sw_sinks.py::test_fwd_bwd -v
"""

import math
from typing import Optional

import pytest
import torch
from flash_attn import flash_attn_varlen_func

from gfused.myfa_varlen_sw_sinks import (
    myfa_varlen_sw_sinks,
    myfa_varlen_sw_sinks_fwd,
    myfa_varlen_sw_sinks_bwd,
    myfa_varlen_sw_sinks_bwd_pre,
    myfa_varlen_sw_sinks_bwd_dsink,
    _get_fwd_block_config,
    _get_bwd_block_config,
    _round_up,
)


def ref_varlen_attn(
    q, k, v, cu_seqlens_q, cu_seqlens_k,
    sinks=None, is_causal=True, window_size=None,
):
    """Reference implementation for varlen GQA attention with SWA and sinks."""
    total_q, HQ, d = q.shape
    _, HK, _ = k.shape
    groups = HQ // HK
    scaling = 1.0 / math.sqrt(d)
    batch_size = len(cu_seqlens_q) - 1

    output = torch.zeros_like(q)

    for b in range(batch_size):
        q_start = int(cu_seqlens_q[b])
        q_end = int(cu_seqlens_q[b + 1])
        k_start = int(cu_seqlens_k[b])
        k_end = int(cu_seqlens_k[b + 1])

        q_len = q_end - q_start
        k_len = k_end - k_start
        if q_len == 0:
            continue

        q_seq = q[q_start:q_end]  # [q_len, HQ, d]
        k_seq = k[k_start:k_end]  # [k_len, HK, d]
        v_seq = v[k_start:k_end]  # [k_len, HK, d]

        # GQA reshape
        q_seq = q_seq.view(q_len, HK, groups, d)
        k_seq = k_seq.unsqueeze(2)  # [k_len, HK, 1, d]
        v_seq = v_seq.unsqueeze(2)  # [k_len, HK, 1, d]

        # [HK, groups, q_len, k_len]
        logits = torch.einsum("qhgd,khgd->hgqk", q_seq.float(), k_seq.float()) * scaling

        offset = k_len - q_len
        pos_keys = torch.arange(k_len, device=q.device)
        pos_queries = torch.arange(q_len, device=q.device) + offset

        mask = torch.zeros(q_len, k_len, device=q.device)
        if is_causal:
            causal_mask = pos_keys[None, :] > pos_queries[:, None]
            mask.masked_fill_(causal_mask, float("-inf"))
        if window_size is not None:
            too_old = pos_keys[None, :] < (pos_queries[:, None] - window_size + 1)
            mask.masked_fill_(too_old, float("-inf"))

        logits = logits + mask[None, None, :, :]

        if sinks is not None:
            sinks_expanded = sinks.view(HK, groups, 1, 1).float()
            logits_max = torch.max(logits, dim=-1, keepdim=True).values
            logits_or_sinks_max = torch.maximum(sinks_expanded, logits_max)
            sinks_exp = torch.exp(sinks_expanded - logits_or_sinks_max)
            unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
            normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks_exp
            scores = unnormalized_scores / normalizer
        else:
            scores = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)

        out = torch.einsum("hgqk,khgd->qhgd", scores, v_seq.float())
        out = out.reshape(q_len, HQ, d).to(q.dtype)
        output[q_start:q_end] = out

    return output


CASES = [
    # causal, no SWA, no sinks
    dict(seqlens=[64], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    dict(seqlens=[128, 256], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    dict(seqlens=[33, 1023, 3, 128], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    dict(seqlens=[7, 13, 65, 129, 255], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    dict(seqlens=[4096], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    # causal + sinks
    dict(seqlens=[64], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[128, 256], HQ=8, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[33, 1023, 3, 128], HQ=12, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[17, 97, 503, 1], HQ=8, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[4096], HQ=8, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    # causal + SWA
    dict(seqlens=[512], HQ=4, HK=4, d=128, is_causal=True, window_size=128, sinks=False),
    dict(seqlens=[1024], HQ=4, HK=4, d=128, is_causal=True, window_size=256, sinks=False),
    dict(seqlens=[128, 256, 512], HQ=8, HK=2, d=128, is_causal=True, window_size=128, sinks=False),
    dict(seqlens=[73, 511, 200], HQ=4, HK=4, d=128, is_causal=True, window_size=128, sinks=False),
    dict(seqlens=[4096], HQ=4, HK=4, d=128, is_causal=True, window_size=512, sinks=False),
    # causal + SWA + sinks
    dict(seqlens=[512], HQ=4, HK=4, d=128, is_causal=True, window_size=128, sinks=True),
    dict(seqlens=[1024], HQ=12, HK=2, d=128, is_causal=True, window_size=256, sinks=True),
    dict(seqlens=[128, 256, 512], HQ=8, HK=2, d=128, is_causal=True, window_size=128, sinks=True),
    dict(seqlens=[63, 127, 513], HQ=8, HK=2, d=128, is_causal=True, window_size=128, sinks=True),
    dict(seqlens=[2048], HQ=4, HK=4, d=128, is_causal=True, window_size=512, sinks=True),
    dict(seqlens=[4096], HQ=4, HK=4, d=128, is_causal=True, window_size=512, sinks=True),
    # non-causal
    dict(seqlens=[128, 256], HQ=4, HK=4, d=128, is_causal=False, window_size=None, sinks=False),
    dict(seqlens=[128, 256], HQ=4, HK=4, d=128, is_causal=False, window_size=None, sinks=True),
    dict(seqlens=[37, 199, 5], HQ=4, HK=4, d=128, is_causal=False, window_size=None, sinks=True),
    dict(seqlens=[4096], HQ=4, HK=4, d=128, is_causal=False, window_size=None, sinks=True),
    # GQA with different group sizes
    dict(seqlens=[256, 128], HQ=24, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[256, 128], HQ=24, HK=2, d=128, is_causal=True, window_size=128, sinks=True),
    dict(seqlens=[4096], HQ=24, HK=2, d=128, is_causal=True, window_size=512, sinks=True),
    # d=256
    dict(seqlens=[256, 128], HQ=12, HK=2, d=256, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[256, 128], HQ=12, HK=2, d=256, is_causal=True, window_size=128, sinks=True),
    dict(seqlens=[2048], HQ=12, HK=2, d=256, is_causal=True, window_size=512, sinks=True),
    # edge case: single token
    dict(seqlens=[1, 1, 1, 1], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=True),
    # edge case: window_size >= seq_len
    dict(seqlens=[64], HQ=4, HK=4, d=128, is_causal=True, window_size=128, sinks=True),

    # kv longer than q (cross-attention style, right-aligned)
    # dict(seqlens=[200], seqlens_k=[1234], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=False),
    # dict(seqlens=[200], seqlens_k=[1234], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=True),
    # dict(seqlens=[200], seqlens_k=[1234], HQ=8, HK=2, d=128, is_causal=True, window_size=512, sinks=True),
    # dict(seqlens=[100, 300], seqlens_k=[500, 1500], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=True),
    # dict(seqlens=[100, 300], seqlens_k=[500, 1500], HQ=4, HK=4, d=128, is_causal=True, window_size=256, sinks=True),
    # dict(seqlens=[64], seqlens_k=[4096], HQ=4, HK=4, d=128, is_causal=True, window_size=512, sinks=True),
    # dict(seqlens=[1], seqlens_k=[1024], HQ=4, HK=4, d=128, is_causal=True, window_size=None, sinks=True),
]


def _case_tag(c):
    ws = c['window_size']
    ws_str = f"sw{ws}" if ws is not None else "causal" if c['is_causal'] else "full"
    tag = f"hq{c['HQ']}hk{c['HK']}d{c['d']}_{ws_str}"
    if c['sinks']:
        tag += "_sink"
    tag += f"_seqs{'x'.join(str(s) for s in c['seqlens'][:3])}"
    if len(c['seqlens']) > 3:
        tag += "..."
    if 'seqlens_k' in c:
        tag += f"_kv{'x'.join(str(s) for s in c['seqlens_k'][:3])}"
    return tag


def _fa2_window_size(is_causal, window_size):
    if window_size is None:
        return (-1, -1)
    return (window_size - 1, 0 if is_causal else -1)


def _make_inputs(seqlens, HQ, HK, d, seqlens_k=None, dtype=torch.float16):
    cu_seqlens_q = [0]
    for sl in seqlens:
        cu_seqlens_q.append(cu_seqlens_q[-1] + sl)
    total_q = cu_seqlens_q[-1]

    if seqlens_k is None:
        seqlens_k = seqlens
    cu_seqlens_k = [0]
    for sl in seqlens_k:
        cu_seqlens_k.append(cu_seqlens_k[-1] + sl)
    total_k = cu_seqlens_k[-1]

    rng = torch.Generator(device='cuda').manual_seed(1919810)
    q = torch.randn(total_q, HQ, d, generator=rng, device="cuda", dtype=dtype)
    k = torch.randn(total_k, HK, d, generator=rng, device="cuda", dtype=dtype)
    v = torch.randn(total_k, HK, d, generator=rng, device="cuda", dtype=dtype)
    sinks = torch.randn(HQ, generator=rng, device="cuda", dtype=dtype)

    cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
    cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cuda")
    max_seqlen = max(seqlens)

    return q, k, v, sinks, cu_seqlens_q_t, cu_seqlens_k_t, max_seqlen


def _check(tag, a, b, atol_avg=0.01, atol_max=0.05):
    avg = (a - b).abs().mean().item()
    mx = (a - b).abs().max().item()
    print(f'{tag} | avg {avg:.6e} | max {mx:.6e}')
    assert avg < atol_avg, f'{tag} avg {avg} >= {atol_avg}'
    assert mx < atol_max, f'{tag} max {mx} >= {atol_max}'


@pytest.mark.parametrize("case", CASES, ids=[_case_tag(c) for c in CASES])
def test_fwd_bwd(case):
    seqlens = case['seqlens']
    seqlens_k = case.get('seqlens_k', None)
    HQ, HK, d = case['HQ'], case['HK'], case['d']
    is_causal = case['is_causal']
    window_size = case['window_size']
    has_sinks = case['sinks']

    q, k, v, sinks_val, cu_seqlens_q, cu_seqlens_k, max_seqlen = _make_inputs(
        seqlens, HQ, HK, d, seqlens_k=seqlens_k, dtype=torch.float16)

    sinks_arg = sinks_val if has_sinks else None

    # reference forward + backward
    with torch.enable_grad():
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        sinks_ref = sinks_val.detach().clone().requires_grad_(True) if has_sinks else None
        ref_o = ref_varlen_attn(q_ref, k_ref, v_ref, cu_seqlens_q, cu_seqlens_k,
                                sinks=sinks_ref, is_causal=is_causal, window_size=window_size)
        dO = torch.rand_like(ref_o)
        ref_o.backward(dO)
    ref_dQ = q_ref.grad
    ref_dK = k_ref.grad
    ref_dV = v_ref.grad
    ref_dsinks = sinks_ref.grad if has_sinks else None

    # myfa forward + backward (via autograd wrapper)
    with torch.enable_grad():
        q_my = q.detach().clone().requires_grad_(True)
        k_my = k.detach().clone().requires_grad_(True)
        v_my = v.detach().clone().requires_grad_(True)
        sinks_my = sinks_val.detach().clone().requires_grad_(True) if has_sinks else None
        my_o = myfa_varlen_sw_sinks(q_my, k_my, v_my, cu_seqlens_q, cu_seqlens_k, max_seqlen,
                                    sinks=sinks_my, is_causal=is_causal, window_size=window_size)
        my_o.backward(dO)
    my_dQ = q_my.grad
    my_dK = k_my.grad
    my_dV = v_my.grad
    my_dsinks = sinks_my.grad if has_sinks else None

    tag = _case_tag(case)
    rtol_avg, rtol_max = (0.015, 0.08) if window_size is not None else (0.01, 0.05)
    _check(f'[{tag}] O', my_o, ref_o, atol_avg=rtol_avg, atol_max=rtol_max)
    _check(f'[{tag}] dQ', my_dQ, ref_dQ, atol_avg=rtol_avg, atol_max=rtol_max)
    _check(f'[{tag}] dK', my_dK, ref_dK, atol_avg=rtol_avg, atol_max=rtol_max)
    _check(f'[{tag}] dV', my_dV, ref_dV, atol_avg=rtol_avg, atol_max=rtol_max)
    if has_sinks:
        _check(f'[{tag}] dSinks', my_dsinks, ref_dsinks, atol_avg=rtol_avg, atol_max=rtol_max)


BENCH_CASES = [
    dict(seqlens=[ 4 * 1024] * 8, HQ=24, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[32 * 1024],     HQ=24, HK=2, d=128, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[ 4 * 1024] * 8, HQ=24, HK=2, d=256, is_causal=True, window_size=None, sinks=True),
    dict(seqlens=[32 * 1024],     HQ=24, HK=2, d=256, is_causal=True, window_size=None, sinks=True),

    dict(seqlens=[ 4 * 1024] * 8, HQ=24, HK=2, d=128, is_causal=False, window_size=None, sinks=True),
    dict(seqlens=[32 * 1024],     HQ=24, HK=2, d=128, is_causal=False, window_size=None, sinks=True),
    dict(seqlens=[ 4 * 1024] * 8, HQ=24, HK=2, d=256, is_causal=False, window_size=None, sinks=True),
    dict(seqlens=[32 * 1024],     HQ=24, HK=2, d=256, is_causal=False, window_size=None, sinks=True),

    dict(seqlens=[ 4 * 1024] * 8, HQ=24, HK=2, d=128, is_causal=True, window_size=512, sinks=True),
    dict(seqlens=[32 * 1024],     HQ=24, HK=2, d=128, is_causal=True, window_size=512, sinks=True),
    dict(seqlens=[ 4 * 1024] * 8, HQ=24, HK=2, d=256, is_causal=True, window_size=512, sinks=True),
    dict(seqlens=[32 * 1024],     HQ=24, HK=2, d=256, is_causal=True, window_size=512, sinks=True),
]


def _bench_fn(fn, warmup=3, rep=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def _get_official_module():
    import sys, pathlib, importlib
    _example_path = str(pathlib.Path(__file__).resolve().parents[2] / "examples" / "nrwu" / "tilelang-exp")
    if _example_path not in sys.path:
        sys.path.insert(0, _example_path)
    return importlib.import_module("example_gqa_sink_bwd_varlen")


@pytest.mark.parametrize("case", BENCH_CASES, ids=[_case_tag(c) for c in BENCH_CASES])
@torch.no_grad()
def test_bench_fwd(case):
    seqlens = case['seqlens']
    HQ, HK, d = case['HQ'], case['HK'], case['d']
    is_causal = case['is_causal']
    window_size = case['window_size']
    groups = HQ // HK

    q, k, v, sinks_val, cu_seqlens_q, cu_seqlens_k, max_seqlen = _make_inputs(
        seqlens, HQ, HK, d, dtype=torch.bfloat16)

    scaling = 1.0 / math.sqrt(d)
    batch_size = cu_seqlens_q.numel() - 1
    total = int(cu_seqlens_q[-1])

    # myfa kernel
    block_q, block_k, num_stages, threads = _get_fwd_block_config(d)
    if window_size is not None:
        block_k = min(block_k, window_size)
    fwd_kernel = myfa_varlen_sw_sinks_fwd.compile(
        HQ=HQ, HK=HK, D=d, scaling=scaling,
        is_causal=is_causal, window_size=window_size, has_sinks=True,
        BLOCK_Q=block_q, BLOCK_K=block_k, num_stages=num_stages, threads=threads)
    max_l = _round_up(max_seqlen, block_q)
    lse = torch.zeros((batch_size, HQ, max_l), device="cuda", dtype=torch.float32)

    def myfa_fwd():
        fwd_kernel(q, k, v, sinks_val, cu_seqlens_q, cu_seqlens_k, max_seqlen, lse)

    # FA2
    def fa2_fwd():
        flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen, max_seqlen,
            softmax_scale=scaling, causal=is_causal,
            window_size=_fa2_window_size(is_causal, window_size),
        )

    # official tilelang example
    _mod = _get_official_module()
    _official_attn = _mod._attention

    def official_fwd():
        _official_attn.apply(
            q, k, v, sinks_val,
            cu_seqlens_q, cu_seqlens_k,
            max_l, max_seqlen, max_seqlen, window_size, groups, is_causal,
        )

    myfa_ms = _bench_fn(myfa_fwd)
    fa2_ms = _bench_fn(fa2_fwd)
    official_ms = _bench_fn(official_fwd)
    total_tokens = int(cu_seqlens_q[-1])
    print(f"\n[fwd HQ={HQ} HK={HK} d={d} total={total_tokens} "
          f"seqlens={seqlens}... causal={is_causal} ws={window_size}]")
    print(f"  MyFA     fwd: {myfa_ms:.3f} ms")
    print(f"  FA2      fwd: {fa2_ms:.3f} ms")
    print(f"  Official fwd: {official_ms:.3f} ms")
    print(f"  Ratio myfa/fa2:      {myfa_ms / fa2_ms:.2f}x")
    print(f"  Ratio myfa/official: {myfa_ms / official_ms:.2f}x")


@pytest.mark.parametrize("case", BENCH_CASES, ids=[_case_tag(c) for c in BENCH_CASES])
@torch.no_grad()
def test_bench_bwd(case):
    seqlens = case['seqlens']
    HQ, HK, d = case['HQ'], case['HK'], case['d']
    is_causal = case['is_causal']
    window_size = case['window_size']
    groups = HQ // HK

    q, k, v, sinks_val, cu_seqlens_q, cu_seqlens_k, max_seqlen = _make_inputs(
        seqlens, HQ, HK, d, dtype=torch.bfloat16)

    scaling = 1.0 / math.sqrt(d)
    batch_size = cu_seqlens_q.numel() - 1
    total = int(cu_seqlens_q[-1])

    block_q_fwd, block_k_fwd, _, _ = _get_fwd_block_config(d)
    max_l = _round_up(max_seqlen, block_q_fwd)
    lse = torch.zeros((batch_size, HQ, max_l), device="cuda", dtype=torch.float32)
    delta = torch.zeros_like(lse)
    dO = torch.randn_like(q)

    # myfa bwd kernel
    block_q, block_k, num_stages, threads = _get_bwd_block_config(d)
    if window_size is not None:
        block_k = min(block_k, window_size)
    bwd_kernel = myfa_varlen_sw_sinks_bwd.compile(
        HQ=HQ, HK=HK, D=d, scaling=scaling,
        is_causal=is_causal, window_size=window_size,
        BLOCK_Q=block_q, BLOCK_K=block_k, num_stages=num_stages, threads=threads)
    dQ = torch.zeros_like(q, dtype=torch.float32)
    dK = torch.zeros(k.shape[0], HK, d, dtype=torch.float32, device="cuda")
    dV = torch.zeros(v.shape[0], HK, d, dtype=torch.float32, device="cuda")

    def myfa_bwd():
        bwd_kernel(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen, lse, dO, delta, dQ, dK, dV)

    # FA2 bwd
    with torch.enable_grad():
        q_fa = q.detach().requires_grad_(True)
        k_fa = k.detach().requires_grad_(True)
        v_fa = v.detach().requires_grad_(True)
        fa2_o = flash_attn_varlen_func(
            q_fa, k_fa, v_fa, cu_seqlens_q, cu_seqlens_k, max_seqlen, max_seqlen,
            softmax_scale=scaling, causal=is_causal,
            window_size=_fa2_window_size(is_causal, window_size),
        )

    def fa2_bwd():
        q_fa.grad = None
        k_fa.grad = None
        v_fa.grad = None
        fa2_o.backward(dO, retain_graph=True)

    # official tilelang example bwd
    import tilelang.language as TL
    _mod = _get_official_module()
    dtype_tl = TL.float16 if q.dtype == torch.float16 else TL.bfloat16

    lse_off = torch.randn((batch_size, HQ, max_seqlen), device="cuda", dtype=torch.float32)
    delta_off = torch.randn((batch_size, HQ, max_seqlen), device="cuda", dtype=torch.float32)
    official_bwd_kernel = _mod.flashattn_bwd(
        batch_size, groups, total, total, max_seqlen, HQ, max_seqlen, d, is_causal,
        window_size=window_size, dtype=dtype_tl)
    dQ_off = torch.zeros_like(q, dtype=torch.float32)
    dK_off = torch.zeros(k.shape[0], HK, d, dtype=torch.float32, device="cuda")
    dV_off = torch.zeros(v.shape[0], HK, d, dtype=torch.float32, device="cuda")

    def official_bwd():
        official_bwd_kernel(q, k, v, dO, lse_off, delta_off,
                            cu_seqlens_q, cu_seqlens_k,
                            dQ_off, dK_off, dV_off)

    myfa_ms = _bench_fn(myfa_bwd)
    fa2_ms = _bench_fn(fa2_bwd)
    official_ms = _bench_fn(official_bwd)
    total_tokens = int(cu_seqlens_q[-1])
    print(f"\n[bwd HQ={HQ} HK={HK} d={d} total={total_tokens} "
          f"seqlens={seqlens}... causal={is_causal} ws={window_size}]")
    print(f"  MyFA     bwd: {myfa_ms:.3f} ms")
    print(f"  FA2      bwd: {fa2_ms:.3f} ms")
    print(f"  Official bwd: {official_ms:.3f} ms")
    print(f"  Ratio myfa/fa2:      {myfa_ms / fa2_ms:.2f}x")
    print(f"  Ratio myfa/official: {myfa_ms / official_ms:.2f}x")
