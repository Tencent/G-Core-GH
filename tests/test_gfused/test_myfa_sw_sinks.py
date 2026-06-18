"""
myfa_sw_sinks forward / backward correctness tests.

GQA attention kernel (BSHD layout) with sliding-window causal masking
and learnable sinks.

Example usage::

    cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/wepsdl/gcore-dev
    pytest -s tests/test_gfused/test_myfa_sw_sinks.py -v
    pytest -s tests/test_gfused/test_myfa_sw_sinks.py::test_fwd_bwd -v
"""

import math
from pathlib import Path

import pytest
import torch

from gfused.myfa_sw_sinks import (
    myfa_sw_sinks_bwd,
    myfa_sw_sinks_bwd_dsink,
    myfa_sw_sinks_bwd_pre,
    myfa_sw_sinks_fwd,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def ref_attn(q, k, v, mask, scaling, sink=None, num_key_value_groups=1):
    """BHSD layout reference."""
    key_states = repeat_kv(k, num_key_value_groups)
    value_states = repeat_kv(v, num_key_value_groups)

    bsz, num_heads, q_len, _ = q.shape

    attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * scaling

    if mask is not None:
        causal_mask = mask[:, :, :, :key_states.shape[-2]]
        if causal_mask.dtype == torch.bool:
            min_dtype = torch.finfo(q.dtype).min
            causal_mask = torch.where(causal_mask, 0.0, min_dtype).to(q.dtype)
        attn_weights = attn_weights + causal_mask

    # Attention sink: append learnable logit column before softmax
    if sink is not None:
        sink_logits = sink.reshape(1, -1, 1, 1).expand(
            bsz, -1, q_len, -1
        )
        attn_weights = torch.cat([attn_weights, sink_logits], dim=-1)

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(q.dtype)

    # Drop the virtual sink column after softmax
    if sink is not None:
        attn_weights = attn_weights[..., :-1]

    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

# Production config: HQ=24, HK=2, D=256, window_size=512 (SWA layers)
CASES = [
    # 全 causal（对应 full causal mask 层，sliding_window=262144）
    dict(b=2, HQ=12, HK=2, s=256,  d=128, window_size=None),
    dict(b=2, HQ=12, HK=2, s=512,  d=128, window_size=None),
    dict(b=2, HQ=12, HK=2, s=1024, d=128, window_size=None),
    # SWA（对应 sliding window causal mask 层，sliding_window=512）
    dict(b=2, HQ=12, HK=2, s=512,  d=128, window_size=512),
    dict(b=2, HQ=12, HK=2, s=1024, d=128, window_size=512),
    dict(b=2, HQ=12, HK=2, s=2048, d=128, window_size=512),
    # MHA (groups=1)
    dict(b=2, HQ=4, HK=4, s=512, d=128, window_size=None),
    dict(b=2, HQ=4, HK=4, s=1024, d=128, window_size=512),
    # D=64
    dict(b=2, HQ=12, HK=2, s=512, d=64, window_size=None),
    dict(b=2, HQ=12, HK=2, s=512, d=64, window_size=512),
    # D=256（实际 head_dim）
    dict(b=2, HQ=24, HK=2, s=512,  d=256, window_size=None),
    dict(b=2, HQ=24, HK=2, s=1024, d=256, window_size=512),
    # no sinks
    dict(b=2, HQ=12, HK=2, s=512,  d=128, window_size=None,  sinks=False),
    dict(b=2, HQ=12, HK=2, s=1024, d=128, window_size=512,   sinks=False),
    dict(b=2, HQ=24, HK=2, s=512,  d=256, window_size=None,  sinks=False),
]


def _make_inputs(b, HQ, HK, s, d, window_size, dtype=torch.float16):
    """Generate BSHD tensors + BHSD mask for ref_attn."""
    scaling = 1.0 / math.sqrt(d)
    groups = HQ // HK

    mask = torch.triu(
        torch.full((b, 1, s, s), float("-inf"), device='cuda', dtype=dtype),
        diagonal=1,
    )
    if window_size is not None:
        row_idx = torch.arange(s, device='cuda').unsqueeze(1)
        col_idx = torch.arange(s, device='cuda').unsqueeze(0)
        too_old = col_idx < (row_idx - window_size + 1)
        mask.masked_fill_(too_old.unsqueeze(0).unsqueeze(0), float("-inf"))

    rng = torch.Generator(device='cuda').manual_seed(1919810)
    q = torch.randn(b, s, HQ, d, generator=rng, device="cuda", dtype=dtype)
    k = torch.randn(b, s, HK, d, generator=rng, device="cuda", dtype=dtype)
    v = torch.randn(b, s, HK, d, generator=rng, device="cuda", dtype=dtype)
    return q, k, v, mask, scaling, groups


def _case_tag(c):
    ws = c['window_size']
    ws_str = f"sw{ws}" if ws is not None else "causal"
    tag = f"b{c['b']}hq{c['HQ']}hkv{c['HK']}s{c['s']}d{c['d']}_{ws_str}"
    if not c.get('sinks', True):
        tag += "_nosink"
    return tag


def _check(tag, a, b, atol_avg=0.01, atol_max=0.05):
    avg = (a - b).abs().mean().item()
    mx = (a - b).abs().max().item()
    print(f'{tag} | avg {avg:.6e} | max {mx:.6e}')
    assert avg < atol_avg, f'{tag} avg {avg} >= {atol_avg}'
    assert mx < atol_max, f'{tag} max {mx} >= {atol_max}'


def _bshd_to_bhsd(x):
    return x.transpose(1, 2)


def _bhsd_to_bshd(x):
    return x.transpose(1, 2)


@pytest.mark.parametrize("case", CASES, ids=[_case_tag(c) for c in CASES])
def test_fwd_bwd(case):
    case_inputs = {k: v for k, v in case.items() if k != 'sinks'}
    q, k, v, mask, scaling, groups = _make_inputs(**case_inputs)
    HQ, HK, d = case['HQ'], case['HK'], case['d']
    window_size = case['window_size']
    has_sinks = case.get('sinks', True)
    dt = q.dtype

    rng = torch.Generator(device='cuda').manual_seed(114514)
    sink = torch.randn(HQ, generator=rng, device='cuda', dtype=dt) if has_sinks else None
    sink_ag = None

    # 1. ref (eager) forward + backward — ref_attn expects BHSD
    with torch.enable_grad():
        q_ag = _bshd_to_bhsd(q).detach().requires_grad_(True)
        k_ag = _bshd_to_bhsd(k).detach().requires_grad_(True)
        v_ag = _bshd_to_bhsd(v).detach().requires_grad_(True)
        if has_sinks:
            sink_ag = sink.detach().requires_grad_(True)
        ref_o_bhsd = ref_attn(
            q_ag, k_ag, v_ag, mask, scaling, sink=sink_ag,
            num_key_value_groups=groups,
        )
        ref_o = _bhsd_to_bshd(ref_o_bhsd).contiguous()
        dO = torch.randn_like(ref_o)
        ref_o_bhsd.backward(_bshd_to_bhsd(dO))
    ref_dQ = _bhsd_to_bshd(q_ag.grad).contiguous()
    ref_dK = _bhsd_to_bshd(k_ag.grad).contiguous()
    ref_dV = _bhsd_to_bshd(v_ag.grad).contiguous()
    ref_dSink = sink_ag.grad if has_sinks else None

    # 2. tilelang forward (BSHD)
    sink_kernel = sink if has_sinks else torch.empty(HQ, dtype=dt, device='cuda')
    fwd_kernel = myfa_sw_sinks_fwd.compile(
        HQ=HQ, HK=HK, D=d, scaling=scaling,
        is_causal=True, window_size=window_size,
        has_sinks=has_sinks, dtype=dt,
    )
    my_o, my_lse = fwd_kernel(q, k, v, sink_kernel)

    # 3. tilelang backward (BSHD)
    bwd_pre_kernel = myfa_sw_sinks_bwd_pre.compile(HQ=HQ, D=d, dtype=dt)
    my_delta = bwd_pre_kernel(my_o, dO)

    bwd_kernel = myfa_sw_sinks_bwd.compile(
        HQ=HQ, HK=HK, D=d, scaling=scaling,
        is_causal=True, window_size=window_size, dtype=dt,
    )
    my_dQ = torch.zeros_like(q, dtype=torch.float32)
    my_dK = torch.zeros(
        q.shape[0], k.shape[1], HK, d,
        dtype=torch.float32, device='cuda',
    )
    my_dV = torch.zeros(
        q.shape[0], v.shape[1], HK, d,
        dtype=torch.float32, device='cuda',
    )
    bwd_kernel(q, k, v, my_lse, dO, my_delta, my_dQ, my_dK, my_dV)
    my_dQ = my_dQ.to(dt)
    my_dK = my_dK.to(dt)
    my_dV = my_dV.to(dt)

    tag = _case_tag(case)
    gqa_atol_max = 0.05 * max(groups, 1)
    _check(f'[{tag}] O',  my_o,  ref_o)
    _check(f'[{tag}] dQ', my_dQ, ref_dQ)
    _check(f'[{tag}] dK', my_dK, ref_dK)
    _check(f'[{tag}] dV', my_dV, ref_dV)
    if has_sinks:
        dsink_kernel = myfa_sw_sinks_bwd_dsink.compile(HQ=HQ, dtype=dt)
        my_dSink = dsink_kernel(sink_kernel, my_delta, my_lse).sum(0).sum(1).to(ref_dSink.dtype)
        _check(f'[{tag}] dSink', my_dSink, ref_dSink)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

BENCH_CASES = [
    dict(b=2, HQ=12, HK=2, s=4096,  d=128, window_size=None),
    dict(b=2, HQ=12, HK=2, s=4096,  d=128, window_size=512),
    dict(b=2, HQ=24, HK=2, s=4096,  d=128, window_size=None),
    # 实际 config: HQ=24, HK=2, D=256
    dict(b=2, HQ=24, HK=2, s=4096,  d=256, window_size=None),
    dict(b=2, HQ=24, HK=2, s=4096,  d=256, window_size=512),
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


@pytest.mark.parametrize("case", BENCH_CASES, ids=[_case_tag(c) for c in BENCH_CASES])
@torch.no_grad()
def test_bench_fwd_throughput(case):
    q, k, v, mask, scaling, groups = _make_inputs(**case, dtype=torch.bfloat16)
    HQ, HK, d = case['HQ'], case['HK'], case['d']
    window_size = case['window_size']
    sink = torch.full((HQ,), float("-inf"), device='cuda', dtype=torch.bfloat16)

    dt = q.dtype
    fwd_kernel = myfa_sw_sinks_fwd.compile(
        HQ=HQ, HK=HK, D=d, scaling=scaling,
        is_causal=True, window_size=window_size, dtype=dt,
    )

    def myfa_fwd():
        fwd_kernel(q, k, v, sink)

    def eager_fwd():
        ref_attn(_bshd_to_bhsd(q), _bshd_to_bhsd(k), _bshd_to_bhsd(v),
                 mask, scaling, sink=sink, num_key_value_groups=groups)

    eager_ms = _bench_fn(eager_fwd)
    myfa_ms = _bench_fn(myfa_fwd)

    tag = _case_tag(case)
    print(f'\n[fwd {tag}]')
    print(f'  eager: {eager_ms:.3f} ms')
    print(f'  myfa : {myfa_ms:.3f} ms')
    print(f'  Ratio eager/myfa: {eager_ms / myfa_ms:.2f}x')


@pytest.mark.parametrize("case", BENCH_CASES, ids=[_case_tag(c) for c in BENCH_CASES])
def test_bench_bwd_throughput(case):
    q, k, v, mask, scaling, groups = _make_inputs(**case, dtype=torch.bfloat16)
    HQ, HK, d = case['HQ'], case['HK'], case['d']
    window_size = case['window_size']
    sink = torch.full((HQ,), float("-inf"), device='cuda', dtype=torch.bfloat16)

    dt = q.dtype
    fwd_kernel = myfa_sw_sinks_fwd.compile(
        HQ=HQ, HK=HK, D=d, scaling=scaling,
        is_causal=True, window_size=window_size, dtype=dt,
    )
    bwd_pre_kernel = myfa_sw_sinks_bwd_pre.compile(HQ=HQ, D=d, dtype=dt)
    bwd_kernel = myfa_sw_sinks_bwd.compile(
        HQ=HQ, HK=HK, D=d, scaling=scaling,
        is_causal=True, window_size=window_size, dtype=dt,
    )

    with torch.no_grad():
        my_o, my_lse = fwd_kernel(q, k, v, sink)
    dO = torch.rand_like(my_o)

    def myfa_bwd():
        my_delta = bwd_pre_kernel(my_o, dO)
        my_dQ = torch.zeros_like(q, dtype=torch.float32)
        my_dK = torch.zeros(
            q.shape[0], k.shape[1], HK, d,
            dtype=torch.float32, device='cuda',
        )
        my_dV = torch.zeros(
            q.shape[0], v.shape[1], HK, d,
            dtype=torch.float32, device='cuda',
        )
        bwd_kernel(q, k, v, my_lse, dO, my_delta, my_dQ, my_dK, my_dV)

    q_bhsd = _bshd_to_bhsd(q)
    k_bhsd = _bshd_to_bhsd(k)
    v_bhsd = _bshd_to_bhsd(v)
    dO_bhsd = _bshd_to_bhsd(dO)

    def eager_bwd():
        with torch.enable_grad():
            q_ag = q_bhsd.detach().requires_grad_(True)
            k_ag = k_bhsd.detach().requires_grad_(True)
            v_ag = v_bhsd.detach().requires_grad_(True)
            eager_o = ref_attn(
                q_ag, k_ag, v_ag, mask, scaling, sink=sink,
                num_key_value_groups=groups,
            )
            eager_o.backward(dO_bhsd)

    eager_ms = _bench_fn(eager_bwd)
    myfa_ms = _bench_fn(myfa_bwd)

    tag = _case_tag(case)
    print(f'\n[bwd {tag}]')
    print(f'  eager: {eager_ms:.3f} ms')
    print(f'  myfa : {myfa_ms:.3f} ms')
    print(f'  Ratio eager/myfa: {eager_ms / myfa_ms:.2f}x')
