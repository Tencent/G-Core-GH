# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""Compare FlashAttention-2 vs FlashAttention-3 throughput (incl. FA3 ``sm_margin``).

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$PYTHONPATH"
    pytest -v -s --timeout=1800 tests/test_gfused/test_fa2_fa3_throughput.py
"""

from __future__ import annotations

import importlib.util
import os
import unittest

import pytest
import torch

_FA2_AVAILABLE = importlib.util.find_spec("flash_attn") is not None
_FA3_AVAILABLE = importlib.util.find_spec("flash_attn_interface") is not None

WARMUP = int(os.environ.get("FA_THPT_WARMUP", "4"))
REPS = int(os.environ.get("FA_THPT_REPS", "16"))
DTYPE = torch.bfloat16

# (batch, seqlen, nheads, nheads_kv, headdim, causal)
BENCH_CASES = [
    dict(b=1, s=2048, h=32, h_kv=32, d=128, causal=True),
    dict(b=1, s=4096, h=32, h_kv=32, d=128, causal=True),
    dict(b=1, s=8192, h=32, h_kv=32, d=128, causal=True),
    dict(b=2, s=4096, h=16, h_kv=16, d=128, causal=True),
    dict(b=1, s=4096, h=32, h_kv=8, d=128, causal=True),  # GQA
]

# FA3 reserves ``sm_margin`` SMs (e.g. for concurrent communication).
# H20 有 78 个 SM，所以 70 会导致速度很慢，很合理。
SM_MARGINS = tuple(
    int(x) for x in os.environ.get("FA3_SM_MARGINS", "0,8,70").split(",") if x.strip()
)


def _flops(b: int, s: int, h: int, d: int, causal: bool) -> float:
    # QK^T + PV; causal ≈ half the non-causal matmul FLOPs.
    flops = 4.0 * b * h * s * s * d
    if causal:
        flops *= 0.5
    return flops


def _bench_ms(fn, warmup: int = WARMUP, reps: int = REPS) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def _make_qkv(
    b: int,
    s: int,
    h: int,
    h_kv: int,
    d: int,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # FA2 / FA3 both take BSHD.
    q = torch.randn(b, s, h, d, device="cuda", dtype=DTYPE, requires_grad=requires_grad)
    k = torch.randn(b, s, h_kv, d, device="cuda", dtype=DTYPE, requires_grad=requires_grad)
    v = torch.randn(b, s, h_kv, d, device="cuda", dtype=DTYPE, requires_grad=requires_grad)
    return q, k, v


def _case_tag(case: dict) -> str:
    causal = "causal" if case["causal"] else "full"
    return (
        f"b{case['b']}s{case['s']}h{case['h']}hkv{case['h_kv']}d{case['d']}_{causal}"
    )


def _print_row(name: str, ms: float, tflops: float) -> None:
    print(f"  {name:<16} {ms:8.3f} ms  {tflops:8.2f} TFLOPS")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _FA2_AVAILABLE, reason="flash_attn (FA2) not installed")
@pytest.mark.skipif(not _FA3_AVAILABLE, reason="flash_attn_interface (FA3) not installed")
@pytest.mark.parametrize("case", BENCH_CASES, ids=[_case_tag(c) for c in BENCH_CASES])
@torch.no_grad()
def test_fa2_vs_fa3_fwd_throughput(case: dict) -> None:
    from flash_attn import flash_attn_func as fa2_func
    from flash_attn_interface import flash_attn_func as fa3_func

    b, s, h, h_kv, d = case["b"], case["s"], case["h"], case["h_kv"], case["d"]
    causal = case["causal"]
    q, k, v = _make_qkv(b, s, h, h_kv, d)
    flops = _flops(b, s, h, d, causal)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    print(
        f"\n[fwd {_case_tag(case)}] gpu={torch.cuda.get_device_name(0)} "
        f"SMs={sm_count} dtype={DTYPE}"
    )

    fa2_ms = _bench_ms(lambda: fa2_func(q, k, v, causal=causal))
    _print_row("fa2", fa2_ms, flops / fa2_ms / 1e9)

    for sm_margin in SM_MARGINS:
        assert 0 <= sm_margin < sm_count, (
            f"sm_margin={sm_margin} out of range for {sm_count} SMs"
        )
        fa3_ms = _bench_ms(
            lambda sm_margin=sm_margin: fa3_func(
                q, k, v, causal=causal, sm_margin=sm_margin
            )
        )
        name = f"fa3(sm_m={sm_margin})"
        _print_row(name, fa3_ms, flops / fa3_ms / 1e9)
        print(f"    ratio fa3/fa2: {fa3_ms / fa2_ms:.3f}x")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _FA2_AVAILABLE, reason="flash_attn (FA2) not installed")
@pytest.mark.skipif(not _FA3_AVAILABLE, reason="flash_attn_interface (FA3) not installed")
@pytest.mark.parametrize("case", BENCH_CASES[:2], ids=[_case_tag(c) for c in BENCH_CASES[:2]])
def test_fa2_vs_fa3_bwd_throughput(case: dict) -> None:
    from flash_attn import flash_attn_func as fa2_func
    from flash_attn_interface import flash_attn_func as fa3_func

    b, s, h, h_kv, d = case["b"], case["s"], case["h"], case["h_kv"], case["d"]
    causal = case["causal"]
    # fwd + bwd ≈ 2.5x fwd FLOPs (common FA bench convention).
    flops = _flops(b, s, h, d, causal) * 2.5
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    dout = torch.randn(b, s, h, d, device="cuda", dtype=DTYPE)

    def _fa2_bwd() -> None:
        q, k, v = _make_qkv(b, s, h, h_kv, d, requires_grad=True)
        out = fa2_func(q, k, v, causal=causal)
        out.backward(dout)

    def _fa3_bwd(sm_margin: int) -> None:
        q, k, v = _make_qkv(b, s, h, h_kv, d, requires_grad=True)
        out = fa3_func(q, k, v, causal=causal, sm_margin=sm_margin)
        out.backward(dout)

    print(
        f"\n[bwd {_case_tag(case)}] gpu={torch.cuda.get_device_name(0)} "
        f"SMs={sm_count} dtype={DTYPE}"
    )

    fa2_ms = _bench_ms(_fa2_bwd)
    _print_row("fa2", fa2_ms, flops / fa2_ms / 1e9)

    for sm_margin in SM_MARGINS:
        assert 0 <= sm_margin < sm_count, (
            f"sm_margin={sm_margin} out of range for {sm_count} SMs"
        )
        fa3_ms = _bench_ms(lambda sm_margin=sm_margin: _fa3_bwd(sm_margin))
        name = f"fa3(sm_m={sm_margin})"
        _print_row(name, fa3_ms, flops / fa3_ms / 1e9)
        print(f"    ratio fa3/fa2: {fa3_ms / fa2_ms:.3f}x")


class TestFa2Fa3Env(unittest.TestCase):
    def test_packages_and_sm_margin_api(self) -> None:
        self.assertTrue(_FA2_AVAILABLE, "flash_attn (FA2) missing")
        self.assertTrue(_FA3_AVAILABLE, "flash_attn_interface (FA3) missing")
        import inspect

        from flash_attn_interface import flash_attn_func as fa3_func

        sig = inspect.signature(fa3_func)
        self.assertIn("sm_margin", sig.parameters)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(
                f"gpu={torch.cuda.get_device_name(0)} "
                f"capability={torch.cuda.get_device_capability(0)} "
                f"SMs={props.multi_processor_count}"
            )
