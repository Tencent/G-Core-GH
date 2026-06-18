"""
Vanilla MHA (BHSD layout) flash attention fwd/bwd correctness test.

Imports kernels from examples/nrwu/tilelang-exp/example_mha_bwd_bhsd.py.

Example usage::

    cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/wepsdl/gcore-dev
    pytest -s tests/test_gfused/test_myfa_bhsd.py -v
    pytest -s tests/test_gfused/test_myfa_bhsd.py::test_fwd_bwd -v
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "nrwu" / "tilelang-exp"))
from example_mha_bwd_bhsd import attention, ref_program


CASES = [
    dict(b=2, h=4, s=256,  d=64,  causal=True),
    dict(b=2, h=4, s=256,  d=64,  causal=False),
    dict(b=2, h=4, s=512,  d=64,  causal=True),
    dict(b=2, h=4, s=512,  d=64,  causal=False),
    dict(b=2, h=4, s=1024, d=64,  causal=True),
    dict(b=2, h=4, s=256,  d=128, causal=True),
    dict(b=2, h=4, s=256,  d=128, causal=False),
    dict(b=2, h=4, s=512,  d=128, causal=True),
    dict(b=2, h=4, s=1024, d=128, causal=True),
]


def _case_tag(c):
    return f"b{c['b']}h{c['h']}s{c['s']}d{c['d']}_{'causal' if c['causal'] else 'full'}"


def _check(tag, a, b, atol_avg=0.01, atol_max=0.05):
    avg = (a - b).abs().mean().item()
    mx = (a - b).abs().max().item()
    print(f'{tag} | avg {avg:.6e} | max {mx:.6e}')
    assert avg < atol_avg, f'{tag} avg {avg} >= {atol_avg}'
    assert mx < atol_max, f'{tag} max {mx} >= {atol_max}'


@pytest.mark.parametrize("case", CASES, ids=[_case_tag(c) for c in CASES])
def test_fwd_bwd(case):
    b, h, s, d, causal = case['b'], case['h'], case['s'], case['d'], case['causal']
    torch.manual_seed(42)
    Q = torch.randn(b, h, s, d, dtype=torch.float16, device='cuda').requires_grad_(True)
    K = torch.randn(b, h, s, d, dtype=torch.float16, device='cuda').requires_grad_(True)
    V = torch.randn(b, h, s, d, dtype=torch.float16, device='cuda').requires_grad_(True)
    dO = torch.randn_like(Q)

    # tilelang fwd+bwd
    O = attention(Q, K, V, causal)
    O.backward(dO, retain_graph=True)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None

    # ref fwd+bwd
    O_ref = ref_program(Q, K, V, causal)
    O_ref.backward(dO, retain_graph=True)
    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None

    tag = _case_tag(case)
    _check(f'[{tag}] O',  O,  O_ref)
    _check(f'[{tag}] dQ', dQ, dQ_ref)
    _check(f'[{tag}] dK', dK, dK_ref)
    _check(f'[{tag}] dV', dV, dV_ref)
