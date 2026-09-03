# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.

import os

import pytest
import torch

from gpatch_v4.core.device import get_device_module


# Rank-68 local-expert occupancy from the step-430 NaN (32 experts).
_OBSERVED_EXPERT_SPLITS = [
    7444,
    133,
    16,
    6365,
    3877,
    2065,
    3994,
    349,
    748,
    7586,
    260,
    1003,
    3296,
    6984,
    5587,
    25283,
    4533,
    4,
    3144,
    1124,
    149282,
    971,
    3025,
    12284,
    8271,
    23447,
    6867,
    51246,
    10679,
    223,
    1140,
    166073,
]

_EP = 8
_NUM_LOCAL = 32


def _require_accelerator():
    if not get_device_module().is_available():
        pytest.skip("accelerator is required")
    return torch.device("cuda")

def _make_ep_sort_chunks_inputs(hidden=8):
    """Build the AlltoAll permute-2 layout: 32 experts x 8 EP ranks.

    Incoming chunks are expert-major (e0r0..e0r7, e1r0, ...). The
    dispatcher then reorders to rank-major via ``sorted_idxs``.
    Each non-empty chunk is filled with a unique fingerprint so a wrong
    gather cannot accidentally match.
    """
    chunk_sizes = []
    chunks_p = []
    chunks_t = []
    for expert_id, count in enumerate(_OBSERVED_EXPERT_SPLITS):
        base, rem = divmod(int(count), _EP)
        parts = [base] * _EP
        parts[-1] += rem
        for rank, n in enumerate(parts):
            chunk_id = expert_id * _EP + rank
            chunk_sizes.append(n)
            if n == 0:
                chunks_p.append(torch.empty(0, dtype=torch.float32))
                chunks_t.append(torch.empty(0, hidden, dtype=torch.bfloat16))
            else:
                chunks_p.append(torch.full((n,), 0.1 + chunk_id * 0.001))
                chunks_t.append(
                    torch.full((n, hidden), float(chunk_id), dtype=torch.bfloat16)
                )
    tokens = torch.cat(chunks_t, dim=0)
    probs = torch.cat(chunks_p, dim=0)
    sorted_idxs = [e * _EP + r for r in range(_EP) for e in range(_NUM_LOCAL)]
    ref_probs = torch.cat([chunks_p[i] for i in sorted_idxs], dim=0)
    ref_tokens = torch.cat([chunks_t[i] for i in sorted_idxs], dim=0)
    return tokens, probs, chunk_sizes, sorted_idxs, ref_tokens, ref_probs


def _prob_mismatch(fused_probs, ref_probs):
    fused = fused_probs.detach().cpu().float().reshape(-1)
    ref = ref_probs.float().reshape(-1)
    if fused.numel() != ref.numel():
        return fused.numel(), None, None
    diff = (fused - ref).abs()
    n_bad = int((diff > 1.0e-6).sum().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    return fused.numel(), n_bad, max_abs


def test_te_sort_chunks_with_probs_corrupts_at_ep_scale():
    """Direct TE kernel call. FAIL + dump = proof the fused-with-probs path is wrong.

    Tokens are expected to match the reference. Probs are not: on the
    machine that reproduced step 430 we saw ``n_mismatch=517263/517303``
    and fused values like ``7e-41``.
    """
    device = _require_accelerator()
    try:
        from transformer_engine.pytorch.permutation import (
            moe_sort_chunks_by_index_with_probs,
        )
    except ImportError:
        pytest.skip("transformer_engine.moe_sort_chunks_by_index_with_probs is missing")
    # Device backend may wrap this symbol; hit the raw kernel to prove
    # the TE implementation itself is wrong.
    # moe_sort_chunks_by_index_with_probs = getattr(
    #     moe_sort_chunks_by_index_with_probs,
    #     "_original",
    #     moe_sort_chunks_by_index_with_probs,
    # )

    tokens, probs, chunk_sizes, sorted_idxs, ref_tokens, ref_probs = (
        _make_ep_sort_chunks_inputs()
    )
    print(
        f"[te-kernel] tokens={tokens.size(0)} nchunks={len(chunk_sizes)} "
        f"zeros={sum(1 for n in chunk_sizes if n == 0)} max_chunk={max(chunk_sizes)}",
        flush=True,
    )
    out_tokens, out_probs = moe_sort_chunks_by_index_with_probs(
        tokens.to(device),
        probs.to(device),
        torch.tensor(chunk_sizes, dtype=torch.int64, device=device),
        torch.tensor(sorted_idxs, dtype=torch.int64, device=device),
    )

    tok_max = float(
        (out_tokens.detach().cpu().float() - ref_tokens.float()).abs().max().item()
    )
    n, n_bad, max_abs = _prob_mismatch(out_probs, ref_probs)
    fused_min = float(out_probs.detach().cpu().float().min())
    fused_max = float(out_probs.detach().cpu().float().max())
    ref_min = float(ref_probs.min())
    ref_max = float(ref_probs.max())
    print(
        f"[te-kernel] tokens_max_abs={tok_max} "
        f"prob_mismatch={n_bad}/{n} prob_max_abs={max_abs} "
        f"fused_range=[{fused_min}, {fused_max}] "
        f"ref_range=[{ref_min}, {ref_max}]",
        flush=True,
    )

    assert tok_max <= 1.0e-3, (
        f"TE sort_chunks also corrupted tokens (max_abs={tok_max}); "
        "this case expected token-correct / prob-wrong"
    )
    if n_bad and n_bad > 0:
        pytest.fail(
            "TE moe_sort_chunks_by_index_with_probs corrupted router probs "
            f"while tokens stayed correct: tokens_max_abs={tok_max} "
            f"prob_mismatch={n_bad}/{n} prob_max_abs={max_abs} "
            f"fused_range=[{fused_min}, {fused_max}] "
            f"ref_range=[{ref_min}, {ref_max}]"
        )
    torch.testing.assert_close(
        out_probs.detach().cpu().float(),
        ref_probs.float(),
        rtol=0.0,
        atol=0.0,
    )


def test_sort_chunks_by_idxs_gathers_probs_correctly():
    """After the device-backend TE wrap, Megatron's fused path is correct."""
    device = _require_accelerator()
    from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs

    tokens, probs, chunk_sizes, sorted_idxs, ref_tokens, ref_probs = (
        _make_ep_sort_chunks_inputs()
    )
    out_tokens, out_probs = sort_chunks_by_idxs(
        tokens.to(device),
        torch.tensor(chunk_sizes, dtype=torch.int64, device=device),
        torch.tensor(sorted_idxs, dtype=torch.int64, device=device),
        probs=probs.to(device),
        fused=True,
    )
    n, n_bad, max_abs = _prob_mismatch(out_probs, ref_probs)
    print(
        f"[megatron-fix] prob_mismatch={n_bad}/{n} max_abs={max_abs}",
        flush=True,
    )
    torch.testing.assert_close(
        out_tokens.detach().cpu().float(),
        ref_tokens.float(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        out_probs.detach().cpu().float(),
        ref_probs.float(),
        rtol=0.0,
        atol=0.0,
    )
