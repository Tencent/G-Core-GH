"""Packed THD MyFA dynamic-CP forward/backward vs unsharded packed reference.

Static BSHD CP uses ``_myfa_attention_with_context_parallel_backward``.
Dynamic CP packed THD uses a different pair:

- ``local_cp=1``: varlen over full packed sequences (already aligned to BSHD).
- ``local_cp>1``: zigzag shard + gathered K/V + ``_run_varlen_myfa_backward``.

This file checks the second path against the first, in one process, by looping
CP ranks. It does **not** use the BSHD kernel as the oracle.

Run::

    pytest -s tests/test_gpatch_v4/test_welm_v45_myfa_packed_cp.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# Direct pytest (no itao_test.sh) does not export PYTHONPATH. gpatch_v4 lives at
# repo root; Megatron-LM is the sibling checkout used by gcore tests.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARENT = _REPO_ROOT.parent
for extra in (
        _REPO_ROOT,
        _REPO_ROOT / "Megatron-LM",
        _PARENT / "Megatron-LM",
        _PARENT / "mbridge",
):
    path = str(extra)
    if extra.exists() and path not in sys.path:
        sys.path.insert(0, path)


def _split_thd_by_padded_seqlens(tensor, padded_lens, cp_size):
    parts = []
    offset = 0
    for padded_len in padded_lens:
        local_len = padded_len // cp_size
        parts.append(tensor[offset:offset + local_len])
        offset += local_len
    assert offset == tensor.shape[0], f"{offset=} != {tensor.shape[0]=}"
    return parts


def _merge_zigzag_chunks(rank_chunks, cp_size):
    if cp_size == 1:
        return rank_chunks[0]
    local_len = rank_chunks[0].shape[0]
    chunk = local_len // 2
    chunks = [None] * (2 * cp_size)
    for rank, local in enumerate(rank_chunks):
        chunks[rank] = local[:chunk]
        chunks[2 * cp_size - rank - 1] = local[chunk:]
    return torch.cat(chunks, dim=0)


def _zigzag_shard_packed(tensor: torch.Tensor, padded_lens: list[int], cp_size: int, rank: int):
    """Natural packed THD ``[T, ...]`` → this rank's zigzag local shard."""
    parts = _split_thd_by_padded_seqlens(tensor, padded_lens, 1)
    shards = []
    for seq, padded_len in zip(parts, padded_lens):
        chunk = padded_len // (2 * cp_size)
        first = seq[rank * chunk:(rank + 1) * chunk]
        second_idx = 2 * cp_size - rank - 1
        second = seq[second_idx * chunk:(second_idx + 1) * chunk]
        shards.append(torch.cat([first, second], dim=0))
    return torch.cat(shards, dim=0)


def _rank_concat_zigzag(tensor: torch.Tensor, padded_lens: list[int], cp_size: int):
    """Natural packed THD → CP all-gather layout (rank-concat zigzag shards)."""
    return torch.cat(
        [_zigzag_shard_packed(tensor, padded_lens, cp_size, rank) for rank in range(cp_size)],
        dim=0,
    )


def _merge_rank_shards_to_natural(
    rank_locals: list[torch.Tensor],
    padded_lens: list[int],
    cp_size: int,
):
    """Per-rank zigzag local packed tensors → natural packed THD."""
    split_by_rank = [
        _split_thd_by_padded_seqlens(local, padded_lens, cp_size) for local in rank_locals
    ]
    merged = []
    for seq_idx in range(len(padded_lens)):
        merged.append(
            _merge_zigzag_chunks([split_by_rank[rank][seq_idx] for rank in range(cp_size)], cp_size)
        )
    return torch.cat(merged, dim=0)


def _gathered_sum_to_natural(
    rank_gathered: list[torch.Tensor],
    padded_lens: list[int],
    cp_size: int,
):
    """Sum CP-all-gather backward outputs, then zigzag-rank-concat → natural."""
    gathered = rank_gathered[0].clone()
    for extra in rank_gathered[1:]:
        gathered = gathered + extra
    local_total = sum(padded_len // cp_size for padded_len in padded_lens)
    shards = list(torch.split(gathered, local_total, dim=0))
    return _merge_rank_shards_to_natural(shards, padded_lens, cp_size)


def _psp(padded_lens: list[int], local_cp_size: int, cp_group=None):
    return SimpleNamespace(
        cp_group=cp_group,
        local_cp_size=local_cp_size,
        _myfa_padded_lens_cache=list(padded_lens),
    )


class _FakeAttention:
    def __init__(self, window_size=None):
        self._window_size = window_size
        self.pg_collection = SimpleNamespace(cp=None)

    def _get_layer_window_size(self):
        return self._window_size


def _assert_close(tag: str, actual: torch.Tensor, reference: torch.Tensor, *, atol_avg: float,
                  atol_max: float):
    delta = (actual.float() - reference.float()).abs()
    avg = delta.mean().item()
    mx = delta.max().item()
    print(f"{tag} | avg {avg:.6e} | max {mx:.6e}")
    assert avg < atol_avg, f"{tag} avg {avg} >= {atol_avg}"
    assert mx < atol_max, f"{tag} max {mx} >= {atol_max}"


def test_zigzag_pack_unpack_is_identity():
    """Layout helper used by the CP=2 vs CP=1 compare: shard then merge."""
    padded_lens = [8, 4]
    cp_size = 2
    natural = torch.arange(sum(padded_lens), dtype=torch.float32)
    shards = [_zigzag_shard_packed(natural, padded_lens, cp_size, rank) for rank in range(cp_size)]
    restored = _merge_rank_shards_to_natural(shards, padded_lens, cp_size)
    torch.testing.assert_close(restored, natural)

    gathered = _rank_concat_zigzag(natural, padded_lens, cp_size)
    # Rank-concat: rank0 packed (seq0 [0,1,6,7] + seq1 [8,11]) then rank1.
    torch.testing.assert_close(
        gathered,
        torch.tensor([0.0, 1.0, 6.0, 7.0, 8.0, 11.0, 2.0, 3.0, 4.0, 5.0, 9.0, 10.0]),
    )
    torch.testing.assert_close(
        _gathered_sum_to_natural([gathered], padded_lens, cp_size),
        natural,
    )


def _run_packed_cp2_vs_cp1(*, padded_lens, window_size, has_sinks):
    from gpatch_v4.training_backend.megatron_backend import welm_v45_myfa as myfa

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    assert all(length % 4 == 0 for length in padded_lens)
    hq, hk, head_dim = 8, 2, 128
    softmax_scale = 1.0 / math.sqrt(head_dim)
    cp_size = 2
    total = sum(padded_lens)

    query = torch.randn(1, total, hq, head_dim, device=device, dtype=dtype)
    key = torch.randn(1, total, hk, head_dim, device=device, dtype=dtype)
    value = torch.randn(1, total, hk, head_dim, device=device, dtype=dtype)
    sinks = torch.randn(hq, device=device, dtype=dtype) if has_sinks else None
    doutput = torch.randn_like(query)

    attention = _FakeAttention(window_size=window_size)
    psp_cp1 = _psp(padded_lens, local_cp_size=1, cp_group=None)

    out_ref, lse_ref = myfa._myfa_attention_packed_thd_forward(
        attention,
        query,
        key,
        value,
        None,
        sinks,
        softmax_scale,
        0.0,
        psp_cp1,
    )
    dq_ref, dk_ref, dv_ref, ds_ref = myfa._myfa_attention_packed_thd_backward(
        attention,
        query,
        key,
        value,
        None,
        sinks,
        softmax_scale,
        0.0,
        out_ref,
        lse_ref,
        doutput,
        psp_cp1,
    )

    gathered_key = _rank_concat_zigzag(key[0], padded_lens, cp_size).unsqueeze(0)
    gathered_value = _rank_concat_zigzag(value[0], padded_lens, cp_size).unsqueeze(0)
    fake_group = object()
    psp_cp2 = _psp(padded_lens, local_cp_size=cp_size, cp_group=fake_group)

    rank_outs = []
    rank_dqs = []
    rank_dks = []
    rank_dvs = []
    rank_dsinks = []
    for rank in range(cp_size):
        q_local = _zigzag_shard_packed(query[0], padded_lens, cp_size, rank).unsqueeze(0)
        dout_local = _zigzag_shard_packed(doutput[0], padded_lens, cp_size, rank).unsqueeze(0)
        with patch.object(myfa.dist, "get_rank", lambda group=None, r=rank: r):
            with patch.object(myfa.dist, "get_world_size", lambda group=None: cp_size):
                out_local, lse_local = myfa._myfa_attention_packed_thd_forward(
                    attention,
                    q_local,
                    gathered_key,
                    gathered_value,
                    None,
                    sinks,
                    softmax_scale,
                    0.0,
                    psp_cp2,
                )
                dq, dk, dv, dsinks = myfa._myfa_attention_packed_thd_backward(
                    attention,
                    q_local,
                    gathered_key,
                    gathered_value,
                    None,
                    sinks,
                    softmax_scale,
                    0.0,
                    out_local,
                    lse_local,
                    dout_local,
                    psp_cp2,
                )
        rank_outs.append(out_local[0])
        rank_dqs.append(dq[0])
        rank_dks.append(dk[0])
        rank_dvs.append(dv[0])
        rank_dsinks.append(dsinks)

    out_cp = _merge_rank_shards_to_natural(rank_outs, padded_lens, cp_size).unsqueeze(0)
    dq_cp = _merge_rank_shards_to_natural(rank_dqs, padded_lens, cp_size).unsqueeze(0)
    dk_cp = _gathered_sum_to_natural(rank_dks, padded_lens, cp_size).unsqueeze(0)
    dv_cp = _gathered_sum_to_natural(rank_dvs, padded_lens, cp_size).unsqueeze(0)

    # Same kernel, different zigzag split: bf16 softmax order can drift a bit.
    _assert_close("out", out_cp, out_ref, atol_avg=2e-2, atol_max=0.15)
    _assert_close("dQ", dq_cp, dq_ref, atol_avg=3e-2, atol_max=0.25)
    _assert_close("dK", dk_cp, dk_ref, atol_avg=3e-2, atol_max=0.25)
    _assert_close("dV", dv_cp, dv_ref, atol_avg=3e-2, atol_max=0.25)
    if has_sinks:
        ds_cp = rank_dsinks[0]
        for extra in rank_dsinks[1:]:
            ds_cp = ds_cp + extra
        _assert_close("dSinks", ds_cp, ds_ref, atol_avg=3e-2, atol_max=0.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("has_sinks", [False, True], ids=["no_sinks", "sinks"])
@pytest.mark.parametrize("window_size", [None, 128], ids=["causal", "swa128"])
def test_packed_thd_cp2_fwd_bwd_matches_unsharded_cp1(has_sinks, window_size):
    # Existing short-seq case: window=128 >= seq=128, so SWA degenerates to causal.
    _run_packed_cp2_vs_cp1(padded_lens=[128, 64], window_size=window_size, has_sinks=has_sinks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("has_sinks", [False, True], ids=["no_sinks", "sinks"])
def test_packed_thd_cp2_swa_window_lt_seq(has_sinks):
    """Production SWA layers use window=512 on ~51k tokens. Short-seq test never clips.

    chunk(1024, cp=2)=256 > window=128, so the second zigzag segment actually
    slides; dQ of SWA+kv_mirror layers was 0 in the 64-GPU dump.
    """
    _run_packed_cp2_vs_cp1(padded_lens=[1024, 512], window_size=128, has_sinks=has_sinks)
