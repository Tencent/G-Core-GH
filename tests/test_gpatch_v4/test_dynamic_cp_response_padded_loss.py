"""Dyn-CP response-padded sequence loss: path probes + numerical e2e compare.

Simulates the mixin pipeline without running a full trainer:

1. Build a pure ``[B, S]`` sequence batch (non-dyn-CP baseline).
2. Pack the same data into THD, CP-shard model outputs, reconstruct, then
   response-pad back to ``[B, max_resp]`` (dyn-CP new path).
3. Assert GRPO (+ TIS sequence) matches, and ``masked_reduce_thd_expand`` is
   never entered once ``cu_seqlens_padded=None``.

``wemm_video`` / ``welm_v4`` are out of scope this round (legacy THD loss).
"""
from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.core import correction_helper
from gpatch_v4.training_backend.loss_factory import (
    PolicyLossInput,
    grpo_loss_func,
    gspo_loss_func,
)
from gpatch_v4.utils.dynamic_cp_utils import (
    as_sequence_sample_mask,
    compute_dyn_cp_response_span,
    dynamic_cp_local_packed_token_count,
    jagged_to_response_padded,
    packed_to_jagged,
    packed_to_response_padded,
    reconstruct_dynamic_cp_packed_tensor,
)
from gpatch_v4.utils.training_utils import masked_sum_per_seq


_REDUCE_METRICS = (
    "gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group"
)
_THD_EXPAND = "gpatch_v4.core.correction_helper.masked_reduce_thd_expand"


def _make_config(
    *,
    loss_func: str = "grpo",
    enable_tis: bool = True,
    tis_level: str = "sequence",
):
    return types.SimpleNamespace(
        ppo=PpoConfig(
            loss_func=loss_func,
            ppo_entropy_bonus=0.0,
            grpo_kl_loss_beta=0.0,
            enable_off_policy_correction=enable_tis,
            off_policy_correction_level=tis_level,
            off_policy_correction_mode="truncate",
            off_policy_correction_upper_bound=2.0,
            ppo_ratio_eps=0.2,
        ),
        debug=DebugConfig(),
    )


def _make_input(
    *,
    curr,
    prev,
    advantages,
    mask,
    rollout=None,
    ref=None,
    sample_mask=None,
    cu_seqlens_padded=None,
    local_cp_size: int = 1,
    calculate_per_token_loss: bool = False,
) -> PolicyLossInput:
    return PolicyLossInput(
        advantages=advantages,
        prev_log_probs=prev,
        ref_log_probs=ref,
        curr_log_probs=curr,
        response_mask=mask,
        scaled_entropy=torch.tensor(0.0),
        rollout_log_probs=rollout if rollout is not None else prev.detach().clone(),
        per_token_entropy=torch.zeros_like(curr),
        sample_mask=sample_mask,
        cu_seqlens_padded=cu_seqlens_padded,
        local_cp_size=local_cp_size,
        calculate_per_token_loss=calculate_per_token_loss,
    )


def _build_sequence_batch(seed: int = 0):
    """Two samples with unequal response lengths and an internal mask hole."""
    torch.manual_seed(seed)
    # Sample meta in token (unshifted) coordinates: prompt_len / sequence_len.
    # After shift, actual_len = sequence_len - 1.
    specs = [
        # prompt=2, seq=6 => start=1, resp_len=4 on actual_len=5; hole at resp pos1
        dict(prompt_len=2, sequence_len=6, pad_to=8, hole_at=2),
        # prompt=1, seq=4 => start=0, resp_len=3 on actual_len=3
        dict(prompt_len=1, sequence_len=4, pad_to=8, hole_at=None),
    ]

    seq_curr = []
    seq_prev = []
    seq_adv = []
    seq_mask = []
    seq_sample_mask = []
    packed_pieces = []
    starts = []
    lengths = []
    original_lens = []
    padded_lens = []

    for i, spec in enumerate(specs):
        actual_len = spec["sequence_len"] - 1
        pad_len = spec["pad_to"]
        start, length = compute_dyn_cp_response_span(
            prompt_len=spec["prompt_len"],
            sequence_len=spec["sequence_len"],
            actual_len=actual_len,
        )
        assert length > 0

        curr = torch.randn(actual_len)
        prev = curr.detach() + 0.05 * torch.randn(actual_len)
        adv = torch.randn(actual_len)
        mask = torch.zeros(actual_len)
        mask[start:start + length] = 1.0
        if spec["hole_at"] is not None:
            mask[spec["hole_at"]] = 0.0

        # Alive / dead sample flag (expanded across tokens during packing).
        sample_alive = 1.0 if i == 0 else 1.0
        sample_mask_tok = torch.full((actual_len,), sample_alive)

        # Sequence baseline: right-pad response window to max_resp later.
        seq_curr.append(curr[start:start + length])
        seq_prev.append(prev[start:start + length])
        seq_adv.append(adv[start:start + length])
        seq_mask.append(mask[start:start + length])
        seq_sample_mask.append(torch.tensor(sample_alive))

        def _pad(t):
            out = torch.zeros(pad_len, dtype=t.dtype)
            out[:actual_len] = t
            return out

        packed_pieces.append(
            {
                "curr_log_probs": _pad(curr),
                "prev_log_probs": _pad(prev),
                "advantages": _pad(adv),
                "mask": _pad(mask),
                "sample_mask": _pad(sample_mask_tok),
            }
        )
        starts.append(start)
        lengths.append(length)
        original_lens.append(actual_len)
        padded_lens.append(pad_len)

    max_resp = max(lengths)

    def _stack_pad(rows):
        out = torch.zeros(len(rows), max_resp)
        for idx, row in enumerate(rows):
            out[idx, :row.numel()] = row
        return out

    return {
        "curr": _stack_pad(seq_curr).requires_grad_(),
        "prev": _stack_pad(seq_prev),
        "advantages": _stack_pad(seq_adv),
        "mask": _stack_pad(seq_mask),
        "sample_mask": torch.stack(seq_sample_mask),
        "starts": torch.tensor(starts, dtype=torch.int32),
        "lengths": torch.tensor(lengths, dtype=torch.int32),
        "original_lens": torch.tensor(original_lens, dtype=torch.int32),
        "padded_lens": torch.tensor(padded_lens, dtype=torch.int32),
        "packed_pieces": packed_pieces,
    }


def _pack_pieces(pieces, original_lens, padded_lens):
    packed = {}
    for key in pieces[0]:
        packed[key] = torch.cat([p[key] for p in pieces], dim=0).unsqueeze(0)
    cu_padded = torch.zeros(len(padded_lens) + 1, dtype=torch.int32)
    cu_padded[1:] = torch.cumsum(padded_lens, dim=0)
    cu = torch.zeros(len(original_lens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(original_lens, dim=0)
    return packed, cu_padded, cu


def _simulate_response_padded_dyn_cp_loss_tensors(
    packed,
    cu_padded,
    cu,
    starts,
    lengths,
    *,
    local_cp_size: int,
    cp_rank: int,
    monkeypatch,
    partition_indices,
):
    """Mirror mixin: shard model outputs, keep rollout replicated, then pad."""
    total = packed["curr_log_probs"].shape[1]
    local_tokens = total // local_cp_size
    assert total == local_tokens * local_cp_size

    class FakeGroup:
        def rank(self):
            return cp_rank

        def size(self):
            return local_cp_size

    shards = []
    for rank in range(local_cp_size):
        idx = partition_indices[rank]
        shards.append(packed["curr_log_probs"].index_select(1, idx))

    local = shards[cp_rank].detach().clone().requires_grad_()

    def fake_all_gather(outputs, _input, group):
        assert isinstance(group, FakeGroup)
        for r, shard in enumerate(shards):
            outputs[r].copy_(shard.detach())

    monkeypatch.setattr(
        "gpatch_v4.utils.dynamic_cp_utils.parallel_state.get_dynamic_data_context_parallel_groups",
        lambda group_size: FakeGroup(),
    )
    monkeypatch.setattr(
        "gpatch_v4.utils.dynamic_cp_utils.torch.distributed.all_gather",
        fake_all_gather,
    )
    monkeypatch.setattr(
        "gpatch_v4.utils.dynamic_cp_utils.get_thd_partitioned_indices",
        lambda _cu, _total, _cp, rank: partition_indices[rank],
    )

    reconstructed = reconstruct_dynamic_cp_packed_tensor(
        local,
        cu_seqlens_padded=cu_padded,
        local_cp_size=local_cp_size,
        cp_partition_mode="zigzag",
    )
    model_resp = jagged_to_response_padded(
        {"curr_log_probs": packed_to_jagged(reconstructed, cu_padded, cu)},
        starts,
        lengths,
    )
    rollout_resp = packed_to_response_padded(
        {
            "mask": packed["mask"],
            "advantages": packed["advantages"],
            "prev_log_probs": packed["prev_log_probs"],
            "sample_mask": packed["sample_mask"],
        },
        cu_padded,
        starts,
        lengths,
    )
    sample_mask = as_sequence_sample_mask(rollout_resp["sample_mask"])
    local_count = dynamic_cp_local_packed_token_count(
        packed["mask"],
        cu_padded,
        local_cp_size,
        cp_partition_mode="zigzag",
    )
    return {
        "curr": model_resp["curr_log_probs"],
        "prev": rollout_resp["prev_log_probs"],
        "advantages": rollout_resp["advantages"],
        "mask": rollout_resp["mask"],
        "sample_mask": sample_mask,
        "local": local,
        "local_count": local_count,
    }


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
@patch(_REDUCE_METRICS)
def test_response_padded_grpo_matches_pure_sequence_baseline(
    _mock_reduce,
    calculate_per_token_loss,
    monkeypatch,
):
    baseline = _build_sequence_batch()
    packed, cu_padded, cu = _pack_pieces(
        baseline["packed_pieces"],
        baseline["original_lens"],
        baseline["padded_lens"],
    )
    # Contiguous CP-2 shards for a deterministic fake partition.
    total = int(cu_padded[-1].item())
    assert total % 2 == 0
    half = total // 2
    indices = [
        torch.arange(0, half, dtype=torch.long),
        torch.arange(half, total, dtype=torch.long),
    ]

    # Use zigzag mode name but feed contiguous indices via monkeypatch.
    dyn = _simulate_response_padded_dyn_cp_loss_tensors(
        packed,
        cu_padded,
        cu,
        baseline["starts"],
        baseline["lengths"],
        local_cp_size=2,
        cp_rank=0,
        monkeypatch=monkeypatch,
        partition_indices=indices,
    )

    torch.testing.assert_close(dyn["curr"], baseline["curr"].detach(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dyn["mask"], baseline["mask"], atol=0, rtol=0)
    torch.testing.assert_close(dyn["sample_mask"], baseline["sample_mask"], atol=0, rtol=0)
    assert dyn["sample_mask"].ndim == 1

    config = _make_config(enable_tis=True, tis_level="sequence")
    with patch(_THD_EXPAND) as thd_expand:
        thd_expand.side_effect = AssertionError("TIS must not enter THD reduce after response-pad")
        base_loss, base_metrics = grpo_loss_func(
            config,
            _make_input(
                curr=baseline["curr"],
                prev=baseline["prev"],
                advantages=baseline["advantages"],
                mask=baseline["mask"],
                sample_mask=baseline["sample_mask"],
                cu_seqlens_padded=None,
                local_cp_size=1,
                calculate_per_token_loss=calculate_per_token_loss,
            ),
        )
        dyn_loss, dyn_metrics = grpo_loss_func(
            config,
            _make_input(
                curr=dyn["curr"],
                prev=dyn["prev"],
                advantages=dyn["advantages"],
                mask=dyn["mask"],
                sample_mask=dyn["sample_mask"],
                cu_seqlens_padded=None,
                local_cp_size=1,
                calculate_per_token_loss=calculate_per_token_loss,
            ),
        )

    torch.testing.assert_close(dyn_loss, base_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        dyn_metrics["ppo_ratio"],
        base_metrics["ppo_ratio"],
        atol=1e-5,
        rtol=1e-5,
    )

    # Local shard token counts across CP ranks must partition the packed mask.
    counts = []
    for rank in range(2):
        monkeypatch.setattr(
            "gpatch_v4.utils.dynamic_cp_utils.parallel_state.get_dynamic_data_context_parallel_groups",
            lambda group_size, _rank=rank: types.SimpleNamespace(rank=lambda: _rank),
        )
        monkeypatch.setattr(
            "gpatch_v4.utils.dynamic_cp_utils.get_thd_partitioned_indices",
            lambda _cu, _total, _cp, r: indices[r],
        )
        counts.append(
            dynamic_cp_local_packed_token_count(
                packed["mask"],
                cu_padded,
                2,
                cp_partition_mode="zigzag",
            )
        )
    torch.testing.assert_close(torch.stack(counts).sum(), packed["mask"].sum())


@patch(_REDUCE_METRICS)
def test_response_padded_gspo_accepts_sequence_format(_mock_reduce, monkeypatch):
    baseline = _build_sequence_batch(seed=1)
    packed, cu_padded, cu = _pack_pieces(
        baseline["packed_pieces"],
        baseline["original_lens"],
        baseline["padded_lens"],
    )
    total = int(cu_padded[-1].item())
    half = total // 2
    indices = [
        torch.arange(0, half, dtype=torch.long),
        torch.arange(half, total, dtype=torch.long),
    ]
    dyn = _simulate_response_padded_dyn_cp_loss_tensors(
        packed,
        cu_padded,
        cu,
        baseline["starts"],
        baseline["lengths"],
        local_cp_size=2,
        cp_rank=0,
        monkeypatch=monkeypatch,
        partition_indices=indices,
    )
    config = _make_config(loss_func="gspo", enable_tis=False)
    # Must not raise the THD assert inside gspo_loss_func.
    loss, _metrics = gspo_loss_func(
        config,
        _make_input(
            curr=dyn["curr"],
            prev=dyn["prev"],
            advantages=dyn["advantages"],
            mask=dyn["mask"],
            sample_mask=dyn["sample_mask"],
            ref=dyn["prev"].detach(),
            cu_seqlens_padded=None,
            local_cp_size=1,
        ),
    )
    assert torch.isfinite(loss)


def test_tis_sequence_uses_masked_sum_expand_not_thd(monkeypatch):
    """Direct probe of correction_helper branching after sequence conversion."""
    b, s = 2, 4
    prev = torch.randn(b, s)
    rollout = prev + 0.1
    mask = torch.ones(b, s)

    config = _make_config(enable_tis=True, tis_level="sequence")
    calls = {"thd": 0, "sum_expand": 0}

    real_sum_expand = correction_helper.masked_sum_expand

    def counting_sum_expand(*args, **kwargs):
        calls["sum_expand"] += 1
        return real_sum_expand(*args, **kwargs)

    def counting_thd(*args, **kwargs):
        calls["thd"] += 1
        raise AssertionError("should not use THD reduce")

    monkeypatch.setattr(correction_helper, "masked_sum_expand", counting_sum_expand)
    monkeypatch.setattr(correction_helper, "masked_reduce_thd_expand", counting_thd)

    weights, _mask, _metrics = correction_helper.compute_off_policy_correction_weights(
        True,
        config,
        prev,
        rollout,
        mask,
        cu_seqlens_padded=None,
        local_cp_group=None,
    )
    assert calls["sum_expand"] == 1
    assert calls["thd"] == 0
    assert weights.shape == mask.shape


def test_sample_mask_shape_keeps_seq_mean_denominator_correct():
    values = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    # Token-expanded sample mask must be collapsed before seq-mean.
    token_expanded = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    sample_mask = as_sequence_sample_mask(token_expanded)
    assert sample_mask.shape == (2,)

    numerator = masked_sum_per_seq(values, mask, sample_mask)
    # Only first sample: mean((1+2)/2) = 1.5
    torch.testing.assert_close(numerator, torch.tensor(1.5))
    torch.testing.assert_close(sample_mask.sum(), torch.tensor(1.0))
