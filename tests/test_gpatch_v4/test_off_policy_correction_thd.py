from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.core.correction_helper import (
    compute_off_policy_correction_weights,
    masked_reduce_thd_expand,
)


def _make_config(level: str, veto_threshold=None):
    return SimpleNamespace(
        ppo=SimpleNamespace(
            off_policy_correction_level=level,
            off_policy_correction_mode="truncate",
            off_policy_correction_upper_bound=10.0,
            off_policy_correction_lower_bound=None,
            off_policy_correction_veto_threshold=veto_threshold,
        )
    )


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("sequence", [[4.0, 4.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0]]),
        ("geometric", [[2.0, 2.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]]),
    ],
)
def test_thd_sequence_levels_match_bshd(level, expected):
    ratios = torch.tensor([[2.0, 2.0, 3.0, 3.0, 0.5, 0.5, 3.0, 3.0]])
    prev_log_probs = ratios.log()
    rollout_log_probs = torch.zeros_like(prev_log_probs)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32)

    thd_weights, _, _ = compute_off_policy_correction_weights(
        True,
        _make_config(level),
        prev_log_probs,
        rollout_log_probs,
        mask,
        cu_seqlens_padded=cu_seqlens_padded,
    )

    bshd_weights, _, _ = compute_off_policy_correction_weights(
        True,
        _make_config(level),
        prev_log_probs.reshape(2, 4),
        rollout_log_probs.reshape(2, 4),
        mask.reshape(2, 4),
    )

    expected_tensor = torch.tensor(expected)
    assert torch.allclose(thd_weights, expected_tensor)
    assert torch.allclose(thd_weights, bshd_weights.reshape_as(thd_weights))


def test_thd_veto_only_rejects_the_affected_segment():
    ratios = torch.tensor([[1.0, 1.0, 3.0, 3.0, 1.0, 0.5, 3.0, 3.0]])
    prev_log_probs = ratios.log()
    rollout_log_probs = torch.zeros_like(prev_log_probs)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])

    _, modified_mask, _ = compute_off_policy_correction_weights(
        True,
        _make_config("token", veto_threshold=0.75),
        prev_log_probs,
        rollout_log_probs,
        mask,
        cu_seqlens_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
    )

    assert torch.equal(
        modified_mask,
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("sum", [4.0, 4.0, 0.25, 0.25]),
        ("mean", [2.0, 2.0, 0.5, 0.5]),
    ],
)
def test_thd_segment_reduce_across_dynamic_cp(reduction, expected):
    local_ratios = torch.tensor([2.0, 3.0, 0.5, 3.0])
    local_values = local_ratios.log()
    local_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    remote_contribution = torch.tensor(
        [
            [torch.log(torch.tensor(2.0)), torch.log(torch.tensor(0.5))],
            [1.0, 1.0],
        ]
    )
    cp_group = object()

    def fake_all_reduce(tensor, group, op):
        assert group is cp_group
        assert op == torch.distributed.ReduceOp.SUM
        tensor.add_(remote_contribution)

    with (
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_reduce", side_effect=fake_all_reduce),
    ):
        reduced = masked_reduce_thd_expand(
            local_values,
            local_mask,
            torch.tensor([0, 4, 8], dtype=torch.int32),
            local_cp_group=cp_group,
            reduction=reduction,
        )

    assert torch.allclose(reduced.exp(), torch.tensor(expected))
