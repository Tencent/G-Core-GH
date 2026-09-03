"""Unit tests for dyn-CP → response-padded sequence reconstruction helpers.

Coverage note: llm / qwen3_vl attach ``dyn_cp_response_*`` and take the new
path. ``wemm_video`` / ``welm_v4`` remain on the legacy THD+CP loss path and
are intentionally uncovered this round.
"""
import pytest
import torch

import gpatch_v4.utils.dynamic_cp_utils as dynamic_cp_utils
from gpatch_v4.utils.dynamic_cp_utils import (
    as_sequence_sample_mask,
    compute_dyn_cp_response_span,
    dynamic_cp_local_packed_token_count,
    jagged_to_response_padded,
    packed_to_jagged,
    packed_to_response_padded,
    reconstruct_dynamic_cp_packed_tensor,
)


def test_packed_to_response_padded_slices_spans_and_preserves_autograd():
    values = torch.arange(7.0).reshape(1, 7).requires_grad_()
    topk_values = torch.arange(14.0).reshape(1, 7, 2).requires_grad_()
    result = packed_to_response_padded(
        {"values": values, "topk_values": topk_values},
        cu_seqlens_padded=torch.tensor([0, 4, 7], dtype=torch.int32),
        response_starts=torch.tensor([1, 0], dtype=torch.int32),
        response_lengths=torch.tensor([2, 2], dtype=torch.int32),
    )

    torch.testing.assert_close(result["values"], torch.tensor([[1.0, 2.0], [4.0, 5.0]]))
    torch.testing.assert_close(
        result["topk_values"],
        torch.tensor([[[2.0, 3.0], [4.0, 5.0]], [[8.0, 9.0], [10.0, 11.0]]]),
    )

    result["values"].sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]]))


def test_packed_to_response_padded_keeps_internal_mask_holes():
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]])
    result = packed_to_response_padded(
        {"mask": mask},
        cu_seqlens_padded=torch.tensor([0, 5], dtype=torch.int32),
        response_starts=torch.tensor([1], dtype=torch.int32),
        response_lengths=torch.tensor([4], dtype=torch.int32),
    )

    torch.testing.assert_close(result["mask"], torch.tensor([[1.0, 0.0, 1.0, 0.0]]))


def test_packed_to_jagged_drops_thd_padding_before_response_slice():
    values = torch.arange(7.0).reshape(1, 7).requires_grad_()
    jagged = packed_to_jagged(
        values,
        cu_seqlens_padded=torch.tensor([0, 4, 7], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
    )

    torch.testing.assert_close(jagged.values(), torch.tensor([0.0, 1.0, 2.0, 4.0, 5.0]))
    response = jagged_to_response_padded(
        {"values": jagged},
        response_starts=torch.tensor([1, 0], dtype=torch.int32),
        response_lengths=torch.tensor([2, 1], dtype=torch.int32),
    )["values"]
    torch.testing.assert_close(response, torch.tensor([[1.0, 2.0], [4.0, 0.0]]))

    response.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]))


def test_packed_to_response_padded_rejects_invalid_span():
    with pytest.raises(ValueError, match="outside"):
        packed_to_response_padded(
            {"values": torch.ones(1, 4)},
            cu_seqlens_padded=torch.tensor([0, 4], dtype=torch.int32),
            response_starts=torch.tensor([3], dtype=torch.int32),
            response_lengths=torch.tensor([2], dtype=torch.int32),
        )


def test_reconstruct_dynamic_cp_packed_tensor_is_identity_for_single_rank():
    values = torch.arange(4.0).reshape(1, 4).requires_grad_()
    reconstructed = reconstruct_dynamic_cp_packed_tensor(
        values,
        cu_seqlens_padded=torch.tensor([0, 4], dtype=torch.int32),
        local_cp_size=1,
    )

    assert reconstructed is values
    reconstructed.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


@pytest.mark.parametrize(
    ("partition_mode", "indices", "expected_counts"),
    [
        ("contiguous", None, [5.0, 12.0]),
        ("zigzag", [torch.tensor([0, 3]), torch.tensor([1, 2])], [9.0, 8.0]),
    ],
)
def test_dynamic_cp_local_packed_token_count_uses_only_local_thd_shard(
    monkeypatch,
    partition_mode,
    indices,
    expected_counts,
):
    class FakeGroup:
        def __init__(self, rank):
            self._rank = rank

        def rank(self):
            return self._rank

    rank = 0
    monkeypatch.setattr(
        dynamic_cp_utils.parallel_state,
        "get_dynamic_data_context_parallel_groups",
        lambda group_size: FakeGroup(rank),
    )
    if partition_mode == "zigzag":
        monkeypatch.setattr(
            dynamic_cp_utils,
            "get_thd_partitioned_indices",
            lambda _cu, _total, _cp_size, cp_rank: indices[cp_rank],
        )

    packed_mask = torch.tensor([[2.0, 3.0, 5.0, 7.0]])
    counts = []
    for rank in range(2):
        counts.append(
            dynamic_cp_local_packed_token_count(
                packed_mask,
                cu_seqlens_padded=torch.tensor([0, 4], dtype=torch.int32),
                local_cp_size=2,
                cp_partition_mode=partition_mode,
            )
        )

    torch.testing.assert_close(torch.stack(counts), torch.tensor(expected_counts))
    torch.testing.assert_close(torch.stack(counts).sum(), packed_mask.sum())


@pytest.mark.parametrize(
    ("partition_mode", "indices"),
    [
        ("contiguous", [torch.tensor([0, 1]), torch.tensor([2, 3])]),
        ("zigzag", [torch.tensor([0, 3]), torch.tensor([1, 2])]),
    ],
)
def test_reconstruct_dynamic_cp_packed_tensor_keeps_only_local_autograd(
    monkeypatch,
    partition_mode,
    indices,
):
    class FakeGroup:
        def rank(self):
            return 0

    local = torch.tensor([[10.0, 11.0]], requires_grad=True)
    remote = torch.tensor([[20.0, 21.0]])

    def fake_all_gather(outputs, _input, group):
        assert isinstance(group, FakeGroup)
        outputs[0].copy_(local.detach())
        outputs[1].copy_(remote)

    monkeypatch.setattr(
        dynamic_cp_utils.parallel_state,
        "get_dynamic_data_context_parallel_groups",
        lambda group_size: FakeGroup(),
    )
    monkeypatch.setattr(dynamic_cp_utils.torch.distributed, "all_gather", fake_all_gather)
    if partition_mode == "zigzag":
        monkeypatch.setattr(
            dynamic_cp_utils,
            "get_thd_partitioned_indices",
            lambda _cu, _total, _cp_size, rank: indices[rank],
        )

    reconstructed = reconstruct_dynamic_cp_packed_tensor(
        local,
        cu_seqlens_padded=torch.tensor([0, 4], dtype=torch.int32),
        local_cp_size=2,
        cp_partition_mode=partition_mode,
    )

    if partition_mode == "contiguous":
        torch.testing.assert_close(reconstructed, torch.tensor([[10.0, 11.0, 20.0, 21.0]]))
    else:
        torch.testing.assert_close(reconstructed, torch.tensor([[10.0, 20.0, 21.0, 11.0]]))
    reconstructed.sum().backward()
    torch.testing.assert_close(local.grad, torch.ones_like(local))


def test_compute_dyn_cp_response_span_from_prompt_sequence_metadata():
    # tokens length N => shifted actual_len = N-1. prompt=3, seq=7 =>
    # response_start = 2, response_len = min(4, 6-2) = 4.
    start, length = compute_dyn_cp_response_span(
        prompt_len=3,
        sequence_len=7,
        actual_len=6,
    )
    assert (start, length) == (2, 4)


def test_compute_dyn_cp_response_span_fallback_preserves_mask_holes():
    # Sparse mask: do not shrink the span to contiguous nonzero runs.
    mask = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    start, length = compute_dyn_cp_response_span(
        prompt_len=None,
        sequence_len=None,
        actual_len=6,
        response_mask=mask,
    )
    assert (start, length) == (2, 4)


def test_as_sequence_sample_mask_collapses_token_expanded_rows():
    token_expanded = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    collapsed = as_sequence_sample_mask(token_expanded)
    torch.testing.assert_close(collapsed, torch.tensor([1.0, 0.0]))
    assert collapsed.ndim == 1

    empty = as_sequence_sample_mask(torch.zeros(2, 0))
    torch.testing.assert_close(empty, torch.zeros(2))


def test_model_and_rollout_response_slices_stay_aligned():
    """Same spans must line up model outputs with replicated rollout fields."""
    cu_padded = torch.tensor([0, 4, 7], dtype=torch.int32)
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    starts = torch.tensor([1, 0], dtype=torch.int32)
    lengths = torch.tensor([2, 1], dtype=torch.int32)

    packed_model = torch.arange(7.0).reshape(1, 7).requires_grad_()
    packed_rollout = torch.arange(100.0, 107.0).reshape(1, 7)

    model_response = jagged_to_response_padded(
        {
            "curr": packed_to_jagged(packed_model, cu_padded, cu),
        },
        starts,
        lengths,
    )["curr"]
    rollout_response = packed_to_response_padded(
        {"prev": packed_rollout},
        cu_padded,
        starts,
        lengths,
    )["prev"]

    torch.testing.assert_close(model_response, torch.tensor([[1.0, 2.0], [4.0, 0.0]]))
    # Rollout still uses padded offsets, so sample1 start=0 keeps the THD pad
    # slot out of the response window via length=1 on the jagged/original path
    # for model, while rollout packed slice [4:5] matches the first original
    # token of sample1.
    torch.testing.assert_close(rollout_response, torch.tensor([[101.0, 102.0], [104.0, 0.0]]))
    assert model_response.shape == rollout_response.shape

    model_response.sum().backward()
    torch.testing.assert_close(
        packed_model.grad,
        torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]),
    )
