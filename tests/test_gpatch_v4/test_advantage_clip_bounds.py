import pytest
import torch

from gpatch_v4.utils.ppo_utils import (
    count_advantage_clip_samples,
    get_advantage_clip_bounds,
)


def test_legacy_advantage_clip_resolves_symmetric_bounds():
    assert get_advantage_clip_bounds(2.0, None, None) == (-2.0, 2.0)


def test_explicit_advantage_clip_bounds():
    assert get_advantage_clip_bounds(None, -1.0, 5.0) == (-1.0, 5.0)


def test_explicit_advantage_clip_upper_bound_only():
    assert get_advantage_clip_bounds(None, None, 5.0) == (None, 5.0)


def test_explicit_advantage_clip_lower_bound_only():
    assert get_advantage_clip_bounds(None, -1.0, None) == (-1.0, None)


def test_advantage_clip_rejects_legacy_and_explicit_bounds():
    with pytest.raises(AssertionError, match="cannot be set together"):
        get_advantage_clip_bounds(2.0, -1.0, 5.0)


def test_advantage_clip_rejects_invalid_symmetric_bound():
    with pytest.raises(AssertionError, match="must be positive"):
        get_advantage_clip_bounds(0.0, None, None)


def test_advantage_clip_rejects_invalid_explicit_bounds():
    with pytest.raises(AssertionError, match="must be less than"):
        get_advantage_clip_bounds(None, 5.0, 5.0)


def test_count_advantage_clip_samples_lower_and_upper():
    # sample0: lower-clipped; sample1: upper-clipped; sample2: untouched
    original = [
        torch.tensor([-3.0, -3.0, 0.0]),
        torch.tensor([3.0, 3.0, 0.0]),
        torch.tensor([0.5, 0.5, 0.0]),
    ]
    clipped = [
        torch.tensor([-1.0, -1.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0]),
        torch.tensor([0.5, 0.5, 0.0]),
    ]
    masks = [
        torch.tensor([1.0, 1.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0]),
    ]
    n_lower, n_upper, n_samples = count_advantage_clip_samples(original, clipped, masks)
    assert (n_lower, n_upper, n_samples) == (1, 1, 3)


def test_count_advantage_clip_samples_ignores_masked_tokens():
    # only padded positions differ; valid tokens identical → no clip
    original = [torch.tensor([0.5, 9.0])]
    clipped = [torch.tensor([0.5, 1.0])]
    masks = [torch.tensor([1.0, 0.0])]
    n_lower, n_upper, n_samples = count_advantage_clip_samples(original, clipped, masks)
    assert (n_lower, n_upper, n_samples) == (0, 0, 1)


def test_count_advantage_clip_samples_all_masked_sample():
    original = [torch.tensor([-3.0, 3.0])]
    clipped = [torch.tensor([-1.0, 1.0])]
    masks = [torch.tensor([0.0, 0.0])]
    n_lower, n_upper, n_samples = count_advantage_clip_samples(original, clipped, masks)
    assert (n_lower, n_upper, n_samples) == (0, 0, 1)
