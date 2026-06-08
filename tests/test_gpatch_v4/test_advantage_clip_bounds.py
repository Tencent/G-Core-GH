import pytest

from gpatch_v4.utils.ppo_utils import get_advantage_clip_bounds


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
