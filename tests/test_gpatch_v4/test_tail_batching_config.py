"""tail-batching 配置项的校验单测（纯 dataclass，不需要 GPU）。"""

import pytest

from gpatch_v4.configs.training_config import RLTrainingConfig


def _make(**overrides) -> RLTrainingConfig:
    return RLTrainingConfig(**overrides)


def test_defaults_disable_tail_batching():
    cfg = _make()
    assert cfg.rollout_over_dispatch_ratio == 1.0
    assert cfg.rollout_reuse_unused_prompts is False


def test_over_dispatch_ratio_below_one_is_rejected():
    with pytest.raises(AssertionError, match="rollout_over_dispatch_ratio must be >= 1.0"):
        _make(rollout_over_dispatch_ratio=0.5)


def test_tail_batching_conflicts_with_ordered_collection():
    with pytest.raises(AssertionError, match="incompatible with rollout_ordered_collection"):
        _make(
            rollout_over_dispatch_ratio=1.5,
            rollout_ordered_collection=True,
            single_controller=True,
        )


def test_reuse_unused_prompts_requires_over_dispatch():
    with pytest.raises(AssertionError, match="requires rollout_over_dispatch_ratio > 1.0"):
        _make(rollout_reuse_unused_prompts=True)


def test_discard_hook_fields_must_be_set_together():
    with pytest.raises(AssertionError, match="must be set together"):
        _make(rollout_over_dispatch_ratio=1.5, tail_batching_discard_py_path="/tmp/x.py")
    with pytest.raises(AssertionError, match="must be set together"):
        _make(rollout_over_dispatch_ratio=1.5, tail_batching_discard_fn_name="f")


def test_discard_hook_requires_over_dispatch():
    with pytest.raises(AssertionError, match="requires rollout_over_dispatch_ratio > 1.0"):
        _make(tail_batching_discard_py_path="/tmp/x.py", tail_batching_discard_fn_name="f")
