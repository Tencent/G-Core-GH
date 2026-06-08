"""CPU unit tests for ``McoreEngine._maybe_post_clip_grad_norm``.

The helper is exercised as an unbound method against minimal stub
objects to avoid pulling cuda/ray dependencies.
"""
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from gpatch_v4.training_backend.megatron_backend.mcore_engine import McoreEngine


def _make_stub_engine(
    *,
    report_post_clip_grad_norm: bool,
    is_stub_optimizer: bool = False,
    reuse_grad_buf_for_mxfp8_param_ag: bool = False,
    optimizer_spec=None,
    grad_norm_value: float = 0.5,
):
    """Build a minimal object exposing only what the helper touches."""
    if optimizer_spec is None:
        # Default: a supported MixedPrecisionOptimizer-like spec.
        from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
        optimizer_spec = MixedPrecisionOptimizer

    optimizer = MagicMock(spec=optimizer_spec)
    optimizer.is_stub_optimizer = is_stub_optimizer
    optimizer.config = SimpleNamespace(
        reuse_grad_buf_for_mxfp8_param_ag=reuse_grad_buf_for_mxfp8_param_ag,
    )
    optimizer.get_grad_norm = MagicMock(return_value=grad_norm_value)

    engine = SimpleNamespace(
        optimizer=optimizer,
        config=SimpleNamespace(
            optimizer=SimpleNamespace(
                report_post_clip_grad_norm=report_post_clip_grad_norm,
            ),
        ),
    )
    return engine


def _call(engine, update_successful: bool) -> Optional[float]:
    """Invoke the unbound helper with the stub engine as ``self``."""
    return McoreEngine._maybe_post_clip_grad_norm(engine, update_successful)


def test_switch_off_returns_none_and_skips_get_grad_norm():
    engine = _make_stub_engine(report_post_clip_grad_norm=False)
    assert _call(engine, update_successful=True) is None
    engine.optimizer.get_grad_norm.assert_not_called()


def test_step_failed_returns_none_and_skips_get_grad_norm():
    engine = _make_stub_engine(report_post_clip_grad_norm=True)
    assert _call(engine, update_successful=False) is None
    engine.optimizer.get_grad_norm.assert_not_called()


def test_stub_optimizer_returns_none_and_skips_get_grad_norm():
    engine = _make_stub_engine(
        report_post_clip_grad_norm=True,
        is_stub_optimizer=True,
    )
    assert _call(engine, update_successful=True) is None
    engine.optimizer.get_grad_norm.assert_not_called()


def test_mxfp8_grad_buf_reuse_raises_assertion_error():
    engine = _make_stub_engine(
        report_post_clip_grad_norm=True,
        reuse_grad_buf_for_mxfp8_param_ag=True,
    )
    with pytest.raises(AssertionError, match="reuse_grad_buf_for_mxfp8_param_ag"):
        _call(engine, update_successful=True)


def test_unsupported_optimizer_returns_none_and_warns_once():
    # A type that is not in the whitelist (and not ChainedOptimizer).
    class _FakeUnknownOptimizer:
        pass

    engine = _make_stub_engine(
        report_post_clip_grad_norm=True,
        optimizer_spec=_FakeUnknownOptimizer,
    )

    with patch(
        "gpatch_v4.training_backend.megatron_backend.mcore_engine.log_info"
    ) as mock_log:
        # First call -> warns once.
        assert _call(engine, update_successful=True) is None
        # Second call -> still skips, must NOT warn again.
        assert _call(engine, update_successful=True) is None

    engine.optimizer.get_grad_norm.assert_not_called()
    # Exactly one warn over two skipped calls.
    assert mock_log.call_count == 1
    warn_msg = mock_log.call_args_list[0].args[0]
    assert "report_post_clip_grad_norm" in warn_msg
    assert "outside the validated whitelist" in warn_msg


def test_supported_optimizer_returns_get_grad_norm_value():
    engine = _make_stub_engine(
        report_post_clip_grad_norm=True,
        grad_norm_value=0.7,
    )
    assert _call(engine, update_successful=True) == 0.7
    engine.optimizer.get_grad_norm.assert_called_once_with()


def test_chained_optimizer_with_all_supported_inner_returns_value():
    """ChainedOptimizer is allowed iff every inner optimizer is supported."""
    from megatron.core.optimizer.optimizer import (
        ChainedOptimizer,
        MixedPrecisionOptimizer,
    )

    inner_a = MagicMock(spec=MixedPrecisionOptimizer)
    inner_b = MagicMock(spec=MixedPrecisionOptimizer)
    chained = MagicMock(spec=ChainedOptimizer)
    chained.is_stub_optimizer = False
    chained.config = SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=False)
    chained.chained_optimizers = [inner_a, inner_b]
    chained.get_grad_norm = MagicMock(return_value=1.25)

    engine = SimpleNamespace(
        optimizer=chained,
        config=SimpleNamespace(
            optimizer=SimpleNamespace(report_post_clip_grad_norm=True),
        ),
    )
    assert _call(engine, update_successful=True) == 1.25
    chained.get_grad_norm.assert_called_once_with()


def test_chained_optimizer_with_unsupported_inner_returns_none():
    from megatron.core.optimizer.optimizer import (
        ChainedOptimizer,
        MixedPrecisionOptimizer,
    )

    class _FakeUnknownOptimizer:
        pass

    inner_a = MagicMock(spec=MixedPrecisionOptimizer)
    inner_b = MagicMock(spec=_FakeUnknownOptimizer)
    chained = MagicMock(spec=ChainedOptimizer)
    chained.is_stub_optimizer = False
    chained.config = SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=False)
    chained.chained_optimizers = [inner_a, inner_b]
    chained.get_grad_norm = MagicMock(return_value=1.25)

    engine = SimpleNamespace(
        optimizer=chained,
        config=SimpleNamespace(
            optimizer=SimpleNamespace(report_post_clip_grad_norm=True),
        ),
    )
    with patch(
        "gpatch_v4.training_backend.megatron_backend.mcore_engine.log_info"
    ) as mock_log:
        assert _call(engine, update_successful=True) is None
    chained.get_grad_norm.assert_not_called()
    # Unsupported chained inner -> still warn-once.
    assert mock_log.call_count == 1
