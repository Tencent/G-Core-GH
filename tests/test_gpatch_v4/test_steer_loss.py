import types
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.training_backend.loss import (
    PolicyLossInput,
    get_loss_fn,
)
from gpatch_v4.training_backend.loss.ppo_loss import (
    _compute_steer_token_weights,
)
from gpatch_v4.training_backend.loss.metrics import (
    _STEER_HISTOGRAM_BIN_COUNT,
    steer_histogram_quantiles,
)
from gpatch_v4.training_backend.loss.registry import finalize_histogram_metrics

_REDUCE_METRICS_PATH = (
    "gpatch_v4.training_backend.loss.ppo_loss.reduce_metrics_across_data_parallel_group"
)


def _make_config(
    loss_func: str = "steer",
    token_weight_min: float = 0.8,
    steer_policy_method: str = "grpo",
):
    return types.SimpleNamespace(
        ppo=PpoConfig(
            loss_func=loss_func,
            steer_token_weight_min=token_weight_min,
            steer_policy_method=steer_policy_method,
            grpo_kl_loss_beta=0.0,
        ),
    )


def _make_loss_input(curr_log_probs: torch.Tensor) -> PolicyLossInput:
    return PolicyLossInput(
        advantages=torch.tensor([[1.0, -0.5, 0.75], [-1.0, 0.25, -0.75]]),
        prev_log_probs=torch.tensor([[-1.0, -2.0, -0.5], [-1.5, -0.25, -3.0]]),
        ref_log_probs=None,
        curr_log_probs=curr_log_probs,
        response_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float),
        scaled_entropy=torch.tensor(0.0),
        per_token_entropy=torch.tensor([[0.8, 1.3, 0.0], [1.1, 0.6, 1.7]]),
        prev_per_token_entropy=torch.tensor([[0.8, 1.3, 0.0], [1.1, 0.6, 1.7]]),
    )


def test_steer_token_weights_are_detached_and_bounded():
    loss_input = _make_loss_input(
        torch.tensor([[-0.8, -1.8, -0.7], [-1.2, -0.4, -2.8]], requires_grad=True)
    )

    token_weights, metrics = _compute_steer_token_weights(
        advantages=loss_input.advantages,
        prev_log_probs=loss_input.prev_log_probs,
        curr_log_probs=loss_input.curr_log_probs,
        prev_per_token_entropy=loss_input.prev_per_token_entropy,
        response_mask=loss_input.response_mask,
        token_weight_min=0.8,
    )

    valid_weights = token_weights[loss_input.response_mask.bool()]
    assert not token_weights.requires_grad
    assert valid_weights.min() >= 0.8
    assert valid_weights.max() <= 1.0
    assert torch.all(token_weights[~loss_input.response_mask.bool()] == 0)
    assert set(metrics) == {
        "steer/token_weight_mean",
        "steer/token_weight_min",
        "steer/token_weight_max",
        "steer/token_weight_lt_0_99_frac",
        "steer/token_weight_lt_0_95_frac",
        "steer/entropy_change_metric_mean",
        "steer/entropy_change_metric_max",
        "steer/token_weight_histogram",
        "steer/entropy_change_metric_histogram",
    }
    valid_token_count = torch.tensor(float(valid_weights.numel()))
    torch.testing.assert_close(
        metrics["steer/token_weight_mean"],
        torch.stack([valid_weights.sum(), valid_token_count]),
    )
    torch.testing.assert_close(
        metrics["steer/token_weight_lt_0_99_frac"],
        torch.stack([(valid_weights < 0.99).sum().float(), valid_token_count]),
    )
    torch.testing.assert_close(
        metrics["steer/token_weight_lt_0_95_frac"],
        torch.stack([(valid_weights < 0.95).sum().float(), valid_token_count]),
    )
    torch.testing.assert_close(metrics["steer/token_weight_histogram"].sum(), valid_token_count)
    torch.testing.assert_close(
        metrics["steer/entropy_change_metric_histogram"].sum(),
        valid_token_count,
    )


def test_steer_histogram_quantiles_merge_microbatch_distributions():
    first_microbatch_histogram = torch.zeros(_STEER_HISTOGRAM_BIN_COUNT)
    second_microbatch_histogram = torch.zeros(_STEER_HISTOGRAM_BIN_COUNT)
    first_microbatch_histogram[0] = 4
    second_microbatch_histogram[-1] = 6

    p50, p90, p99 = steer_histogram_quantiles(
        first_microbatch_histogram + second_microbatch_histogram,
        value_min=0.0,
        value_max=1.0,
        log_space=False,
    )

    expected_upper_bin_center = torch.tensor(
        1.0 - 0.5 / _STEER_HISTOGRAM_BIN_COUNT
    )
    torch.testing.assert_close(p50, expected_upper_bin_center)
    torch.testing.assert_close(p90, expected_upper_bin_center)
    torch.testing.assert_close(p99, expected_upper_bin_center)


def test_steer_histogram_finalizer_reports_global_quantiles():
    histograms = {
        "steer/token_weight_histogram": torch.ones(_STEER_HISTOGRAM_BIN_COUNT),
        "steer/entropy_change_metric_histogram": torch.ones(_STEER_HISTOGRAM_BIN_COUNT + 1),
    }

    metrics = finalize_histogram_metrics("steer", _make_config(), histograms)

    assert set(metrics) == {
        "steer/token_weight_p50",
        "steer/token_weight_p90",
        "steer/token_weight_p99",
        "steer/entropy_change_metric_p50",
        "steer/entropy_change_metric_p90",
        "steer/entropy_change_metric_p99",
        "steer/entropy_change_metric_histogram_overflow_frac",
    }
    assert 0.8 <= metrics["steer/token_weight_p50"] <= 1.0
    torch.testing.assert_close(
        metrics["steer/entropy_change_metric_histogram_overflow_frac"],
        torch.tensor(1.0 / (_STEER_HISTOGRAM_BIN_COUNT + 1)),
    )


def test_histogram_finalizer_rejects_unregistered_loss():
    with pytest.raises(ValueError, match="has no registered finalizer"):
        finalize_histogram_metrics(
            "grpo",
            _make_config(loss_func="grpo"),
            {"grpo/entropy_histogram": torch.ones(_STEER_HISTOGRAM_BIN_COUNT)},
        )


@patch(_REDUCE_METRICS_PATH)
def test_steer_loss_flows_gradient_and_reports_metrics(_mock_reduce):
    loss_input = _make_loss_input(
        torch.tensor([[-0.8, -1.8, -0.7], [-1.2, -0.4, -2.8]], requires_grad=True)
    )

    loss, _, metrics = get_loss_fn("mcore", "steer")(_make_config(), loss_input)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "steer/token_weight_mean" in metrics
    loss.backward()
    assert loss_input.curr_log_probs.grad is not None
    assert loss_input.curr_log_probs.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("steer_policy_method", "method_metric"),
    (
        ("grpo", "ppo_ratio_clamped"),
        ("cispo", "cispo/clipfrac"),
        ("gspo", "ppo_ratio_clamped_seq_frac"),
        ("sapo", "sapo/gate_mean"),
    ),
)
@patch(_REDUCE_METRICS_PATH)
def test_steer_reweights_each_policy_method(_mock_reduce, steer_policy_method, method_metric):
    loss_input = _make_loss_input(
        torch.tensor([[-0.8, -1.8, -0.7], [-1.2, -0.4, -2.8]], requires_grad=True)
    )

    loss, _, metrics = get_loss_fn("mcore", "steer")(
        _make_config(steer_policy_method=steer_policy_method), loss_input
    )

    assert method_metric in metrics
    assert "steer/token_weight_mean" in metrics
    loss.backward()
    assert loss_input.curr_log_probs.grad is not None
    assert loss_input.curr_log_probs.grad.abs().sum() > 0


@patch(_REDUCE_METRICS_PATH)
def test_steer_with_unit_minimum_matches_grpo(_mock_reduce):
    curr_log_probs = torch.tensor(
        [[-0.8, -1.8, -0.7], [-1.2, -0.4, -2.8]], requires_grad=True
    )
    grpo_input = _make_loss_input(curr_log_probs)
    steer_input = _make_loss_input(curr_log_probs.detach().clone().requires_grad_(True))

    grpo_loss, grpo_count, _ = get_loss_fn("mcore", "grpo")(
        _make_config(loss_func="grpo"), grpo_input
    )
    steer_loss, steer_count, _ = get_loss_fn("mcore", "steer")(
        _make_config(token_weight_min=1.0), steer_input
    )

    torch.testing.assert_close(steer_loss, grpo_loss)
    torch.testing.assert_close(steer_count, grpo_count)


def test_steer_rejects_skip_prev_logps():
    config = _make_config()
    config.ppo.skip_prev_logps = True
    loss_input = _make_loss_input(
        torch.tensor([[-0.8, -1.8, -0.7], [-1.2, -0.4, -2.8]], requires_grad=True)
    )

    with pytest.raises(AssertionError, match="requires prev_log_probs"):
        get_loss_fn("mcore", "steer")(config, loss_input)


def test_steer_rejects_unknown_policy_method():
    with pytest.raises(AssertionError, match="steer_policy_method must be one of"):
        PpoConfig(loss_func="steer", steer_policy_method="unknown")
