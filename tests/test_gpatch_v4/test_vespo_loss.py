import math
import types
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.training_backend.loss import (
    PolicyLossInput,
    get_loss_fn,
)

_REDUCE_METRICS_PATH = (
    "gpatch_v4.training_backend.loss.ppo_loss.reduce_metrics_across_data_parallel_group"
)

_ADVANTAGES = torch.tensor([[1.0, 1.0, 0.0], [-2.0, -2.0, -2.0]])
_PREV_LOG_PROBS = torch.tensor([[-1.0, -2.0, -0.5], [-1.5, -0.25, -3.0]])
_RESPONSE_MASK = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float)


def _make_config(**overrides):
    kwargs = dict(
        loss_func="vespo",
        use_legacy_loss=False,
        grpo_kl_loss_beta=0.0,
        ppo_entropy_bonus=0.0,
        # VESPO 用它同时截 per-token log-ratio 与序列级 log W，不能为 None
        ppo_logps_ratio_clamp=20.0,
    )
    kwargs.update(overrides)
    return types.SimpleNamespace(ppo=PpoConfig(**kwargs))


def _make_loss_input(curr_log_probs: torch.Tensor, **overrides) -> PolicyLossInput:
    kwargs = dict(
        advantages=_ADVANTAGES,
        prev_log_probs=_PREV_LOG_PROBS,
        ref_log_probs=None,
        curr_log_probs=curr_log_probs,
        response_mask=_RESPONSE_MASK,
        scaled_entropy=torch.tensor(0.0),
        per_token_entropy=torch.tensor([[0.8, 1.3, 0.0], [1.1, 0.6, 1.7]]),
        # 默认让 rollout == prev，折叠项为 0，W 退化成 curr/prev
        rollout_log_probs=_PREV_LOG_PROBS,
    )
    kwargs.update(overrides)
    return PolicyLossInput(**kwargs)


def _phi_max(c1: float, c2: float) -> float:
    """VESPO 命题 C.1：sup_W phi(W) = (c1/c2)^c1 * exp(c2 - c1)。"""
    return (c1 / c2)**c1 * math.exp(c2 - c1)


def _expected_seq_mean_sum(per_token: torch.Tensor) -> torch.Tensor:
    per_seq_mean = (per_token * _RESPONSE_MASK).sum(dim=-1) / _RESPONSE_MASK.sum(dim=-1)
    return per_seq_mean.sum()


@patch(_REDUCE_METRICS_PATH)
def test_vespo_on_policy_reduces_to_reinforce(_mock_reduce):
    # curr == prev 时 W = 1、phi(1) = 1，loss 退化成 -A * log(pi) 的逐样本均值和
    curr_log_probs = _PREV_LOG_PROBS.clone().requires_grad_(True)
    loss_input = _make_loss_input(curr_log_probs)

    loss, count, metrics = get_loss_fn("mcore", "vespo")(_make_config(), loss_input)

    torch.testing.assert_close(
        loss, _expected_seq_mean_sum(-_ADVANTAGES * _PREV_LOG_PROBS)
    )
    torch.testing.assert_close(count, torch.tensor(2.0))
    torch.testing.assert_close(metrics["vespo/phi_mean"][0] / metrics["vespo/phi_mean"][1],
                               torch.tensor(1.0))
    torch.testing.assert_close(metrics["vespo/w_seq_max"], torch.tensor(1.0))


@patch(_REDUCE_METRICS_PATH)
def test_vespo_kernel_matches_closed_form_with_asymmetric_coefficients(_mock_reduce):
    # 两条序列的 log W 相同但优势符号相反，phi 必须取不同的 (c1, c2)
    curr_log_probs = _PREV_LOG_PROBS + torch.tensor([[0.3, -0.1, 5.0], [0.1, 0.05, 0.05]])
    loss_input = _make_loss_input(curr_log_probs.requires_grad_(True))
    config = _make_config()

    _, _, metrics = get_loss_fn("mcore", "vespo")(config, loss_input)

    log_w = torch.tensor([0.2, 0.2])
    w = log_w.exp()
    expected_pos = math.exp(3.0 + 2.0 * 0.2 - 3.0 * w[0].item())
    expected_neg = math.exp(2.0 + 3.0 * 0.2 - 2.0 * w[1].item())
    torch.testing.assert_close(
        metrics["vespo/log_w_seq_mean"][0], log_w.sum(), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        metrics["vespo/phi_pos_mean"][0], torch.tensor(expected_pos), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        metrics["vespo/phi_neg_mean"][0], torch.tensor(expected_neg), rtol=1e-5, atol=1e-5
    )
    assert expected_pos != pytest.approx(expected_neg)


@pytest.mark.parametrize("log_ratio_scale", (-8.0, -1.0, 1.0, 8.0, 40.0))
@patch(_REDUCE_METRICS_PATH)
def test_vespo_stays_bounded_under_extreme_staleness(_mock_reduce, log_ratio_scale):
    curr_log_probs = (_PREV_LOG_PROBS + log_ratio_scale).requires_grad_(True)
    loss_input = _make_loss_input(curr_log_probs)
    config = _make_config()

    loss, _, metrics = get_loss_fn("mcore", "vespo")(config, loss_input)
    loss.backward()

    bound = max(_phi_max(2.0, 3.0), _phi_max(3.0, 2.0))
    assert torch.isfinite(loss)
    assert metrics["vespo/phi_max"] <= bound + 1e-5
    assert torch.isfinite(curr_log_probs.grad).all()


@patch(_REDUCE_METRICS_PATH)
def test_vespo_gradient_only_flows_through_log_probs(_mock_reduce):
    # phi 已 detach，梯度必须等于 -phi * A（在 mask 内），与 W 的求导无关
    curr_log_probs = (_PREV_LOG_PROBS + 0.2).requires_grad_(True)
    loss_input = _make_loss_input(curr_log_probs)

    loss, _, metrics = get_loss_fn("mcore", "vespo")(_make_config(), loss_input)
    loss.backward()

    phi = torch.tensor(
        [
            metrics["vespo/phi_pos_mean"][0].item(),
            metrics["vespo/phi_neg_mean"][0].item(),
        ]
    ).unsqueeze(-1)
    seq_scale = 1.0 / _RESPONSE_MASK.sum(dim=-1, keepdim=True)
    expected_grad = -phi * _ADVANTAGES * _RESPONSE_MASK * seq_scale
    torch.testing.assert_close(curr_log_probs.grad, expected_grad, rtol=1e-5, atol=1e-6)


@patch(_REDUCE_METRICS_PATH)
def test_vespo_folds_rollout_importance_ratio_into_w(_mock_reduce):
    # 折叠后 log W = sum(curr - rollout)，而不是 sum(curr - prev)
    curr_log_probs = (_PREV_LOG_PROBS + 0.2).requires_grad_(True)
    rollout_log_probs = _PREV_LOG_PROBS - 0.1
    loss_input = _make_loss_input(
        curr_log_probs, rollout_log_probs=rollout_log_probs
    )

    _, _, metrics = get_loss_fn("mcore", "vespo")(_make_config(), loss_input)

    expected_log_w = ((curr_log_probs - rollout_log_probs) * _RESPONSE_MASK).sum(dim=-1)
    torch.testing.assert_close(
        metrics["vespo/log_w_seq_mean"][0],
        expected_log_w.sum().detach(),
        rtol=1e-5,
        atol=1e-5,
    )


@patch(_REDUCE_METRICS_PATH)
def test_vespo_requires_rollout_log_probs(_mock_reduce):
    loss_input = _make_loss_input(
        (_PREV_LOG_PROBS + 0.2).requires_grad_(True), rollout_log_probs=None
    )

    with pytest.raises(AssertionError, match="always requires"):
        get_loss_fn("mcore", "vespo")(_make_config(), loss_input)


def test_vespo_config_rejects_invalid_coefficients():
    with pytest.raises(AssertionError, match="vespo_c1_pos must be >= 1.0"):
        _make_config(vespo_c1_pos=0.5)
    with pytest.raises(AssertionError, match="vespo_c2_neg must be > 0.0"):
        _make_config(vespo_c2_neg=0.0)


def test_vespo_config_rejects_double_counted_rollout_is():
    with pytest.raises(AssertionError, match="a second time"):
        _make_config(enable_off_policy_correction=True)


def test_vespo_config_rejects_skip_prev_logps_and_legacy_path():
    with pytest.raises(AssertionError, match="requires prev_log_probs"):
        _make_config(skip_prev_logps=True)
    with pytest.raises(AssertionError, match="use_legacy_loss=False"):
        _make_config(use_legacy_loss=True)
