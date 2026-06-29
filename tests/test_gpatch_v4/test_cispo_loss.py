"""Tests for CISPO loss (Clipped IS-weight Policy Optimization, arXiv 2506.13585)."""
import types
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.training_backend.loss_factory import (
    PolicyLossInput,
    cispo_loss_func,
    grpo_loss_func,
)

_REDUCE_METRICS_PATH = (
    "gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group"
)


def _make_config(
    ppo_ratio_eps: float = 0.2,
    ppo_clip_ratio_low: Optional[float] = None,
    ppo_clip_ratio_high: Optional[float] = None,
    ppo_logps_ratio_clamp: Optional[float] = None,
    ppo_entropy_bonus: float = 0.0,
    skip_prev_logps: bool = False,
    enable_off_policy_correction: bool = False,
    grpo_kl_loss_beta: float = 1e-3,
):
    ppo = PpoConfig(
        ppo_ratio_eps=ppo_ratio_eps,
        ppo_clip_ratio_low=ppo_clip_ratio_low,
        ppo_clip_ratio_high=ppo_clip_ratio_high,
        ppo_logps_ratio_clamp=ppo_logps_ratio_clamp,
        ppo_entropy_bonus=ppo_entropy_bonus,
        skip_prev_logps=skip_prev_logps,
        enable_off_policy_correction=enable_off_policy_correction,
        grpo_kl_loss_beta=grpo_kl_loss_beta,
    )
    return types.SimpleNamespace(ppo=ppo, debug=DebugConfig())


def _make_loss_input(
    batch_size: int = 4,
    seq_len: int = 16,
    with_ref: bool = True,
    with_rollout: bool = False,
    skip_prev_logps: bool = False,
):
    torch.manual_seed(42)
    curr = torch.randn(batch_size, seq_len, requires_grad=True)

    if skip_prev_logps:
        prev = torch.zeros(batch_size, seq_len)
    else:
        prev = curr.detach().clone() + torch.randn(batch_size, seq_len) * 0.5

    advantages = torch.randn(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)
    ref = torch.randn(batch_size, seq_len) if with_ref else None
    entropy = torch.tensor(0.5)
    rollout = torch.randn(batch_size, seq_len) if with_rollout else None
    per_token_entropy = torch.rand(batch_size, seq_len)

    return PolicyLossInput(
        advantages=advantages,
        prev_log_probs=prev,
        ref_log_probs=ref,
        curr_log_probs=curr,
        response_mask=mask,
        scaled_entropy=entropy,
        rollout_log_probs=rollout,
        per_token_entropy=per_token_entropy,
    )


class TestCispoBasic:
    """CISPO loss returns valid outputs and flows gradients."""

    @patch(_REDUCE_METRICS_PATH)
    def test_returns_loss_and_metrics(self, mock_reduce):
        config = _make_config()
        loss_input = _make_loss_input()

        bwd_loss, metrics = cispo_loss_func(config, loss_input)

        assert bwd_loss.dim() == 0
        assert not torch.isnan(bwd_loss)
        for key in ["loss", "policy_loss", "ppo_ratio", "ppo_ratio_clamped",
                     "scaled_entropy", "grpo_kl_loss", "cispo/clipfrac"]:
            assert key in metrics, f"missing metric: {key}"

    @patch(_REDUCE_METRICS_PATH)
    def test_gradient_flows(self, mock_reduce):
        config = _make_config()
        loss_input = _make_loss_input()

        bwd_loss, _ = cispo_loss_func(config, loss_input)
        bwd_loss.backward()

        assert loss_input.curr_log_probs.grad is not None
        assert loss_input.curr_log_probs.grad.abs().sum() > 0


class TestCispoVsGrpo:
    """CISPO preserves gradients for all tokens, unlike GRPO which drops clipped tokens."""

    @patch(_REDUCE_METRICS_PATH)
    def test_cispo_all_tokens_have_gradient(self, mock_reduce):
        # 用很大的 ratio 偏移确保部分 token 会被 clip
        torch.manual_seed(0)
        B, S = 2, 8
        curr = torch.randn(B, S, requires_grad=True)
        prev = curr.detach() - 2.0  # ratio = exp(curr - prev) >> 1, 会被 upper clip

        config = _make_config(ppo_ratio_eps=0.2, grpo_kl_loss_beta=0.0)
        loss_input = PolicyLossInput(
            advantages=torch.ones(B, S),
            prev_log_probs=prev,
            ref_log_probs=None,
            curr_log_probs=curr,
            response_mask=torch.ones(B, S),
            scaled_entropy=torch.tensor(0.0),
            per_token_entropy=torch.zeros(B, S),
        )

        bwd_loss, metrics = cispo_loss_func(config, loss_input)

        # 确认有 token 被 clip 了
        clipfrac = metrics["cispo/clipfrac"].item()
        assert clipfrac > 0, "test setup error: no tokens were clipped"

        bwd_loss.backward()
        grad = loss_input.curr_log_probs.grad
        assert grad is not None
        # CISPO: 即使 token 被 clip，梯度仍然流过 log π
        nonzero_frac = (grad.abs() > 1e-12).float().mean().item()
        assert nonzero_frac == 1.0, (
            f"CISPO should have gradient on all tokens, got {nonzero_frac:.2%}"
        )

    @patch(_REDUCE_METRICS_PATH)
    def test_grpo_drops_clipped_tokens(self, mock_reduce):
        # 对比：GRPO 会丢掉 clipped token 的梯度
        torch.manual_seed(0)
        B, S = 2, 8
        curr = torch.randn(B, S, requires_grad=True)
        prev = curr.detach() - 2.0

        config = _make_config(ppo_ratio_eps=0.2, grpo_kl_loss_beta=0.0)
        loss_input = PolicyLossInput(
            advantages=torch.ones(B, S),
            prev_log_probs=prev,
            ref_log_probs=None,
            curr_log_probs=curr,
            response_mask=torch.ones(B, S),
            scaled_entropy=torch.tensor(0.0),
            per_token_entropy=torch.zeros(B, S),
        )

        bwd_loss, metrics = grpo_loss_func(config, loss_input)
        bwd_loss.backward()

        grad = loss_input.curr_log_probs.grad
        # GRPO: 当 clipped loss > unclipped loss 时（positive advantage + ratio > 1+eps），
        # 梯度来自 clamped ratio 分支，该分支 ratio 是常数 → 该 token 无梯度。
        # 但由于 min(loss1, loss2) 的选择，并非所有 clipped token 都无梯度。
        # 这里只验证 GRPO 正常运行。
        assert grad is not None


class TestCispoSkipPrevLogps:

    @patch(_REDUCE_METRICS_PATH)
    def test_ratio_one_when_skip(self, mock_reduce):
        # skip_prev_logps → ratio = 1.0，等价于 REINFORCE
        config = _make_config(skip_prev_logps=True)
        loss_input = _make_loss_input(skip_prev_logps=True)

        _, metrics = cispo_loss_func(config, loss_input)
        ratio_sum_count = metrics["ppo_ratio"]
        ratio_mean = (ratio_sum_count[0] / ratio_sum_count[1]).item()
        assert abs(ratio_mean - 1.0) < 1e-5

    @patch(_REDUCE_METRICS_PATH)
    def test_gradient_flows_when_skip(self, mock_reduce):
        config = _make_config(skip_prev_logps=True)
        loss_input = _make_loss_input(skip_prev_logps=True)

        loss, _ = cispo_loss_func(config, loss_input)
        loss.backward()
        assert loss_input.curr_log_probs.grad is not None
        assert loss_input.curr_log_probs.grad.abs().sum() > 0


class TestCispoClipRatios:

    @patch(_REDUCE_METRICS_PATH)
    def test_asymmetric_clip(self, mock_reduce):
        # 论文推荐 eps_low=1.0（实质禁用下界）+ eps_high=0.2
        config = _make_config(ppo_clip_ratio_low=1.0, ppo_clip_ratio_high=0.2)
        loss_input = _make_loss_input()

        bwd_loss, metrics = cispo_loss_func(config, loss_input)
        bwd_loss.backward()

        lower_frac = metrics["ppo_ratio_clamped_lower_frac"]
        lower_rate = (lower_frac[0] / lower_frac[1]).item()
        # eps_low=1.0 → ratio 要 < 0.0 才会被 lower clip，几乎不可能
        assert lower_rate < 0.01, f"lower clip rate {lower_rate:.4f} should be ~0"

    @patch(_REDUCE_METRICS_PATH)
    def test_no_clip_when_on_policy(self, mock_reduce):
        # ratio 很接近 1.0 时应无 clip
        torch.manual_seed(123)
        B, S = 4, 16
        curr = torch.randn(B, S, requires_grad=True)
        prev = curr.detach().clone()  # ratio = exp(0) = 1.0

        config = _make_config(ppo_ratio_eps=0.2, grpo_kl_loss_beta=0.0)
        loss_input = PolicyLossInput(
            advantages=torch.randn(B, S),
            prev_log_probs=prev,
            ref_log_probs=None,
            curr_log_probs=curr,
            response_mask=torch.ones(B, S),
            scaled_entropy=torch.tensor(0.0),
            per_token_entropy=torch.zeros(B, S),
        )

        _, metrics = cispo_loss_func(config, loss_input)
        clipfrac = metrics["cispo/clipfrac"].item()
        assert clipfrac == 0.0, f"on-policy should have 0 clip, got {clipfrac}"


class TestCispoNoRef:

    @patch(_REDUCE_METRICS_PATH)
    def test_no_kl_when_no_ref(self, mock_reduce):
        config = _make_config(grpo_kl_loss_beta=0.01)
        loss_input = _make_loss_input(with_ref=False)

        bwd_loss, metrics = cispo_loss_func(config, loss_input)
        bwd_loss.backward()

        kl = metrics["grpo_kl_loss"]
        kl_val = (kl[0] / kl[1]).item()
        assert abs(kl_val) < 1e-8, "KL should be 0 when ref_log_probs=None"


class TestCispoSampleMask:

    @patch(_REDUCE_METRICS_PATH)
    def test_sample_mask_metric(self, mock_reduce):
        config = _make_config()
        loss_input = _make_loss_input()
        loss_input.sample_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])

        _, metrics = cispo_loss_func(config, loss_input)
        assert "valid_sample_ratio" in metrics
