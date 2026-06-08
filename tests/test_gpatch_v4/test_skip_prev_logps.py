"""Tests for skip_prev_logps optimization (on-policy PPO ratio shortcut)."""
import types
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.training_backend.loss_factory import PolicyLossInput, grpo_loss_func, gspo_loss_func
from gpatch_v4.utils.ppo_utils import create_response_mask

_REDUCE_METRICS_PATH = (
    "gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    skip_prev_logps: bool = False,
    enable_off_policy_correction: bool = False,
    ppo_ratio_eps: float = 0.2,
    ppo_logps_ratio_clamp: Optional[float] = None,
    ppo_entropy_bonus: float = 0.0,
    ppo_dual_clip_ratio_c: Optional[float] = None,
    ppo_clip_ratio_low: Optional[float] = None,
    ppo_clip_ratio_high: Optional[float] = None,
):
    ppo = PpoConfig(
        skip_prev_logps=skip_prev_logps,
        enable_off_policy_correction=enable_off_policy_correction,
        ppo_ratio_eps=ppo_ratio_eps,
        ppo_logps_ratio_clamp=ppo_logps_ratio_clamp,
        ppo_entropy_bonus=ppo_entropy_bonus,
        ppo_dual_clip_ratio_c=ppo_dual_clip_ratio_c,
        ppo_clip_ratio_low=ppo_clip_ratio_low,
        ppo_clip_ratio_high=ppo_clip_ratio_high,
    )
    return types.SimpleNamespace(ppo=ppo, debug=DebugConfig())


def _make_loss_input(
    batch_size: int = 2,
    seq_len: int = 8,
    skip_prev_logps: bool = False,
    with_rollout_logprobs: bool = False,
):
    """Build a minimal PolicyLossInput for grpo_loss_func."""
    curr = torch.randn(batch_size, seq_len, requires_grad=True)

    if skip_prev_logps:
        prev = torch.zeros(batch_size, seq_len)
    else:
        prev = curr.detach().clone() + torch.randn(batch_size, seq_len) * 0.01

    advantages = torch.randn(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)
    ref = torch.randn(batch_size, seq_len)
    entropy = torch.tensor(0.5)
    rollout = torch.randn(batch_size, seq_len) if with_rollout_logprobs else None

    return PolicyLossInput(
        advantages=advantages,
        prev_log_probs=prev,
        ref_log_probs=ref,
        curr_log_probs=curr,
        response_mask=mask,
        scaled_entropy=entropy,
        rollout_log_probs=rollout,
    )


def _extract_ratio_mean(metrics):
    """Extract scalar ratio mean from the stacked ppo_ratio metric."""
    ppo_ratio = metrics["ppo_ratio"]
    return (ppo_ratio[0] / ppo_ratio[1]).item()


# ---------------------------------------------------------------------------
# T1: Ratio correctness when skip_prev_logps=True
# ---------------------------------------------------------------------------

class TestSkipPrevLogpsRatio:

    @patch(_REDUCE_METRICS_PATH)
    def test_ratio_is_one_when_skip(self, mock_reduce):
        """With skip, log_ratio = curr - curr.detach() = 0, so ratio = 1.0."""
        config = _make_config(skip_prev_logps=True)
        loss_input = _make_loss_input(skip_prev_logps=True)

        loss, metrics = grpo_loss_func(config, loss_input)

        ratio_mean = _extract_ratio_mean(metrics)
        assert abs(ratio_mean - 1.0) < 1e-5, f"ratio_mean={ratio_mean}, expected ~1.0"

    @patch(_REDUCE_METRICS_PATH)
    def test_gradient_flows_when_skip(self, mock_reduce):
        """Even though ratio=1.0, gradient should flow through curr_log_probs."""
        config = _make_config(skip_prev_logps=True)
        loss_input = _make_loss_input(skip_prev_logps=True)

        loss, _ = grpo_loss_func(config, loss_input)
        loss.backward()

        assert loss_input.curr_log_probs.grad is not None
        assert loss_input.curr_log_probs.grad.abs().sum() > 0

    @patch(_REDUCE_METRICS_PATH)
    def test_ratio_clamp_with_skip(self, mock_reduce):
        """ppo_logps_ratio_clamp should still work (clamping log_ratio=0 is a no-op)."""
        config = _make_config(skip_prev_logps=True, ppo_logps_ratio_clamp=5.0)
        loss_input = _make_loss_input(skip_prev_logps=True)

        loss, metrics = grpo_loss_func(config, loss_input)
        loss.backward()
        assert loss_input.curr_log_probs.grad is not None


# ---------------------------------------------------------------------------
# T2: Bit-exact regression — skip=False unchanged
# ---------------------------------------------------------------------------

class TestSkipFalseRegression:

    @patch(_REDUCE_METRICS_PATH)
    def test_skip_false_uses_prev(self, mock_reduce):
        """When skip=False, ratio uses actual prev_log_probs."""
        config = _make_config(skip_prev_logps=False)
        loss_input = _make_loss_input(skip_prev_logps=False)

        loss, metrics = grpo_loss_func(config, loss_input)
        loss.backward()

        assert loss_input.curr_log_probs.grad is not None

    @patch(_REDUCE_METRICS_PATH)
    def test_skip_false_ratio_differs_from_one(self, mock_reduce):
        """With real prev != curr, ratio should NOT be exactly 1.0."""
        torch.manual_seed(42)
        config = _make_config(skip_prev_logps=False)

        curr = torch.randn(2, 8, requires_grad=True)
        prev = curr.detach() + 0.5

        loss_input = PolicyLossInput(
            advantages=torch.ones(2, 8),
            prev_log_probs=prev,
            ref_log_probs=torch.zeros(2, 8),
            curr_log_probs=curr,
            response_mask=torch.ones(2, 8),
            scaled_entropy=torch.tensor(0.0),
        )

        _, metrics = grpo_loss_func(config, loss_input)
        ratio_mean = _extract_ratio_mean(metrics)
        assert abs(ratio_mean - 1.0) > 0.01, (
            f"ratio_mean={ratio_mean}, should differ from 1.0 with skip=False"
        )


# ---------------------------------------------------------------------------
# T3: TIS with skip
# ---------------------------------------------------------------------------

class TestTISWithSkip:

    @patch(_REDUCE_METRICS_PATH)
    def test_tis_uses_curr_detach_when_skip(self, mock_reduce):
        """When skip=True and TIS enabled, effective_prev = curr.detach()."""
        config = _make_config(
            skip_prev_logps=True,
            enable_off_policy_correction=True,
        )
        loss_input = _make_loss_input(
            skip_prev_logps=True,
            with_rollout_logprobs=True,
        )

        loss, metrics = grpo_loss_func(config, loss_input)
        loss.backward()

        assert loss_input.curr_log_probs.grad is not None
        assert any(k.startswith("off_policy_correction/") for k in metrics)

    @patch(_REDUCE_METRICS_PATH)
    def test_tis_skip_false_uses_prev(self, mock_reduce):
        """When skip=False and TIS enabled, effective_prev = prev_log_probs."""
        config = _make_config(
            skip_prev_logps=False,
            enable_off_policy_correction=True,
        )
        loss_input = _make_loss_input(
            skip_prev_logps=False,
            with_rollout_logprobs=True,
        )

        loss, metrics = grpo_loss_func(config, loss_input)
        loss.backward()
        assert loss_input.curr_log_probs.grad is not None


class TestGspoWithSkip:

    @patch(_REDUCE_METRICS_PATH)
    def test_gspo_ratio_is_one_when_skip(self, mock_reduce):
        config = _make_config(skip_prev_logps=True)
        loss_input = _make_loss_input(skip_prev_logps=True)

        _, metrics = gspo_loss_func(config, loss_input)
        ratio_mean = _extract_ratio_mean(metrics)
        assert abs(ratio_mean - 1.0) < 1e-5, f"ratio_mean={ratio_mean}, expected ~1.0"

    @patch(_REDUCE_METRICS_PATH)
    def test_gspo_gradient_flows_when_skip(self, mock_reduce):
        config = _make_config(skip_prev_logps=True)
        loss_input = _make_loss_input(skip_prev_logps=True)

        loss, _ = gspo_loss_func(config, loss_input)
        loss.backward()
        assert loss_input.curr_log_probs.grad is not None
        assert loss_input.curr_log_probs.grad.abs().sum() > 0

    @patch(_REDUCE_METRICS_PATH)
    def test_gspo_tis_with_skip(self, mock_reduce):
        config = _make_config(skip_prev_logps=True, enable_off_policy_correction=True)
        loss_input = _make_loss_input(skip_prev_logps=True, with_rollout_logprobs=True)

        loss, metrics = gspo_loss_func(config, loss_input)
        loss.backward()
        assert loss_input.curr_log_probs.grad is not None
        assert any(k.startswith("off_policy_correction/") for k in metrics)


# ---------------------------------------------------------------------------
# T4: Assert validations in _validate_skip_prev_logps
# ---------------------------------------------------------------------------

class TestValidateSkipPrevLogps:
    """Test the validation logic extracted from GrpoTrainActor."""

    def _make_actor_config(
        self,
        train_gbs=64,
        rollout_gbs=8,
        sampling_keep_n=8,
        ppo_max_epochs_2=1,
        loss_func="grpo",
        advantage_type="grpo",
        ppo_initial_policy_kl_penalty=0.0,
    ):
        training = types.SimpleNamespace(
            train_gbs=train_gbs,
            rollout_gbs=rollout_gbs,
            sampling_keep_n=sampling_keep_n,
            ppo_max_epochs_2=ppo_max_epochs_2,
        )
        ppo = PpoConfig(
            skip_prev_logps=True,
            loss_func=loss_func,
            advantage_type=advantage_type,
            ppo_initial_policy_kl_penalty=ppo_initial_policy_kl_penalty,
        )
        return types.SimpleNamespace(training=training, ppo=ppo)

    @staticmethod
    def _run_validation(config):
        from gpatch_v4.actor.grpo_train_actor import GrpoTrainActor
        actor = object.__new__(GrpoTrainActor)
        actor.config = config
        actor._validate_skip_prev_logps()

    def test_valid_config_passes(self):
        config = self._make_actor_config()
        self._run_validation(config)

    def test_non_on_policy_fails(self):
        config = self._make_actor_config(train_gbs=32)
        with pytest.raises(AssertionError, match="strict on-policy"):
            self._run_validation(config)

    def test_multi_epoch_fails(self):
        config = self._make_actor_config(ppo_max_epochs_2=2)
        with pytest.raises(AssertionError, match="ppo_max_epochs_2"):
            self._run_validation(config)

    def test_gspo_loss_passes(self):
        config = self._make_actor_config(loss_func="gspo")
        self._run_validation(config)

    def test_unsupported_loss_fails(self):
        config = self._make_actor_config(loss_func="fipo")
        with pytest.raises(AssertionError, match="only supports loss_func"):
            self._run_validation(config)

    def test_incompatible_advantage_type_fails(self):
        config = self._make_actor_config(advantage_type="on_policy_distill")
        with pytest.raises(AssertionError, match="incompatible"):
            self._run_validation(config)

    def test_kl_penalty_fails(self):
        config = self._make_actor_config(ppo_initial_policy_kl_penalty=0.01)
        with pytest.raises(AssertionError, match="ppo_initial_policy_kl_penalty"):
            self._run_validation(config)


# ---------------------------------------------------------------------------
# T5: Placeholder shape correctness
# ---------------------------------------------------------------------------

class TestPlaceholderShape:

    def test_zeros_shape_matches_tokens_minus_one(self):
        """Dummy logprobs placeholder should have shape [len(tokens) - 1]."""
        tokens_list = [
            torch.randint(0, 100, (seq_len,))
            for seq_len in [32, 64, 48]
        ]
        logprobs = [
            torch.zeros(len(t) - 1, dtype=torch.float32)
            for t in tokens_list
        ]
        for t, lp in zip(tokens_list, logprobs):
            assert lp.shape == (len(t) - 1,)
            assert lp.dtype == torch.float32

    def test_mask_from_placeholder_is_correct(self):
        """create_response_mask should produce correct masks from zeros placeholder."""
        prompt_len = 10
        seq_len = 30
        token_len = seq_len

        placeholder = torch.zeros(token_len - 1, dtype=torch.float32)
        mask = create_response_mask(
            values=[placeholder],
            prompt_lengths=[torch.tensor(prompt_len)],
            sequence_lengths=[torch.tensor(seq_len)],
        )

        assert len(mask) == 1
        m = mask[0]
        assert m.shape == (token_len - 1,)
        assert m[:prompt_len - 1].sum() == 0, "prompt region should be 0"
        assert m[prompt_len - 1:seq_len - 1].sum() == seq_len - prompt_len, (
            "response region should be 1"
        )
        assert m[seq_len - 1:].sum() == 0, "padding region should be 0"

    def test_placeholder_different_sample_lengths(self):
        """Multiple samples with different lengths should each get correct masks."""
        samples = [
            {"prompt_len": 5, "seq_len": 20},
            {"prompt_len": 8, "seq_len": 15},
            {"prompt_len": 3, "seq_len": 40},
        ]
        placeholders = [
            torch.zeros(s["seq_len"] - 1, dtype=torch.float32) for s in samples
        ]
        masks = create_response_mask(
            values=placeholders,
            prompt_lengths=[torch.tensor(s["prompt_len"]) for s in samples],
            sequence_lengths=[torch.tensor(s["seq_len"]) for s in samples],
        )

        for i, (m, s) in enumerate(zip(masks, samples)):
            assert m.shape == (s["seq_len"] - 1,), f"sample {i}: shape mismatch"
            response_sum = m[s["prompt_len"] - 1:s["seq_len"] - 1].sum().item()
            expected = s["seq_len"] - s["prompt_len"]
            assert response_sum == expected, (
                f"sample {i}: response region sum={response_sum}, expected={expected}"
            )
