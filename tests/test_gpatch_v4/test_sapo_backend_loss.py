"""Tests for SAPO on the ``ppo.use_legacy_loss=False`` loss path.

New loss is ``[B, S]`` only: THD / dyn-CP packs must be response-padded before
entering ``sapo_loss``. Packed-THD semantics remain covered by the legacy
``test_sapo_loss.py`` suite.
"""
import types
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.training_backend.loss import PolicyLossInput, get_loss_fn
from gpatch_v4.training_backend.loss.ppo_loss import gspo_loss, sapo_loss


_REDUCE_METRICS_PATH = (
    "gpatch_v4.training_backend.loss.ppo_loss.reduce_metrics_across_data_parallel_group"
)


def _make_config():
    return types.SimpleNamespace(
        ppo=PpoConfig(
            loss_func="sapo",
            grpo_kl_loss_beta=0.0,
            sapo_tau_pos=1.0,
            sapo_tau_neg=1.25,
        ),
        debug=DebugConfig(),
    )


def _make_input(
    curr: torch.Tensor,
    prev: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    cu_seqlens_padded: torch.Tensor | None = None,
    calculate_per_token_loss: bool = False,
    local_cp_size: int = 1,
) -> PolicyLossInput:
    return PolicyLossInput(
        advantages=advantages,
        prev_log_probs=prev,
        ref_log_probs=None,
        curr_log_probs=curr,
        response_mask=mask,
        scaled_entropy=torch.tensor(0.0),
        per_token_entropy=torch.zeros_like(curr),
        cu_seqlens_padded=cu_seqlens_padded,
        calculate_per_token_loss=calculate_per_token_loss,
        local_cp_size=local_cp_size,
    )


class TestSapoBackendLoss:
    def test_registry_selects_new_sapo_loss(self):
        assert get_loss_fn("mcore", "sapo") is sapo_loss

    @patch(_REDUCE_METRICS_PATH)
    def test_bshd_loss_and_gradients(self, mock_reduce):
        config = _make_config()
        curr = torch.tensor(
            [[-0.2, -0.4, 0.0], [-0.1, -0.3, -0.5]],
            requires_grad=True,
        )
        prev = torch.tensor([[-0.3, -0.35, 0.0], [-0.2, -0.25, -0.6]])
        advantages = torch.tensor([[1.0, -0.5, 0.0], [0.0, 0.25, 0.75]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

        loss, count, _ = sapo_loss(
            config,
            _make_input(curr, prev, advantages, mask),
        )
        assert loss.ndim == 0
        assert count.ndim == 0
        loss.backward()
        assert curr.grad is not None
        assert torch.count_nonzero(curr.grad[0, 2]) == 0
        assert torch.count_nonzero(curr.grad[1, 0]) == 0

    @patch(_REDUCE_METRICS_PATH)
    def test_thd_is_rejected_for_new_loss(self, mock_reduce):
        curr = torch.zeros(1, 4, requires_grad=True)
        loss_input = _make_input(
            curr,
            torch.zeros_like(curr),
            torch.ones_like(curr),
            torch.ones_like(curr),
            cu_seqlens_padded=torch.tensor([0, 4]),
        )
        with pytest.raises(AssertionError, match="response-padded"):
            sapo_loss(_make_config(), loss_input)
        with pytest.raises(AssertionError, match="response-padded"):
            gspo_loss(_make_config(), loss_input)

    @patch(_REDUCE_METRICS_PATH)
    def test_per_token_mode_returns_token_sum(self, mock_reduce):
        curr = torch.zeros(1, 3, requires_grad=True)
        bwd_loss, bwd_count, _ = sapo_loss(
            _make_config(),
            _make_input(
                curr,
                torch.zeros_like(curr),
                torch.ones_like(curr),
                torch.ones_like(curr),
                calculate_per_token_loss=True,
            ),
        )
        # At ratio=1 and tau_pos=1, gate=2, so token sum is -2 * 3 = -6.
        assert torch.allclose(bwd_loss, torch.tensor(-6.0))
        assert torch.allclose(bwd_count, torch.tensor(3.0))
