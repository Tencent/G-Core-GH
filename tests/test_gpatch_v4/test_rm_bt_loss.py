"""Tests for rm_bt loss (Bradley-Terry sequence-level reward model)."""
import types
from unittest.mock import patch

import torch

from gpatch_v4.training_backend.loss_factory import (
    FinetuneLossInput,
    rm_bt_loss_func,
)

_AVG_METRICS_PATH = (
    "gpatch_v4.training_backend.loss_factory.average_losses_across_data_parallel_group"
)


def _make_loss_input(scores: torch.Tensor, sequence_lengths: torch.Tensor):
    # scores: [B, S, 1] 标量头输出; sequence_lengths: [B] 有效 token 数
    return FinetuneLossInput(
        logits=scores,
        batch={"sequence_lengths": sequence_lengths},
    )


class TestRmBtPooling:
    @patch(_AVG_METRICS_PATH, side_effect=lambda xs: xs)
    def test_reads_last_valid_token(self, _mock_avg):
        # B=2 (1 对): chosen 末位=3, rejected 末位=2; 其余位置放噪声确认不被取到
        scores = torch.zeros(2, 5, 1)
        scores[0, 3, 0] = 2.0  # chosen reward
        scores[0, 4, 0] = 99.0  # padding 位, 不应被取到
        scores[1, 2, 0] = 0.5  # rejected reward
        scores[1, 3, 0] = -99.0
        sequence_lengths = torch.tensor([4, 3])

        loss, metrics = rm_bt_loss_func(None, _make_loss_input(scores, sequence_lengths))

        expected = -torch.nn.functional.logsigmoid(torch.tensor(2.0 - 0.5))
        assert torch.allclose(loss, expected, atol=1e-6)
        assert torch.allclose(metrics["rm-metrics/reward_chosen"], torch.tensor(2.0))
        assert torch.allclose(metrics["rm-metrics/reward_rejected"], torch.tensor(0.5))
        assert torch.allclose(metrics["rm-metrics/reward_margin"], torch.tensor(1.5))
        assert metrics["rm-metrics/acc"].item() == 1.0


class TestRmBtLossMath:
    @patch(_AVG_METRICS_PATH, side_effect=lambda xs: xs)
    def test_equal_reward_gives_log2(self, _mock_avg):
        # chosen==rejected -> -logsigmoid(0) = log(2)
        scores = torch.zeros(2, 3, 1)
        sequence_lengths = torch.tensor([3, 3])
        loss, metrics = rm_bt_loss_func(None, _make_loss_input(scores, sequence_lengths))
        assert torch.allclose(loss, torch.log(torch.tensor(2.0)), atol=1e-6)
        assert metrics["rm-metrics/acc"].item() == 0.0

    @patch(_AVG_METRICS_PATH, side_effect=lambda xs: xs)
    def test_multi_pair_acc(self, _mock_avg):
        # B=4 (2 对): pair0 chosen>rejected, pair1 chosen<rejected -> acc=0.5
        scores = torch.zeros(4, 2, 1)
        # chosen 在前半 [0,1], rejected 在后半 [2,3]
        scores[0, 1, 0] = 1.0
        scores[1, 1, 0] = -1.0
        scores[2, 1, 0] = 0.0  # pair0 rejected
        scores[3, 1, 0] = 0.0  # pair1 rejected
        sequence_lengths = torch.tensor([2, 2, 2, 2])
        loss, metrics = rm_bt_loss_func(None, _make_loss_input(scores, sequence_lengths))
        assert metrics["rm-metrics/acc"].item() == 0.5
        assert not torch.isnan(loss)

    @patch(_AVG_METRICS_PATH, side_effect=lambda xs: xs)
    def test_gradient_flows(self, _mock_avg):
        scores = torch.randn(2, 4, 1, requires_grad=True)
        sequence_lengths = torch.tensor([4, 3])
        loss, _ = rm_bt_loss_func(None, _make_loss_input(scores, sequence_lengths))
        loss.backward()
        assert scores.grad is not None
        assert scores.grad.abs().sum() > 0
