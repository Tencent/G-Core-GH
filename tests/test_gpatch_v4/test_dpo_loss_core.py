"""compute_dpo_loss_core 单元测试（纯 CPU，无 GPU 依赖）。

验证：
1. policy == ref 时 loss = ln(2)；
2. chosen 优于 rejected 时 loss < ln(2)，反之 loss > ln(2)；
3. label_smoothing=0.5 时对称退化；
4. 梯度正常回传且有限；
5. 非法 loss_type 抛 ValueError；
6. 多 pair batch 与单 pair 一致。
"""

import math

import pytest
import torch
import torch.nn.functional as F

from gpatch_v4.training_backend.loss_factory import compute_dpo_loss_core

LN2 = math.log(2.0)
BETA = 0.1


class TestEqualLogps:
    """policy == ref 时 chosen_rewards == rejected_rewards == 0, loss = ln(2)。"""

    def test_single_pair(self):
        logps = torch.tensor([-5.0])
        losses, cr, rr = compute_dpo_loss_core(logps, logps, logps, logps, beta=BETA)
        assert torch.allclose(losses, torch.tensor([LN2]), atol=1e-6)
        assert torch.allclose(cr, torch.zeros(1), atol=1e-8)
        assert torch.allclose(rr, torch.zeros(1), atol=1e-8)

    def test_multi_pair(self):
        logps = torch.randn(8)
        losses, cr, rr = compute_dpo_loss_core(logps, logps, logps, logps, beta=BETA)
        assert torch.allclose(losses, torch.full((8,), LN2), atol=1e-6)


class TestLossDirection:
    """chosen 优于 rejected 时 loss 更低。"""

    def test_chosen_better(self):
        # policy chosen logps 高于 rejected，ref 相同
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-3.0])
        ref = torch.tensor([-2.0])
        losses, cr, rr = compute_dpo_loss_core(pc, pr, ref, ref, beta=BETA)
        assert losses.item() < LN2

    def test_rejected_better(self):
        pc = torch.tensor([-3.0])
        pr = torch.tensor([-1.0])
        ref = torch.tensor([-2.0])
        losses, cr, rr = compute_dpo_loss_core(pc, pr, ref, ref, beta=BETA)
        assert losses.item() > LN2

    def test_reward_signs(self):
        # chosen 的 policy > ref → chosen_reward > 0
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-3.0])
        rc = torch.tensor([-2.0])
        rr = torch.tensor([-2.0])
        _, chosen_rewards, rejected_rewards = compute_dpo_loss_core(
            pc, pr, rc, rr, beta=BETA
        )
        assert chosen_rewards.item() > 0
        assert rejected_rewards.item() < 0


class TestLabelSmoothing:

    def test_no_smoothing_matches_logsigmoid(self):
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-3.0])
        rc = torch.tensor([-2.0])
        rr = torch.tensor([-2.0])
        losses, _, _ = compute_dpo_loss_core(pc, pr, rc, rr, beta=BETA, label_smoothing=0.0)
        logits = BETA * ((pc - rc) - (pr - rr))
        expected = -F.logsigmoid(logits)
        assert torch.allclose(losses, expected, atol=1e-7)

    def test_full_smoothing_symmetric(self):
        # label_smoothing=0.5: loss = 0.5*(-logsigmoid(x)) + 0.5*(-logsigmoid(-x))
        # = 0.5 * (log(1+e^-x) + log(1+e^x)) = 0.5 * (x + 2*log(1+e^-x)) 当 x>0
        # 关键性质：翻转 chosen/rejected 结果相同
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-3.0])
        ref = torch.tensor([-2.0])
        loss_fwd, _, _ = compute_dpo_loss_core(pc, pr, ref, ref, beta=BETA, label_smoothing=0.5)
        loss_rev, _, _ = compute_dpo_loss_core(pr, pc, ref, ref, beta=BETA, label_smoothing=0.5)
        assert torch.allclose(loss_fwd, loss_rev, atol=1e-7)

    def test_zero_logits_any_smoothing(self):
        # logits=0 时，-logsigmoid(0) == -logsigmoid(-0) == ln(2)
        # 所以任意 label_smoothing 下 loss = ln(2)
        logps = torch.tensor([-5.0])
        for ls in [0.0, 0.1, 0.3, 0.5, 1.0]:
            losses, _, _ = compute_dpo_loss_core(logps, logps, logps, logps, beta=BETA, label_smoothing=ls)
            assert torch.allclose(losses, torch.tensor([LN2]), atol=1e-6), f"ls={ls}"


class TestGradient:

    def test_backward_runs(self):
        pc = torch.randn(4, requires_grad=True)
        pr = torch.randn(4, requires_grad=True)
        rc = torch.randn(4)
        rr = torch.randn(4)
        losses, _, _ = compute_dpo_loss_core(pc, pr, rc, rr, beta=BETA)
        losses.sum().backward()
        assert pc.grad is not None
        assert pr.grad is not None
        assert torch.isfinite(pc.grad).all()
        assert torch.isfinite(pr.grad).all()

    def test_rewards_are_detached(self):
        pc = torch.randn(2, requires_grad=True)
        pr = torch.randn(2, requires_grad=True)
        rc = torch.randn(2)
        rr = torch.randn(2)
        _, cr, rr_out = compute_dpo_loss_core(pc, pr, rc, rr, beta=BETA)
        assert not cr.requires_grad
        assert not rr_out.requires_grad


class TestInvalidLossType:

    def test_raises_on_unknown(self):
        logps = torch.tensor([0.0])
        with pytest.raises(ValueError, match="unknown DPO loss type"):
            compute_dpo_loss_core(logps, logps, logps, logps, beta=BETA, loss_type="hinge")


class TestBetaScaling:

    def test_larger_beta_larger_margin(self):
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-3.0])
        ref = torch.tensor([-2.0])
        _, cr_small, _ = compute_dpo_loss_core(pc, pr, ref, ref, beta=0.1)
        _, cr_large, _ = compute_dpo_loss_core(pc, pr, ref, ref, beta=1.0)
        assert abs(cr_large.item()) > abs(cr_small.item())
