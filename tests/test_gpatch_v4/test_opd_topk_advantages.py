import os
import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.core.advantage_impl import calculate_topk_advantages
from gpatch_v4.trainer import OnPolicyDistillTrainer
from gpatch_v4.training_backend.loss_factory import PolicyLossInput, opd_loss_func
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config, requires_sglang


def _cfg(**ppo_kw):
    d = dict(
        ppo_logps_ratio_clamp=None, ppo_ratio_eps=0.2,
        ppo_clip_ratio_low=None, ppo_clip_ratio_high=None,
        ppo_dual_clip_ratio_c=None, ppo_clamp_kl_val=None,
        ppo_entropy_bonus=0.0, grpo_kl_loss_beta=0.0,
        opd_teacher_kl_loss_beta=0.0, enable_off_policy_correction=False, log_prob_top_k=16,
        ppo_entropy_regularization_type=None,
    )
    d.update(ppo_kw)
    return SimpleNamespace(
        ppo=SimpleNamespace(**d),
        policy=SimpleNamespace(override_transformer_config={}),
    )


@patch("gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group")
def test_topk_pure_kl_adv_3d_ratio_3d_loss_scalar(mock_reduce):
    """no reward, log_prob_top_k=16: adv is 3D → ratio is 3D → loss is scalar, numerically correct."""
    B, S, K = 2, 6, 16
    stu = [torch.randn(S, K) for _ in range(B)]
    tea = [torch.randn(S, K) for _ in range(B)]
    masks = [torch.cat([torch.zeros(2), torch.ones(S - 2)]) for _ in range(B)]

    # step 1: advantage is 3D [S, K], pure reverse-KL, no reward
    advs, metrics = calculate_topk_advantages(
        mask_lst=masks, stu_topk_logprobs=stu, teacher_topk_logprobs=tea,
    )
    assert all(a.shape == (S, K) for a in advs)
    assert all((a[:2] == 0).all() for a in advs)
    for i in range(B):
        expected = -(stu[i].float() - tea[i].float()) * torch.softmax(stu[i].float(), dim=-1)
        expected[:2] = 0
        assert torch.allclose(advs[i], expected, atol=1e-6)

    # step 2: feed 3D advantage into opd_loss → ratio is 3D, final loss is 0-d scalar
    prev_topk = torch.randn(B, S, K) - 2.0
    curr_topk = prev_topk + torch.randn(B, S, K) * 0.01  # small perturbation → ratio ≈ 1
    li = PolicyLossInput(
        advantages=torch.stack(advs),
        prev_log_probs=torch.randn(B, S), ref_log_probs=torch.randn(B, S),
        curr_log_probs=torch.randn(B, S), response_mask=torch.stack(masks),
        scaled_entropy=torch.tensor(0.1), teacher_log_probs=torch.randn(B, S),
        prev_topk_logprobs=prev_topk, curr_topk_logprobs=curr_topk.requires_grad_(True),
    )
    loss, m = opd_loss_func(_cfg(), li)

    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert li.curr_topk_logprobs.grad is not None and torch.isfinite(li.curr_topk_logprobs.grad).all()

    ratio = m["ppo_ratio"][0] / m["ppo_ratio"][1]
    assert 0.9 < ratio < 1.1, f"ppo_ratio should be ~1.0 with small perturbation, got {ratio:.4f}"


# -- E2E test: full pipeline with log_prob_top_k=16, no reward --

class TestOnPolicyOPDTopK(unittest.IsolatedAsyncioTestCase):

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    @requires_sglang
    async def test_opd_topk_train_one_step(self):
        config = load_config('test_on_policy_opd_topk', OnPolicyDistillConfig)

        save_path = config.checkpoint.save_ckpt_path
        if os.path.exists(os.path.join(save_path, "latest_checkpointed_iteration.txt")):
            os.remove(os.path.join(save_path, "latest_checkpointed_iteration.txt"))

        trainer = OnPolicyDistillTrainer()
        metrics_list = await trainer.launch_then_run_with_recovery(config)

        assert metrics_list is not None and len(metrics_list) > 0
        for dp_metrics in metrics_list:
            for step_metric in dp_metrics:
                assert 'policy/loss' in step_metric
                assert 'policy/ppo_ratio' in step_metric
                assert 'policy/grad_norm' in step_metric

                assert -0.5 < step_metric['policy/loss'] < 0.5
                assert 0 <= step_metric['policy/grad_norm'] < 10.0
                assert 0.5 < step_metric['policy/ppo_ratio'] < 1.5

        if os.path.exists(save_path):
            shutil.rmtree(save_path)
