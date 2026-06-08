import os
import shutil
import unittest

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.trainer import OnPolicyDistillTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)


class TestOnPolicyOPD(unittest.IsolatedAsyncioTestCase):
    """Test on-policy distillation with opd_loss_func."""

    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    @requires_sglang
    async def test_opd_loss_train_one_step(self):
        config = load_config('test_on_policy_opd', OnPolicyDistillConfig)

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)

        trainer = OnPolicyDistillTrainer()
        metrics_list = await trainer.launch_then_run_with_recovery(config)

        # Verify metrics contain opd-specific keys and reasonable ranges
        assert metrics_list is not None
        assert len(metrics_list) > 0
        for dp_metrics in metrics_list:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert 'policy/loss' in step_metric, f"Missing policy/loss: {step_metric.keys()}"
                assert 'policy/grad_norm' in step_metric
                assert 'policy/ppo_ratio' in step_metric
                assert 'policy/policy_ref_kl_loss' in step_metric, (
                    f"Missing policy/policy_ref_kl_loss: {step_metric.keys()}"
                )
                assert 'policy/teacher_student_kl_loss' in step_metric, (
                    f"Missing policy/teacher_student_kl_loss: {step_metric.keys()}"
                )

                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']

                assert -0.5 < loss < 0.5, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 10.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.5 < ppo_ratio < 1.5, f"policy/ppo_ratio out of range: {ppo_ratio}"

        if os.path.exists(save_path):
            shutil.rmtree(save_path)
