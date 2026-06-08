import os
import shutil
import unittest

import ray

from gpatch_v4.configs.config import OffPolicyDistillConfig
from gpatch_v4.trainer import OffPolicyDistillTrainer
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class TestOffPolicyDistill(unittest.IsolatedAsyncioTestCase):
    """Test OffPolicyDistill"""
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def test_train_one_step_and_save(self):
        config = load_config("test_offpd_distill", OffPolicyDistillConfig)

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)
        hf_ckpt_path = os.path.join(save_path, "hf")
        if os.path.exists(hf_ckpt_path):
            shutil.rmtree(hf_ckpt_path)

        trainer = OffPolicyDistillTrainer()
        metrics = await trainer.launch_then_run_with_recovery(config)
        # print(f"\n====== OFFPD METRICS ======\n {metrics=}")
        # NOTE: loss metrics is highly relevant to the gap between teacher and student.
        # if models change, the loss metrics should also be modified.
        assert metrics is not None
        assert len(metrics) >= 1
        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert 'finetune/loss' in step_metric
                assert 'finetune/lm_loss' in step_metric
                assert 'finetune/kl_loss' in step_metric
                assert 'finetune/grad_norm' in step_metric

                loss = step_metric['finetune/loss']
                lm_loss = step_metric['finetune/lm_loss']
                kl_loss = step_metric['finetune/kl_loss']
                grad_norm = step_metric['finetune/grad_norm']

                assert 0 < loss < 1, f"finetune/loss out of range: {loss}"
                assert 0 <= lm_loss < 1, f"finetune/lm_loss out of range: {lm_loss}"
                assert 0 <= kl_loss < 1, f"finetune/kl_loss out of range: {kl_loss}"
                assert 0 <= grad_norm < 20, f"finetune/grad_norm out of range: {grad_norm}"

        assert os.path.exists(latest_ckpt_path)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
