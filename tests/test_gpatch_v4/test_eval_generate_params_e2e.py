"""GPU e2e: train/eval generate_params diverge over ~10 PPO steps + 2 evals.

Config: ``test_eval_generate_params_e2e``
  - train data: dapo-math-17k
  - eval data: aime-2024
  - exit_step=10, eval_before_train=True, eval_interval=10
    → evals at ppo_step 0 and 10

Asserts returned train metrics only (math_r3 style). Sampling temperature
split is covered by L3 unit tests + ``[sampling]`` log probe.
"""

import os
import shutil
import unittest

import numpy as np

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.trainer import GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)

HF_MODEL_PATH = (
    "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Qwen/Qwen2.5-Math-1.5B"
)
TRAIN_DATA_PATH = (
    "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/zhuzilin/dapo-math-17k/"
)
EVAL_DATA_PATH = (
    "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/zhuzilin/aime-2024/"
)


@requires_sglang
class EvalGenerateParamsE2ETest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def test_ten_steps_two_evals_sglang(self):
        self.assertTrue(os.path.isdir(HF_MODEL_PATH), f"missing model: {HF_MODEL_PATH}")
        self.assertTrue(os.path.isdir(TRAIN_DATA_PATH), f"missing train data: {TRAIN_DATA_PATH}")
        self.assertTrue(os.path.isdir(EVAL_DATA_PATH), f"missing eval data: {EVAL_DATA_PATH}")

        config = load_config("test_eval_generate_params_e2e", RlConfig)
        self.assertEqual(config.training.exit_step, 10)
        self.assertTrue(config.training.eval_before_train)
        self.assertEqual(config.training.eval_interval, 10)
        ie = config.sampler.infer_engine_configs[0]
        self.assertEqual(ie.generate_params.temperature, 1.0)
        self.assertIsNotNone(ie.eval_generate_params)
        self.assertEqual(ie.eval_generate_params.temperature, 0.0)

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            trainer = GrpoTrainer()
            metrics = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)

        self.assertIsNotNone(metrics)
        self.assertGreater(len(metrics), 0)
        for dp_metrics in metrics:
            self.assertEqual(
                len(dp_metrics),
                10,
                f"expected 10 train ppo steps, got {len(dp_metrics)}",
            )
            for step_metric in dp_metrics:
                self.assertIn("policy/loss", step_metric)
                self.assertIn("policy/grad_norm", step_metric)
                self.assertIn("policy/ppo_ratio", step_metric)

                loss = step_metric["policy/loss"]
                grad_norm = step_metric["policy/grad_norm"]
                ppo_ratio = step_metric["policy/ppo_ratio"]

                self.assertTrue(
                    np.isfinite(loss), f"policy/loss not finite: {loss}"
                )
                self.assertTrue(
                    -0.5 < float(loss) < 0.5, f"policy/loss out of range: {loss}"
                )
                self.assertTrue(
                    np.isfinite(grad_norm) and 0 <= float(grad_norm) < 50.0,
                    f"policy/grad_norm out of range: {grad_norm}",
                )
                self.assertTrue(
                    np.isfinite(ppo_ratio) and 0.1 < float(ppo_ratio) < 2.0,
                    f"policy/ppo_ratio out of range: {ppo_ratio}",
                )


if __name__ == "__main__":
    unittest.main()
