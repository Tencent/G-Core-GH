# coding=utf-8
"""DSpark 一步训练 E2E，覆盖正式 FinetuneTrainer 路径。"""

import math
import os
import shutil
import unittest

import pytest

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.trainer import FinetuneTrainer
from test_gpatch_v4.gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
)


HF_MODEL_PATH = os.environ.get(
    "DSPARK_HF_MODEL_PATH",
    "hf-hub/deepseek-ai/DeepSeek-V4-Flash-0731",
)
DATA_PATH = os.environ.get(
    "DSPARK_E2E_DATA_PATH",
    "hf-hub/AI-MO/NuminaMath-CoT-jsonl/train",
)


class TestDeepseekV4DSparkSmoke(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    @pytest.mark.timeout(2400)
    async def test_dspark_one_training_step(self):
        hf_model_path = os.path.abspath(HF_MODEL_PATH)
        data_path = os.path.abspath(DATA_PATH)
        if not os.path.isdir(hf_model_path):
            raise unittest.SkipTest(
                f"DSpark checkpoint not found: {hf_model_path}; "
                "set DSPARK_HF_MODEL_PATH"
            )
        if not os.path.isdir(data_path):
            raise unittest.SkipTest(
                f"DSpark E2E data not found: {data_path}; "
                "set DSPARK_E2E_DATA_PATH"
            )

        config = load_config("test_dsv4_sft_dspark", FinetuneConfig)
        assert config.policy.ppo_pack_seq, "DSpark smoke covers the THD pack path"
        assert config.policy.dist_config.context_parallel_size == 1
        config.policy.hf_model_path = hf_model_path
        config.policy.hf_tokenizer_path = hf_model_path
        config.data.data_pathes = [data_path]
        # Numina samples exceed yaml seq_length=128; pack asserts without a larger budget.
        config.training.seq_length = 4096

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
        try:
            metrics = await FinetuneTrainer().launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
            shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)

        assert metrics is not None
        step_metrics = [
            step
            for actor_metrics in metrics
            for step in actor_metrics
        ]
        assert step_metrics
        expected_keys = (
            "finetune/lm_loss",
            "finetune/total_loss",
            "finetune/dspark_loss",
            "finetune/dspark_ce_loss",
            "finetune/dspark_l1_loss",
            "finetune/dspark_confidence_loss",
            "finetune/dspark_tau",
            "finetune/grad_norm",
        ) + tuple(
            f"finetune/dspark_accept_rate_{position}"
            for position in range(5)
        )
        for step in step_metrics:
            for key in expected_keys:
                assert key in step
                assert math.isfinite(step[key]), f"{key}={step[key]}"
            expected_total_loss = (
                step["finetune/lm_loss"]
                + config.training.dspark_loss_scaling_factor
                * step["finetune/dspark_loss"]
            )
            assert math.isclose(
                step["finetune/total_loss"],
                expected_total_loss,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
            assert step["finetune/dspark_loss"] > 0
            assert step["finetune/grad_norm"] > 0
            assert step["finetune/dspark_tau"] >= 1.0


if __name__ == "__main__":
    unittest.main()
