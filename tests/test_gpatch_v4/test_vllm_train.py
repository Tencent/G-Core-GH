import os
import shutil
import unittest

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.trainer import GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_vllm,
)


class Qwen36MoeGrpoVllmTest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def reuse_test_train(self, config):
        tmp_dir = "tests/test_gpatch_v4/gsm8k_small"
        os.makedirs(tmp_dir, exist_ok=True)
        src = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
        dst = os.path.join(tmp_dir, "train.jsonl")
        with open(src, 'r') as fin, open(dst, 'w') as fout:
            for i, line in enumerate(fin):
                if i >= 256:
                    break
                fout.write(line)

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            trainer = GrpoTrainer()
            metrics = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
        return metrics

    @requires_vllm
    async def test_train_qwen3_6_moe_35b_a3b_vllm(self):
        config = load_config('test_qwen3_6_moe_grpo_vllm', RlConfig)
        config.sampler.backend = "vllm"

        metrics = await self.reuse_test_train(config)

        assert metrics is not None
        assert len(metrics) > 0

        has_policy_step = False
        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert 'policy/loss' in step_metric
                assert 'policy/grad_norm' in step_metric
                assert 'policy/ppo_ratio' in step_metric
                has_policy_step = True

        assert has_policy_step
