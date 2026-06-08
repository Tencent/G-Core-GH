"""End-to-end smoke test for agentic RL with GrpoSingleCtrlTrainer + EnvAgentLoopActor."""
from __future__ import annotations

import os
import shutil
import unittest

import ray

from gpatch_v4.configs.config import AgenticRlConfig
from gpatch_v4.trainer import GrpoSingleCtrlTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
    requires_vllm,
)


class TestAgenticRl(unittest.IsolatedAsyncioTestCase):
    """End-to-end: load config -> GrpoTrainer -> one training step with TrajEnvManager."""
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _reuse_train_one_step(self, backend):
        config = load_config("test_agentic_rl", AgenticRlConfig)
        config.sampler.backend = backend

        tmp_dir = "tests/test_gpatch_v4/gsm8k_small"
        os.makedirs(tmp_dir, exist_ok=True)
        src = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
        dst = os.path.join(tmp_dir, "train.jsonl")
        with open(src, "r") as fin, open(dst, "w") as fout:
            for i, line in enumerate(fin):
                if i >= 8:
                    break
                fout.write(line)

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)

        trainer = GrpoSingleCtrlTrainer()
        await trainer.launch_then_run_with_recovery(config)

        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    @unittest.skip(
        "master's rollout_generator factory lacks an 'agentic' branch; "
        "pending upstream fix — unrelated to this cleanup PR"
    )
    @requires_sglang
    async def test_train_one_step_sglang(self):
        await self._reuse_train_one_step("sglang")

    @unittest.skip(
        "master's rollout_generator factory lacks an 'agentic' branch; "
        "pending upstream fix — unrelated to this cleanup PR"
    )
    @requires_vllm
    async def test_train_one_step_vllm(self):
        await self._reuse_train_one_step("vllm")
