import os
import shutil
import unittest

import numpy as np
import pytest
import ray
import torch
from packaging.version import Version

from megatron.core import package_info

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.orches.placement_group import create_placement_groups, create_train_group
from gpatch_v4.trainer import OnPolicyDistillTrainer
from gpatch_v4.utils import logging_memory_usage
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
    requires_vllm,
)


class TestQwen3VLOnPolicyDistill(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _reuse_train_one_step_and_save(self, backend):
        mcore_version = Version(package_info.__version__)
        if mcore_version > Version("0.13.1"):
            return

        config = load_config('test_qwen3_vl_on_distill', OnPolicyDistillConfig)
        config.sampler.backend = backend

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)
        if os.path.exists(os.path.join(save_path, "hf")):
            shutil.rmtree(os.path.join(save_path, "hf"))

        trainer = OnPolicyDistillTrainer()
        metrics_list = await trainer.launch_then_run_with_recovery(config)
        for metrics in metrics_list:
            for metric in metrics:
                assert metric['policy/grpo_kl_loss'] == 0, f"{metric=}"
                assert metric['rollout-rewards/global_acc_reward'] >= 0.25, f"{metric=}"
                assert metric['rollout-rewards/global_fmt_reward'] >= 0.5, f"{metric=}"

    @requires_sglang
    async def test_train_one_step_and_save_sglang(self):
        await self._reuse_train_one_step_and_save("sglang")

    @requires_vllm
    async def test_train_one_step_and_save_vllm(self):
        await self._reuse_train_one_step_and_save("vllm")
