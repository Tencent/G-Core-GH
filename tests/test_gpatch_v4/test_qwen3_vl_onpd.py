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


class TestQwen3VLOriginalLogprob(unittest.IsolatedAsyncioTestCase):
    """Sampler vs actor logprob scale at temperature=0.5.

    Both ``use_original_logprob`` True (raw / T=1 recompute) and False
    (processed / T=0.5 recompute) should keep
    ``policy/off_policy_correction/kl`` near zero.
    """

    KL_KEY = "policy/off_policy_correction/kl"
    # sglang vs megatron residual at T=0.5 on this VL e2e is ~7e-4 (raw)
    # / ~1.4e-3 (processed); 5e-4 is too tight. Scale match is still
    # ppl_ratio≈1.001 / ppo_ratio=1.0.
    KL_ABS_TOL = 2e-3

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _run_one_step(self, use_original_logprob):
        config = load_config("test_qwen3_vl_on_distill", OnPolicyDistillConfig)
        config.sampler.backend = "sglang"
        ie = config.sampler.infer_engine_configs[0]
        ie.temperature = 0.5
        ie.generate_params.temperature = 0.5
        config.ppo.use_original_logprob = use_original_logprob
        config.data.eval_data_pathes = list(config.data.data_pathes)
        config.training.eval_before_train = False
        config.training.total_eval_step = 0

        save_path = f"unittest_onpd_orig_logprob_{int(use_original_logprob)}"
        config.checkpoint.load_ckpt_path = save_path
        config.checkpoint.save_ckpt_path = save_path
        shutil.rmtree(save_path, ignore_errors=True)
        try:
            trainer = OnPolicyDistillTrainer()
            metrics_list = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(save_path, ignore_errors=True)

        assert metrics_list is not None and len(metrics_list) > 0
        for dp_metrics in metrics_list:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert self.KL_KEY in step_metric, (
                    f"missing {self.KL_KEY}, keys={list(step_metric.keys())}"
                )
                kl = float(step_metric[self.KL_KEY])
                assert np.isfinite(kl), f"{self.KL_KEY} not finite: {kl}"
                assert abs(kl) < self.KL_ABS_TOL, (
                    f"{self.KL_KEY}={kl} exceeds {self.KL_ABS_TOL} "
                    f"(use_original_logprob={use_original_logprob}, temperature=0.5)"
                )

    @requires_sglang
    async def test_use_original_logprob_true(self):
        await self._run_one_step(True)

    @requires_sglang
    async def test_use_original_logprob_false(self):
        await self._run_one_step(False)
