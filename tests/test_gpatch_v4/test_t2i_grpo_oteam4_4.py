import asyncio
import shutil
import unittest

import numpy as np
import pynvml
import pytest
import ray
import torch
from PIL import Image

from gpatch_v4.actor.t2i_grpo_gen_rm_actor import find_images_recursively
from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.orches.placement_group import (
    create_gen_rm_group,
    create_placement_groups,
    create_train_group,
)
from gpatch_v4.trainer import T2iGrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
    requires_vllm,
)


class T2iGrpoOteam4_4Test(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def reuse_test_train(self, config):
        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            trainer = T2iGrpoTrainer()
            ret = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
        return ret

    @requires_sglang
    async def test_1_sglang(self):
        config = load_config('test_t2i_grpo_oteam4_4', T2iRlConfig)
        await self.reuse_test_train(config)

    @requires_sglang
    async def test_disable_cfg_uncond_grad_sglang(self):
        config = load_config('test_t2i_grpo_oteam4_4', T2iRlConfig)
        config.training.disable_cfg_uncond_grad = True
        await self.reuse_test_train(config)

    @requires_sglang
    async def test_t2i_cfg_logps_impl_v2_sglang(self):
        config = load_config('test_t2i_grpo_oteam4_4', T2iRlConfig)
        config.training.disable_cfg_uncond_grad = True
        config.training.t2i_cfg_logps_impl_v2 = True
        metrics = await self.reuse_test_train(config)

        assert len(metrics) == 16  # 16 gpus
        assert len(metrics[0]) == 1  # 1 ppo step
        for dp_rank in range(len(metrics)):
            assert abs(metrics[dp_rank][0]['policy/loss']) < 0.001
            assert abs(metrics[dp_rank][0]['policy/ppo_ratio']) > 0.995

    @requires_vllm
    async def test_1_vllm(self):
        config = load_config('test_t2i_grpo_oteam4_4', T2iRlConfig)
        config.gen_rm.backend = "vllm"
        await self.reuse_test_train(config)

    @requires_vllm
    async def test_disable_cfg_uncond_grad_vllm(self):
        config = load_config('test_t2i_grpo_oteam4_4', T2iRlConfig)
        config.gen_rm.backend = "vllm"
        config.training.disable_cfg_uncond_grad = True
        await self.reuse_test_train(config)

    @requires_vllm
    async def test_t2i_cfg_logps_impl_v2_vllm(self):
        config = load_config('test_t2i_grpo_oteam4_4', T2iRlConfig)
        config.gen_rm.backend = "vllm"
        config.training.disable_cfg_uncond_grad = True
        config.training.t2i_cfg_logps_impl_v2 = True
        metrics = await self.reuse_test_train(config)
        assert len(metrics) == 16  # 16 gpus
        assert len(metrics[0]) == 1  # 1 ppo step
        for dp_rank in range(len(metrics)):
            assert abs(metrics[dp_rank][0]['policy/loss']) < 0.001
            assert abs(metrics[dp_rank][0]['policy/ppo_ratio']) > 0.995
