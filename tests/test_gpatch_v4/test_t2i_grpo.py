import asyncio
import unittest
import shutil

import numpy as np
import pytest
import ray
import torch
import pynvml
from PIL import Image

from gpatch_v4.actor.t2i_grpo_gen_rm_actor import find_images_recursively
from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.orches.placement_group import (
    create_gen_rm_group,
    create_placement_groups,
    create_train_group,
)
from gpatch_v4.trainer import T2iGrpoTrainer
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class T2iGrpoTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def reuse_test_train(self, config):
        pin = 'tasks/t2i_grpo_tv4/prompts.txt'
        pout = 'tasks/t2i_grpo_tv4/prompts_for_test.txt'
        with open(pin, 'r') as fin, open(pout, 'w') as fout:
            for _ in range(16):
                l = fin.readline()
                fout.write(l)

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            trainer = T2iGrpoTrainer()
            await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)

    async def test_train(self):
        # TODO(@nrwu)：外传指标，做 verify。
        config = load_config('test_t2i_grpo', T2iRlConfig)
        await self.reuse_test_train(config)

    async def test_train_allow_tf32_and_autocast(self):
        config = load_config('test_t2i_grpo', T2iRlConfig)
        config.training.allow_tf32 = True
        config.training.use_torch_autocast = True
        await self.reuse_test_train(config)

    async def test_train_rollout_model_mbs(self):
        config = load_config('test_t2i_grpo', T2iRlConfig)
        config.training.rollout_model_mbs = 1
        await self.reuse_test_train(config)
