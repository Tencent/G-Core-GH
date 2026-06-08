import asyncio
import unittest

import numpy as np

import ray
import torch
import os
import shutil

from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config

from gpatch_v4.orches.placement_group import (
    create_train_group,
    create_placement_groups,
)


class CkptIntervalNamingTest(unittest.IsolatedAsyncioTestCase):
    CKPT_PATH = "tests/test_gpatch_v4/ckpt/test_ckpt_interval_naming"

    def test_(self):
        ray.init()

        # Ensure checkpoint directory is empty
        if os.path.exists(self.CKPT_PATH):
            shutil.rmtree(self.CKPT_PATH)
        os.makedirs(self.CKPT_PATH)

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()
        # Clean up checkpoint directory
        if os.path.exists(self.CKPT_PATH):
            shutil.rmtree(self.CKPT_PATH)

    async def test_ckpt_interval_naming(self):
        config = load_config('test_t2i_grpo_ckpt_interval_naming', T2iRlConfig)
        pgs = create_placement_groups(config)
        train_group = create_train_group(config, pgs)
        await train_group.init()
        await train_group.train_loop()

        # Test save latest_checkpoint.txt
        assert os.path.exists(
            os.path.join(config.checkpoint.save_ckpt_path, 'latest_checkpoint.txt')
        )

        # Check step-3 directory (multiple of save_interval)
        assert os.path.exists(os.path.join(config.checkpoint.save_ckpt_path, 'step-3'))

        # Test latest_checkpoint.txt content
        with open(
            os.path.join(config.checkpoint.save_ckpt_path, 'latest_checkpoint.txt'), 'r'
        ) as f:
            assert f.read() == '4'

        # Test resume 暂时没有续训入口，先创建个新的train_group.
        # Kill all Actors
        for actor in train_group._actor_handlers:
            try:
                ray.kill(actor)  # Force kill Actor
            except Exception as e:
                print(f"Error killing Actor: {e}")

        # Wait for resource release
        import asyncio
        await asyncio.sleep(3)  # Wait 3 seconds to ensure resources are fully released

        # config_resume = load_config('test_t2i_grpo_ckpt_interval_naming', T2iRlConfig)
        # pgs_resume = create_placement_groups(config_resume)
        train_group_resume = create_train_group(config, pgs)
        await train_group_resume.init()

        # Test load latest_checkpoint.txt
        for actor in train_group_resume._actor_handlers:
            prev_ppo_step = ray.get(actor.get_prev_ppo_step.remote())
            assert prev_ppo_step == 4
