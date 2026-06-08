import asyncio
import unittest

import numpy as np
import pytest
import ray
import torch
from PIL import Image

from gpatch_v4.actor.t2i_grpo_gen_rm_actor import find_images_recursively
from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.orches.placement_group import create_bt_rm_group, create_placement_groups
from gpatch_v4.utils import logging_memory_usage, logging_memory_usage_details
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class BtRmTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init()

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def test_bt_rm_setup(self):
        config = load_config('test_t2i_grpo_bt_rm', T2iRlConfig)
        pgs = create_placement_groups(config)
        bt_rm_group = create_bt_rm_group(config, pgs)
        assert config.training.use_bt_rm_reward
        await bt_rm_group.init()
        return bt_rm_group, config

    async def test_bt_rm_generate_reward(self):
        bt_rm_group, config = await self.test_bt_rm_setup()

        rm_idx = 0
        ep_idx = 0

        await bt_rm_group.wake_up(rm_idx)
        bt_rm_actor = bt_rm_group._actor_handlers[rm_idx][ep_idx]

        req_dict = {
            'batched_data':
                {
                    'prompt': [
                        'A picture of random noise',
                        'A picture of random noise',
                    ],
                    'images':
                        [
                            Image.fromarray((np.random.rand(512, 512, 3) * 255).astype('uint8')
                                           ).convert('RGB'),
                            Image.fromarray((np.random.rand(512, 512, 3) * 255).astype('uint8')
                                           ).convert('RGB'),
                        ],
                }
        }
        ret = await bt_rm_actor.generate_rewards.remote(req_dict)
        rewards = ret[f'reward_bt_rm_{rm_idx}']
        print(f"rewards {rewards}")
        assert len(rewards) == config.training.sampling_repeat_n
        for t in rewards:
            assert isinstance(t, torch.Tensor)

        # TODO 应该 check 下 mem usage，而不是肉眼观察
        logging_memory_usage_details("before sleep")
        await bt_rm_actor.sleep.remote()
        logging_memory_usage_details("after sleep")

        logging_memory_usage_details("before wake_up")
        await bt_rm_actor.wake_up.remote()
        logging_memory_usage_details("after wake_up")
