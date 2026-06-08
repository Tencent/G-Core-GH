import asyncio
import unittest

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
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class GenRmTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init()

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def test_rm_re_setup(self):
        config = load_config('test_t2i_grpo_gen_rm', T2iRlConfig)
        pgs = create_placement_groups(config)
        gen_rm_groups = create_gen_rm_group(config, pgs)
        assert config.training.use_gen_rm_reward
        for grp in gen_rm_groups:
            await grp.init()

    def test_find_images(self):
        img1 = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype('uint8')).convert('RGB')
        img2 = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype('uint8')).convert('RGB')
        msg_i = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "You are a helpful assistant."
                }, ],
            },
            {
                "role":
                    "user",
                "content":
                    [
                        {
                            "type": "image",
                            "image": img2
                        },
                        {
                            "type": "text",
                            "text": 'which one is better?'
                        },
                    ],
            },
            {
                "role":
                    "user",
                "content":
                    [
                        {
                            "type": "image",
                            "image": img1
                        },
                        {
                            "type": "image",
                            "image": img2
                        },
                        {
                            "type": "text",
                            "text": 'which one is better?'
                        },
                    ],
            },
        ]
        d = find_images_recursively(msg_i)
        assert len(d) == 3
        assert d[0] is img2
        assert d[1] is img1
        assert d[2] is img2

    async def test_rm_setup(self):
        config = load_config('test_t2i_grpo_gen_rm', T2iRlConfig)
        pgs = create_placement_groups(config)
        gen_rm_groups = create_gen_rm_group(config, pgs)
        assert config.training.use_gen_rm_reward
        for grp in gen_rm_groups:
            await grp.init()

        rm_idx = 0
        ep_idx = 0
        gen_rm_group = gen_rm_groups[rm_idx]
        await gen_rm_group.wake_up()

        gen_rm_actor = gen_rm_group._actor_handlers[ep_idx]
        img_wh = (512, 512)

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
                },
        }
        ret = await gen_rm_actor.generate_rewards.remote(req_dict)
        rewards = ret[f'reward_gen_rm_{rm_idx}']
        assert len(rewards) == config.training.sampling_repeat_n
        for t in rewards:
            assert isinstance(t, torch.Tensor)

    async def test_multi_tp4(self):
        pynvml.nvmlInit()
        nvml_hdls = [pynvml.nvmlDeviceGetHandleByIndex(gpu_id) for gpu_id in range(8)]
        gb = 2**30

        config = load_config('test_t2i_grpo_gen_rm_multi', T2iRlConfig)
        pgs = create_placement_groups(config)
        gen_rm_groups = create_gen_rm_group(config, pgs)
        assert config.training.use_gen_rm_reward
        for grp in gen_rm_groups:
            await grp.init()

        for gpu_id, h in enumerate(nvml_hdls):
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(h)
            assert meminfo.used <= 12 * gb

        pynvml.nvmlShutdown()
