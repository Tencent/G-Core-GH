import asyncio
import unittest

import numpy as np
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
from gpatch_v4.utils import repeat_interleave_tensor_or_list
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class DataFlowTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init()

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def test_repeat_interleave(self):
        prompt = ['a man', 'a cat']
        embed = torch.rand(2, 10)
        prompt2 = repeat_interleave_tensor_or_list(prompt, 2)
        embed2 = repeat_interleave_tensor_or_list(embed, 2)
        assert len(prompt2) == 4
        assert prompt2[0] == 'a man'
        assert prompt2[1] == 'a man'
        assert prompt2[2] == 'a cat'
        assert prompt2[3] == 'a cat'
        assert embed2.shape == (4, 10)
        assert (embed2[0] == embed[0]).all().item()
        assert (embed2[1] == embed[0]).all().item()
        assert (embed2[2] == embed[1]).all().item()
        assert (embed2[3] == embed[1]).all().item()

    async def test_policy_setup(self):
        config = load_config('test_t2i_grpo_data_flow', T2iRlConfig)
        rmbs = config.training.rollout_mbs
        rgbs = config.training.rollout_gbs
        repeat_n = config.training.sampling_repeat_n
        assert rmbs == 2
        assert rgbs == 32
        assert repeat_n == 4
        rgas = 2  # 32 / (2 * 8)

        pgs = create_placement_groups(config)
        train_group = create_train_group(config, pgs)
        gen_rm_groups = create_gen_rm_group(config, pgs)

        await train_group.init()
        if config.training.use_gen_rm_reward:
            for grp in gen_rm_groups:
                await grp.init()
        await train_group.setup_client()

        world_size = len(train_group._actor_handlers)
        assert world_size == 8
        futs = [actor.maybe_set_epoch.remote(0) for actor in train_group._actor_handlers]
        [await fut for fut in futs]

        assert rgas == await train_group._actor_handlers[0].get_num_rollout_micro_batches.remote()

        epoch_i = 0
        ppo_step_i = 0
        futs = [
            actor.rollout_remotable.remote(
                epoch_i, ppo_step_i, rgas, debug_disable_advantage=False
            ) for actor in train_group._actor_handlers
        ]
        rets = [(await fut) for fut in futs]
        dp_rbs = [ret[0] for ret in rets]

        # 标准输出
        # num_rollout_micro_batches 个 dict
        #   每个 dict 的 value，都是一个 list，长度是 mbs * repeat
        for rank, rbs in enumerate(dp_rbs):
            assert isinstance(rbs, list), f'{type(rbs)}'
            assert len(rbs) == rgas, f'{len(rbs)=}'
            for rb in rbs:
                assert 'images' in rb
                assert 'log_probs' in rb
                assert len(rb['images']) == rmbs * repeat_n
                for v in rb.values():
                    assert isinstance(v, list)
                    assert len(v) == rmbs * repeat_n
