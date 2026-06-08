import asyncio
import unittest

import numpy as np
import pytest
import ray
import torch

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.orches.placement_group import (
    create_train_group,
    create_placement_groups,
)
from gpatch_v4.utils import logging_memory_usage
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class CkptTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init()

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def train_setup(self):
        config = load_config('test_ckpt_distill.yaml', OnPolicyDistillConfig)
        pgs = create_placement_groups(config)
        policy_group = create_train_group(config, pgs)
        await policy_group.init()
        return policy_group, config

    async def test_policy_load_save_ckpt(self):
        policy_group, config = await self.train_setup()
        # load from hf
        print("load from hf...")
        futs = [actor.setup_model_and_optimizer.remote() for actor in policy_group._actor_handlers]
        [await fut for fut in futs]
        # save to torch dist ckpt
        print("save to torch dist...")
        futs = [actor.save_checkpoint.remote(10) for actor in policy_group._actor_handlers]
        [await fut for fut in futs]

        # load from torch dist ckpt
        print("load from torch dist...")
        futs = [actor.setup_model_and_optimizer.remote() for actor in policy_group._actor_handlers]
        [await fut for fut in futs]
        print(f"-" * 1000)
