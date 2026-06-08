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
from gpatch_v4_test_helper import load_config


class CustomConfigTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    async def test_1(self):
        config = load_config('test_custom_config', T2iRlConfig)
        assert config.task['trick1'] == 'a long string'
        assert config.task['attr3'] == 2
        assert config.task['attr4'] == 2.0
