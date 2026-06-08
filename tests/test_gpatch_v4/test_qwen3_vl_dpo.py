import os
import shutil
import unittest

import numpy as np
import pytest
import ray
import torch

from gpatch_v4.configs.config import DpoConfig
from gpatch_v4.trainer import DpoTrainer
from gpatch_v4.orches.placement_group import (
    create_train_group,
    create_placement_groups,
)
from gpatch_v4.utils import logging_memory_usage
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class TestQwen3VLDpo(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def test_train_one_step_and_save(self):
        config = load_config('test_qwen3_vl_dpo', DpoConfig)

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)
        if os.path.exists(os.path.join(save_path, "hf")):
            shutil.rmtree(os.path.join(save_path, "hf"))

        trainer = DpoTrainer()
        metrics_list = await trainer.launch_then_run_with_recovery(config)
        for metrics in metrics_list:
            for metric in metrics:
                assert abs(
                    metric['finetune/dpo-metrics/loss'] - torch.log(torch.tensor(2)).item()
                ) < 0.01, f"{metric=}"

        assert os.path.exists(latest_ckpt_path)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
