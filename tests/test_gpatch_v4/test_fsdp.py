import os
import shutil
import unittest

import ray

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.trainer import FinetuneTrainer
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class TestFsdp2(unittest.IsolatedAsyncioTestCase):
    """Test OffPolicyDistill"""
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def test_sft_train_one_step_and_save(self):
        config = load_config("test_fsdp_finetune", FinetuneConfig)

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)

        trainer = FinetuneTrainer()
        await trainer.launch_then_run_with_recovery(config)

        assert os.path.exists(latest_ckpt_path)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
