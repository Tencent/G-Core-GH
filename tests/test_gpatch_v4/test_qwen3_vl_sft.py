import os
import shutil
import unittest

import ray
from packaging.version import Version

from megatron.core import package_info

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.trainer import FinetuneTrainer
from gpatch_v4_test_helper import load_config


class TestQwen3VLFinetune(unittest.IsolatedAsyncioTestCase):
    """Test Qwen3VL SFT"""
    def setUp(self):
        pass

    def tearDown(self):
        ray.shutdown()

    async def test_train_one_step_and_save(self):
        mcore_version = Version(package_info.__version__)

        config = load_config("test_qwen3_vl_sft", FinetuneConfig)

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)
        if os.path.exists(os.path.join(save_path, "hf")):
            shutil.rmtree(os.path.join(save_path, "hf"))

        try:
            trainer = FinetuneTrainer()
            await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(save_path, ignore_errors=True)
