import os
import shutil
import unittest
from packaging.version import Version

import pytest
import ray

from megatron.core import package_info

try:
    from gpatch_v4.configs.config import FinetuneConfig
    from gpatch_v4.trainer import FinetuneTrainer
    from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config
except ImportError:
    pass


class TestOffPolicyDistill(unittest.IsolatedAsyncioTestCase):
    """Test OffPolicyDistill"""
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def test_load_from_megatron_bridge(self):
        if True:
            pytest.skip("skip")

        mcore_version = Version(package_info.__version__)
        # 跳过测试
        if mcore_version <= Version("0.13.1"):
            return

        config = load_config("test_megatron_bridge", FinetuneConfig)

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)
        hf_ckpt_path = os.path.join(save_path, "hf")
        if os.path.exists(hf_ckpt_path):
            shutil.rmtree(hf_ckpt_path)

        trainer = FinetuneTrainer()
        config.training.build_from_mbridge = False
        config.training.exit_step = 12
        await trainer.launch_then_run_with_recovery(config)

        assert os.path.exists(latest_ckpt_path)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
