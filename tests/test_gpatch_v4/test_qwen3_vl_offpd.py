import os
import shutil
import unittest
from packaging.version import Version

import ray

from megatron.core import package_info

from gpatch_v4.configs.config import OffPolicyDistillConfig
from gpatch_v4.trainer import OffPolicyDistillTrainer
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class TestQwen3VLOffPolicyDistill(unittest.IsolatedAsyncioTestCase):
    """Test OffPolicyDistill"""
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def test_train_one_step_and_save(self):
        mcore_version = Version(package_info.__version__)
        # TODO(guanyouhe): 现在跳过测试，后面升级到 dev 版本中，去除
        if mcore_version > Version("0.13.1"):
            return

        config = load_config("test_qwen3_vl_off_distill", OffPolicyDistillConfig)

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)
        if os.path.exists(os.path.join(save_path, "hf")):
            shutil.rmtree(os.path.join(save_path, "hf"))

        trainer = OffPolicyDistillTrainer()
        metrics_list = await trainer.launch_then_run_with_recovery(config)
        for metrics in metrics_list:
            for metric in metrics:
                assert metric["finetune/kl_loss"] == 0, f"{metric=}"
