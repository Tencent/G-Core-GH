import os
import shutil
import unittest

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.trainer import FinetuneTrainer
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class TestFsdp2Muon(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    @unittest.skip(
        "FSDP2 muon is temporarily disabled in setup_optimizer until NS-on-shard is fixed"
    )
    async def test_muon_train_save_load(self):
        config = load_config("test_fsdp_muon", FinetuneConfig)
        save_path = config.checkpoint.save_ckpt_path
        ckpt_file = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)

        # Round 1: train 10 steps, save ckpt
        config.training.exit_step = 10
        trainer = FinetuneTrainer()
        await trainer.launch_then_run_with_recovery(config)
        assert os.path.exists(ckpt_file), "checkpoint should be saved after round 1"
        kill_all_actors_and_shutdown_ray()

        # Round 2: new trainer, load ckpt, resume to step 15
        config.training.exit_step = 15
        trainer2 = FinetuneTrainer()
        await trainer2.launch_then_run_with_recovery(config)

        if os.path.exists(save_path):
            shutil.rmtree(save_path)
