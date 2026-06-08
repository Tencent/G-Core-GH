import os
import shutil
import time
import unittest

import ray

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.trainer import FinetuneTrainer
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


def wait_until(condition_fn, timeout=15, interval=0.5):
    """Wait until condition_fn returns True or timeout"""
    start = time.time()
    while time.time() - start < timeout:
        if condition_fn():
            return True
        time.sleep(interval)
    return False


class TestOffPolicyDistill(unittest.IsolatedAsyncioTestCase):
    CKPT_PATH = "test_save_ckpt"

    def setUp(self):
        # Ensure checkpoint directory is empty
        if os.path.exists(self.CKPT_PATH):
            shutil.rmtree(self.CKPT_PATH)
        os.makedirs(self.CKPT_PATH)

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()
        # Clean up checkpoint directory
        if os.path.exists(self.CKPT_PATH):
            shutil.rmtree(self.CKPT_PATH)

    async def test_load_from_megatron_bridge(self):
        config = load_config("test_saving_ckpt", FinetuneConfig)

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)

        trainer = FinetuneTrainer()
        # save_interval=1, save_total_limit=3, save_retain_interval=3
        # Train 7 steps: saves at 1..7, retains multiples of 3 (step 3,6),
        # plus the last step (7). Older non-retained ckpts get deleted.
        config.training.exit_step = 7
        await trainer.launch_then_run_with_recovery(config)

        def old_ckpt_gone():
            return not (
                os.path.exists(os.path.join(save_path, 'iter_0000005')) or
                os.path.exists(os.path.join(save_path, 'hf/5'))
            )

        assert wait_until(old_ckpt_gone, timeout=15), \
            "Old checkpoint (iter_0000005 / hf/5) not deleted"

        assert os.path.exists(os.path.join(save_path, 'iter_0000003'))
        assert os.path.exists(os.path.join(save_path, 'iter_0000006'))
        assert os.path.exists(os.path.join(save_path, 'iter_0000007'))
        assert os.path.exists(os.path.join(save_path, 'hf/3'))
        assert os.path.exists(os.path.join(save_path, 'hf/6'))
        assert os.path.exists(os.path.join(save_path, 'hf/7'))

        folders = [
            f for f in os.listdir(save_path)
            if os.path.isdir(os.path.join(save_path, f)) and f not in ('hf', 'logs')
        ]
        assert len(
            folders
        ) == 3, f"Expected 3 folders in {save_path} (excluding 'hf' and 'logs'), but found {len(folders)}: {folders}"

        hf_path = os.path.join(save_path, 'hf')
        hf_items = [f for f in os.listdir(hf_path) if os.path.exists(os.path.join(hf_path, f))]
        assert len(
            hf_items
        ) == 3, f"Expected 3 items in {hf_path}, but found {len(hf_items)}: {hf_items}"
