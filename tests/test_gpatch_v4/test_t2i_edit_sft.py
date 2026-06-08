import asyncio
import os
import shutil
import unittest
import uuid

import pytest
import ray

from gpatch_v4.configs.config import T2iEditSftConfig
from gpatch_v4.trainer import T2iEditSftTrainer
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class T2iEditSftTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Use a unique KV root per test run to avoid LMDB residual data
        # (stale guard keys) from previous runs causing image read failures.
        self._kv_root = f"/root/kv_test_{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def test_train(self):
        config = load_config('test_t2i_edit_sft', T2iEditSftConfig)
        config.kv.root = self._kv_root
        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            trainer = T2iEditSftTrainer()
            await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
            # KV LMDB dirs live on each node under self._kv_root/{rank}.
            # The actors are already dead after cleanup, so we clean up
            # what we can reach from the driver node.  Remote nodes' dirs will
            # be orphaned but won't conflict with future runs (unique path).
            shutil.rmtree(self._kv_root, ignore_errors=True)
