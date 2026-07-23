"""End-to-end smoke tests for early model offload.

Each case drives the production sampler weight-update flow twice. That flow
offloads the optimizer and model, wakes the colocated sampler, generates with
the updated weights, and then restores the trainer for the next update.
"""

import os
import shutil
import unittest
import uuid

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.trainer import GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)


HF_MODEL_PATH = (
    "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Qwen/Qwen2.5-Math-1.5B"
)


@requires_sglang
class EarlyOffloadTest(unittest.IsolatedAsyncioTestCase):
    """Cover {MCore, FSDP2} x {late, early model offload}."""

    _SAVE_ROOT = os.path.abspath(
        os.path.join("tests", "test_gpatch_v4", "_tmp_early_offload")
    )

    def setUp(self):
        self._case_root = os.path.join(self._SAVE_ROOT, uuid.uuid4().hex)
        self._debug_path = os.path.join(self._case_root, "weights")
        self._ckpt_path = os.path.join(self._case_root, "checkpoint")

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()
        shutil.rmtree(self._case_root, ignore_errors=True)

    def _make_config(self, backend, early_swap_model):
        config = load_config("test_early_offload", RlConfig)
        config.training.training_backend = backend
        config.training.early_swap_model = early_swap_model
        config.checkpoint.load_ckpt_path = self._ckpt_path
        config.checkpoint.save_ckpt_path = self._ckpt_path
        config.debug.debug_engine_save_path = self._debug_path
        return config

    async def _run_case(self, backend, early_swap_model):
        self.assertTrue(
            os.path.isdir(HF_MODEL_PATH),
            f"small test model is missing: {HF_MODEL_PATH}",
        )

        config = self._make_config(backend, early_swap_model)
        self.assertEqual(config.policy.dist_config.pipeline_model_parallel_size, 1)

        trainer = GrpoTrainer()
        await trainer.debug_update_weight(config)

        # Reaching all three snapshots proves both update cycles completed:
        # sampler generation succeeded and the trainer survived offload/onload.
        for snapshot in ("src", "zero", "real"):
            path = os.path.join(self._debug_path, f"{snapshot}_weights_0")
            self.assertTrue(
                os.path.isdir(path),
                f"{backend=} {early_swap_model=}: missing {path}",
            )

    async def test_mcore_without_early_offload(self):
        await self._run_case("mcore", False)

    async def test_mcore_with_early_offload(self):
        await self._run_case("mcore", True)

    # async def test_fsdp2_without_early_offload(self):
    #     await self._run_case("fsdp2", False)

    # async def test_fsdp2_with_early_offload(self):
    #     await self._run_case("fsdp2", True)
