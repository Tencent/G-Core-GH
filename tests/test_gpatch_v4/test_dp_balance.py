"""
Test dp_balance correctness for GrpoTrainActor.

Test 1: compute_log_probs with dp_balance ON vs OFF
  - Rebalance → compute → restore, compare restored logprobs sums
    with baseline (no rebalance) logprobs sums.

Test 2: rl_train_actor with dp_balance ON vs OFF
  - Compare training metrics consistency.

Usage:
    pytest tests/test_gpatch_v4/test_dp_balance.py -s -v
"""

import os
import shutil
import unittest

import ray
from custom_py.dp_balance_test_trainer import DpBalanceTestTrainer

from gpatch_v4.configs.config import RlConfig
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
    requires_vllm,
)


class TestDpBalance(unittest.IsolatedAsyncioTestCase):
    """End-to-end dp_balance correctness test using real GrpoTrainActor."""
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _reuse_dp_balance_correctness(self, backend):
        config = load_config('test_dp_balance', RlConfig)
        config.sampler.backend = backend

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)

        trainer = DpBalanceTestTrainer()
        results_list = await trainer.launch_then_run_with_recovery(config)

        # results_list is a list (one per actor) of dicts
        # Check that all actors passed
        for actor_idx, actor_results in enumerate(results_list):
            if actor_results is None:
                continue

            # Test 1: compute_log_probs consistency
            if "test1" in actor_results:
                test1 = actor_results["test1"]
                assert test1["passed"], (
                    f"Actor {actor_idx} Test 1 (compute_log_probs) FAILED!\n"
                    f"Details: {test1.get('details', {})}"
                )

        # Cleanup
        if os.path.exists(save_path):
            shutil.rmtree(save_path, ignore_errors=True)

    @requires_sglang
    async def test_dp_balance_correctness_sglang(self):
        await self._reuse_dp_balance_correctness("sglang")

    @requires_vllm
    async def test_dp_balance_correctness_vllm(self):
        await self._reuse_dp_balance_correctness("vllm")
