import os
import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.training_backend.fsdp2_backend.mixin import ForwardStepMixin
from gpatch_v4.trainer import GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
    requires_vllm,
)


class PackedRouterReplayUnitTest(unittest.TestCase):
    @patch("gpatch_v4.training_backend.fsdp2_backend.mixin.mpu")
    def test_thd_replay_indices_follow_packed_cp_slice(self, mock_mpu):
        mock_mpu.get_context_parallel_world_size.return_value = 2
        mock_mpu.get_context_parallel_rank.return_value = 1
        mock_mpu.get_data_parallel_rank.return_value = 0

        mixin = ForwardStepMixin.__new__(ForwardStepMixin)
        mixin._topk_layer_indices = [0, 2]
        routed0 = torch.arange(3 * 3 * 2).view(3, 3, 2)
        routed1 = 100 + torch.arange(3 * 3 * 2).view(3, 3, 2)
        batches = [
            {
                "routed_experts": routed0
            },
            {
                "routed_experts": routed1
            },
        ]
        psp = SimpleNamespace(
            cu_seqlens_q_padded=torch.tensor([0, 4, 8]),
            total_seqlen=8,
        )

        per_layer = mixin._prepare_replay_indices(
            batches,
            seq_length=8,
            packed_seq_params=psp,
        )

        self.assertEqual(len(per_layer), 2)
        # CP rank 1 owns packed positions [4:8], i.e. the second segment.
        expected_rows = torch.tensor([0, 1, 0, 1])
        self.assertTrue(torch.equal(per_layer[0], routed1[expected_rows, 0, :].long()))
        self.assertTrue(torch.equal(per_layer[1], routed1[expected_rows, 2, :].long()))

    @patch("gpatch_v4.training_backend.fsdp2_backend.mixin.mpu")
    def test_thd_replay_cp_slice_can_start_inside_segment(self, mock_mpu):
        mock_mpu.get_context_parallel_world_size.return_value = 2
        mock_mpu.get_context_parallel_rank.return_value = 1
        mock_mpu.get_data_parallel_rank.return_value = 0

        mixin = ForwardStepMixin.__new__(ForwardStepMixin)
        mixin._topk_layer_indices = [0]
        routed0 = torch.arange(4 * 2).view(4, 1, 2)
        routed1 = 100 + torch.arange(6 * 2).view(6, 1, 2)
        batches = [
            {
                "routed_experts": routed0
            },
            {
                "routed_experts": routed1
            },
        ]
        psp = SimpleNamespace(
            cu_seqlens_q_padded=torch.tensor([0, 4, 16]),
            total_seqlen=16,
        )

        per_layer = mixin._prepare_replay_indices(
            batches,
            seq_length=16,
            packed_seq_params=psp,
        )

        # Segment 1 occupies packed [4:16]. Rank 1 owns global [8:16],
        # i.e. offsets [4:12] within that segment. Its five real route rows
        # are repeated only for the segment's padding rows.
        expected_rows = torch.tensor([4, 0, 1, 2, 3, 4, 0, 1])
        self.assertEqual(len(per_layer), 1)
        self.assertTrue(torch.equal(per_layer[0], routed1[expected_rows, 0, :].long()))


class RouterReplayR3Test(unittest.IsolatedAsyncioTestCase):
    """End-to-end R3 (router replay) tests for SGLang and vLLM backends.

    Validates that moe_router_replay=True works through the full
    rollout → logprob → training pipeline without errors, and that
    the resulting training metrics are within expected ranges.
    """
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _run_r3_train(self, config):
        tmp_dir = "tests/test_gpatch_v4/gsm8k_small"
        os.makedirs(tmp_dir, exist_ok=True)
        src = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
        dst = os.path.join(tmp_dir, "train.jsonl")
        with open(src, 'r') as fin, open(dst, 'w') as fout:
            for i, line in enumerate(fin):
                if i >= 256:
                    break
                fout.write(line)

        assert config.training.moe_router_replay is True

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            trainer = GrpoTrainer()
            metrics = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
        return metrics

    def _assert_metrics(self, metrics):
        assert metrics is not None
        assert len(metrics) > 0

        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert 'policy/loss' in step_metric
                assert 'policy/grad_norm' in step_metric
                assert 'policy/ppo_ratio' in step_metric

                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']

                assert -0.02 < loss < 0.02, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 0.2, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.5 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

    @requires_sglang
    async def test_r3_sglang(self):
        config = load_config('test_math_rl_r3', RlConfig)
        metrics = await self._run_r3_train(config)
        self._assert_metrics(metrics)

    @requires_vllm
    async def test_r3_vllm(self):
        config = load_config('test_math_rl_r3', RlConfig)
        config.sampler.backend = "vllm"
        metrics = await self._run_r3_train(config)
        self._assert_metrics(metrics)
