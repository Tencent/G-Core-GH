import os
import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.extended_model.welm_v4 import (
    WelmV4PrepareDataForwardLLM,
    _flatten_routed_experts_for_dynamic_cp,
    _restore_packed_routed_experts,
)
from gpatch_v4.training_backend.fsdp2_backend.mixin import ForwardStepMixin
from gpatch_v4.training_backend.megatron_backend.mixin import (
    ForwardStepMixin as McoreForwardStepMixin,
)
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


class DynamicCPRouterReplayUnitTest(unittest.TestCase):
    def test_welm_reroute_packs_multisegment_experts_without_losing_layout(self):
        num_layers = 2
        topk = 2
        routed0 = torch.arange(4 * num_layers * topk).view(4, num_layers, topk)
        routed1 = 100 + torch.arange(6 * num_layers * topk).view(6, num_layers, topk)

        flat0 = _flatten_routed_experts_for_dynamic_cp(
            routed0, actual_len=3, padded_len=4, dp_rank=0
        )
        flat1 = _flatten_routed_experts_for_dynamic_cp(
            routed1, actual_len=5, padded_len=8, dp_rank=0
        )
        packed = [{
            "tokens": torch.arange(12),
            "routed_experts": torch.cat([flat0, flat1]),
        }]
        _restore_packed_routed_experts(packed, num_layers=num_layers, topk=topk)

        expected0 = routed0[torch.tensor([0, 1, 2, 0])]
        expected1 = routed1[torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])]
        self.assertEqual(packed[0]["routed_experts"].shape, (12, num_layers, topk))
        self.assertTrue(
            torch.equal(packed[0]["routed_experts"], torch.cat([expected0, expected1]))
        )

    @patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    @patch("gpatch_v4.extended_model.welm_v4.get_thd_partitioned_indices")
    @patch("gpatch_v4.extended_model.welm_v4.parallel_state")
    def test_welm_dynamic_cp_slices_experts_with_token_thd_index(
        self, mock_parallel_state, mock_partition
    ):
        class _Group:
            def size(self):
                return 2

            def rank(self):
                return 1

        class _TPGroup:
            def size(self):
                return 1

        cp_group = _Group()
        mock_parallel_state.get_dynamic_data_context_parallel_groups.return_value = cp_group
        mock_parallel_state.get_tensor_model_parallel_group.return_value = _TPGroup()
        index = torch.tensor([4, 1, 6, 3])
        mock_partition.return_value = index

        prepare = WelmV4PrepareDataForwardLLM.__new__(WelmV4PrepareDataForwardLLM)
        prepare._oe_grams = []
        prepare._ngram_vocab_size = None
        routed_experts = torch.arange(8 * 3 * 2).view(8, 3, 2)
        batch = {
            "tokens": torch.arange(8),
            "labels": 10 + torch.arange(8),
            "position_ids": torch.arange(8),
            "advantages": torch.arange(8, dtype=torch.float32),
            "prev_log_probs": torch.arange(8, dtype=torch.float32),
            "loss_mask": torch.ones(8),
            "routed_experts": routed_experts.clone(),
            "cu_seqlens_padded": torch.tensor([0, 8], dtype=torch.int32),
            "max_seqlen": torch.tensor(8, dtype=torch.int32),
            "local_cp_size": torch.tensor(2, dtype=torch.int32),
        }

        prepared_batch, fwd_kwargs = prepare.grpo_train_with_dynamic_cp(
            [batch],
            seqlen=8,
            pad_token_id=0,
            ppo_pack_seq=False,
        )

        self.assertTrue(torch.equal(prepared_batch["tokens"], index.view(1, -1)))
        self.assertEqual(prepared_batch["routed_experts"].shape, (4, 3, 2))
        self.assertTrue(
            torch.equal(prepared_batch["routed_experts"], routed_experts.index_select(0, index))
        )
        self.assertEqual(fwd_kwargs["input_ids"].shape, (1, 4))

    @patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    @patch(
        "gpatch_v4.training_backend.megatron_backend.mixin.get_transformer_layer_offset",
        return_value=1,
    )
    @patch(
        "gpatch_v4.training_backend.megatron_backend.mixin.get_num_layers_to_build",
        return_value=1,
    )
    @patch("gpatch_v4.training_backend.megatron_backend.mixin.mpu")
    def test_mcore_packed_thd_replay_skips_fixed_cp_and_keeps_tp_sp(
        self, mock_mpu, _mock_num_layers, _mock_layer_offset
    ):
        mixin = McoreForwardStepMixin.__new__(McoreForwardStepMixin)
        mixin.get_mcore_config = lambda: SimpleNamespace(
            context_parallel_size=4,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            moe_router_topk=2,
        )
        mock_mpu.get_tensor_model_parallel_rank.return_value = 1

        captured = []
        mixin.get_router_replay_manager = lambda: SimpleNamespace(
            append_micro_batch=lambda value: captured.append(value)
        )
        routed_experts = torch.arange(8 * 3 * 2).view(8, 3, 2)
        original_arange = torch.arange

        def _cpu_arange(*args, **kwargs):
            kwargs.pop("device", None)
            return original_arange(*args, **kwargs)

        with patch(
            "gpatch_v4.training_backend.megatron_backend.mixin.torch.arange",
            side_effect=_cpu_arange,
        ):
            mixin.prepare_for_router_replay(
                [{"routed_experts": routed_experts}],
                seqlen=8,
                packed_thd=True,
            )

        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), 1)
        expected = routed_experts[torch.tensor([4, 5, 6, 7]), 1, :].long()
        self.assertTrue(torch.equal(captured[0][0], expected))


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
