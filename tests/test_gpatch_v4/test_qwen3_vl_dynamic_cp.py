import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from gpatch_v4.extended_model.qwen3_vl import Qwen3VLPrepareDataForward


class TestQwen3VLDynamicCp(unittest.TestCase):

    def _reroute_rl_text_only_batch(self, token_weights=None, routed_experts=None):
        dist_config = SimpleNamespace(max_seqlen_per_dp_cp_rank=8192)
        config = SimpleNamespace(
            policy=SimpleNamespace(
                dist_config=dist_config,
                model_arch="qwen3_5_moe",
            ),
            training=SimpleNamespace(moe_router_replay=routed_experts is not None),
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        batch = {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "prompt_lengths": torch.tensor(1),
            "sequence_lengths": torch.tensor(4),
            "position_ids": torch.arange(12).reshape(1, 3, 4),
            "image_input_mask": torch.zeros(4, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
            "mask": torch.tensor([True, True, True]),
            "sample_mask": torch.tensor(1.0),
        }
        if token_weights is not None:
            batch["token_weights"] = token_weights
        if routed_experts is not None:
            batch["routed_experts"] = routed_experts
        captured = {}

        def schedule(samples, *args, **kwargs):
            captured["packed_keys"] = kwargs["packed_keys"]
            return samples, 1, 3.0, 9.0, {}

        with patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_data_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_data_parallel_rank",
            return_value=0,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.torch.cuda.current_device",
            return_value=0,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.dyn_cp_schedule_default",
            side_effect=schedule,
        ):
            samples, _, _, _, _ = prepare_data.rl_reroute_data_for_dynamic_cp(
                [batch],
                pad_token_id=0,
            )

        return samples[0], captured["packed_keys"]

    def test_rl_text_only_batch_expands_scalar_token_weight(self):
        sample, packed_keys = self._reroute_rl_text_only_batch(torch.tensor([2.5]))

        self.assertIn("token_weights", packed_keys)
        self.assertEqual(sample["token_weights"].tolist(), [2.5, 2.5, 2.5])

    def test_rl_text_only_batch_preserves_per_token_weights(self):
        sample, packed_keys = self._reroute_rl_text_only_batch(
            torch.tensor([0.5, 1.0, 1.5])
        )

        self.assertIn("token_weights", packed_keys)
        self.assertEqual(sample["token_weights"].tolist(), [0.5, 1.0, 1.5])

    def test_rl_text_only_batch_keeps_token_weights_optional(self):
        sample, packed_keys = self._reroute_rl_text_only_batch()

        self.assertNotIn("token_weights", packed_keys)
        self.assertNotIn("token_weights", sample)

    def test_rl_reroute_packs_routed_experts(self):
        """moe_router_replay 时 routed_experts 要进 packed_keys，并和 shifted tokens 对齐。"""
        routed = torch.arange(4 * 2 * 2).view(4, 2, 2)
        sample, packed_keys = self._reroute_rl_text_only_batch(routed_experts=routed)

        self.assertIn("routed_experts", packed_keys)
        self.assertEqual(sample["routed_experts"].shape, (3, 2, 2))
        self.assertTrue(torch.equal(sample["routed_experts"], routed[:3]))

    def test_grpo_dynamic_cp_keeps_full_routed_experts(self):
        """Qwen3-VL tokens 不按 CP 切，routed_experts 保持 packed 全长。"""
        dist_config = SimpleNamespace(max_seqlen_per_dp_cp_rank=8192)
        config = SimpleNamespace(
            policy=SimpleNamespace(
                dist_config=dist_config,
                model_arch="qwen3_5_moe",
            )
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        routed = torch.arange(3 * 2 * 2).view(3, 2, 2)
        batch = {
            "tokens": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([2, 3, 4]),
            "position_ids": torch.arange(9),
            "image_input_mask": torch.zeros(3, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
            "routed_experts": routed.clone(),
            "local_cp_size": torch.tensor(1),
            "cu_seqlens_padded": torch.tensor([0, 3], dtype=torch.int32),
            "max_seqlen": torch.tensor(3),
        }

        with patch.object(
            torch.Tensor,
            "cuda",
            lambda tensor, *args, **kwargs: tensor,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.parallel_state."
            "get_dynamic_data_context_parallel_groups",
            return_value=group,
            create=True,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.parallel_state."
            "get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.PackedSeqParams",
        ):
            prepared, _ = prepare_data.grpo_train_with_dynamic_cp(
                [batch],
                seqlen=3,
                pad_token_id=0,
                ppo_pack_seq=True,
            )

        self.assertEqual(prepared["routed_experts"].shape, (3, 2, 2))
        self.assertTrue(torch.equal(prepared["routed_experts"], routed))

    def test_grpo_dynamic_cp_keeps_token_weights_for_loss(self):
        dist_config = SimpleNamespace(max_seqlen_per_dp_cp_rank=8192)
        config = SimpleNamespace(
            policy=SimpleNamespace(
                dist_config=dist_config,
                model_arch="qwen3_5_moe",
            )
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        batch = {
            "tokens": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([2, 3, 4]),
            "position_ids": torch.arange(9),
            "image_input_mask": torch.zeros(3, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
            "token_weights": torch.tensor([2.5, 2.5, 2.5]),
            "local_cp_size": torch.tensor(1),
            "cu_seqlens_padded": torch.tensor([0, 3], dtype=torch.int32),
            "max_seqlen": torch.tensor(3),
        }

        with patch.object(
            torch.Tensor,
            "cuda",
            lambda tensor, *args, **kwargs: tensor,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.parallel_state."
            "get_dynamic_data_context_parallel_groups",
            return_value=group,
            create=True,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.parallel_state."
            "get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.PackedSeqParams",
        ):
            prepared, _ = prepare_data.grpo_train_with_dynamic_cp(
                [batch],
                seqlen=3,
                pad_token_id=0,
                ppo_pack_seq=True,
            )

        self.assertEqual(prepared["token_weights"].shape, (1, 3))
        self.assertEqual(prepared["token_weights"].tolist(), [[2.5, 2.5, 2.5]])

    def test_sft_text_only_batch_accepts_none_vision_fields(self):
        dist_config = SimpleNamespace(
            dynamic_cp_scheduler_type="smart_padding",
            max_seqlen_per_dp_cp_rank=8192,
        )
        config = SimpleNamespace(
            policy=SimpleNamespace(
                dist_config=dist_config,
                model_arch="qwen3_vl",
            )
        )
        prepare_data = Qwen3VLPrepareDataForward(config)
        group = MagicMock()
        group.size.return_value = 1
        batches = [{
            "tokens": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.arange(12).reshape(1, 3, 4),
            "image_input_mask": torch.zeros(4, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
        }]

        def schedule(samples, *args, **kwargs):
            return samples, 1, 4.0, 16.0

        with patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_data_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_tensor_model_parallel_group",
            return_value=group,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.mpu.get_context_parallel_world_size",
            return_value=1,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.torch.cuda.current_device",
            return_value=0,
        ), patch(
            "gpatch_v4.extended_model.qwen3_vl.dyn_cp_schedule_smart_padding",
            side_effect=schedule,
        ):
            samples, _, _, _ = prepare_data.sft_reroute_data_for_dynamic_cp(
                batches,
                pad_token_id=0,
            )

        self.assertIsNone(samples[0]["vision_data"])
        self.assertIsNone(samples[0]["vision_grid_thw"])


if __name__ == "__main__":
    unittest.main()
