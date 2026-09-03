"""DeepSeek-V4 regression tests for two-environment, two-teacher FrozenLake MOPD."""

import importlib.util
import math
import os
import shutil
import unittest

from hydra import compose, initialize

from gpatch_v4.configs.config import AgenticOnPolicyDistillConfig
from gpatch_v4.configs.utils import merge_hydra_config


requires_sglang = unittest.skipUnless(
    importlib.util.find_spec("sglang") is not None,
    "sglang not installed in this image",
)


class TestFrozenLakeMopd(unittest.IsolatedAsyncioTestCase):
    """Exercise config routing everywhere and the full MOPD path with sglang."""

    checkpoint_path = "unittest_frozen_lake_mopd"

    def tearDown(self):
        if importlib.util.find_spec("ray") is not None:
            from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

            kill_all_actors_and_shutdown_ray()

    @staticmethod
    def _load_config():
        with initialize(config_path="configs/test_yaml", version_base=None):
            config = compose(config_name="test_frozen_lake_mopd")
        return merge_hydra_config(AgenticOnPolicyDistillConfig, config)

    def test_config_routes_each_environment_to_a_teacher(self):
        config = self._load_config()
        templates = config.training.agentic.resolved_env_templates()

        self.assertEqual(config.ppo.advantage_type, "g_opd")
        self.assertEqual(config.ppo.g_opd_teacher_routing_field, "teacher_type")
        self.assertEqual(config.training.training_backend, "fsdp2")
        self.assertEqual(config.policy.model_arch, "deepseek_v4")
        self.assertTrue(config.debug.disable_save_checkpoint)
        self.assertEqual(
            [template.env_config["teacher_type"] for template in templates],
            ["teacher_a", "teacher_b"],
        )

    @requires_sglang
    async def test_train_one_step_sglang(self):
        # Keep direct pytest invocations on the same safe DSV4 path as the
        # production launcher. topk_v2 enters an incompatible CUDA 12.8 JIT.
        os.environ["SGLANG_OPT_USE_TOPK_V2"] = "0"
        propagated_env = {
            name
            for name in os.environ.get("GPATCH_EXTRA_PROPAGATE_ENV", "").split(",")
            if name
        }
        propagated_env.add("SGLANG_OPT_USE_TOPK_V2")
        os.environ["GPATCH_EXTRA_PROPAGATE_ENV"] = ",".join(sorted(propagated_env))

        from gpatch_v4.trainer import GrpoSingleCtrlTrainer

        config = self._load_config()
        required_model_files = (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        model_paths = {
            config.policy.hf_model_path,
            config.policy.hf_tokenizer_path,
            *(teacher.hf_model_path for teacher in config.teachers.values()),
        }
        for model_path in model_paths:
            self.assertTrue(os.path.isdir(model_path), f"missing model directory: {model_path}")
            for filename in required_model_files:
                path = os.path.join(model_path, filename)
                self.assertTrue(os.path.isfile(path), f"missing model file: {path}")

        shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)

        trainer = GrpoSingleCtrlTrainer()
        metrics_by_rank = await trainer.launch_then_run_with_recovery(config)

        self.assertIsNotNone(metrics_by_rank)
        expected_world_size = (
            config.policy.dist_config.nnodes *
            config.policy.dist_config.num_gpus_per_node
        )
        self.assertEqual(len(metrics_by_rank), expected_world_size)

        expected_keys = (
            "policy/loss",
            "policy/grad_norm",
            "policy/teacher_student_kl_loss",
            "rollout-metrics/teacher_a/num_samples",
            "rollout-metrics/teacher_b/num_samples",
            "rollout-rewards/teacher_a/rewards",
            "rollout-rewards/teacher_b/rewards",
        )
        for rank, rank_metrics in enumerate(metrics_by_rank):
            self.assertEqual(len(rank_metrics), 1, f"rank {rank} did not return exactly one step")
            metrics = rank_metrics[0]
            for key in expected_keys:
                self.assertIn(key, metrics, f"rank {rank} is missing {key}")
                self.assertTrue(
                    math.isfinite(metrics[key]),
                    f"rank {rank} {key} is not finite",
                )

            self.assertGreater(metrics["rollout-metrics/teacher_a/num_samples"], 0)
            self.assertGreater(metrics["rollout-metrics/teacher_b/num_samples"], 0)

        # Keep logs/checkpoint artifacts when launch, training, or assertions
        # fail. A successful run cleans up its smoke-test directory here.
        shutil.rmtree(self.checkpoint_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
