import math
import os
import shutil
import unittest

from gpatch_v4.configs.config import FinetuneConfig, RlConfig
from gpatch_v4.trainer import FinetuneTrainer, GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)

_MODEL_PATH = "hf-hub/Qwen/Qwen3-0.6B"


class TestLossBackendSmoke(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def _set_checkpoint_paths(self, config, name: str) -> None:
        config.checkpoint.load_ckpt_path = f"unittest_loss_smoke_{name}"
        config.checkpoint.save_ckpt_path = f"unittest_loss_smoke_{name}"

    def _set_small_model(self, config) -> None:
        config.policy.model_arch = "qwen3"
        config.policy.hf_model_path = _MODEL_PATH
        config.policy.hf_tokenizer_path = _MODEL_PATH
        if not config.policy.without_ref:
            config.policy.ref_hf_model_path = _MODEL_PATH
        config.policy.dist_config.nnodes = 1
        config.policy.dist_config.tensor_model_parallel_size = 1
        config.policy.dist_config.pipeline_model_parallel_size = 1
        config.policy.dist_config.expert_model_parallel_size = 1
        config.policy.dist_config.expert_tensor_parallel_size = 1

    def _prepare_grpo_data(self, backend: str) -> str:
        data_dir = f"tests/test_gpatch_v4/unittest_loss_smoke_{backend}_grpo"
        os.makedirs(data_dir, exist_ok=True)
        source_path = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
        target_path = os.path.join(data_dir, "train.jsonl")
        with open(source_path) as source, open(target_path, "w") as target:
            for index, line in enumerate(source):
                if index == 32:
                    break
                target.write(line)
        return data_dir

    def _assert_metrics(self, metrics, keys: tuple[str, ...]) -> None:
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(len(metrics), 1)
        for dp_metrics in metrics:
            self.assertGreaterEqual(len(dp_metrics), 2)
            for step_metrics in dp_metrics:
                for key in keys:
                    self.assertIn(key, step_metrics)
                    self.assertTrue(math.isfinite(step_metrics[key]), f"{key}={step_metrics[key]}")

    async def _run_finetune(self, backend: str) -> None:
        config = load_config("test_fsdp_muon", FinetuneConfig)
        config.training.training_backend = backend
        config.training.exit_step = 2
        config.training.auto_load_from_save_ckpt = False
        config.training.build_from_mbridge = backend == "mcore"
        config.debug.trainer_return_ppo_step_metrics = True
        config.optimizer.optimizer_type = "adam"
        self._set_checkpoint_paths(config, f"{backend}_finetune")
        self._set_small_model(config)

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            metrics = await FinetuneTrainer().launch_then_run_with_recovery(config)
            self._assert_metrics(metrics, ("finetune/lm_loss", "finetune/grad_norm"))
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
            shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)

    def _make_grpo_config(self, backend: str) -> RlConfig:
        config = load_config("test_math_rl", RlConfig)
        config.training.training_backend = backend
        config.training.exit_step = 2
        config.training.auto_load_from_save_ckpt = False
        config.training.build_from_mbridge = backend == "mcore"
        config.debug.trainer_return_ppo_step_metrics = True
        config.training.rollout_gbs = 8
        config.training.train_gbs = 16
        config.training.sampling_repeat_n = 2
        config.training.sampling_keep_n = 2
        config.training.seq_length = 1024
        config.training.rollout_mbs = 1
        config.training.train_mbs = 1
        config.ppo.use_legacy_loss = backend != "mcore"
        self._set_checkpoint_paths(config, f"{backend}_grpo")
        self._set_small_model(config)

        config.sampler.dist_config.nnodes = 1
        config.sampler.model_info[0].model_arch = "qwen3"
        config.sampler.model_info[0].hf_model_path = _MODEL_PATH
        config.sampler.infer_engine_configs[0].dist_config.nnodes = 1
        config.sampler.infer_engine_configs[0].dist_config.tensor_model_parallel_size = 1
        config.sampler.infer_engine_configs[0].generate_max_tokens = 128
        config.bt_rm.infer_engine_configs[0].dist_config.nnodes = 1
        return config

    async def _run_grpo(self, backend: str) -> None:
        config = self._make_grpo_config(backend)
        data_dir = self._prepare_grpo_data(backend)
        config.data.data_pathes = [data_dir]
        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            metrics = await GrpoTrainer().launch_then_run_with_recovery(config)
            self._assert_metrics(
                metrics,
                ("policy/loss", "policy/grad_norm", "policy/ppo_ratio"),
            )
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
            shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
            shutil.rmtree(data_dir, ignore_errors=True)

    async def test_mcore_finetune(self):
        await self._run_finetune("mcore")

    async def test_fsdp2_finetune(self):
        await self._run_finetune("fsdp2")

    @requires_sglang
    async def test_mcore_grpo(self):
        await self._run_grpo("mcore")

    @requires_sglang
    async def test_fsdp2_grpo(self):
        await self._run_grpo("fsdp2")
