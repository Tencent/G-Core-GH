"""Coverage for generate_params / eval_generate_params and eval wiring.

Layers
------
L1 config unit: legacy fill, explicit generate_params, eval resolve
L2 hydra merge: yaml-like DictConfig → InferEngineConfig / SamplerConfig
L3 sampling build: get_sampling_params_from_config(is_eval=...)
L4 actor wiring: sampler generate gets is_eval; _eval_loop passes real ppo_step

GPU e2e: ``test_eval_generate_params_e2e.py`` (10 steps + 2 evals).
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from omegaconf import OmegaConf

from gpatch_v4.configs.infer_engine_config import GenerateParams, InferEngineConfig
from gpatch_v4.configs.sampler_config import SamplerConfig
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.generation_backend.infer_engine import InferEngine


class _StubInferEngine:
    def get_sampling_params(self, **kwargs):
        return SimpleNamespace(**kwargs)


class TestGenerateParamsConfig(unittest.TestCase):
    def test_legacy_flat_fields_fill_generate_params(self):
        cfg = InferEngineConfig(temperature=1.0, top_k=0, top_p=0.9, min_p=0.1)
        self.assertIsNotNone(cfg.generate_params)
        self.assertEqual(cfg.generate_params.temperature, 1.0)
        self.assertEqual(cfg.generate_params.top_k, 0)
        self.assertEqual(cfg.generate_params.top_p, 0.9)
        self.assertEqual(cfg.generate_params.min_p, 0.1)

    def test_explicit_generate_params_not_overwritten(self):
        cfg = InferEngineConfig(
            temperature=0.5,
            top_k=-1,
            top_p=1.0,
            generate_params=GenerateParams(temperature=1.0, top_k=0, top_p=0.9, min_p=0.0),
        )
        self.assertEqual(cfg.generate_params.temperature, 1.0)
        self.assertEqual(cfg.generate_params.top_k, 0)
        self.assertEqual(cfg.generate_params.top_p, 0.9)

    def test_eval_falls_back_to_generate_params(self):
        cfg = InferEngineConfig(temperature=1.0, top_k=0, top_p=0.9)
        gp = cfg.resolve_generate_params(is_eval=True)
        self.assertIs(gp, cfg.generate_params)
        self.assertEqual(gp.temperature, 1.0)

    def test_eval_uses_eval_generate_params(self):
        cfg = InferEngineConfig(
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            eval_generate_params=GenerateParams(
                temperature=0.0, top_k=1, top_p=0.9, min_p=0.0
            ),
        )
        train_gp = cfg.resolve_generate_params(is_eval=False)
        eval_gp = cfg.resolve_generate_params(is_eval=True)
        self.assertEqual(train_gp.temperature, 1.0)
        self.assertEqual(eval_gp.temperature, 0.0)
        self.assertEqual(eval_gp.top_k, 1)


class TestGenerateParamsHydraMerge(unittest.TestCase):
    def test_legacy_yaml_fields_fill_on_merge(self):
        cfg = merge_hydra_config(
            InferEngineConfig,
            OmegaConf.create(
                {
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                    "min_p": 0.05,
                    "generate_max_tokens": 256,
                }
            ),
        )
        self.assertIsNotNone(cfg.generate_params)
        self.assertEqual(cfg.generate_params.temperature, 0.7)
        self.assertEqual(cfg.generate_params.top_k, 50)
        self.assertEqual(cfg.generate_params.top_p, 0.95)
        self.assertEqual(cfg.generate_params.min_p, 0.05)

    def test_eval_generate_params_from_yaml(self):
        cfg = merge_hydra_config(
            InferEngineConfig,
            OmegaConf.create(
                {
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "top_k": 0,
                    "eval_generate_params": {
                        "temperature": 0.0,
                        "top_k": 1,
                        "top_p": 0.9,
                        "min_p": 0.0,
                    },
                }
            ),
        )
        self.assertEqual(cfg.resolve_generate_params(is_eval=False).temperature, 1.0)
        eval_gp = cfg.resolve_generate_params(is_eval=True)
        self.assertEqual(eval_gp.temperature, 0.0)
        self.assertEqual(eval_gp.top_k, 1)

    def test_sampler_config_ensures_nested_ie_configs(self):
        # Bypass IE __post_init__ fill, then SamplerConfig must ensure.
        ie = object.__new__(InferEngineConfig)
        ie.temperature = 0.8
        ie.top_k = 20
        ie.top_p = 0.85
        ie.min_p = 0.0
        ie.generate_params = None
        ie.eval_generate_params = None

        sampler = SamplerConfig(infer_engine_configs=[ie])
        self.assertIsNotNone(sampler.infer_engine_configs[0].generate_params)
        self.assertEqual(sampler.infer_engine_configs[0].generate_params.temperature, 0.8)


class TestGetSamplingParamsFromConfig(unittest.TestCase):
    def test_train_reads_generate_params(self):
        ie = InferEngineConfig(
            temperature=0.5,
            generate_params=GenerateParams(temperature=1.0, top_k=0, top_p=0.9, min_p=0.0),
            generate_max_tokens=128,
            seed=7,
        )
        params = InferEngine.get_sampling_params_from_config(
            _StubInferEngine(), ie, stop_at_token_id=2, is_eval=False
        )
        self.assertEqual(params.temperature, 1.0)
        self.assertEqual(params.top_p, 0.9)
        self.assertEqual(params.top_k, -1)  # top_k<=0 → -1
        self.assertEqual(params.max_tokens, 128)

    def test_eval_reads_eval_generate_params(self):
        ie = InferEngineConfig(
            temperature=1.0,
            top_p=0.9,
            eval_generate_params=GenerateParams(
                temperature=0.0, top_k=1, top_p=0.9, min_p=0.0
            ),
            generate_max_tokens=64,
        )
        params = InferEngine.get_sampling_params_from_config(
            _StubInferEngine(), ie, stop_at_token_id=2, is_eval=True
        )
        self.assertEqual(params.temperature, 0.0)
        self.assertEqual(params.top_k, 1)


class TestEvalWiring(unittest.IsolatedAsyncioTestCase):
    async def test_sampler_actor_forwards_is_eval_to_generate_func(self):
        from gpatch_v4.actor.grpo_sampler_actor import GrpoSamplerActor

        actor = object.__new__(GrpoSamplerActor)
        actor._is_master_node = True
        actor.config = SimpleNamespace()  # no training → skip mbs assert
        actor.infer_engine = object()
        actor.idx = 0
        actor.tokenizer = object()
        actor._extra_gen_args = ()
        seen = {}

        async def _gen_fn(**kwargs):
            seen.update(kwargs)
            return {"ok": True}

        actor.generate_func = _gen_fn
        out = await GrpoSamplerActor.generate(
            actor,
            {
                "batched_data": {"tokens": [1]},
                "sampling_repeat": 2,
                "is_eval": True,
            },
        )
        self.assertEqual(out, {"ok": True})
        self.assertTrue(seen["is_eval"])
        self.assertEqual(seen["sampling_repeat_n"], 2)

    async def test_eval_loop_passes_real_ppo_step_to_external_reward(self):
        from gpatch_v4.actor import grpo_train_actor as actor_module
        from gpatch_v4.actor.grpo_train_actor import GrpoTrainActor

        class _EvalRolloutGenerator:
            def __init__(self):
                self.clear_data_cache = MagicMock()
                self.seen_ppo_steps = []

            async def __call__(self, data_iter, num_microbatches, ppo_step):
                self.seen_ppo_steps.append(ppo_step)
                return [{"tokens": []}]

            def add_back_rollout_attr_after_sampling(self, rollout_batches):
                return rollout_batches

        class _ExternalReward:
            def __init__(self):
                self.calls = []

            async def calc_external_reward(
                self, rollout_batches, ppo_step, is_eval=False, _started_event=None
            ):
                self.calls.append((ppo_step, is_eval))
                return [{} for _ in rollout_batches]

        actor = object.__new__(GrpoTrainActor)
        actor.config = SimpleNamespace(
            training=SimpleNamespace(
                use_external_reward=True,
                total_eval_step=2,
            ),
            external_reward=SimpleNamespace(async_eval=False),
        )
        actor.external_reward = _ExternalReward()
        actor._pending_eval_external_rewards = None
        actor.eval_rollout_generator = _EvalRolloutGenerator()
        actor.eval_dataloader = [object(), object()]
        actor.policy_engine = SimpleNamespace(
            set_model_eval=MagicMock(),
            set_model_train=MagicMock(),
        )
        actor.get_num_eval_rollout_micro_batches = MagicMock(return_value=1)
        actor.compute_rollout_metrics = MagicMock(return_value={"rewards": 1.0})
        actor.eval_logging = MagicMock()

        timers = MagicMock()
        with (
            patch(
                "gpatch_v4.actor.grpo_train_actor.TimerSingleton.get_timer",
                return_value=timers,
            ),
            patch("gpatch_v4.actor.grpo_train_actor.cpu_barrier"),
            patch("gpatch_v4.actor.grpo_train_actor.clear_memory"),
            patch("gpatch_v4.actor.grpo_train_actor.check_rollout_batches", return_value=True),
            patch.object(
                actor_module.BroadcastUtils,
                "broadcast_rollout_batch",
                side_effect=lambda batches: batches,
            ),
        ):
            await actor._eval_loop(42)

        self.assertEqual(actor.eval_rollout_generator.seen_ppo_steps, [42, 42])
        self.assertEqual(actor.external_reward.calls, [(42, True), (42, True)])


if __name__ == "__main__":
    unittest.main()
