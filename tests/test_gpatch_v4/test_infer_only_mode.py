"""End-to-end mock tests for infer_only mode.

Validates the full inference entry chain:
  infer_entry.py  ->  InferenceRunner.start
    -> orches.init(config)
    -> create_placement_groups(config)
    -> create_infer_group(config, pgs)  ->  RaySamplerGroup.__init__
    -> RaySamplerGroup.init()
      -> GrpoSamplerActor.init(config)           [BaseActor.init]
      -> GrpoSamplerActor.init_infer_engine(...)  [build_tokenizer, load_hf_config, post_init]
    -> InferenceWorker.init(config)

Strategy: Since the environment lacks heavy dependencies (megatron, Ray actors,
CUDA, etc.), we avoid importing gpatch_v4's actor/orches modules directly.
Instead we:
  1. Build a real InferenceConfig via OmegaConf (same as infer_entry.py).
  2. Replay every attribute-access path the production code performs on the config.
  3. Read the source files and verify key guard patterns (hasattr, isinstance)
     exist, so regressions that remove a guard will be caught.

Any missing ``hasattr`` guard, missing field, or wrong fallback path will cause
an AttributeError / AssertionError here.
"""
import os
import re
import unittest

from omegaconf import OmegaConf

from gpatch_v4.configs.config import InferenceConfig
from gpatch_v4.configs.utils import merge_hydra_config


# ---------------------------------------------------------------------------
# Helper: build a realistic InferenceConfig matching infer_only yaml shape
# ---------------------------------------------------------------------------
def _build_infer_config():
    """Mimic infer_entry.py: merge yaml-like dict onto InferenceConfig defaults."""
    yaml_like = {
        'data':
            {
                'data_pathes': ['hf-hub/openai/gsm8k-jsonl/eval'],
                'py_path': 'tasks/infer_only/simple_dataset.py',
                'fn_name': 'get_dataset_and_dataloader',
            },
        'sampler':
            {
                'backend':
                    'sglang',
                'sampler_type':
                    'sampler',
                'dist_config': {
                    'nnodes': 1,
                    'num_gpus_per_node': 1,
                },
                'model_info':
                    [
                        {
                            'model_arch': 'qwen3',
                            'hf_model_path': '/tmp/fake_model_path',
                            'gen_rollout_py_path': 'tasks/infer_only/sample_demo.py',
                            'gen_rollout_fn_name': 'generate_func',
                        }
                    ],
                'infer_engine_configs':
                    [
                        {
                            'dist_config':
                                {
                                    'nnodes': 1,
                                    'tensor_model_parallel_size': 1,
                                    'num_gpus_per_node': 1,
                                },
                            'gpu_memory_utilization': 0.8,
                            'max_running_requests': 64,
                            'use_fast_tokenizer': False,
                        }
                    ],
            },
        'infer_result': {
            'output_dir': '/tmp/infer_only_test_results',
        },
    }
    return merge_hydra_config(InferenceConfig, OmegaConf.create(yaml_like))


def _read_source(rel_path):
    """Read a source file relative to project root."""
    proj_root = os.path.join(os.path.dirname(__file__), '..', '..')
    with open(os.path.join(proj_root, rel_path)) as f:
        return f.read()


class TestInferEntryEndToEnd(unittest.TestCase):
    """Single end-to-end test that replays every attribute-access the production
    code performs on InferenceConfig, in the exact order of the call chain."""
    def setUp(self):
        self.cfg = _build_infer_config()

    # -- Step 0: config build (infer_entry.py:25-27) -----------------------

    def test_config_type_and_missing_fields(self):
        self.assertIsInstance(self.cfg, InferenceConfig)
        self.assertFalse(hasattr(self.cfg, 'policy'))
        self.assertFalse(hasattr(self.cfg, 'training'))

    # -- Step 1-3: placement_group attribute accesses ----------------------

    def test_placement_group_attribute_accesses(self):
        """Replay create_placement_groups InferenceConfig branch."""
        cfg = self.cfg
        sampler_nnodes = cfg.sampler.dist_config.nnodes
        sampler_gpus = cfg.sampler.dist_config.num_gpus_per_node
        self.assertEqual(sampler_nnodes * sampler_gpus, 1)

    # -- Step 4: BaseActor.init attribute accesses -------------------------

    def test_base_actor_init_attribute_accesses(self):
        """Replay BaseActor.init() with InferenceConfig."""
        cfg = self.cfg
        infer_only_mode = not hasattr(cfg, 'policy') and not hasattr(cfg, 'training')
        self.assertTrue(infer_only_mode)

        # offload_process_group guard – must not raise
        if not infer_only_mode and cfg.training.offload_process_group:
            self.fail("Should not reach here for InferenceConfig")

        # timeout fallback
        timeout = (
            cfg.sampler.dist_config.torch_dist_timeout_minutes
            if infer_only_mode else cfg.policy.dist_config.torch_dist_timeout_minutes
        )
        self.assertIsNotNone(timeout)

        # num_gpus_per_node fallback
        num_gpus = (
            cfg.sampler.dist_config.num_gpus_per_node
            if infer_only_mode else cfg.policy.dist_config.num_gpus_per_node
        )
        self.assertEqual(num_gpus, 1)

    # -- Step 5: init_infer_engine attribute accesses ----------------------

    def test_init_infer_engine_attribute_accesses(self):
        """Replay GrpoSamplerActor.init_infer_engine() config reads."""
        cfg = self.cfg
        idx = 0

        infer_engine_config = cfg.sampler.infer_engine_configs[idx]
        dist_config = infer_engine_config.dist_config
        model_arch = cfg.sampler.model_info[idx].model_arch
        hf_model_path = cfg.sampler.model_info[idx].hf_model_path
        self.assertEqual(model_arch, 'qwen3')
        self.assertEqual(hf_model_path, '/tmp/fake_model_path')

        # load_format access
        load_format = infer_engine_config.load_format
        if hasattr(cfg, 'debug') and cfg.debug.debug_engine_update_weight:
            load_format = "auto"
        self.assertIsNotNone(load_format)

        # InferEngine.from_engine_args kwargs
        _ = cfg.sampler.backend
        _ = infer_engine_config.dtype
        _ = dist_config.tensor_model_parallel_size
        _ = dist_config.pipeline_model_parallel_size
        _ = dist_config.expert_model_parallel_size
        _ = infer_engine_config.enable_deepep_moe
        _ = infer_engine_config.gpu_memory_utilization
        _ = dist_config.num_gpus_per_node
        _ = cfg.sampler.sampler_type
        _ = infer_engine_config.use_fast_tokenizer
        _ = infer_engine_config.max_running_requests
        _ = infer_engine_config.allow_auto_truncate
        _ = infer_engine_config.mm_per_request_timeout

        # attention_backend guard
        sgl_backend = (cfg.infer_result.attention_backend if hasattr(cfg, 'infer_result') else None)
        self.assertEqual(sgl_backend, 'flashinfer')

        # moe_router_replay guard
        enable_routed = (cfg.training.moe_router_replay if hasattr(cfg, 'training') else False)
        self.assertFalse(enable_routed)

    # -- Step 6: build_tokenizer attribute accesses ------------------------

    def test_build_tokenizer_attribute_accesses(self):
        """Replay TokenizerMixin.build_tokenizer() config reads."""
        cfg = self.cfg
        infer_only_mode = not hasattr(cfg, 'policy') and not hasattr(cfg, 'training')
        self.assertTrue(infer_only_mode)

        train_config = cfg.training if not infer_only_mode else None
        self.assertIsNone(train_config)

        # actor_tokenizer path
        if not infer_only_mode:
            _ = cfg.policy.hf_tokenizer_path
            _ = train_config.use_fast_tokenizer
        else:
            actor_tok_path = cfg.sampler.model_info[0].hf_model_path
            use_fast = cfg.sampler.infer_engine_configs[0].use_fast_tokenizer
        self.assertEqual(actor_tok_path, '/tmp/fake_model_path')
        self.assertFalse(use_fast)

        # sampler_tokenizers loop
        for sidx, model_info in enumerate(cfg.sampler.model_info):
            s_use_fast = (
                train_config.use_fast_tokenizer if not infer_only_mode else
                cfg.sampler.infer_engine_configs[sidx].use_fast_tokenizer
            )
            self.assertEqual(model_info.hf_model_path, '/tmp/fake_model_path')
            self.assertFalse(s_use_fast)

    # -- Step 7: load_hf_config attribute accesses -------------------------

    def test_load_hf_config_attribute_accesses(self):
        """Replay BaseActor.load_hf_config() config reads."""
        cfg = self.cfg
        if hasattr(cfg, 'policy'):
            load_path = cfg.policy.hf_model_path
        else:
            load_path = cfg.sampler.model_info[0].hf_model_path
        self.assertEqual(load_path, '/tmp/fake_model_path')

        # policy.hf_config should not be set for InferenceConfig
        self.assertFalse(hasattr(cfg, 'policy'))

    # -- Step 8: generate guard --------------------------------------------

    def test_generate_guard(self):
        """Replay training.rollout_mbs guard in generate()."""
        cfg = self.cfg
        if hasattr(cfg, 'training'):
            _ = cfg.training.rollout_mbs
            self.fail("Should not reach here for InferenceConfig")

    # -- Step 9: InferenceWorker attribute accesses ------------------------

    def test_inference_worker_attribute_accesses(self):
        """Replay InferenceWorker.init() config reads."""
        cfg = self.cfg
        self.assertIsNotNone(cfg.data.py_path)
        self.assertIsNotNone(cfg.data.fn_name)
        self.assertEqual(cfg.infer_result.output_dir, '/tmp/infer_only_test_results')


class TestSourceCodeGuards(unittest.TestCase):
    """Verify that critical guard patterns exist in the source code.

    If someone removes a hasattr/isinstance guard, these tests will fail
    even without importing the heavy modules.
    """
    def test_base_actor_init_has_infer_only_guard(self):
        src = _read_source('gpatch_v4/orches/train_actor.py')
        self.assertIn('infer_only_mode', src)
        self.assertIn("not hasattr(self.config", src)
        self.assertIn("if not infer_only_mode and self.config.training.offload_process_group", src)

    def test_load_hf_config_has_policy_guard(self):
        src = _read_source('gpatch_v4/orches/train_actor.py')
        self.assertIn("hasattr(self.config, 'policy')", src)
        self.assertIn('self.config.sampler.model_info[self.idx].hf_model_path', src)

    def test_build_tokenizer_has_infer_only_guard(self):
        src = _read_source('gpatch_v4/actor/mixin.py')
        self.assertIn('infer_only_mode', src)
        self.assertIn('if infer_only_mode:', src)

    def test_init_infer_engine_has_training_guard(self):
        src = _read_source('gpatch_v4/actor/grpo_sampler_actor.py')
        self.assertIn("hasattr(config, 'training')", src)
        self.assertIn("hasattr(config, 'infer_result')", src)

    def test_placement_group_has_inference_config_branch(self):
        src = _read_source('gpatch_v4/orches/placement_group.py')
        self.assertIn('isinstance(config, InferenceConfig)', src)

    def test_sampler_group_has_inference_config_check(self):
        src = _read_source('gpatch_v4/orches/sampler_group.py')
        self.assertIn('InferenceConfig', src)


if __name__ == '__main__':
    unittest.main()
