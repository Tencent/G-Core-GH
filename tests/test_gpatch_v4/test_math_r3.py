import asyncio
import shutil
import unittest

import numpy as np
import pynvml
import pytest
import ray
import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.trainer import GrpoSingleCtrlTrainer, GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
    requires_vllm,
)


class LLmGrpoTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def reuse_test_train(self, config, trainer_cls=GrpoTrainer):
        # Create a small temp dataset (32 lines) to speed up the test
        import os
        tmp_dir = "tests/test_gpatch_v4/gsm8k_small"
        os.makedirs(tmp_dir, exist_ok=True)
        src = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
        dst = os.path.join(tmp_dir, "train.jsonl")
        with open(src, 'r') as fin, open(dst, 'w') as fout:
            for i, line in enumerate(fin):
                if i >= 256:
                    break
                fout.write(line)

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            trainer = trainer_cls()

            ret = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
        return ret

    async def _reuse_test_train_basic(self, backend, config_name='test_math_rl'):
        config = load_config(config_name, RlConfig)
        config.sampler.backend = backend
        metrics = await self.reuse_test_train(config)

        # metrics[dp_rank][ppo_step] = dict of metric values
        assert metrics is not None
        assert len(metrics) > 0  # at least dp_size returns

        for dp_metrics in metrics:  # each dp rank
            assert len(dp_metrics) >= 1  # at least 1 ppo step
            for step_metric in dp_metrics:
                # Baseline values (2 steps observed):
                #   policy/loss:      7.49e-05 ~ 1.21e-04
                #   policy/grad_norm: 8.00e-02 ~ 8.49e-02
                #   policy/ppo_ratio: ~1.001
                assert 'policy/loss' in step_metric
                assert 'policy/grad_norm' in step_metric
                assert 'policy/ppo_ratio' in step_metric

                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']

                assert -0.01 < loss < 0.01, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 0.2, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.5 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

                assert 'policy/grpo_kl_loss' in step_metric
                grpo_kl_loss = step_metric['policy/grpo_kl_loss']
                assert 0 <= grpo_kl_loss < 0.001, f"policy/grpo_kl_loss out of range: {grpo_kl_loss}"

    @requires_sglang
    async def test_train_sglang(self):
        await self._reuse_test_train_basic("sglang")

    @requires_vllm
    async def test_train_vllm(self):
        await self._reuse_test_train_basic(
            "vllm",
            config_name='test_math_rl_vllm',
        )

    @requires_sglang
    async def test_train_skip_prev_logps_sglang(self):
        import glob

        config = load_config('test_math_rl', RlConfig)
        config.sampler.backend = "sglang"
        config.ppo.skip_prev_logps = True
        shutil.rmtree("debug-tmp", ignore_errors=True)
        metrics = await self.reuse_test_train(config)

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

                assert -0.01 < loss < 0.01, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 0.2, f"policy/grad_norm out of range: {grad_norm}"
                # skip_prev_logps path should keep PPO ratio near 1.0
                assert 0.99 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

        # Branch-specific oracle: in skip_prev_logps path, rollout_batch["logprobs"]
        # is populated with zeros placeholders before prepare_data.
        debug_files = sorted(glob.glob("debug-tmp/rollout_batches_*.pt"))
        assert debug_files, "debug rollout files not found under debug-tmp/"
        for dump_file in debug_files:
            rollout_batches = torch.load(dump_file, map_location="cpu", weights_only=False)
            assert rollout_batches, f"empty rollout dump in {dump_file}"
            for batch_idx, one_batch in enumerate(rollout_batches):
                assert "logprobs" in one_batch, f"logprobs missing in {dump_file}"
                for sample_idx, logps in enumerate(one_batch["logprobs"]):
                    assert torch.count_nonzero(logps).item() == 0, (
                        f"{dump_file} batch={batch_idx} sample={sample_idx} has non-zero "
                        f"logprobs in skip_prev_logps mode"
                    )

    async def _reuse_filter_stage_metrics(
        self,
        config_name,
        expected_stage,
        backend,
        loss_bound=0.01,
    ):
        config = load_config(config_name, RlConfig)
        config.sampler.backend = backend
        # assert config.training.sampling_repeat_n == 8
        # assert config.training.sampling_keep_n == 4
        assert config.training.filter_sampling_stage == expected_stage
        metrics = await self.reuse_test_train(config)

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

                assert -loss_bound < loss < loss_bound, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 5.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.5 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

    def _apply_filter_overrides(self, config, stage):
        """Apply filter_sampling overrides to a config."""
        config.training.sampling_repeat_n = 8
        config.training.sampling_keep_n = 4
        config.training.sampling_keeping_strategy = "random_filter"
        config.training.ppo_filter_samplings_path = (
            "tests/test_gpatch_v4/custom_py/random_filter.py"
        )
        config.training.ppo_filter_samplings_name = "random_filter"
        config.training.filter_sampling_stage = stage
        config.training.train_gbs = 512

    @requires_sglang
    async def test_filter_sampling_pre_sglang(self):
        await self._reuse_filter_stage_metrics(
            'test_math_rl_filter_pre', 'pre', 'sglang', loss_bound=0.15,
        )

    @requires_vllm
    async def test_filter_sampling_pre_vllm(self):
        config = load_config('test_math_rl_disaggregated', RlConfig)
        config.sampler.backend = "vllm"
        config.gen_rm.backend = "vllm"
        self._apply_filter_overrides(config, "pre")
        assert config.training.filter_sampling_stage == "pre"
        metrics = await self.reuse_test_train(config)

        assert metrics is not None
        assert len(metrics) > 0
        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']
                assert -0.01 < loss < 0.01, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 5.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.5 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

    @requires_sglang
    async def test_filter_sampling_post_sglang(self):
        # loss_bound relaxed to 0.1 for post-filter:
        # In post-filter mode, GRPO advantages are computed on all
        # ``repeat_n=8`` samples (per-group mean == 0), then
        # ``random_filter`` retains only ``keep_n=4``.  The kept subset's
        # advantage mean is generally != 0 (biased by which samples the
        # filter selects), so the resulting policy loss is non-zero even
        # at ppo_ratio == 1.  Empirically loss lands in [0.03, 0.07].
        await self._reuse_filter_stage_metrics(
            'test_math_rl_filter_post',
            'post',
            'sglang',
            loss_bound=0.25,
        )

    @requires_vllm
    async def test_filter_sampling_post_vllm(self):
        config = load_config('test_math_rl_disaggregated', RlConfig)
        config.sampler.backend = "vllm"
        config.gen_rm.backend = "vllm"
        self._apply_filter_overrides(config, "post")
        assert config.training.filter_sampling_stage == "post"
        metrics = await self.reuse_test_train(config)

        assert metrics is not None
        assert len(metrics) > 0
        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']
                assert -0.05 < loss < 0.05, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 5.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.5 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

    # ------------------------------------------------------------------ #
    #  test_train_gspo / test_train_fipo
    # ------------------------------------------------------------------ #

    @requires_sglang
    async def test_train_gspo_sglang(self):
        config = load_config('test_math_rl', RlConfig)
        config.ppo.loss_func = "gspo"
        config.sampler.backend = "sglang"
        metrics = await self.reuse_test_train(config)

        assert metrics is not None
        assert len(metrics) > 0
        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert 'policy/loss' in step_metric
                assert 'policy/grad_norm' in step_metric
                assert 'policy/ppo_ratio' in step_metric
                assert 'policy/grpo_kl_loss' in step_metric

                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']

                assert -0.01 < loss < 0.01, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 5.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.5 < ppo_ratio < 1.5, f"policy/ppo_ratio out of range: {ppo_ratio}"

    @requires_sglang
    async def test_train_fipo_sglang(self):
        config = load_config('test_math_rl', RlConfig)
        config.ppo.loss_func = "fipo"
        config.ppo.ppo_logps_ratio_clamp = 20.0
        config.ppo.ppo_dual_clip_ratio_c = 3.0
        config.sampler.backend = "sglang"
        metrics = await self.reuse_test_train(config)

        assert metrics is not None
        assert len(metrics) > 0
        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert 'policy/loss' in step_metric
                assert 'policy/grad_norm' in step_metric
                assert 'policy/ppo_ratio' in step_metric
                assert 'policy/grpo_kl_loss' in step_metric

                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']

                assert -0.05 < loss < 0.05, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 5.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.5 < ppo_ratio < 1.5, f"policy/ppo_ratio out of range: {ppo_ratio}"

    # ------------------------------------------------------------------ #
    #  test_train_ppo (placeholder)
    # ------------------------------------------------------------------ #

    async def test_train_ppo(self):
        pass

    async def _reuse_test_train_disaggregated(self, backend):
        config = load_config('test_math_rl_disaggregated', RlConfig)
        config.sampler.backend = backend
        config.gen_rm.backend = backend
        assert config.placement_type == "disaggregated"
        metrics = await self.reuse_test_train(config)

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

                assert -0.01 < loss < 0.01, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 0.2, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.5 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

    @requires_sglang
    async def test_train_disaggregated_sglang(self):
        await self._reuse_test_train_disaggregated("sglang")

    @requires_vllm
    async def test_train_disaggregated_vllm(self):
        await self._reuse_test_train_disaggregated("vllm")

    async def reuse_test_train_async(self, config):
        """Run async trainer and return (metrics, trainer) for overlap check."""
        import os
        tmp_dir = "tests/test_gpatch_v4/gsm8k_small"
        os.makedirs(tmp_dir, exist_ok=True)
        src = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
        dst = os.path.join(tmp_dir, "train.jsonl")
        with open(src, 'r') as fin, open(dst, 'w') as fout:
            for i, line in enumerate(fin):
                if i >= 64:
                    break
                fout.write(line)

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            trainer = GrpoSingleCtrlTrainer()
            ret = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
        return ret, trainer

    async def _reuse_test_train_async_rollout(self, backend):
        config = load_config('test_math_rl_async_rollout', RlConfig)
        config.sampler.backend = backend
        config.gen_rm.backend = backend
        assert config.placement_type == "disaggregated"
        assert config.training.async_rollout
        metrics, trainer = await self.reuse_test_train_async(config)

        assert metrics is not None
        assert len(metrics) >= 1

        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert 'policy/loss' in step_metric
                assert 'policy/grad_norm' in step_metric
                assert 'policy/ppo_ratio' in step_metric

                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']

                assert -0.01 < loss < 0.01, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 3.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.99 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

        # Verify pipeline overlap: the second batch's fire time should be
        # earlier than the first batch's last train_done, proving that the
        # controller was generating while GPU actors were training.
        pipeline_stats = trainer.get_pipeline_stats()
        ppo_steps = sorted(pipeline_stats.keys())
        assert len(ppo_steps) == 4, f"Expected 4 ppo steps, got {len(ppo_steps)}"

        t0 = min(ts.get("fire", float("inf")) for ts in pipeline_stats.values())
        print("\n" + "=" * 60)
        print("  Pipeline Timeline (driver-side)")
        print("=" * 60)
        for step in ppo_steps:
            ts = pipeline_stats[step]
            parts = []
            for key in ("fire", "collect", "train_done"):
                if key in ts:
                    parts.append(f"{key}={ts[key] - t0:.1f}s")
            print(f"  Step {step}: {' → '.join(parts)}")

        # Check rolling-prefetch overlap: for consecutive training steps,
        # the later step's ``fire`` timestamp should precede the earlier
        # step's ``train_done``.  In the rolling-prefetch loop, the later
        # step is either co-fired in the same cold-start burst as the
        # current step (identical fire time) or prefetched by a follow-up
        # ``fire_up_to`` call before the current step finishes training.
        # Either way, the controller is generating while the GPU trains.
        overlap_count = 0
        for j in range(len(ppo_steps) - 1):
            cur, nxt = ppo_steps[j], ppo_steps[j + 1]
            cur_ts, nxt_ts = pipeline_stats[cur], pipeline_stats[nxt]
            if "train_done" in cur_ts and "fire" in nxt_ts:
                fire_t = nxt_ts["fire"] - t0
                train_done_t = cur_ts["train_done"] - t0
                if nxt_ts["fire"] < cur_ts["train_done"]:
                    overlap_count += 1
                    delta = cur_ts["train_done"] - nxt_ts["fire"]
                    print(
                        f"  Overlap: fire({nxt})={fire_t:.1f}s < "
                        f"train_done({cur})={train_done_t:.1f}s "
                        f"-> overlap {delta:.1f}s"
                    )
                else:
                    print(
                        f"  No overlap: fire({nxt})={fire_t:.1f}s >= "
                        f"train_done({cur})={train_done_t:.1f}s"
                    )
        assert overlap_count > 0, (
            f"Expected pipeline overlap but none detected. "
            f"Stats: {pipeline_stats}"
        )
        print("=" * 60 + "\n")

    @requires_sglang
    async def test_train_async_rollout_sglang(self):
        await self._reuse_test_train_async_rollout("sglang")

    @requires_vllm
    async def test_train_async_rollout_vllm(self):
        await self._reuse_test_train_async_rollout("vllm")

    async def _reuse_test_train_async_rollout_rule_only(self, backend):
        config = load_config('test_math_rl_async_rollout_rule_only', RlConfig)
        config.sampler.backend = backend
        assert config.placement_type == "disaggregated"
        assert config.training.async_rollout
        assert config.training.use_bt_rm_reward
        assert not config.training.use_gen_rm_reward
        metrics, trainer = await self.reuse_test_train_async(config)

        assert metrics is not None
        assert len(metrics) >= 1

        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert 'policy/loss' in step_metric
                assert 'policy/grad_norm' in step_metric
                assert 'policy/ppo_ratio' in step_metric

                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']

                assert -0.01 < loss < 0.01, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 3.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.99 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

        pipeline_stats = trainer.get_pipeline_stats()
        ppo_steps = sorted(pipeline_stats.keys())
        assert len(ppo_steps) == 4, f"Expected 4 ppo steps, got {len(ppo_steps)}"

        t0 = min(ts.get("fire", float("inf")) for ts in pipeline_stats.values())
        print("\n" + "=" * 60)
        print("  Pipeline Timeline (driver-side)")
        print("=" * 60)
        for step in ppo_steps:
            ts = pipeline_stats[step]
            parts = []
            for key in ("fire", "collect", "train_done"):
                if key in ts:
                    parts.append(f"{key}={ts[key] - t0:.1f}s")
            print(f"  Step {step}: {' → '.join(parts)}")

        overlap_count = 0
        for j in range(len(ppo_steps) - 1):
            cur, nxt = ppo_steps[j], ppo_steps[j + 1]
            cur_ts, nxt_ts = pipeline_stats[cur], pipeline_stats[nxt]
            if "train_done" in cur_ts and "fire" in nxt_ts:
                fire_t = nxt_ts["fire"] - t0
                train_done_t = cur_ts["train_done"] - t0
                if nxt_ts["fire"] < cur_ts["train_done"]:
                    overlap_count += 1
                    delta = cur_ts["train_done"] - nxt_ts["fire"]
                    print(
                        f"  Overlap: fire({nxt})={fire_t:.1f}s < "
                        f"train_done({cur})={train_done_t:.1f}s "
                        f"-> overlap {delta:.1f}s"
                    )
                else:
                    print(
                        f"  No overlap: fire({nxt})={fire_t:.1f}s >= "
                        f"train_done({cur})={train_done_t:.1f}s"
                    )
        assert overlap_count > 0, (
            f"Expected pipeline overlap but none detected. "
            f"Stats: {pipeline_stats}"
        )
        print("=" * 60 + "\n")

    @requires_sglang
    async def test_train_async_rollout_rule_only_sglang(self):
        await self._reuse_test_train_async_rollout_rule_only("sglang")

    @requires_vllm
    async def test_train_async_rollout_rule_only_vllm(self):
        await self._reuse_test_train_async_rollout_rule_only("vllm")

    async def _reuse_async_rollout_grpo_two_node(self, backend):
        config = load_config('test_math_rl_async_rollout_2node', RlConfig)
        config.sampler.backend = backend
        assert config.placement_type == "disaggregated"
        assert config.training.async_rollout
        metrics, trainer = await self.reuse_test_train_async(config)

        assert metrics is not None
        assert len(metrics) >= 1

        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert 'policy/loss' in step_metric
                assert 'policy/grad_norm' in step_metric
                assert 'policy/ppo_ratio' in step_metric

                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']

                assert -0.01 < loss < 0.01, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 3.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.99 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

    @requires_sglang
    async def test_async_rollout_grpo_two_node_sglang(self):
        await self._reuse_async_rollout_grpo_two_node("sglang")

    @requires_vllm
    async def test_async_rollout_grpo_two_node_vllm(self):
        await self._reuse_async_rollout_grpo_two_node("vllm")

    async def _reuse_async_rollout_grpo_two_turns_rollout(self, backend):
        config = load_config('test_math_rl_async_rollout_2turns_rollout', RlConfig)
        config.sampler.backend = backend
        assert config.placement_type == "disaggregated"
        assert config.training.async_rollout
        metrics, trainer = await self.reuse_test_train_async(config)

        assert metrics is not None
        assert len(metrics) >= 1

        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1

        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert 'policy/loss' in step_metric
                assert 'policy/grad_norm' in step_metric
                assert 'policy/ppo_ratio' in step_metric

                loss = step_metric['policy/loss']
                grad_norm = step_metric['policy/grad_norm']
                ppo_ratio = step_metric['policy/ppo_ratio']

                # Relaxed bound: two-turn reflect produces uneven response
                # lengths across repeat_n samples, so the token-level
                # advantage mean is non-zero even when scalar advantages
                # are group-symmetric.
                assert -0.05 < loss < 0.05, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 3.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.99 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"

    @requires_sglang
    async def test_async_rollout_grpo_two_turns_rollout_sglang(self):
        await self._reuse_async_rollout_grpo_two_turns_rollout("sglang")

    @requires_vllm
    async def test_async_rollout_grpo_two_turns_rollout_vllm(self):
        await self._reuse_async_rollout_grpo_two_turns_rollout("vllm")
