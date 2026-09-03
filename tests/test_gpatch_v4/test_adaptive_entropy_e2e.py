"""Adaptive entropy GRPO e2e: lag-1 coef metrics + feature-store sidecar.

Smoke: 2 ppo_steps on Qwen2.5-Math-1.5B colocated sglang with
``use_adaptive_entropy`` + high ``entropy_target`` so coef monotonically
increases; assert train metrics and ``ppo_feature_store.pt`` keys.
"""
import os
import shutil
import unittest

import pytest
import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.adaptive_entropy import ENTROPY_COEF_KEY, LAST_WORLD_ENTROPY_KEY
from gpatch_v4.core.ppo_feature_store import (
    PPO_FEATURE_STORE_FILENAME,
    iter_checkpoint_dir,
)
from gpatch_v4.trainer import GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)

_CKPT = "save_adaptive_entropy_e2e"
_DATASET_N_LINES = 32
_CONFIG_NAME = "test_adaptive_entropy_e2e"


def _prepare_dataset(n_lines: int = _DATASET_N_LINES) -> None:
    tmp_dir = "tests/test_gpatch_v4/gsm8k_small"
    os.makedirs(tmp_dir, exist_ok=True)
    src = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
    dst = os.path.join(tmp_dir, "train.jsonl")
    with open(src, "r") as fin, open(dst, "w") as fout:
        for i, line in enumerate(fin):
            if i >= n_lines:
                break
            fout.write(line)


def _first_rank_steps(metrics) -> list:
    assert metrics is not None and len(metrics) > 0
    steps = metrics[0]
    assert isinstance(steps, list) and len(steps) > 0
    return steps


def _metric_float(step_metrics: dict, key: str) -> float:
    assert key in step_metrics, f"missing {key}; keys={sorted(step_metrics.keys())}"
    val = step_metrics[key]
    if isinstance(val, list):
        assert len(val) > 0
        val = val[-1]
    return float(val)


def _load_feature_store(ckpt_root: str, step: int) -> dict:
    path = os.path.join(iter_checkpoint_dir(ckpt_root, step), PPO_FEATURE_STORE_FILENAME)
    assert os.path.isfile(path), f"missing PpoFeatureStore sidecar: {path}"
    return torch.load(path, map_location="cpu", weights_only=False)


class TestAdaptiveEntropyE2e(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()
        shutil.rmtree(_CKPT, ignore_errors=True)

    async def _run_train(self, config: RlConfig):
        trainer = GrpoTrainer()
        return await trainer.launch_then_run_with_recovery(config)

    @requires_sglang
    async def test_adaptive_entropy_metrics_and_sidecar(self):
        _prepare_dataset()
        shutil.rmtree(_CKPT, ignore_errors=True)

        config = load_config(_CONFIG_NAME, RlConfig)
        config.training.exit_step = 2
        config.training.save_interval = 1
        assert config.ppo.feature_store_enable
        assert config.ppo.use_adaptive_entropy
        assert config.ppo.entropy_target == pytest.approx(100.0)
        assert config.ppo.ppo_entropy_bonus == pytest.approx(0.01)
        assert config.ppo.entropy_coef_delta == pytest.approx(0.005)

        metrics = await self._run_train(config)
        steps = _first_rank_steps(metrics)
        assert len(steps) == 2, f"expected 2 ppo steps, got {len(steps)}"

        coefs = []
        last_hs = []
        for i, step in enumerate(steps):
            assert "policy/loss" in step, f"step{i} missing policy/loss; keys={sorted(step.keys())}"
            assert "policy/scaled_entropy" in step, (
                f"step{i} missing policy/scaled_entropy; keys={sorted(step.keys())}"
            )
            coef = _metric_float(step, "policy/entropy_coef")
            last_h = _metric_float(step, "policy/last_world_entropy")
            coefs.append(coef)
            last_hs.append(last_h)
            assert last_h < float("inf")
            assert last_h == pytest.approx(_metric_float(step, "policy/scaled_entropy"), rel=1e-4)

        # High entropy_target ⇒ each train_step increments coef from 0.01.
        assert coefs[0] == pytest.approx(0.015)
        assert coefs[1] == pytest.approx(0.02)
        assert coefs[1] > coefs[0]

        state = _load_feature_store(_CKPT, step=2)
        data = state["data"]
        assert LAST_WORLD_ENTROPY_KEY in data
        assert ENTROPY_COEF_KEY in data
        assert float(data[ENTROPY_COEF_KEY]) == pytest.approx(coefs[-1])
        assert float(data[LAST_WORLD_ENTROPY_KEY]) == pytest.approx(last_hs[-1], rel=1e-4)
