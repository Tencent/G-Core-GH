"""EPO-lite GRPO L2 e2e: history baseline, accumulate, sidecar save/load.

Run A: exit_step=2, save_interval=1 → metrics + disk ppo_feature_store.pt
Run B: resume from Run A ckpt, exit_step=3 → history prefix preserved, len+1
"""
import os
import shutil
import unittest

import pytest
import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.ppo_feature_store import (
    PPO_FEATURE_STORE_FILENAME,
    feature_history_key,
    feature_pending_key,
    iter_checkpoint_dir,
)
from gpatch_v4.trainer import GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)

_CKPT = "save_epo_grpo_e2e"
_DATASET_N_LINES = 32
_CONFIG_NAME = "test_epo_grpo_e2e"


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


def _extra_pt_path(ckpt_root: str, step: int) -> str:
    return os.path.join(iter_checkpoint_dir(ckpt_root, step), PPO_FEATURE_STORE_FILENAME)


def _load_extra_state(ckpt_root: str, step: int) -> dict:
    path = _extra_pt_path(ckpt_root, step)
    assert os.path.isfile(path), f"missing PpoFeatureStore sidecar: {path}"
    return torch.load(path, map_location="cpu", weights_only=False)


def _history_from_pt(state: dict, feature: str = "epo") -> list:
    data = state["data"]
    pending_key = feature_pending_key(feature)
    history_key = feature_history_key(feature)
    assert pending_key not in data
    hist = data[history_key]
    assert isinstance(hist, list)
    return [float(x) for x in hist]


def _assert_adv_stats_metrics(step_metrics: dict, *, history_len: float) -> None:
    for name in ("adv_mean", "adv_min", "adv_max"):
        assert f"extra/{name}" in step_metrics, step_metrics.keys()
        assert step_metrics[f"extra/{name}_history_len"] == pytest.approx(history_len)


def _assert_adv_stats_in_ckpt(state: dict, *, history_len: int) -> None:
    for name in ("adv_mean", "adv_min", "adv_max"):
        hist = _history_from_pt(state, name)
        assert len(hist) == history_len
        assert state["data"].get(feature_pending_key(name)) is None


def _first_rank_steps(metrics) -> list:
    assert metrics is not None and len(metrics) > 0
    steps = metrics[0]
    assert isinstance(steps, list) and len(steps) > 0
    return steps


class TestEpoGrpoE2e(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _run_train(self, config: RlConfig):
        trainer = GrpoTrainer()
        return await trainer.launch_then_run_with_recovery(config)

    @requires_sglang
    async def test_history_accumulate_save_load(self):
        _prepare_dataset()
        shutil.rmtree(_CKPT, ignore_errors=True)

        # --- Run A: from scratch, 2 ppo steps ---
        config_a = load_config(_CONFIG_NAME, RlConfig)
        config_a.training.exit_step = 2
        config_a.training.save_interval = 1
        assert config_a.ppo.feature_store_enable
        assert config_a.task is not None
        assert config_a.task["epo_mask_mode"] == "token"
        assert config_a.ppo.loss_func == "epo_grpo"
        assert config_a.ppo.loss_func_py_path.endswith("tasks/math_rl_v4/epo_grpo_loss.py")
        assert config_a.ppo.loss_func_py_name == "epo_grpo_loss_func"
        assert config_a.ppo.advantage_type == "custom_grpo_adv_stats"
        assert config_a.ppo.custom_advantage_py_path.endswith(
            "tasks/math_rl_v4/custom_advantage.py"
        )
        assert config_a.ppo.custom_advantage_py_name == "custom_grpo_advantage_with_adv_stats"
        metrics_a = await self._run_train(config_a)
        kill_all_actors_and_shutdown_ray()

        steps_a = _first_rank_steps(metrics_a)
        assert len(steps_a) == 2, f"expected 2 ppo steps, got {len(steps_a)}"

        m0, m1 = steps_a[0], steps_a[1]
        assert "extra/epo_history_len" in m0
        assert m0["extra/epo_history_len"] == pytest.approx(1.0)
        assert "extra/epo" in m0
        assert "policy/epo_baseline_H" not in m0
        _assert_adv_stats_metrics(m0, history_len=1.0)

        assert m1["extra/epo_history_len"] == pytest.approx(2.0)
        assert "extra/epo" in m1
        assert "policy/epo_baseline_H" in m1
        baseline_step1 = float(m1["policy/epo_baseline_H"])
        assert baseline_step1 > 0.0
        _assert_adv_stats_metrics(m1, history_len=2.0)

        state1 = _load_extra_state(_CKPT, 1)
        hist1 = _history_from_pt(state1)
        assert len(hist1) == 1
        assert state1.get("ppo_step") == 1
        assert hist1[0] == pytest.approx(float(m0["extra/epo"]), rel=1e-4)
        _assert_adv_stats_in_ckpt(state1, history_len=1)
        assert _history_from_pt(state1, "adv_mean")[0] == pytest.approx(
            float(m0["extra/adv_mean"]), rel=1e-4
        )

        state2 = _load_extra_state(_CKPT, 2)
        hist2 = _history_from_pt(state2)
        assert len(hist2) == 2
        assert state2.get("ppo_step") == 2
        assert hist2[0] == pytest.approx(hist1[0], rel=1e-5)
        assert hist2[1] == pytest.approx(float(m1["extra/epo"]), rel=1e-4)
        assert baseline_step1 == pytest.approx(hist2[0], rel=1e-4)
        _assert_adv_stats_in_ckpt(state2, history_len=2)

        # --- Run B: resume, train one more step ---
        config_b = load_config(_CONFIG_NAME, RlConfig)
        config_b.training.exit_step = 3
        config_b.training.save_interval = 1
        config_b.checkpoint.load_ckpt_path = _CKPT
        config_b.checkpoint.save_ckpt_path = _CKPT
        metrics_b = await self._run_train(config_b)

        steps_b = _first_rank_steps(metrics_b)
        assert len(steps_b) == 1, f"resume should train 1 step, got {len(steps_b)}"
        m_resume = steps_b[0]
        assert m_resume["extra/epo_history_len"] == pytest.approx(3.0)
        assert "policy/epo_baseline_H" in m_resume
        resume_baseline = float(m_resume["policy/epo_baseline_H"])
        expected_baseline = sum(hist2) / len(hist2)
        assert resume_baseline == pytest.approx(expected_baseline, rel=1e-4)
        _assert_adv_stats_metrics(m_resume, history_len=3.0)

        state3 = _load_extra_state(_CKPT, 3)
        hist3 = _history_from_pt(state3)
        assert len(hist3) == 3
        assert hist3[0] == pytest.approx(hist2[0], rel=1e-5)
        assert hist3[1] == pytest.approx(hist2[1], rel=1e-5)
        assert hist3[2] == pytest.approx(float(m_resume["extra/epo"]), rel=1e-4)
        _assert_adv_stats_in_ckpt(state3, history_len=3)

        shutil.rmtree(_CKPT, ignore_errors=True)
