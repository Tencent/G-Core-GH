"""Prev-entropy quantile GRPO e2e: postprocess cuts + no feature-store sidecar.

Smoke: 1 ppo_step on Qwen2.5-Math-1.5B colocated sglang; assert train finishes
and that ``set_step_local`` cuts are NOT written into ``ppo_feature_store.pt``.
"""
import os
import shutil
import unittest

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.ppo_feature_store import PPO_FEATURE_STORE_FILENAME
from gpatch_v4.trainer import GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)

_CKPT = "save_prev_entropy_quantile_e2e"
_DATASET_N_LINES = 32
_CONFIG_NAME = "test_prev_entropy_quantile_e2e"


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


def _assert_model_ckpt_without_feature_store_sidecar(ckpt_root: str) -> None:
    assert os.path.isdir(ckpt_root), f"expected ckpt root {ckpt_root}"
    iter_dirs = [
        name for name in os.listdir(ckpt_root) if name.startswith("iter_")
    ]
    assert iter_dirs, f"expected model iter_* under {ckpt_root}, got {os.listdir(ckpt_root)}"
    found = []
    for root, _dirs, files in os.walk(ckpt_root):
        if PPO_FEATURE_STORE_FILENAME in files:
            found.append(os.path.join(root, PPO_FEATURE_STORE_FILENAME))
    assert not found, (
        "prev_entropy_quantile set_step_local must not write PpoFeatureStore sidecar; "
        f"found {found}"
    )


class TestPrevEntropyQuantileE2e(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()
        shutil.rmtree(_CKPT, ignore_errors=True)

    async def _run_train(self, config: RlConfig):
        trainer = GrpoTrainer()
        return await trainer.launch_then_run_with_recovery(config)

    @requires_sglang
    async def test_postprocess_cuts_and_plain_grpo(self):
        _prepare_dataset()
        shutil.rmtree(_CKPT, ignore_errors=True)

        config = load_config(_CONFIG_NAME, RlConfig)
        config.training.exit_step = 1
        config.training.save_interval = 1
        assert config.ppo.feature_store_enable
        assert config.ppo.use_legacy_loss
        assert config.ppo.post_compute_logprobs == "prev_entropy_quantile"
        assert config.ppo.post_compute_logprobs_py_path.endswith(
            "tasks/math_rl_v4/pre_entropy_quantile.py"
        )
        assert config.ppo.post_compute_logprobs_py_name == "prev_entropy_quantile_postprocess"
        assert config.ppo.loss_func == "prev_entropy_quantile_grpo"
        assert config.ppo.loss_func_py_path.endswith("tasks/math_rl_v4/pre_entropy_quantile.py")
        assert config.ppo.loss_func_py_name == "prev_entropy_quantile_grpo_loss_func"
        assert config.task is not None
        assert int(config.task["prev_entropy_quantile_num_bins"]) == 10

        metrics = await self._run_train(config)
        steps = _first_rank_steps(metrics)
        assert len(steps) == 1, f"expected 1 ppo step, got {len(steps)}"
        step0 = steps[0]
        assert "policy/loss" in step0, f"missing policy/loss; keys={sorted(step0.keys())}"

        # Model ckpt saved, but set_step_local cuts must not produce sidecar.
        _assert_model_ckpt_without_feature_store_sidecar(_CKPT)
