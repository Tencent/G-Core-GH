# coding=utf-8
"""DSpark BSHD vs THD：正式 ``FinetuneTrainer`` 路径对比。

对齐 ``test_deepseek_v4_sft_thd.py``：同一 yaml / 数据，只切
``ppo_pack_seq``，跑 ``exit_step=1``，比较 ``dspark_loss`` /
``total_loss`` / ``grad_norm``。

与 ``test_deepseek_v4_dspark_bshd_vs_thd.py``（Ray worker 精简对照）互补：
本文件走 mixin 真路径，不做 router/anchor replay，阈值更宽。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=3600 \\
      tests/test_gfused/test_deepseek_v4_dspark_sft_bshd_vs_thd.py
"""

from __future__ import annotations

import math
import os
import shutil
import unittest

import pytest

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.trainer import FinetuneTrainer
from test_gpatch_v4.gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
)

HF_MODEL_PATH = os.environ.get(
    "DSPARK_HF_MODEL_PATH",
    "hf-hub/deepseek-ai/DeepSeek-V4-Flash-0731",
)
DATA_PATH = os.environ.get(
    "DSPARK_E2E_DATA_PATH",
    "hf-hub/AI-MO/NuminaMath-CoT-jsonl/train",
)


class TestDeepseekV4DSparkSftBshdVsThd(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _run_one_step(self, *, pack_seq: bool) -> list:
        hf_model_path = os.path.abspath(HF_MODEL_PATH)
        data_path = os.path.abspath(DATA_PATH)
        if not os.path.isdir(hf_model_path):
            raise unittest.SkipTest(
                f"DSpark checkpoint not found: {hf_model_path}; "
                "set DSPARK_HF_MODEL_PATH"
            )
        if not os.path.isdir(data_path):
            raise unittest.SkipTest(
                f"DSpark E2E data not found: {data_path}; "
                "set DSPARK_E2E_DATA_PATH"
            )

        config = load_config("test_dsv4_sft_dspark", FinetuneConfig)
        assert config.training.enable_dspark
        assert config.training.online_train_dspark
        assert config.policy.dist_config.context_parallel_size == 1
        config.training.exit_step = 1
        config.policy.ppo_pack_seq = pack_seq
        config.policy.hf_model_path = hf_model_path
        config.policy.hf_tokenizer_path = hf_model_path
        config.data.data_pathes = [data_path]
        tag = "thd" if pack_seq else "bshd"
        config.checkpoint.load_ckpt_path = f"unittest_dsv4_sft_dspark_bshd_vs_thd_{tag}"
        config.checkpoint.save_ckpt_path = f"unittest_dsv4_sft_dspark_bshd_vs_thd_{tag}"
        # yaml 默认 seq_length=128；Numina 样本远长于此，pack 会 assert。
        # 对齐 test_deepseek_v4_sft_thd：两侧共用更大预算，BSHD 再开 pad-to-max。
        config.training.seq_length = 4096
        if not pack_seq:
            config.debug.experimental_pad_to_max_length = True

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
        try:
            metrics = await FinetuneTrainer().launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
            shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        return metrics

    @staticmethod
    def _extract_step0(metrics) -> dict:
        assert metrics is not None
        assert len(metrics) >= 1
        dp0 = metrics[0]
        assert len(dp0) >= 1
        return dp0[0]

    @pytest.mark.timeout(3600)
    async def test_dspark_sft_bshd_vs_thd(self):
        """FinetuneTrainer：BSHD (pad-to-max) vs THD (pack-seq)。"""
        metrics_thd = await self._run_one_step(pack_seq=True)
        self.tearDown()

        metrics_bshd = await self._run_one_step(pack_seq=False)

        step_thd = self._extract_step0(metrics_thd)
        step_bshd = self._extract_step0(metrics_bshd)

        keys = (
            "finetune/dspark_loss",
            "finetune/total_loss",
            "finetune/grad_norm",
        )
        for step in (step_thd, step_bshd):
            for key in keys:
                assert key in step, f"missing metric {key} in {sorted(step)}"
                assert math.isfinite(step[key]), f"{key}={step[key]}"

        print("\n" + "=" * 60)
        print("  DSpark SFT FinetuneTrainer: BSHD vs THD")
        print("=" * 60)
        for key in keys:
            print(
                f"  {key}: BSHD={step_bshd[key]:.6f}  THD={step_thd[key]:.6f}"
            )

        # 无 router/anchor replay，阈值对齐 test_deepseek_v4_sft_thd。
        for key, loss_tol, gn_tol in (
            ("finetune/dspark_loss", 0.05, None),
            ("finetune/total_loss", 0.05, None),
            ("finetune/grad_norm", None, 0.10),
        ):
            rel = abs(step_bshd[key] - step_thd[key]) / (abs(step_bshd[key]) + 1e-8)
            print(f"  {key} rel_diff={rel:.6f}")
            if loss_tol is not None:
                self.assertLess(
                    rel,
                    loss_tol,
                    f"{key} rel_diff {rel:.6f} > {loss_tol * 100:.0f}%",
                )
            if gn_tol is not None:
                self.assertLess(
                    rel,
                    gn_tol,
                    f"{key} rel_diff {rel:.6f} > {gn_tol * 100:.0f}%",
                )
        print("=" * 60 + "\n")


if __name__ == "__main__":
    unittest.main()
