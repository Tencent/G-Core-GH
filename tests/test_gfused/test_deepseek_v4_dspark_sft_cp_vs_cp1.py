# coding=utf-8
"""DSpark THD CP=1 vs CP=2：正式 ``FinetuneTrainer`` 路径对照。

对齐 ``test_deepseek_v4_dspark_sft_bshd_vs_thd.py``：同一 yaml / 数据，
只切 ``context_parallel_size``（以及为对齐 ``dp_size`` 而按比例改
``nnodes``），跑 ``exit_step=1``，比较 DSpark loss components /
``total_loss`` / ``grad_norm``。

FSDP2 dataloader 的 ``dp_size = world / cp``。同卡数只切 CP 会改 DP
分片与 GAS。本测试在 yaml ``nnodes=4``（32 GPU）上：

- CP=1：``nnodes=2`` → world=16, dp=16
- CP=2：``nnodes=4`` → world=32, dp=16

两侧 ``train_gbs`` / ``sampler_seed`` / packed THD 相同。不做
router/anchor replay，阈值宽于 Ray L2。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=3600 \\
      tests/test_gfused/test_deepseek_v4_dspark_sft_cp_vs_cp1.py
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
CP_SIZE = 2
LOSS_RTOL = 0.10
GRAD_RTOL = 0.15


class TestDeepseekV4DSparkSftCpVsCp1(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _run_one_step(self, *, cp_size: int) -> list:
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
        assert config.policy.ppo_pack_seq
        assert config.policy.dist_config.context_parallel_size == 1
        assert config.training.moe_balance_loss_coef == 0
        assert not config.policy.dist_config.dynamic_context_parallel
        assert not config.policy.moe_router_force_load_balancing
        assert not config.policy.balance_dp_seqlen
        assert not config.training.sort_batched
        yaml_nnodes = config.policy.dist_config.nnodes
        assert yaml_nnodes % CP_SIZE == 0, (
            f"yaml nnodes={yaml_nnodes} must be divisible by cp={CP_SIZE} "
            "so CP=1 can keep the same dp_size"
        )

        config.training.exit_step = 1
        config.policy.hf_model_path = hf_model_path
        config.policy.hf_tokenizer_path = hf_model_path
        config.data.data_pathes = [data_path]
        config.data.sampler_seed = 42
        config.training.auto_load_from_save_ckpt = False
        config.policy.dist_config.context_parallel_size = cp_size
        # 对齐 dp = world/cp：CP=1 用一半节点，CP=2 用 yaml 全量节点。
        if cp_size == 1:
            config.policy.dist_config.nnodes = yaml_nnodes // CP_SIZE
        else:
            assert cp_size == CP_SIZE
            config.policy.dist_config.nnodes = yaml_nnodes
        tag = f"cp{cp_size}"
        config.checkpoint.load_ckpt_path = f"unittest_dsv4_sft_dspark_cp_vs_cp1_{tag}"
        config.checkpoint.save_ckpt_path = f"unittest_dsv4_sft_dspark_cp_vs_cp1_{tag}"

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
    async def test_dspark_sft_cp1_vs_cp2(self):
        """FinetuneTrainer：THD pack CP=1 vs CP=2。"""
        metrics_cp1 = await self._run_one_step(cp_size=1)
        self.tearDown()

        metrics_cp = await self._run_one_step(cp_size=CP_SIZE)

        step_cp1 = self._extract_step0(metrics_cp1)
        step_cp = self._extract_step0(metrics_cp)

        keys = (
            "finetune/dspark_loss",
            "finetune/dspark_ce_loss",
            "finetune/dspark_l1_loss",
            "finetune/dspark_confidence_loss",
            "finetune/total_loss",
            "finetune/grad_norm",
        )
        for step in (step_cp1, step_cp):
            for key in keys:
                assert key in step, f"missing metric {key} in {sorted(step)}"
                assert math.isfinite(step[key]), f"{key}={step[key]}"

        print("\n" + "=" * 60)
        print(f"  DSpark SFT FinetuneTrainer: CP=1 vs CP={CP_SIZE}")
        print("=" * 60)
        for key in keys:
            print(
                f"  {key}: CP1={step_cp1[key]:.6f}  CP{CP_SIZE}={step_cp[key]:.6f}"
            )

        # 无 router/anchor replay；阈值宽于 Ray L2 的 0.5%/1%。
        for key, rtol in (
            ("finetune/dspark_loss", LOSS_RTOL),
            ("finetune/dspark_ce_loss", LOSS_RTOL),
            ("finetune/dspark_l1_loss", LOSS_RTOL),
            ("finetune/dspark_confidence_loss", LOSS_RTOL),
            ("finetune/total_loss", LOSS_RTOL),
            ("finetune/grad_norm", GRAD_RTOL),
        ):
            rel = abs(step_cp1[key] - step_cp[key]) / (abs(step_cp1[key]) + 1e-8)
            print(f"  {key} rel_diff={rel:.6f}")
            self.assertLess(
                rel,
                rtol,
                f"{key} rel_diff {rel:.6f} > {rtol * 100:.0f}%",
            )
        print("=" * 60 + "\n")


if __name__ == "__main__":
    unittest.main()
