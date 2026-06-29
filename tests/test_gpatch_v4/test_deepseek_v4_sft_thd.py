# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""DSV4 THD pack-seq SFT trainer test.

Runs ``FinetuneTrainer`` with DSV4 4-layer, EP=8 CP=4 on 32 GPUs,
comparing BSHD (ppo_pack_seq=False, pad-to-max) against THD
(ppo_pack_seq=True, pack-seq). Both run 1 global step on the same
data; loss and grad_norm should be close.

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 tests/test_gpatch_v4/test_dsv4_sft_thd.py
"""

import shutil
import unittest

import pytest

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.trainer import FinetuneTrainer
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class TestDsv4SftThd(unittest.IsolatedAsyncioTestCase):

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _run_one_step(self, pack_seq: bool, enable_mtp: bool = False) -> dict:
        config = load_config("test_dsv4_sft_thd", FinetuneConfig)
        config.training.exit_step = 1
        config.training.enable_mtp = enable_mtp
        config.policy.ppo_pack_seq = pack_seq
        if not pack_seq:
            config.training.seq_length = 4096
            config.debug.experimental_pad_to_max_length = True

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
        try:
            trainer = FinetuneTrainer()
            metrics = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)
            shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        return metrics

    @pytest.mark.timeout(2400)
    async def test_thd_vs_bshd(self):
        """BSHD (pad-to-max) vs THD (pack-seq): loss/grad 接近。"""

        # 2. THD pack-seq
        metrics_thd = await self._run_one_step(pack_seq=True)
        self.tearDown()

        # 1. BSHD baseline
        metrics_bshd = await self._run_one_step(pack_seq=False)

        def _extract(metrics):
            assert metrics is not None
            assert len(metrics) >= 1
            dp0 = metrics[0]
            assert len(dp0) >= 1
            step0 = dp0[0]
            loss = step0.get("finetune/lm_loss", step0.get("finetune/total_loss"))
            grad_norm = step0.get("finetune/grad_norm")
            return loss, grad_norm

        loss_bshd, gn_bshd = _extract(metrics_bshd)
        loss_thd, gn_thd = _extract(metrics_thd)

        print("\n" + "=" * 60)
        print("  DSV4 SFT: BSHD vs THD pack-seq")
        print("=" * 60)
        print(f"  BSHD: loss={loss_bshd:.6f}  grad_norm={gn_bshd:.6f}")
        print(f"  THD:  loss={loss_thd:.6f}  grad_norm={gn_thd:.6f}")

        loss_rel = abs(loss_bshd - loss_thd) / (abs(loss_bshd) + 1e-8)
        gn_rel = abs(gn_bshd - gn_thd) / (abs(gn_bshd) + 1e-8)
        print(f"  loss rel_diff={loss_rel:.6f}  grad_norm rel_diff={gn_rel:.6f}")
        print("=" * 60 + "\n")

        self.assertLess(loss_rel, 0.01, f"loss rel_diff {loss_rel:.6f} > 1%")
        self.assertLess(gn_rel, 0.05, f"grad_norm rel_diff {gn_rel:.6f} > 5%")

    @pytest.mark.timeout(2400)
    async def test_thd_mtp_vs_bshd_mtp(self):
        """BSHD+MTP vs THD+MTP: total_loss/grad 接近。"""

        metrics_thd = await self._run_one_step(pack_seq=True, enable_mtp=True)
        self.tearDown()

        metrics_bshd = await self._run_one_step(pack_seq=False, enable_mtp=True)

        def _extract(metrics):
            assert metrics is not None
            assert len(metrics) >= 1
            dp0 = metrics[0]
            assert len(dp0) >= 1
            step0 = dp0[0]
            loss = step0.get("finetune/total_loss")
            grad_norm = step0.get("finetune/grad_norm")
            return loss, grad_norm

        loss_bshd, gn_bshd = _extract(metrics_bshd)
        loss_thd, gn_thd = _extract(metrics_thd)

        print("\n" + "=" * 60)
        print("  DSV4 SFT+MTP: BSHD vs THD pack-seq")
        print("=" * 60)
        print(f"  BSHD+MTP: loss={loss_bshd:.6f}  grad_norm={gn_bshd:.6f}")
        print(f"  THD+MTP:  loss={loss_thd:.6f}  grad_norm={gn_thd:.6f}")

        loss_rel = abs(loss_bshd - loss_thd) / (abs(loss_bshd) + 1e-8)
        gn_rel = abs(gn_bshd - gn_thd) / (abs(gn_bshd) + 1e-8)
        print(f"  loss rel_diff={loss_rel:.6f}  grad_norm rel_diff={gn_rel:.6f}")
        print("=" * 60 + "\n")

        self.assertLess(loss_rel, 0.01, f"loss rel_diff {loss_rel:.6f} > 1%")
        self.assertLess(gn_rel, 0.05, f"grad_norm rel_diff {gn_rel:.6f} > 5%")


if __name__ == "__main__":
    unittest.main()
