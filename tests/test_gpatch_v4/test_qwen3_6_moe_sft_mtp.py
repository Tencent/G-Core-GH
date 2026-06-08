"""Text-only SFT one-step smoke test with online MTP SFT enabled.

Runs a single finetune step (``exit_step: 1``, ``lr: 0``) on
Qwen3.6-35B-A3B (MoE, ``model_arch: qwen3_5_moe``) with both
``enable_mtp`` and ``online_mtp_sft`` turned on. This exercises the SFT
path that now feeds ``labels``/``loss_mask`` into the model forward so
``process_mtp_loss`` actually computes the MTP loss instead of early
returning on ``labels is None``.

Manual-run only -- not in CI. Requires 8 GPUs (TP=2, PP=2, EP=4) and the
Qwen3.6-35B-A3B checkpoint downloaded into ``hf-hub/``.

Usage::

    cd /work/wepsdl/gcore_mtp
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore_mtp:$RCDIR/gcore_mtp/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=1800 \\
        tests/test_gpatch_v4/test_qwen3_6_moe_sft_mtp.py
"""

import os
import shutil
import unittest

import ray

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.trainer import FinetuneTrainer
from gpatch_v4_test_helper import load_config


class TestQwen36MoESftMtp(unittest.IsolatedAsyncioTestCase):
    """Qwen3.6-35B-A3B text-only SFT with online MTP SFT, one step."""
    def setUp(self):
        pass

    def tearDown(self):
        ray.shutdown()

    async def test_train_one_step_with_mtp(self):
        config = load_config("test_qwen3_6_moe_sft_mtp", FinetuneConfig)

        assert config.training.enable_mtp, "enable_mtp must be on for this test"
        assert config.training.online_mtp_sft, "online_mtp_sft must be on for this test"

        save_path = config.checkpoint.save_ckpt_path
        latest_ckpt_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_ckpt_path):
            os.remove(latest_ckpt_path)
        if os.path.exists(os.path.join(save_path, "hf")):
            shutil.rmtree(os.path.join(save_path, "hf"))

        try:
            trainer = FinetuneTrainer()
            await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(save_path, ignore_errors=True)
