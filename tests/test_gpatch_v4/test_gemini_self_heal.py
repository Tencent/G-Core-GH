"""Tests for GeminiNodeReplacer in the training retry loop.

Mocks: crash (via RayActorError) and abnormal-node detection API only.
All other Gemini APIs (tag_recovery, exec_cmd, check_recovery) use real calls.
Must run on a Gemini MPI launcher pod (for env vars / hostfile).

Run:  python -m pytest tests/test_gpatch_v4/test_gemini_self_heal.py -v -s
"""

import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from gpatch_v4.orches.exceptions import RayActorError
from gpatch_v4.orches.gemini_self_heal import SelfHealExitError
from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin

_ON_GEMINI = bool(
    os.environ.get("__SYS_TASK_RUNTIME_ID__") and
    os.environ.get("__SYS_JOB_INSTANCE_SIGNATURE__") and os.environ.get("__HOST_IP__") and
    os.path.isfile("/root/hostfile")
)

_real_post = requests.Session.post


def _mock_abnormal_nodes(abnormal_data: dict):
    """Only mock get_task_abnormal_nodes; all other APIs use real calls."""
    def _patched(self, url, **kw):
        if "get_task_abnormal_nodes" in url:
            r = MagicMock()
            r.status_code = 200
            r.raise_for_status = MagicMock()
            r.json.return_value = {"data": abnormal_data, "errcode": 0, "errmsg": ""}
            return r
        return _real_post(self, url, **kw)

    return patch.object(requests.Session, "post", _patched)


class _MockTrainGroup:
    """Train group that crashes immediately on train_loop."""
    async def train_loop(self):
        raise RayActorError()

    async def check_liveness(self):
        await asyncio.sleep(9999)


class _CrashingTrainer(TrainerRetryMixin):
    """Trainer whose launch always succeeds but train_loop always crashes."""
    async def launch(self, config):
        self.train_group = _MockTrainGroup()


def _load_config():
    from gpatch_v4.configs.config import FinetuneConfig
    from gpatch_v4_test_helper import load_config

    cfg = load_config("test_launch_retry_sft", FinetuneConfig)
    cfg.policy.dist_config.nnodes = 4
    cfg.training.train_gbs = 32
    cfg.training.enable_self_heal = True
    cfg.training.max_restart_attempts = 1
    cfg.training.__post_init__()
    return cfg


def _get_worker_ips() -> list:
    """Get all worker IPs from Ray cluster (exclude launcher)."""
    import ray
    launcher_ip = os.environ["__HOST_IP__"]
    ray.init("auto", namespace="train", log_to_driver=False, ignore_reinit_error=True)
    ips = sorted(
        n["NodeManagerAddress"]
        for n in ray.nodes() if n.get("Alive") and n["NodeManagerAddress"] != launcher_ip
    )
    ray.shutdown()
    return ips


@unittest.skipUnless(_ON_GEMINI, "Not on Gemini launcher pod")
class TestSelfHealE2E(unittest.IsolatedAsyncioTestCase):
    async def test_launcher_bad_gpu(self):
        """Launcher IP in abnormal → SelfHealExitError, no pod replacement."""
        cfg = _load_config()
        launcher_ip = os.environ["__HOST_IP__"]
        old_worker_ips = _get_worker_ips()

        with _mock_abnormal_nodes({launcher_ip: "GPU ECC error"}):
            with self.assertRaises(SelfHealExitError):
                await _CrashingTrainer().launch_then_run_with_recovery(cfg)

        new_worker_ips = _get_worker_ips()
        self.assertEqual(
            old_worker_ips,
            new_worker_ips,
            "Worker IPs should NOT change when launcher exits",
        )

    async def test_worker_bad_gpu(self):
        """Worker IP in abnormal → real exec_cmd + tag_recovery + recovery.
        Asserts that the bad worker IP is no longer in the cluster."""
        cfg = _load_config()
        cfg.training.self_heal_config.recovery_max_wait = 600
        old_worker_ips = _get_worker_ips()
        bad_ip = old_worker_ips[0]

        with _mock_abnormal_nodes({bad_ip: "GPU ECC error"}):
            with self.assertRaises(RuntimeError):
                await _CrashingTrainer().launch_then_run_with_recovery(cfg)

        new_worker_ips = _get_worker_ips()
        self.assertNotIn(
            bad_ip,
            new_worker_ips,
            f"Bad worker IP {bad_ip} should be gone after replacement, "
            f"got: {new_worker_ips}",
        )

    async def test_no_bad_gpu(self):
        """No abnormal nodes → no restart, no replacement."""
        cfg = _load_config()
        cfg.training.max_restart_attempts = 0
        old_worker_ips = _get_worker_ips()

        with _mock_abnormal_nodes({}):
            with self.assertRaises(RuntimeError):
                await _CrashingTrainer().launch_then_run_with_recovery(cfg)

        new_worker_ips = _get_worker_ips()
        self.assertEqual(
            old_worker_ips,
            new_worker_ips,
            "Worker IPs should NOT change when no bad GPU detected",
        )


if __name__ == "__main__":
    unittest.main()
