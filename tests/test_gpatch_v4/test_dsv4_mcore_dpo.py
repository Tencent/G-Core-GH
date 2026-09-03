import unittest

import torch

from gpatch_v4.configs.config import DpoConfig
from gpatch_v4.trainer import DpoTrainer
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


class TestDSv4McoreDpo(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    @staticmethod
    def _configure_one_step(config: DpoConfig) -> None:
        """Make the smoke test independent of persisted checkpoint state."""
        config.training.exit_step = 1
        config.training.auto_load_from_save_ckpt = False
        config.debug.disable_save_checkpoint = True

    async def test_cp8_thd_policy_equals_reference_loss(self):
        """Contiguous CP THD DPO keeps chosen/rejected log-probs aligned."""
        config = load_config("test_dsv4_mcore_dpo", DpoConfig)
        self._configure_one_step(config)

        metrics_list = await DpoTrainer().launch_then_run_with_recovery(config)
        expected_loss = torch.log(torch.tensor(2.0)).item()
        for metrics in metrics_list:
            for metric in metrics:
                assert abs(metric["finetune/dpo-metrics/loss"] - expected_loss) < 0.01, metric
