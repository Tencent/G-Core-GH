"""Unit tests for ``_get_expected_nnodes`` after removing the hostfile dependency."""

import unittest
from unittest.mock import MagicMock

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.configs.dist_config import DistConfig


def _dist_config(nnodes=1, gpus_per_node=8, tp=1, pp=1):
    return DistConfig(
        nnodes=nnodes,
        num_gpus_per_node=gpus_per_node,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
    )


class _Dummy:
    """Bare stand-in that lets us call the mixin method bound to *self*."""

    pass


class TestExpectedNnodes(unittest.TestCase):
    """Validate aggregation semantics of :meth:`TrainerRetryMixin._get_expected_nnodes`."""
    def _get_expected_nnodes(self, config):
        # Avoid pulling in the full trainer hierarchy by calling the unbound
        # method on a minimal stand-in.
        from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin
        return TrainerRetryMixin._get_expected_nnodes(_Dummy(), config)

    def test_colocate_returns_max(self):
        """colocate 下各 role 共用节点，expected_nnodes = max(role_nnodes)。"""
        cfg = RlConfig(placement_type="colocate")
        cfg.policy.dist_config = _dist_config(nnodes=4, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=4, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)

        assert self._get_expected_nnodes(cfg) == 4

    def test_disagg_sums_roles(self):
        """disagg 下各 role 独占节点，expected_nnodes = Σ role_nnodes（不含 shared）。"""
        cfg = RlConfig(placement_type="disaggregated")
        # gen_rm 占 1 个节点，必须同时打开 use_gen_rm_reward。
        cfg.training.use_gen_rm_reward = True
        cfg.policy.dist_config = _dist_config(nnodes=4, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=2, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)

        assert self._get_expected_nnodes(cfg) == 7

    def test_excludes_training_plt(self):
        """training_plt 共享 policy 前缀（nnodes=1），不应影响总节点数。"""
        cfg = RlConfig(placement_type="disaggregated")
        cfg.policy.dist_config = _dist_config(nnodes=4, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=2, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)

        # 结果应为 policy(4) + sampler(2) = 6，而非 7（+ training_plt）。
        assert self._get_expected_nnodes(cfg) == 6


if __name__ == "__main__":
    unittest.main()
