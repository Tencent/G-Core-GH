"""Unit tests for :func:`gpatch_v4.orches.placement_group.create_placement_groups`.

Tests stub out :func:`_create_placement_group` (the only ray-touching
helper) so ``allocation_from_config`` runs against real configs, and
the observed call arguments plus the returned ``groups`` slice lengths
fully characterize the behavior under test.
"""

import unittest
from unittest.mock import MagicMock, patch

from gpatch_v4.configs.config import (
    OffPolicyDistillConfig,
    OnPolicyDistillConfig,
    RlConfig,
)
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.policy_config import BasePolicyConfig


def _dist_config(nnodes=1, gpus_per_node=8, tp=1, pp=1):
    return DistConfig(
        nnodes=nnodes,
        num_gpus_per_node=gpus_per_node,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
    )


class _PGFixture:
    """Helper to stub ``_create_placement_group`` and capture its args."""
    def __init__(self):
        self.received_num_gpus = None
        self.fake_pg = MagicMock(name="fake_pg")

    def __call__(self, num_gpus):
        self.received_num_gpus = num_gpus
        # Identity bundle indices — slice lengths reflect GPU counts.
        return (self.fake_pg, list(range(num_gpus)))


class TestCreatePlacementGroupsDisagg(unittest.TestCase):
    """Disaggregated placement: each role occupies distinct bundles."""
    def _build_rl_config(self):
        cfg = RlConfig(placement_type="disaggregated")
        # 期望 gen_rm 真正分配 1 个节点，必须显式打开 use_gen_rm_reward
        # （RLTrainingConfig 默认 False 会让 allocation_from_config 把 gen_rm
        #  的 nnodes 强制清零，下游 placement_group 算到的总 GPU 数也会少 2）。
        cfg.training.use_gen_rm_reward = True
        cfg.policy.dist_config = _dist_config(nnodes=2, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=4)
        cfg.gen_rm.dist_config = _dist_config(nnodes=1, gpus_per_node=2)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        return cfg

    def test_num_gpus_and_slice_lengths(self):
        """disagg RL: num_gpus = policy+sampler+gen_rm+bt_rm（不含 kv/training_plt）。
        各 role 切片长度应与对应 role 的 num_gpus 相等。"""
        from gpatch_v4.orches import placement_group as pg_mod

        cfg = self._build_rl_config()
        fixture = _PGFixture()
        with patch.object(pg_mod, "_create_placement_group", side_effect=fixture):
            groups = pg_mod.create_placement_groups(cfg)

        # policy(16) + sampler(4) + gen_rm(2) + bt_rm(0)
        assert fixture.received_num_gpus == 22
        assert len(groups["policy"][1]) == 16
        assert len(groups["sampler"][1]) == 4
        assert len(groups["gen_rm"][1]) == 2
        assert len(groups["bt_rm"][1]) == 0
        # training_plt 始终切 pg[0:1]，不依赖 disagg/colocate。
        assert len(groups["training_plt"][1]) == 1


class TestCreatePlacementGroupsColocate(unittest.TestCase):
    """Colocate: sampler/gen_rm/bt_rm/off-policy teacher 共享整个 policy 区间。"""
    def _build_rl_colocate(self):
        cfg = RlConfig(placement_type="colocate")
        cfg.policy.dist_config = _dist_config(nnodes=2, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=2, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=1, gpus_per_node=2)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        return cfg

    def test_num_gpus_equals_policy(self):
        """colocate: num_gpus 必须 = policy_num_gpus（其它 role 共享，不累加）。"""
        from gpatch_v4.orches import placement_group as pg_mod

        cfg = self._build_rl_colocate()
        fixture = _PGFixture()
        with patch.object(pg_mod, "_create_placement_group", side_effect=fixture):
            groups = pg_mod.create_placement_groups(cfg)

        policy_num_gpus = 16
        assert fixture.received_num_gpus == policy_num_gpus
        # colocate: sampler/gen_rm/bt_rm 切片全长 = policy 全长。
        assert len(groups["policy"][1]) == policy_num_gpus
        assert len(groups["sampler"][1]) == policy_num_gpus
        assert len(groups["gen_rm"][1]) == policy_num_gpus
        assert len(groups["bt_rm"][1]) == policy_num_gpus

    def test_offpolicy_colocate_teacher_equals_policy(self):
        """off-policy colocate: 即使 teacher_nnodes < policy_nnodes，
        ``groups["teacher"]`` 切片长度仍 == policy_num_gpus（共享整个 policy 区间）。"""
        from gpatch_v4.orches import placement_group as pg_mod

        cfg = OffPolicyDistillConfig(placement_type="colocate")
        cfg.policy.dist_config = _dist_config(nnodes=2, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=2, gpus_per_node=8)
        cfg.teacher.dist_config = _dist_config(nnodes=1, gpus_per_node=8)

        fixture = _PGFixture()
        with patch.object(pg_mod, "_create_placement_group", side_effect=fixture):
            groups = pg_mod.create_placement_groups(cfg)

        assert len(groups["teacher"][1]) == 16  # = policy_num_gpus

    def test_colocate_teacher_exceeds_policy_asserts(self):
        """colocate 下 Σ teacher_nnodes > policy_nnodes 时 create_placement_groups
        必须 AssertionError（保持现状前置不变式）。"""
        from gpatch_v4.orches import placement_group as pg_mod

        cfg = OnPolicyDistillConfig(placement_type="colocate")
        cfg.policy.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.teachers = {
            "a": BasePolicyConfig(dist_config=_dist_config(nnodes=1, gpus_per_node=8)),
            "b": BasePolicyConfig(dist_config=_dist_config(nnodes=1, gpus_per_node=8)),
        }

        fixture = _PGFixture()
        with patch.object(pg_mod, "_create_placement_group", side_effect=fixture):
            with self.assertRaises(AssertionError):
                pg_mod.create_placement_groups(cfg)


class TestCreatePlacementGroupsOnPolicyMultiTeacher(unittest.TestCase):
    """On-policy distill: teacher_{name} 逐个递进切前缀。"""
    def test_multiple_teachers_offsets_advance(self):
        """2 个 teacher 各占 1 个节点；切片 offset 应递进，互不重叠。"""
        from gpatch_v4.orches import placement_group as pg_mod

        cfg = OnPolicyDistillConfig(placement_type="disaggregated")
        cfg.policy.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.teachers = {
            "math": BasePolicyConfig(dist_config=_dist_config(nnodes=1, gpus_per_node=8)),
            "code": BasePolicyConfig(dist_config=_dist_config(nnodes=1, gpus_per_node=8)),
        }

        fixture = _PGFixture()
        with patch.object(pg_mod, "_create_placement_group", side_effect=fixture):
            groups = pg_mod.create_placement_groups(cfg)

        # teacher_num_gpus = (1 + 1) * policy_gpus_per_node = 16
        assert fixture.received_num_gpus == 8 + 8 + 16
        math_slice = groups["teacher_math"][1]
        code_slice = groups["teacher_code"][1]
        assert len(math_slice) == 8
        assert len(code_slice) == 8
        # No overlap.
        assert set(math_slice).isdisjoint(set(code_slice))


class TestTeacherUsesPolicyGpusPerNode(unittest.TestCase):
    """Regression: teacher / kv 切片长度使用 **policy 的** num_gpus_per_node，
    即使 teacher/kv 自己 dist_config 里声明的 num_gpus_per_node 不同。"""
    def test_teacher_uses_policy_gpn(self):
        """teacher.dist_config.num_gpus_per_node = 4，policy = 8；
        teacher 切片长度应是 teacher_nnodes * 8 = 8，而不是 * 4 = 4。"""
        from gpatch_v4.orches import placement_group as pg_mod

        cfg = OnPolicyDistillConfig(placement_type="disaggregated")
        cfg.policy.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.teachers = {
            "t": BasePolicyConfig(dist_config=_dist_config(nnodes=1, gpus_per_node=4)),
        }

        fixture = _PGFixture()
        with patch.object(pg_mod, "_create_placement_group", side_effect=fixture):
            groups = pg_mod.create_placement_groups(cfg)

        # teacher_t slice width = 1 * 8 (policy gpn), NOT 1 * 4 (teacher gpn).
        assert len(groups["teacher_t"][1]) == 8


class TestKvTrainingPltSharePrefix(unittest.TestCase):
    """kv / training_plt 永远切 pg 前缀，且不计入总 num_gpus。"""
    def test_training_plt_slice_is_zero_to_one(self):
        """training_plt 切片永远是 pg[0:1]。"""
        from gpatch_v4.orches import placement_group as pg_mod

        cfg = RlConfig(placement_type="disaggregated")
        cfg.policy.dist_config = _dist_config(nnodes=1, gpus_per_node=4)
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=2)
        cfg.gen_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)

        fixture = _PGFixture()
        with patch.object(pg_mod, "_create_placement_group", side_effect=fixture):
            groups = pg_mod.create_placement_groups(cfg)

        assert groups["training_plt"][1] == [0]
        # training_plt 没加到 num_gpus 总和里（policy+sampler = 6）。
        assert fixture.received_num_gpus == 6


if __name__ == "__main__":
    unittest.main()
