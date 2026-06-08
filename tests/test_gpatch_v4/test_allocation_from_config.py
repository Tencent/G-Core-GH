"""Unit tests for :func:`gpatch_v4.orches.resource_allocator.allocation_from_config`."""

import unittest

from gpatch_v4.configs.config import (
    DpoConfig,
    EvaluateConfig,
    FinetuneConfig,
    InferenceConfig,
    OffPolicyDistillConfig,
    OnPolicyDistillConfig,
    RlConfig,
    T2iRlConfig,
)
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.kv_config import KvConfig
from gpatch_v4.configs.policy_config import BasePolicyConfig
from gpatch_v4.orches.resource_allocator import allocation_from_config


def _dist_config(nnodes=1, gpus_per_node=8, tp=1, pp=1, cp=1, ep=1):
    """Shorthand for constructing a DistConfig."""
    return DistConfig(
        nnodes=nnodes,
        num_gpus_per_node=gpus_per_node,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
    )


class TestAllocationFromConfig(unittest.TestCase):
    """Coverage of every config type accepted by allocation_from_config."""
    def test_rl_disagg(self):
        """RlConfig disagg: policy/sampler/gen_rm/bt_rm/training_plt 都应出现在 allocation 中。"""
        cfg = RlConfig(placement_type="disaggregated")
        # 期望 gen_rm 真正分配 1 个节点，必须显式打开 use_gen_rm_reward（默认 False）。
        # bt_rm.nnodes=0 时开关保持默认 False 即可。
        cfg.training.use_gen_rm_reward = True
        cfg.policy.dist_config = _dist_config(nnodes=2, gpus_per_node=8, tp=4, pp=1)
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=4, tp=2)
        cfg.gen_rm.dist_config = _dist_config(nnodes=1, gpus_per_node=2, tp=2)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)

        alloc = allocation_from_config(cfg)
        assert alloc.role_nnodes == {
            "policy": 2,
            "training_plt": 1,
            "sampler": 1,
            "gen_rm": 1,
            "bt_rm": 0,
        }
        assert alloc.role_gpus_per_node["policy"] == 8
        assert alloc.role_gpus_per_node["sampler"] == 4
        assert alloc.role_gpus_per_node["gen_rm"] == 2
        assert alloc.role_min_gpus_per_replica["policy"] == 4
        assert alloc.role_min_gpus_per_replica["sampler"] == 2
        assert alloc.role_min_gpus_per_replica["gen_rm"] == 2
        assert alloc.role_min_gpus_per_replica["training_plt"] == 1
        # kv 没有字段（RlConfig 不含 kv），不应出现。
        assert "kv" not in alloc.role_nnodes

    def test_rl_colocate_does_not_inherit_policy(self):
        """colocate 下 allocation 应忠实记录 sampler 自身 nnodes，不被改写成 policy 的值。"""
        cfg = RlConfig(placement_type="colocate")
        # 期望 gen_rm/bt_rm 真正分配节点，必须显式打开（开关默认 False 会把 nnodes 清零）。
        cfg.training.use_gen_rm_reward = True
        cfg.training.use_bt_rm_reward = True
        cfg.policy.dist_config = _dist_config(nnodes=4, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=4, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=2, gpus_per_node=2)
        cfg.bt_rm.dist_config = _dist_config(nnodes=1, gpus_per_node=1)

        alloc = allocation_from_config(cfg)
        # colocate 语义由 placement_group.py 体现，allocation 本身不"继承"。
        assert alloc.role_nnodes["sampler"] == 4
        assert alloc.role_nnodes["gen_rm"] == 2
        assert alloc.role_nnodes["bt_rm"] == 1

    def test_finetune(self):
        """FinetuneConfig: 只有 policy / training_plt，sampler/gen_rm/bt_rm 均不存在。"""
        cfg = FinetuneConfig(placement_type="colocate")
        cfg.policy.dist_config = _dist_config(nnodes=1, gpus_per_node=4)

        alloc = allocation_from_config(cfg)
        assert alloc.role_nnodes == {"policy": 1, "training_plt": 1}
        assert "sampler" not in alloc.role_nnodes
        assert "gen_rm" not in alloc.role_nnodes
        assert "bt_rm" not in alloc.role_nnodes

    def test_dpo_has_no_sampler(self):
        """DpoConfig 也没有 sampler / gen_rm / bt_rm。"""
        cfg = DpoConfig(placement_type="colocate")
        cfg.policy.dist_config = _dist_config(nnodes=1, gpus_per_node=4)

        alloc = allocation_from_config(cfg)
        assert "sampler" not in alloc.role_nnodes
        assert "gen_rm" not in alloc.role_nnodes

    def test_evaluate_has_sampler(self):
        """EvaluateConfig 有 sampler，没有 gen_rm / bt_rm。"""
        cfg = EvaluateConfig(placement_type="disaggregated")
        cfg.policy.dist_config = _dist_config(nnodes=1, gpus_per_node=4)
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=2)

        alloc = allocation_from_config(cfg)
        assert "sampler" in alloc.role_nnodes
        assert alloc.role_nnodes["sampler"] == 1
        assert "gen_rm" not in alloc.role_nnodes
        assert "bt_rm" not in alloc.role_nnodes

    def test_inference(self):
        """InferenceConfig: 只有 sampler。"""
        cfg = InferenceConfig()
        cfg.sampler.dist_config = _dist_config(nnodes=2, gpus_per_node=8, tp=2, pp=2)

        alloc = allocation_from_config(cfg)
        assert alloc.role_nnodes == {"sampler": 2}
        assert alloc.role_gpus_per_node == {"sampler": 8}
        assert alloc.role_min_gpus_per_replica == {"sampler": 4}

    def test_on_policy_distill_multi_teacher(self):
        """OnPolicyDistillConfig: teacher_{name} 每个 teacher 作为独立条目出现。
        Teacher 跑 Megatron 算 logps，走 training 公式（tp*pp*min(cp,ep)）。"""
        cfg = OnPolicyDistillConfig(placement_type="disaggregated")
        cfg.policy.dist_config = _dist_config(nnodes=2, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=1, gpus_per_node=2)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.teachers = {
            "math":
                BasePolicyConfig(dist_config=_dist_config(nnodes=1, gpus_per_node=8, tp=4, pp=1)),
            "code":
                BasePolicyConfig(dist_config=_dist_config(nnodes=1, gpus_per_node=4, tp=2, pp=1)),
        }

        alloc = allocation_from_config(cfg)
        assert "teacher_math" in alloc.role_nnodes
        assert "teacher_code" in alloc.role_nnodes
        assert alloc.role_nnodes["teacher_math"] == 1
        assert alloc.role_gpus_per_node["teacher_math"] == 8
        # CP/EP 默认 1 -> min(1,1)=1 -> min_gpus = tp*pp*1 = 4 / 2
        assert alloc.role_min_gpus_per_replica["teacher_math"] == 4
        assert alloc.role_gpus_per_node["teacher_code"] == 4
        assert alloc.role_min_gpus_per_replica["teacher_code"] == 2

    def test_off_policy_distill_has_teacher(self):
        """OffPolicyDistillConfig: role 名就是 ``teacher``（单数），不加后缀。"""
        cfg = OffPolicyDistillConfig(placement_type="colocate")
        cfg.policy.dist_config = _dist_config(nnodes=2, gpus_per_node=8)
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.teacher.dist_config = _dist_config(nnodes=1, gpus_per_node=8, tp=2, pp=2)

        alloc = allocation_from_config(cfg)
        assert "teacher" in alloc.role_nnodes
        assert alloc.role_nnodes["teacher"] == 1
        assert alloc.role_min_gpus_per_replica["teacher"] == 4
        # teacher_{name} 不应出现。
        assert not any(r.startswith("teacher_") for r in alloc.role_nnodes)

    def test_t2i_rl_no_sampler(self):
        """T2iRlConfig 没有 sampler；有 gen_rm / bt_rm。"""
        cfg = T2iRlConfig(placement_type="colocate")
        cfg.policy.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)

        alloc = allocation_from_config(cfg)
        assert "sampler" not in alloc.role_nnodes
        assert "gen_rm" in alloc.role_nnodes
        assert "bt_rm" in alloc.role_nnodes

    def test_min_gpus_per_replica_training_workload(self):
        """训练 workload（policy / bt_rm / teacher，跑 megatron/fsdp）的
        per-replica 最小 GPU = tp * pp * min(cp, ep)；较大的 cp/ep 跨 DP 复制，
        不进入单副本最小占用。"""
        cfg = OnPolicyDistillConfig(placement_type="disaggregated")
        # policy tp=2, pp=2, cp=4, ep=2 -> min = 2*2*min(4,2) = 8
        cfg.policy.dist_config = _dist_config(
            nnodes=2,
            gpus_per_node=8,
            tp=2,
            pp=2,
            cp=4,
            ep=2,
        )
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        cfg.gen_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)
        # bt_rm tp=2, pp=1, cp=2, ep=1 -> min = 2*1*min(2,1) = 2 (FSDP)
        cfg.bt_rm.dist_config = _dist_config(
            nnodes=1,
            gpus_per_node=8,
            tp=2,
            pp=1,
            cp=2,
            ep=1,
        )
        # teacher tp=2, pp=2, cp=2, ep=4 -> min = 2*2*min(2,4) = 8 (Megatron logps)
        cfg.teachers = {
            "math":
                BasePolicyConfig(
                    dist_config=_dist_config(
                        nnodes=1,
                        gpus_per_node=8,
                        tp=2,
                        pp=2,
                        cp=2,
                        ep=4,
                    )
                ),
        }

        alloc = allocation_from_config(cfg)
        assert alloc.role_min_gpus_per_replica["policy"] == 8
        assert alloc.role_min_gpus_per_replica["bt_rm"] == 2
        assert alloc.role_min_gpus_per_replica["teacher_math"] == 8

    def test_min_gpus_per_replica_generate_workload(self):
        """生成 workload（sampler / gen_rm / teacher 跑 sglang/vllm）的
        per-replica 最小 GPU = tp * pp；EP 被推理引擎藏在 TP/PP 内，
        CP 对生成无意义。"""
        cfg = RlConfig(placement_type="disaggregated")
        cfg.policy.dist_config = _dist_config(nnodes=1, gpus_per_node=8)
        # sampler tp=2, pp=2, cp=4, ep=2 -> min 应为 tp*pp = 4，不是 4*min(4,2)=8。
        cfg.sampler.dist_config = _dist_config(
            nnodes=1,
            gpus_per_node=8,
            tp=2,
            pp=2,
            cp=4,
            ep=2,
        )
        cfg.gen_rm.dist_config = _dist_config(
            nnodes=1,
            gpus_per_node=8,
            tp=4,
            pp=1,
            cp=2,
            ep=4,
        )
        cfg.bt_rm.dist_config = _dist_config(nnodes=0, gpus_per_node=8)

        alloc = allocation_from_config(cfg)
        assert alloc.role_min_gpus_per_replica["sampler"] == 4
        assert alloc.role_min_gpus_per_replica["gen_rm"] == 4

    def test_evaluate_policy_uses_generate_formula(self):
        """EvaluateConfig 的 ``policy`` 角色名虽叫 policy，实际跑的是生成 workload
        （sglang/vllm），所以 per-replica 最小 GPU 必须按 tp*pp 计算，
        不带 cp/ep 系数。"""
        cfg = EvaluateConfig(placement_type="disaggregated")
        # tp=2, pp=2, cp=4, ep=2 —— 若按 training 公式 = 8，按 generate 公式 = 4。
        cfg.policy.dist_config = _dist_config(
            nnodes=1,
            gpus_per_node=8,
            tp=2,
            pp=2,
            cp=4,
            ep=2,
        )
        cfg.sampler.dist_config = _dist_config(nnodes=1, gpus_per_node=8)

        alloc = allocation_from_config(cfg)
        assert alloc.role_min_gpus_per_replica["policy"] == 4


if __name__ == "__main__":
    unittest.main()
