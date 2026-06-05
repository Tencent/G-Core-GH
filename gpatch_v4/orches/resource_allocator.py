"""Per-role GPU resource allocation used to drive placement-group layout.

:class:`ResourceAllocation` is the single source of truth for how many
nodes / GPUs each role wants; :func:`allocation_from_config` derives it
from a top-level config.  :func:`~gpatch_v4.orches.placement_group.create_placement_groups`
and :meth:`~gpatch_v4.trainer.trainer_mixin.TrainerRetryMixin._get_expected_nnodes`
both consume it.

The dynamic-reallocation path (an allocator that proposes a new
allocation based on utilization metrics and triggers a graceful
restart) is intentionally out of scope for this module right now;
the abstract interface will be introduced together with its first
consumer.  :meth:`ResourceAllocation.validate` already encodes the
divisibility constraints an allocator must respect, so it will have a
caller the moment the allocator is wired up.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet

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

#: Roles that always share the ``policy`` placement-group prefix and
#: never occupy their own nodes.  Their GPU counts must *not* be summed
#: into the total placement-group size, and they are excluded from the
#: node-count aggregation in :func:`_get_expected_nnodes`.
SHARED_WITH_POLICY: FrozenSet[str] = frozenset({"kv", "training_plt"})

#: Roles whose workload is **training** (Megatron / FSDP).  Their
#: per-replica minimum GPU count is ``tp * pp * min(cp, ep)`` because a
#: single DP replica must span all model-parallel / context-parallel /
#: expert-parallel dimensions.
#:
#: - ``policy``: the student model, Megatron-trained.
#: - ``bt_rm``: a reward model running FSDP despite its "reward model" naming.
#: - teacher roles (``teacher`` / ``teacher_{name}``) run Megatron to compute
#:   log-probs for distillation.  Tested via :func:`_is_teacher_role` since
#:   their names are dynamic.
_TRAINING_ROLES: FrozenSet[str] = frozenset({"policy", "bt_rm"})

#: Roles whose workload is **generation** (SGLang / vLLM).  Their
#: per-replica minimum GPU count is ``tp * pp``.  EP is absorbed inside
#: the inference engine's TP/PP rather than living as a separate axis,
#: and CP has no meaning for single-request decoding.
_GENERATE_ROLES: FrozenSet[str] = frozenset({"sampler", "gen_rm"})

#: Roles that must declare an explicit per-replica minimal GPU count.
#: ``kv`` / ``training_plt`` default to 1 when absent.
_MIN_GPUS_REQUIRED_ROLES: FrozenSet[str] = _TRAINING_ROLES | _GENERATE_ROLES


def _is_teacher_role(role: str) -> bool:
    """Return True for off-policy ``"teacher"`` or on-policy ``"teacher_{name}"``."""
    return role == "teacher" or role.startswith("teacher_")


def _training_min_gpus_per_replica(dist_config) -> int:
    """Per-replica minimal GPU count for a **training** workload.

    A single DP replica must span at least ``tp * pp * min(cp, ep)`` GPUs:

    - ``tp * pp`` is the dense Megatron shard footprint.
    - ``cp`` (context-parallel) and ``ep`` (expert-model-parallel) are
      orthogonal dimensions that can coexist with ``tp * pp``.  Only
      the smaller of the two factors into a single replica's *minimum*
      because the larger one contributes to replication across DP
      groups rather than to one replica's shape.

    Still an approximation — ``expert_tensor_parallel_size`` and similar
    refinements aren't modelled yet.  Tighten here when a caller needs
    stricter accounting.
    """
    tp = dist_config.tensor_model_parallel_size
    pp = dist_config.pipeline_model_parallel_size
    cp = dist_config.context_parallel_size
    ep = dist_config.expert_model_parallel_size
    return tp * pp * min(cp, ep)


def _generate_min_gpus_per_replica(dist_config) -> int:
    """Per-replica minimal GPU count for a **generate** workload.

    SGLang / vLLM absorb EP inside TP/PP, so the per-replica minimum
    collapses to ``tp * pp``.  CP is a training-only concern.
    """
    return (dist_config.tensor_model_parallel_size * dist_config.pipeline_model_parallel_size)


def _min_gpus_per_replica(role: str, dist_config) -> int:
    """Dispatch to the workload-specific per-replica minimum.

    Parameters
    ----------
    role : str
        Must be in ``_TRAINING_ROLES`` / ``_GENERATE_ROLES``, or be a
        teacher role (training workload).
    dist_config : DistConfig

    Returns
    -------
    int
    """
    if role in _TRAINING_ROLES or _is_teacher_role(role):
        return _training_min_gpus_per_replica(dist_config)
    assert role in _GENERATE_ROLES, (f"role '{role}' has no declared workload")
    return _generate_min_gpus_per_replica(dist_config)


@dataclass
class ResourceAllocation:
    """Per-role GPU resource intent.

    This describes *what each role wants*, not *how the placement group
    is carved*.  Sharing relationships (colocate vs disaggregated, the
    ``SHARED_WITH_POLICY`` prefix) are encoded in the placement-group
    builder, not here.

    Attributes
    ----------
    role_nnodes : dict[str, int]
        Number of nodes requested per role.
    role_gpus_per_node : dict[str, int]
        Number of GPUs per node per role.
    role_min_gpus_per_replica : dict[str, int]
        Per-replica minimal GPU count per role.  See
        :func:`_per_replica_min_gpus`.  Roles with no model parallelism
        (e.g. ``kv``, ``training_plt``) default to 1.
    """

    role_nnodes: Dict[str, int] = field(default_factory=dict)
    role_gpus_per_node: Dict[str, int] = field(default_factory=dict)
    role_min_gpus_per_replica: Dict[str, int] = field(default_factory=dict)

    def role_num_gpus(self, role: str) -> int:
        """Return ``role_nnodes[role] * role_gpus_per_node[role]`` (0 if missing).

        Parameters
        ----------
        role : str

        Returns
        -------
        int
        """
        return self.role_nnodes.get(role, 0) * self.role_gpus_per_node.get(role, 0)

    def validate(self, train_gbs: int) -> bool:
        """Validate allocation constraints.

        Constraints
        -----------
        1. Each non-empty role's total GPU count must be a multiple of
           its per-replica minimum.
        2. ``policy`` / ``sampler`` / ``gen_rm`` / ``bt_rm`` and teacher
           roles must declare an explicit ``min_gpus_per_replica``.
           ``kv`` / ``training_plt`` default to 1 when absent.
        3. If ``policy`` is present, ``train_gbs`` must be divisible by
           the policy DP-size
           (``policy_num_gpus // policy_min_gpus_per_replica``).

        Parameters
        ----------
        train_gbs : int

        Returns
        -------
        bool

        Raises
        ------
        AssertionError
            If any constraint is violated.
        """
        for role, nn in self.role_nnodes.items():
            # Zero-GPU roles are allowed to appear (e.g. FinetuneConfig
            # keeps sampler absent entirely).  Skip validation for them.
            if nn == 0:
                continue
            if role in _MIN_GPUS_REQUIRED_ROLES or _is_teacher_role(role):
                assert role in self.role_min_gpus_per_replica, (
                    f"role '{role}' must declare min_gpus_per_replica"
                )
            min_gpus = self.role_min_gpus_per_replica.get(role, 1)
            assert self.role_num_gpus(role) % min_gpus == 0, (
                f"role '{role}': {self.role_num_gpus(role)} GPUs is not "
                f"divisible by min_gpus_per_replica {min_gpus}"
            )

        if "policy" in self.role_nnodes and self.role_nnodes["policy"] > 0:
            policy_min = self.role_min_gpus_per_replica["policy"]
            policy_dp = self.role_num_gpus("policy") // policy_min
            assert train_gbs % policy_dp == 0, (
                f"train_gbs={train_gbs} is not divisible by "
                f"policy DP-size={policy_dp}"
            )

        return True


def allocation_from_config(config) -> ResourceAllocation:
    """Build a :class:`ResourceAllocation` from a top-level config.

    Faithfully records each role's own ``dist_config``.  Does **not**
    express colocate/disagg sharing — that is the responsibility of the
    placement-group builder.

    Exception: ``training_plt`` is hardcoded to
    ``(nnodes=1, gpus_per_node=1, min_gpus_per_replica=1)`` because the
    placement group always reserves ``pg[0:1]`` for it regardless of any
    config.  ``kv`` / ``training_plt`` also hardcode
    ``min_gpus_per_replica = 1`` even though their ``DistConfig`` has
    tp/pp/cp/ep fields — neither role runs model parallelism.

    Parameters
    ----------
    config : object
        One of ``InferenceConfig``, ``EvaluateConfig``, ``FinetuneConfig``,
        ``DpoConfig``, ``RlConfig``, ``T2iRlConfig``, ``OnPolicyDistillConfig``,
        ``OffPolicyDistillConfig``.

    Returns
    -------
    ResourceAllocation
    """
    alloc = ResourceAllocation()

    if isinstance(config, InferenceConfig):
        dc = config.sampler.dist_config
        alloc.role_nnodes["sampler"] = dc.nnodes
        alloc.role_gpus_per_node["sampler"] = dc.num_gpus_per_node
        alloc.role_min_gpus_per_replica["sampler"] = _min_gpus_per_replica(
            "sampler",
            dc,
        )
        return alloc

    assert isinstance(
        config,
        (
            EvaluateConfig, FinetuneConfig, RlConfig, T2iRlConfig, OnPolicyDistillConfig,
            OffPolicyDistillConfig, DpoConfig
        ),
    ), f"unknown config {type(config).__name__}"

    # policy (always present).  In EvaluateConfig the "policy" role is
    # a lightweight coordinator (EvaluateActor) with no model parallelism
    # — it delegates generation to the sampler.  Use the generate formula
    # (tp * pp) which gives the correct result when tp=pp=1.
    policy_dc = config.policy.dist_config
    alloc.role_nnodes["policy"] = policy_dc.nnodes
    alloc.role_gpus_per_node["policy"] = policy_dc.num_gpus_per_node
    if isinstance(config, EvaluateConfig):
        alloc.role_min_gpus_per_replica["policy"] = _generate_min_gpus_per_replica(policy_dc, )
    else:
        alloc.role_min_gpus_per_replica["policy"] = _min_gpus_per_replica(
            "policy",
            policy_dc,
        )

    # training_plt: hardcoded shared-prefix bundle.
    alloc.role_nnodes["training_plt"] = 1
    alloc.role_gpus_per_node["training_plt"] = 1
    alloc.role_min_gpus_per_replica["training_plt"] = 1

    # kv (optional).  min_gpus_per_replica hardcoded to 1 —
    # we ignore dist_config.tp/pp/cp/ep.
    if hasattr(config, "kv"):
        kv_dc = config.kv.dist_config
        alloc.role_nnodes["kv"] = kv_dc.nnodes
        alloc.role_gpus_per_node["kv"] = kv_dc.num_gpus_per_node
        alloc.role_min_gpus_per_replica["kv"] = 1

    # sampler: absent in FinetuneConfig / T2iRlConfig / DpoConfig.
    if not isinstance(config, (FinetuneConfig, T2iRlConfig, DpoConfig)):
        sampler_dc = config.sampler.dist_config
        alloc.role_nnodes["sampler"] = sampler_dc.nnodes
        alloc.role_gpus_per_node["sampler"] = sampler_dc.num_gpus_per_node
        alloc.role_min_gpus_per_replica["sampler"] = _min_gpus_per_replica(
            "sampler",
            sampler_dc,
        )

    # gen_rm / bt_rm: absent in Finetune / OffPolicyDistill / Evaluate / Dpo.
    # 对其它 config，``gen_rm`` / ``bt_rm`` 这两个 *role 条目* 永远会出现在
    # allocation 里，以便下游（placement_group / _get_expected_nnodes /
    # validate）能用统一的 dict 接口查询。是否真正分配 GPU 由
    # ``training.use_{gen,bt}_rm_reward`` 开关决定 —— 开关关闭时把
    # ``nnodes`` 强制设为 0：
    #   - ``role_num_gpus`` 返回 0（不占用 placement group bundle）；
    #   - ``_get_expected_nnodes`` 已显式过滤 ``nn > 0``，自动跳过；
    #   - ``validate`` 已显式 ``if nn == 0: continue``，自动跳过。
    # 没有 ``training`` 字段（旧 config）时按旧行为默认启用，避免破坏既有 yaml。
    if not isinstance(
        config,
        (FinetuneConfig, OffPolicyDistillConfig, EvaluateConfig, DpoConfig),
    ):
        training_cfg = getattr(config, "training", None)
        use_gen_rm = getattr(training_cfg, "use_gen_rm_reward", True)
        use_bt_rm = getattr(training_cfg, "use_bt_rm_reward", True)

        gen_rm_dc = config.gen_rm.dist_config
        alloc.role_nnodes["gen_rm"] = gen_rm_dc.nnodes if use_gen_rm else 0
        alloc.role_gpus_per_node["gen_rm"] = gen_rm_dc.num_gpus_per_node
        alloc.role_min_gpus_per_replica["gen_rm"] = _min_gpus_per_replica(
            "gen_rm",
            gen_rm_dc,
        )

        bt_rm_dc = config.bt_rm.dist_config
        alloc.role_nnodes["bt_rm"] = bt_rm_dc.nnodes if use_bt_rm else 0
        alloc.role_gpus_per_node["bt_rm"] = bt_rm_dc.num_gpus_per_node
        alloc.role_min_gpus_per_replica["bt_rm"] = _min_gpus_per_replica(
            "bt_rm",
            bt_rm_dc,
        )

    # teachers — Megatron training workloads (compute logps).
    if isinstance(config, OnPolicyDistillConfig):
        # OnPolicyDistillConfig.__post_init__ normalizes teacher -> teachers,
        # but if neither was provided the attribute can still be None.
        teachers = config.teachers or {}
        for t_name, t_cfg in teachers.items():
            t_dc = t_cfg.dist_config
            role = f"teacher_{t_name}"
            alloc.role_nnodes[role] = t_dc.nnodes
            alloc.role_gpus_per_node[role] = t_dc.num_gpus_per_node
            alloc.role_min_gpus_per_replica[role] = _min_gpus_per_replica(role, t_dc)
    elif isinstance(config, OffPolicyDistillConfig):
        t_dc = config.teacher.dist_config
        alloc.role_nnodes["teacher"] = t_dc.nnodes
        alloc.role_gpus_per_node["teacher"] = t_dc.num_gpus_per_node
        alloc.role_min_gpus_per_replica["teacher"] = _min_gpus_per_replica(
            "teacher",
            t_dc,
        )

    return alloc
