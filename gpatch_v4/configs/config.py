import collections
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.agentic_config import AgenticConfig
from gpatch_v4.configs.checkpoint_config import CheckpointConfig
from gpatch_v4.configs.data_config import DataConfig
from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ema_config import EmaConfig
from gpatch_v4.configs.evaluate_config import EvaluateResultConfig
from gpatch_v4.configs.inference_config import InferResultConfig
from gpatch_v4.configs.kv_config import KvConfig
from gpatch_v4.configs.optimizer_config import OptimizerConfig
from gpatch_v4.configs.policy_config import (
    BasePolicyConfig,
    PolicyConfig,
    StudentConfig,
    T2iPolicyConfig,
)
from gpatch_v4.configs.ppo_config import DistillConfig, PpoConfig, T2iPpoConfig
from gpatch_v4.configs.report_config import MonitorConfig, ReportConfig
from gpatch_v4.configs.reward_config import (
    BtRewardConfig,
    ExternalRewardConfig,
    GenRewardConfig,
    T2iBtRewardConfig,
)
from gpatch_v4.configs.sampler_config import SamplerConfig
from gpatch_v4.configs.t2i_dpo_config import T2iDpoConfig
from gpatch_v4.configs.tq_config import TqConfig
from gpatch_v4.configs.training_config import (
    DpoTrainingConfig,
    EmbeddingTrainingConfig,
    FinetuneTrainingConfig,
    OffPolicyDistillTrainingConfig,
    RewardTrainingConfig,
    RLTrainingConfig,
    T2iRlTrainingConfig,
    T2iSftTrainingConfig,
    TrainingConfig,
)
from gpatch_v4.configs.utils import MappingProtocol

RL_PLACEMENT_TYPES = ("colocate", "disaggregated", "partial_colocated")


def _assert_deterministic_mode_constraints(training, checkpoint) -> None:
    if not training.apply_deterministic_mode:
        return
    assert not checkpoint.skip_save_mcore_model, (
        "skip_save_mcore_model=True is incompatible with apply_deterministic_mode=True. "
        "The HF bridge roundtrip introduces precision loss that breaks deterministic resume."
    )
    if training.attention_backend == "flash":
        training.disable_flash_attn_3 = True
        logging.info(
            "Deterministic mode + flash backend: disable_flash_attn_3=True "
            "to force FA2 fallback"
        )


def _maybe_force_router_counting_for_dump(training, debug) -> None:
    if debug.debug_dump_expert_token_counts:
        training.freeze_router_correction_bias = False
        training.router_correction_bias_update_speed = 0


def _assert_ce_compaction_model_arch(training, policy) -> None:
    if not training.ce_compaction:
        return
    assert training.build_from_mbridge, (
        "ce_compaction requires training.build_from_mbridge=True"
    )
    # Delayed import: config.py cannot load gpatch_v4.core at module import time.
    from gpatch_v4.core.constants import MODEL_ARCH
    assert policy.model_arch in (
        MODEL_ARCH.QWEN3_VL,
        MODEL_ARCH.QWEN3_VL_MOE,
        MODEL_ARCH.QWEN3_5,
        MODEL_ARCH.QWEN3_5_MOE,
        MODEL_ARCH.WELMV4_MOE,
    ), (
        "ce_compaction is only supported for Qwen3-VL/Qwen3.5/Qwen3.6 or WeLM v4.5, got "
        f"{policy.model_arch!r}"
    )


def _assert_dynamic_cp_requires(training, *policies) -> None:
    attention_backend = training.attention_backend
    for policy in policies:
        if policy is None:
            continue
        dist_config = getattr(policy, "dist_config", None)
        if dist_config is not None and getattr(dist_config, "dynamic_context_parallel", False):
            from gpatch_v4.core.constants import MODEL_ARCH

            is_mlite_welm = (
                training.training_backend == "mlite" and policy.model_arch == MODEL_ARCH.WELMV4_MOE
            )
            if not is_mlite_welm:
                assert attention_backend == "flash", (
                    "dynamic_context_parallel requires training.attention_backend='flash', "
                    f"otherwise the grad_norm nan, got {attention_backend!r}"
                )
            if dist_config.dynamic_cp_scheduler_type == "default":
                assert dist_config.context_parallel_size == 1, (
                    "default scheduler 需要 dist_config.context_parallel_size=1, "
                    f"不然在重新分配数据时会有性能问题 {dist_config.context_parallel_size}"
                )


@dataclass
class FinetuneConfig(MappingProtocol):
    """Configuration for supervised fine-tuning.

    Attributes
    ----------
    placement_type : str or None
        ``"colocate"`` or ``"disaggregated"``.
    data : DataConfig
    training : FinetuneTrainingConfig
    policy : BasePolicyConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    task : Any
        Arbitrary task-specific configuration.
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: FinetuneTrainingConfig = field(default_factory=FinetuneTrainingConfig)
    policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        _maybe_force_router_counting_for_dump(self.training, self.debug)
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        _assert_dynamic_cp_requires(self.training, self.policy)
        _assert_ce_compaction_model_arch(self.training, self.policy)
        self.policy.dist_config.assert_vpp_compatible(
            use_dynamic_mbs=self.training.use_dynamic_mbs,
            smart_pad_train=self.policy.smart_pad_train,
        )
        #TODO(xiaotaoliu): 暂时先限制一下
        if self.training.training_backend == "fsdp2" and self.training.use_linear_ce:
            assert self.training.loss_func == "cross_entropy", (
                "FSDP2 linear CE currently supports SFT cross_entropy only"
            )
        if self.policy.dist_config.dynamic_context_parallel:
            # mlite owns pool-token loss normalization via its dynamic-CP plugin;
            # mcore still requires calculate_per_token_loss in transformer overrides.
            if self.training.training_backend != "mlite":
                assert self.policy.override_transformer_config.get(
                    "calculate_per_token_loss", False
                ), ("dynamic_context_parallel requires calculate_per_token_loss=True ")


@dataclass
class PretrainConfig(FinetuneConfig):
    """Packed pretrain: dataset packing + lean ``PretrainActor``.

    Hot path uses ``prepare_data.pretrain_packed`` (skips dyn-CP /
    ``expand_rollout_batches`` / ``sft_train``). Requires mcore backend,
    ``dynamic_context_parallel=False``, ``smart_pad_train=False``, and
    ``train_mbs=1``.
    """
    def __post_init__(self):
        super().__post_init__()
        assert self.training.training_backend == "mcore", (
            "PretrainConfig requires training.training_backend='mcore' "
            f"(got {self.training.training_backend!r})"
        )
        assert not self.policy.dist_config.dynamic_context_parallel, (
            "PretrainConfig is incompatible with dynamic_context_parallel"
        )
        assert not self.policy.smart_pad_train, ("PretrainConfig does not support smart_pad_train")
        assert self.training.train_mbs == 1, ("PretrainConfig requires training.train_mbs=1")


# embedding config
@dataclass
class EmbeddingConfig(FinetuneConfig):
    """Configuration for supervised fine-tuning with embedding model.

    ``training`` is :class:`EmbeddingTrainingConfig` so YAML keys such as
    ``training.use_gbs_embedding_in_loss`` merge into the structured schema.
    """
    training: EmbeddingTrainingConfig = field(default_factory=EmbeddingTrainingConfig)

    def __post_init__(self):
        super().__post_init__()
        assert self.training.use_linear_ce, "training.use_linear_ce must be True"
        assert not self.training.use_dynamic_mbs, "training.use_dynamic_mbs must be False"
        if getattr(self.training, "pack_seq", False):
            assert self.policy.dist_config.context_parallel_size == 1, (
                "training.pack_seq requires context_parallel_size == 1, "
                f"got {self.policy.dist_config.context_parallel_size}"
            )
            assert not self.policy.ppo_pack_seq, (
                "training.pack_seq is incompatible with policy.ppo_pack_seq "
                "(use collator THD packing + use_linear_ce instead)"
            )
            # THD packing 必须走 flash；auto/其它后端可能静默忽略 packed_seq_params。
            assert self.training.attention_backend == "flash", (
                "training.pack_seq requires training.attention_backend='flash', "
                f"got {self.training.attention_backend!r}"
            )
        if self.training.use_gbs_embedding_in_loss:
            assert self.training.gradcache_loss_py_path is not None, (
                "use_gbs_embedding_in_loss=True requires training.gradcache_loss_py_path to be set"
            )
            assert self.training.gradcache_loss_py_name is not None, (
                "use_gbs_embedding_in_loss=True requires training.gradcache_loss_py_name to be set"
            )


# rl config
@dataclass
class RlConfig(MappingProtocol):
    """Configuration for reinforcement learning (text).

    Attributes
    ----------
    placement_type : str or None
        ``"colocate"`` or ``"disaggregated"``.
    data : DataConfig
    training : RLTrainingConfig
    policy : PolicyConfig
    critic : PolicyConfig
    sampler : SamplerConfig
    gen_rm : GenRewardConfig
    bt_rm : BtRewardConfig
    ppo : PpoConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    critic: PolicyConfig = field(default_factory=PolicyConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    gen_rm: GenRewardConfig = field(default_factory=GenRewardConfig)
    bt_rm: BtRewardConfig = field(default_factory=BtRewardConfig)
    external_reward: ExternalRewardConfig = field(default_factory=ExternalRewardConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        assert self.placement_type in RL_PLACEMENT_TYPES, (
            f"unsupported RL placement_type={self.placement_type!r}"
        )
        _maybe_force_router_counting_for_dump(self.training, self.debug)
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        _assert_dynamic_cp_requires(self.training, self.policy, self.critic)
        if self.ppo.loss_func == "gspo":
            assert not self.policy.override_transformer_config.get(
                "calculate_per_token_loss", False
            ), ("gspo requires seq-mean aggregation; calculate_per_token_loss must be False")
        if self.training.return_hidden_states_for_ce:
            assert getattr(self.ppo, 'log_prob_top_k', 0) == 0, (
                "use_linear_ce/ce_compaction 与 log_prob_top_k > 0 不兼容，"
                "Linear CE and compact CE 不生成完整 logits，无法计算 top-K"
            )
            assert self.training.dump_metrics_logprobs_topk == 0, (
                "use_linear_ce/ce_compaction 与 "
                "dump_metrics_logprobs_topk > 0 不兼容，Linear CE and compact CE 无法 dump top-K"
            )
            assert not self.training.im_end_metrics_enable, (
                "use_linear_ce/ce_compaction 与 im_end_metrics_enable 不兼容，"
                "Linear CE and compact CE 无法计算 im_end 指标"
            )
        _assert_ce_compaction_model_arch(self.training, self.policy)
        if self.training.ce_compaction:
            assert self.ppo.loss_func == "grpo", (
                "ce_compaction currently supports GRPO loss only"
            )
            assert not self.policy.smart_pad_infer, (
                "ce_compaction is incompatible with smart_pad_infer"
            )
            assert not (
                self.policy.ppo_pack_seq and
                not self.policy.dist_config.dynamic_context_parallel
            ), ("ce_compaction does not support static ppo_pack_seq")
        # Global covariance centers on the global (cross-DP) mean, so it does not
        # degenerate at train_mbs == 1; only the per-micro-batch path needs mbs > 1.
        if (
            self.ppo.ppo_entropy_regularization_type is not None and
            not self.ppo.ppo_entropy_global_cov
        ):
            use_dynamic_mbs = self.policy.dynamic_mbs_target_seqlen is not None
            assert self.training.train_mbs > 1 or use_dynamic_mbs, (
                f"ppo_entropy_regularization_type requires train_mbs > 1 or dynamic mbs enabled "
                f"(policy.dynamic_mbs_target_seqlen is not None), "
                f"got train_mbs={self.training.train_mbs}, "
                f"dynamic_mbs_target_seqlen={self.policy.dynamic_mbs_target_seqlen}."
            )

        if self.training.moe_router_replay:
            infer_engine_configs = self.sampler.infer_engine_configs or ()
            model_override_args = (
                infer_engine_configs[0].model_override_args if infer_engine_configs else None
            )
            self.training.resolve_moe_router_replay_shape(
                self.policy.hf_model_path,
                model_override_args,
            )


@dataclass
class T2iRlConfig(MappingProtocol):
    """Configuration for text-to-image reinforcement learning.

    Attributes
    ----------
    placement_type : str or None
    data : DataConfig
    training : T2iRlTrainingConfig
    policy : T2iPolicyConfig
    gen_rm : GenRewardConfig
    bt_rm : T2iBtRewardConfig
    ema : EmaConfig
    ppo : T2iPpoConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    debug : DebugConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: T2iRlTrainingConfig = field(default_factory=T2iRlTrainingConfig)
    policy: T2iPolicyConfig = field(default_factory=T2iPolicyConfig)
    gen_rm: GenRewardConfig = field(default_factory=GenRewardConfig)
    bt_rm: T2iBtRewardConfig = field(default_factory=T2iBtRewardConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    ppo: T2iPpoConfig = field(default_factory=T2iPpoConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})


# distill config
@dataclass
class OnPolicyDistillConfig(MappingProtocol):
    """Configuration for on-policy distillation.

    Attributes
    ----------
    placement_type : str or None
    data : DataConfig
    training : RLTrainingConfig
    policy : StudentConfig
    teacher : dict or None
        **Legacy / backward-compat.** Auto-promoted to ``teachers["default"]``
        when ``teachers`` is empty; ignored when ``teachers`` is set.
    teachers : dict[str, Any]
        Named teacher configs; keys are routing names. Single teacher → no
        routing required. Example::

            teachers:
              math:
                hf_model_path: /path/to/math-teacher
                dist_config: ...
              code:
                hf_model_path: /path/to/code-teacher
                dist_config: ...
    sampler : SamplerConfig
    gen_rm : GenRewardConfig
    bt_rm : BtRewardConfig
    ppo : DistillConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    policy: StudentConfig = field(default_factory=StudentConfig)
    teacher: Optional[BasePolicyConfig] = field(
        default=None,
        metadata={
            "help":
                "Legacy single-teacher config (backward compat). "
                "Ignored when 'teachers' is set."
        }
    )
    teachers: Optional[dict[str, BasePolicyConfig]] = field(
        default=None,
        metadata={
            "help":
                "Named teacher models. Keys are teacher names for per-sample routing. "
                "Single teacher: routing field not required."
        }
    )
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    gen_rm: GenRewardConfig = field(default_factory=GenRewardConfig)
    bt_rm: BtRewardConfig = field(default_factory=BtRewardConfig)
    #TODO: 把 ppo rename
    ppo: DistillConfig = field(default_factory=DistillConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        teachers = tuple(self.teachers.values()) if self.teachers else ()
        _assert_dynamic_cp_requires(self.training, self.policy, self.teacher, *teachers)
        if self.ppo.loss_func == "gspo":
            assert not self.policy.override_transformer_config.get(
                "calculate_per_token_loss", False
            ), ("gspo requires seq-mean aggregation; calculate_per_token_loss must be False")
        if self.training.use_linear_ce:
            assert getattr(
                self.ppo, 'log_prob_top_k', 0
            ) == 0, ("use_linear_ce 与 log_prob_top_k > 0 不兼容，"
                     "linear_ce 不生成完整 logits，无法计算 top-K")
            assert not self.policy.ppo_pack_seq, ("use_linear_ce 与 ppo_pack_seq 不兼容")
            assert self.training.dump_metrics_logprobs_topk == 0, (
                "use_linear_ce 与 dump_metrics_logprobs_topk > 0 不兼容，"
                "linear_ce 不生成完整 logits，无法 dump top-K"
            )
            assert not self.training.im_end_metrics_enable, (
                "use_linear_ce 与 im_end_metrics_enable 不兼容，"
                "linear_ce 不生成完整 logits，无法计算 im_end 指标"
            )
        assert not self.training.ce_compaction, (
            "ce_compaction currently supports GRPO only, not on-policy distillation"
        )
        if not self.teachers and self.teacher is not None:
            self.teachers = {"default": self.teacher}
        self.teacher = None

        if self.training.moe_router_replay:
            infer_engine_configs = self.sampler.infer_engine_configs or ()
            model_override_args = (
                infer_engine_configs[0].model_override_args if infer_engine_configs else None
            )
            self.training.resolve_moe_router_replay_shape(
                self.policy.hf_model_path,
                model_override_args,
            )


@dataclass
class DpoConfig(MappingProtocol):
    """Configuration for Direct Preference Optimization (DPO).

    Attributes
    ----------
    placement_type : str or None
        ``"colocate"`` or ``"disaggregated"``.
    data : DataConfig
    training : DpoTrainingConfig
    policy : BasePolicyConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: DpoTrainingConfig = field(default_factory=DpoTrainingConfig)
    policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        assert self.placement_type in ["colocate", "disaggregated"]
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        _assert_dynamic_cp_requires(self.training, self.policy)
        assert not self.training.return_hidden_states_for_ce, (
            "Linear CE and compact CE are not supported for DPO: dpo_loss_func requires "
            "vocab-parallel logits to aggregate chosen/rejected log-probs"
        )


@dataclass
class RewardConfig(MappingProtocol):
    """Configuration for Bradley-Terry reward-model training (output_scalar).

    Mirrors :class:`DpoConfig`: pairwise (chosen|rejected) preference data,
    no reference model, colocate placement. The policy model is built with a
    scalar reward head via ``training.build_reward_head``.

    Attributes
    ----------
    placement_type : str or None
        ``"colocate"`` or ``"disaggregated"``.
    data : DataConfig
    training : RewardTrainingConfig
    policy : BasePolicyConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: RewardTrainingConfig = field(default_factory=RewardTrainingConfig)
    policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        assert self.placement_type in ["colocate", "disaggregated"]
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        _assert_dynamic_cp_requires(self.training, self.policy)


@dataclass
class OffPolicyDistillConfig(MappingProtocol):
    """Configuration for off-policy distillation.

    Attributes
    ----------
    placement_type : str or None
        ``"colocate"`` or ``"disaggregated"``.
    data : DataConfig
    training : OffPolicyDistillTrainingConfig
    policy : StudentConfig
    teacher : BasePolicyConfig
    distill : DistillConfig
    sampler : SamplerConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    tq : TqConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: OffPolicyDistillTrainingConfig = field(default_factory=OffPolicyDistillTrainingConfig)
    policy: StudentConfig = field(default_factory=StudentConfig)
    teacher: BasePolicyConfig = field(default_factory=BasePolicyConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    tq: TqConfig = field(default_factory=TqConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        assert self.placement_type in ["colocate", "disaggregated"]
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        _assert_dynamic_cp_requires(self.training, self.policy, self.teacher)

        if self.training.enable_teacher_kl_loss:
            assert self.training.use_linear_ce, (
                "Teacher hidden-state transfer requires use_linear_ce=true"
            )
            if self.training.setup_teacher_in_independent_topo:
                assert self.tq.enable, (
                    "Independent Teacher hidden-state transfer requires "
                    "tq.enable=true"
                )

        if self.tq.enable:
            assert self.training.enable_teacher_kl_loss, (
                "off-policy TQ requires enable_teacher_kl_loss=true"
            )
            assert self.training.setup_teacher_in_independent_topo, (
                "off-policy TQ requires setup_teacher_in_independent_topo=true"
            )

        if not self.training.setup_teacher_in_independent_topo:
            # When CP > 1, teacher smart_pad_infer and student smart_pad_train must match
            # to ensure CP splitting consistency between teacher logits and student training.
            if self.policy.dist_config.context_parallel_size > 1:
                assert self.teacher.smart_pad_infer == self.policy.smart_pad_train, (
                    f"When setup_teacher_in_independent_topo=False and CP>1, "
                    f"teacher.smart_pad_infer and policy.smart_pad_train must match for "
                    f"CP alignment, got teacher.smart_pad_infer={self.teacher.smart_pad_infer}, "
                    f"policy.smart_pad_train={self.policy.smart_pad_train}"
                )
            # When both smart_pad_infer and smart_pad_train are enabled,
            # teacher.forward_only_mbs must equal train_mbs so that the smart_pad
            # grouping (seqlen buckets) is identical between teacher and student.
            if self.teacher.smart_pad_infer and self.policy.smart_pad_train:
                assert self.teacher.forward_only_mbs == self.training.train_mbs, (
                    f"smart_pad_train + smart_pad_infer requires "
                    f"teacher.forward_only_mbs == train_mbs so that smart_pad grouping "
                    f"is consistent, got "
                    f"{self.teacher.forward_only_mbs} != {self.training.train_mbs}"
                )


@dataclass
class EvaluateConfig(MappingProtocol):
    """Configuration for model evaluation.

    Attributes
    ----------
    placement_type : str or None
    data : DataConfig
    training : RLTrainingConfig
        Only ``use_fast_tokenizer`` and ``eval_sampling_repeat_n`` are read.
    policy : PolicyConfig
    sampler : SamplerConfig
    evaluate_result : EvaluateResultConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    # training 里面实际只用到了 use_fast_tokenizer/eval_sampling_repeat_n 这个参数，其他参数均不会使用
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    evaluate_result: EvaluateResultConfig = field(default_factory=EvaluateResultConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})


@dataclass
class InferenceConfig(MappingProtocol):
    """Configuration for standalone inference.

    Attributes
    ----------
    data : DataConfig
    sampler : SamplerConfig
    infer_result : InferResultConfig
    """
    data: DataConfig = field(default_factory=DataConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    infer_result: InferResultConfig = field(
        default_factory=lambda: InferResultConfig(output_dir="./inference_output")
    )


@dataclass
class T2iEditSftConfig(FinetuneConfig):
    """Configuration for T2I editing SFT.

    Attributes
    ----------
    placement_type : str or None
    data : DataConfig
    training : T2iSftTrainingConfig
    policy : T2iPolicyConfig
    kv : KvConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    ema : EmaConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: T2iSftTrainingConfig = field(default_factory=T2iSftTrainingConfig)
    policy: T2iPolicyConfig = field(default_factory=T2iPolicyConfig)
    kv: KvConfig = field(default_factory=KvConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})


@dataclass
class AgenticRLTrainingConfig(RLTrainingConfig):
    agentic: AgenticConfig = field(default_factory=AgenticConfig)


@dataclass
class AgenticRlConfig(RlConfig):
    training: AgenticRLTrainingConfig = field(default_factory=AgenticRLTrainingConfig)


@dataclass
class AgenticOnPolicyDistillConfig(OnPolicyDistillConfig):
    """Agentic on-policy distillation: agentic env rollouts + teacher distillation.

    Inherits from ``OnPolicyDistillConfig`` so all teacher / student infra
    (``create_teacher_groups``, ``DistillStudentActor`` tokenizer setup,
    ``opd`` loss path) light up unchanged. The only delta vs.
    ``OnPolicyDistillConfig`` is the ``training`` field swapped to
    ``AgenticRLTrainingConfig`` so the YAML's ``training.agentic`` block
    deserializes correctly and ``agent_loop_actor_cls`` is reachable.
    """
    training: AgenticRLTrainingConfig = field(default_factory=AgenticRLTrainingConfig)
    policy: StudentConfig = field(default_factory=StudentConfig)

    def __post_init__(self):
        super().__post_init__()
        self._validate_multi_env_teacher_routing()

    def _validate_multi_env_teacher_routing(self):
        """Validate multi-env → multi-teacher routing consistency.

        For a multi-env run (``training.agentic.env_cfg_templates`` set), each
        env's samples route to a teacher by the configured routing field in
        ``EnvTemplateConfig.env_config``.
        With more than one teacher the loss path requires a per-sample routing
        field on every sample, so every env template must select a teacher that
        exists in ``teachers``, and env_types must be distinct (workers bind to
        templates by ``env_type``). Single-teacher runs skip the routing check
        (routing is unambiguous).
        """
        agentic = getattr(self.training, "agentic", None)
        templates = getattr(agentic, "env_cfg_templates", None) if agentic else None
        if not templates:
            return

        env_types = [t.env_type for t in templates]
        assert len(env_types) == len(set(env_types)), (
            f"env_cfg_templates must have distinct env_type values (workers bind "
            f"to a template by env_type); got {env_types}"
        )

        teachers = self.teachers or {}
        if len(teachers) <= 1:
            return

        teacher_keys = set(teachers.keys())
        routing_field = self.ppo.g_opd_teacher_routing_field
        for t in templates:
            teacher_type = t.env_config.get(routing_field)
            assert teacher_type, (
                f"multi-teacher OPD: env template '{t.env_type}' must set "
                f"env_config.{routing_field} so its samples can be routed; configured "
                f"teachers={sorted(teacher_keys)}"
            )
            assert teacher_type in teacher_keys, (
                f"env template '{t.env_type}' routes to teacher "
                f"'{teacher_type}' which is not in teachers={sorted(teacher_keys)}"
            )
