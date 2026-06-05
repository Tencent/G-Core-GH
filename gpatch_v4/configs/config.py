import collections
import os
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
from gpatch_v4.configs.training_config import (
    DpoTrainingConfig,
    FinetuneTrainingConfig,
    OffPolicyDistillTrainingConfig,
    RLTrainingConfig,
    T2iRlTrainingConfig,
    T2iSftTrainingConfig,
    TrainingConfig,
)
from gpatch_v4.configs.utils import MappingProtocol


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
        if not self.teachers and self.teacher is not None:
            self.teachers = {"default": self.teacher}
        self.teacher = None


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
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        assert self.placement_type in ["colocate", "disaggregated"]

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
