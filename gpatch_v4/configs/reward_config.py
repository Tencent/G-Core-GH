import collections
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.infer_engine_config import (
    BaseInferEngineConfig,
    InferEngineConfig,
)
from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class BaseRewardConfig(MappingProtocol):
    """Base configuration shared by all reward model types.

    Attributes
    ----------
    dist_config : DistConfig
    hf_tokenizer_pathes : list of str or None
    num_reward_models : int
    custom_reward_model_path : str or None
    reward_ips : list of str or None
        IPs of external reward server instances.
    reward_ports : list of int or None
        Ports of external reward server instances.
    server_timeout_keep_alive : int
        Reward server keep-alive timeout (seconds).
    extra_input_keys : list of str
        Extra input keys passed to the reward computation function.
    """
    dist_config: DistConfig = field(default_factory=DistConfig)

    #TODO: 看起来下面的参数不太需要了，考虑删除
    hf_tokenizer_pathes: Optional[List[str]
                                 ] = field(default=None, metadata={"help": "Tokenizer model"})
    num_reward_models: int = field(default=1, metadata={"help": "Number of reward models."})
    custom_reward_model_path: Optional[str] = field(
        default=None, metadata={"help": "Custom reward model path."}
    )
    reward_ips: Optional[List[str]] = field(default=None, metadata={"help": "reward server ips."})
    reward_ports: Optional[List[int]
                          ] = field(default=None, metadata={"help": "reward server ports."})
    server_timeout_keep_alive: int = field(
        default=5, metadata={"help": "Server timeout keep alive."}
    )
    extra_input_keys: List[str] = field(
        default_factory=list, metadata={"help": "extra inputs keys for reward calculate."}
    )


@dataclass
class RewardModelInfo(MappingProtocol):
    """Information about a single reward model.

    Attributes
    ----------
    model_arch : str or None
    hf_model_path : str or None
    reward_py_path : str or None
        Python file containing the reward implementation.
    prompt_fn_name : str or None
    parse_reward_fn_name : str or None
    reward_weight : float
        Weight when combining multiple reward signals.
    input_token_key : list of str or None
        Batches containing ANY of these keys are routed to this RM.
        ``None`` → broadcast to all.
    """
    model_arch: Optional[str] = field(
        default=None,
        metadata={"help": "model arch, see gpatch_v4.core.constants.MODEL_ARCH for enums"},
    )
    hf_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "hf model path"},
    )
    reward_py_path: Optional[str] = field(
        default=None,
        metadata={"help": "reward impl file"},
    )
    prompt_fn_name: Optional[str] = field(
        default=None,
        metadata={"help": "prompt function name, only used for t2i"},
    )
    parse_reward_fn_name: Optional[str] = field(
        default=None,
        metadata={"help": "reward function name, only used for t2i"},
    )
    gen_reward_fn_name: Optional[str] = field(
        default=None,
        metadata={"help": "generative reward function name, used for llm/vlm"},
    )
    gen_reward_repeat_n: Optional[int] = field(
        default=1,
        metadata={"help": "number of times to repeat generation for reward calculation"},
    )
    reward_weight: float = field(
        default=1.0,
        metadata={"help": "reward weights"},
    )
    input_token_key: Optional[List[str]] = field(
        default=None,
        metadata={
            "help":
                "List of batch keys for routing. Batches containing ANY of "
                "these keys are sent to this RM. None = broadcast to all."
        },
    )


@dataclass
class T2iBtRewardModelInfo(RewardModelInfo):
    """Reward model info for T2I batch reward.

    Attributes
    ----------
    processor_path : str or None
    model_weight_path : str or None
    rm_cls_name : str or None
        Reward model class name for dynamic loading.
    """
    processor_path: Optional[str] = field(
        default=None,
        metadata={"help": "processor path"},
    )
    model_weight_path: Optional[str] = field(
        default=None,
        metadata={"help": "model weight path"},
    )
    rm_cls_name: Optional[str] = field(
        default=None,
        metadata={"help": "reward model class name"},
    )
    # HPSv3 specific paths
    hpsv3_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "HPSv3 config yaml path"},
    )
    hpsv3_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "HPSv3 model checkpoint path"},
    )


@dataclass
class GenRewardConfig(BaseRewardConfig):
    """Configuration for generative reward models.

    Attributes
    ----------
    backend : str
        Inference engine backend, e.g. ``"sglang"``.
    reward_model_info : list of RewardModelInfo or None
    infer_engine_configs : list of InferEngineConfig or None
    destroy_engine_after_generation : bool
        Destroy sglang gen-RM engines after each gen-RM generation phase.
    """
    backend: str = field(default="sglang", metadata={"help": "infer engine impl"})
    reward_model_info: Optional[List[RewardModelInfo]] = field(default=None)
    infer_engine_configs: Optional[List[InferEngineConfig]] = field(default=None)
    destroy_engine_after_generation: bool = field(
        default=False,
        metadata={
            "help":
                "Destroy sglang gen-RM engines after each gen-RM generation phase "
                "and recreate them on the next phase."
        },
    )


@dataclass
class BtRewardConfig(BaseRewardConfig):
    """Configuration for batch (rule/RM) reward models.

    Attributes
    ----------
    reward_type : str
        ``"rule_only"`` / ``"rm_only"`` / ``"rm_with_rule"``.
    reward_model_info : list of RewardModelInfo or None
    infer_engine_configs : list of BaseInferEngineConfig or None
    """
    reward_type: str = field(
        default="rule_only",
        metadata={"help": "Reward type. [rule_only, rm_only, rm_with_rule]"},
    )
    reward_model_info: Optional[List[RewardModelInfo]] = field(default=None)
    infer_engine_configs: Optional[List[BaseInferEngineConfig]] = field(default=None)


@dataclass
class T2iBtRewardConfig(BtRewardConfig):
    """Batch reward config for T2I.

    Attributes
    ----------
    reward_model_info : list of T2iBtRewardModelInfo or None
    """
    reward_model_info: Optional[List[T2iBtRewardModelInfo]] = field(default=None)


@dataclass
class ExternalRewardInfo(MappingProtocol):
    """Information about an external reward.

    Attributes
    ----------
    reward_py_path : str or None
        Python file containing the external reward implementation.
    reward_cls_name : str or None
    reward_weight : float
        Weight when combining multiple reward signals.
    """
    reward_py_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the external reward implementation file"},
    )
    reward_cls_name: Optional[str] = field(
        default=None,
        metadata={"help": "Class name of the external reward"},
    )
    reward_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of this external reward"},
    )


@dataclass
class ExternalRewardConfig(MappingProtocol):
    """Configuration for external rewards.

    Attributes
    ----------
    reward_info : list of ExternalRewardInfo or None
    """
    reward_info: Optional[List[ExternalRewardInfo]] = field(default=None)
