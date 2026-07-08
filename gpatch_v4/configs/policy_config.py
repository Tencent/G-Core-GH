import collections
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.client_config import ClientConfig
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.lora_config import LoRAConfig
from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class BasePolicyConfig(MappingProtocol):
    """Base policy configuration shared by all policy types.

    Attributes
    ----------
    dist_config : DistConfig
    model_arch : str or None
        See ``gpatch_v4.core.constants.MODEL_ARCH`` for valid values.
    hf_model_path : str or None
    hf_tokenizer_path : str or None
    ref_hf_model_path : str or None
        Reference model checkpoint (for KL computation).
    attn_implementation : str
        FSDP2 backend: ``"flash_attention_2"`` / ``"sdpa"`` / ``"eager"``.
    ep_backend : str
        DeepSeek-V4 expert dispatch backend: ``"eager"`` / ``"deepep"``.
    deepep_num_sms : int
        Number of SMs assigned to DeepEP kernels.
    rollout_gen_type : str or None
        ``"base"`` / ``"replay"`` / ``"dynamic_sampling"``.
    forward_only_mbs : int
        Micro batch size for forward-only passes (e.g. reference model).
    dynamic_mbs_target_seqlen : int or None
    dynamic_mbs_limit : int or None
    ppo_pack_seq : bool
        Pack multiple sequences into a single micro batch.
    without_ref : bool
        Skip the reference model entirely.
    without_optim : bool
        Disable optimizer (debug only).
    smart_pad_infer : bool
    smart_pad_train : bool
    wrap_with_ddp : bool
        Whether to wrap the model with DistributedDataParallel.
    post_wrap_with_ddp : bool
        Whether to delay DDP wrapping until after mbridge weight loading.
    override_transformer_config : dict
        Overrides applied to the transformer model config.
    freeze_patterns : list of str
        Wildcard patterns (``*`` matches any substring, see
        ``mbridge.peft.utils.wildcard_match``) matched against parameter
        names; matching parameters have ``requires_grad`` set to ``False``.
    unfreeze_patterns : list of str
        Same wildcard syntax as ``freeze_patterns``, but for keeping/forcing
        parameters trainable. Takes priority over ``freeze_patterns``: a
        parameter matching both stays trainable.
    """
    dist_config: DistConfig = field(default_factory=DistConfig)

    model_arch: Optional[str] = field(
        default=None,
        metadata={"help": "model arch, see gpatch_v4.core.constants.MODEL_ARCH for enums"},
    )
    hf_model_path: Optional[str] = field(default=None, metadata={"help": "Model path"})
    hf_tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Tokenizer model"})
    ref_hf_model_path: Optional[str] = field(
        default=None, metadata={"help": "Reference model path"}
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "fsdp Attention implementation.: flash_attention_2, sdpa, eager"}
    )
    indexer_backend: str = field(
        default="eager", metadata={"help": "DSV4 indexer backend: eager, fused"}
    )
    ep_backend: str = field(default="eager", metadata={"help": "DSV4 EP backend: eager, deepep"})
    deepep_num_sms: int = field(
        default=24, metadata={"help": "Number of SMs used by DeepEP kernels"}
    )
    fp8_qat: bool = field(
        default=False, metadata={"help": "Enable FP8 activation/weight fake-quant (QAT)"}
    )

    rollout_gen_type: Optional[str] = field(
        default="base",
        metadata={"help": "Rollout generator. [base, replay, dynamic_sampling, external]"}
    )
    rollout_gen_py_path: Optional[str] = field(
        default=None,
        metadata={"help": "Python file path for external rollout generator class."},
    )
    rollout_gen_cls_name: Optional[str] = field(
        default=None,
        metadata={"help": "Class name of the external rollout generator."},
    )
    forward_only_mbs: int = field(default=1, metadata={"help": "Forward micro batch size."})
    dynamic_mbs_target_seqlen: Optional[int] = field(
        default=None, metadata={"help": "Target sequence length for dynamic micro batch size."}
    )
    dynamic_mbs_limit: Optional[int] = field(
        default=None, metadata={"help": "Micro batch size limit for dynamic micro batch size."}
    )
    dynamic_mbs_target_seqlen_fwd_only: Optional[int] = field(
        default=None,
        metadata={
            "help": "Target sequence length for dynamic micro batch size in forward-only mode."
        }
    )
    dynamic_mbs_limit_fwd_only: Optional[int] = field(
        default=None,
        metadata={
            "help": "Micro batch size limit for dynamic micro batch size in forward-only mode."
        }
    )
    ppo_pack_seq: bool = field(default=False, metadata={"help": "Whether to pack sequence."})
    without_ref: bool = field(default=False, metadata={"help": "Whether to use ref model."})
    without_optim: bool = field(
        default=False, metadata={"help": "disable optim for debugging purpose"}
    )
    balance_dp_seqlen: bool = field(
        default=False, metadata={"help": "Whether to balance dp seqlen."}
    )
    smart_pad_infer: bool = field(
        default=False, metadata={"help": "Whether to do smart pad in infer."}
    )
    smart_pad_train: bool = field(
        default=False, metadata={"help": "Whether to do smart pad in train."}
    )
    wrap_with_ddp: bool = field(default=True, metadata={"help": "Whether to wrap with ddp."})
    post_wrap_with_ddp: bool = field(
        default=False,
        metadata={"help": "Delay DDP wrapping until after mbridge load_weights."},
    )
    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    freeze_patterns: List[str] = field(
        default_factory=list,
        metadata={
            "help":
                "Wildcard patterns ('*' = any substring) matched against parameter "
                "names to freeze (requires_grad=False)."
        },
    )
    unfreeze_patterns: List[str] = field(
        default_factory=list,
        metadata={
            "help":
                "Wildcard patterns ('*' = any substring) matched against parameter "
                "names to keep/force trainable. Takes priority over freeze_patterns."
        },
    )
    lora: LoRAConfig = field(
        default_factory=LoRAConfig,
        metadata={"help": "LoRA/PEFT config (rank=0 disables)"},
    )
    # TODO: 其它算法也加上这个功能
    offload_process_group: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to destroy/recreate NCCL process groups on sleep/wake_up to free GPU memory."
        }
    )
    manual_clear_memory: bool = field(
        default=True,
        metadata={"help": "Whether to manually clear memory. Only support for sft now."}
    )
    manual_clear_memory_interval: int = field(
        default=1,
        metadata={
            "help": "Interval (in steps) to manually clear memory. Only support for sft now."
        }
    )
    export_weights_buffer_max_size_mb: int = field(
        default=2048,
        metadata={
            "help":
                "Max buffer size (MB) for TP/EP all-gather during export_weights. "
                "Lower values reduce GPU memory peak at the cost of more communication rounds."
        }
    )

    def __post_init__(self):
        if isinstance(self.dist_config, dict):
            self.dist_config = DistConfig(**self.dist_config)
        if isinstance(self.lora, dict):
            self.lora = LoRAConfig(**self.lora)

        if self.dist_config.dynamic_context_parallel:
            _dcp_incompatible = {
                "balance_dp_seqlen":
                    self.balance_dp_seqlen,
                "smart_pad_infer":
                    self.smart_pad_infer,
                "smart_pad_train":
                    self.smart_pad_train,
                "dynamic_mbs_target_seqlen":
                    self.dynamic_mbs_target_seqlen is not None,
                "dynamic_mbs_limit":
                    self.dynamic_mbs_limit is not None,
                "dynamic_mbs_target_seqlen_fwd_only":
                    self.dynamic_mbs_target_seqlen_fwd_only is not None,
                "dynamic_mbs_limit_fwd_only":
                    self.dynamic_mbs_limit_fwd_only is not None,
            }
            _violations = [k for k, v in _dcp_incompatible.items() if v]
            assert not _violations, (
                f"dynamic_context_parallel is incompatible with: {', '.join(_violations)}. "
                "Dynamic CP handles sequence packing and load balancing internally."
            )


@dataclass
class PolicyConfig(BasePolicyConfig):
    """Policy configuration with client connections.

    Attributes
    ----------
    sampler_client : ClientConfig
    bt_rm_client : ClientConfig
    gen_rm_client : ClientConfig
    """
    sampler_client: ClientConfig = field(default_factory=ClientConfig)
    bt_rm_client: ClientConfig = field(default_factory=ClientConfig)
    gen_rm_client: ClientConfig = field(default_factory=ClientConfig)


@dataclass
class StudentConfig(PolicyConfig):
    """Student policy configuration for distillation.

    Attributes
    ----------
    teacher_client : ClientConfig
    """
    teacher_client: ClientConfig = field(default_factory=ClientConfig)


@dataclass
class T2iTextEncoderConfig(MappingProtocol):
    pass


@dataclass
class T2iPolicyConfig(BasePolicyConfig):
    """Policy configuration for text-to-image models.

    Attributes
    ----------
    sampler_client : ClientConfig
    bt_rm_client : ClientConfig
    gen_rm_client : ClientConfig
    kv_store_client : ClientConfig
    fwd_only_mbs : int
        Forward-only micro batch size.
    """
    sampler_client: ClientConfig = field(default_factory=ClientConfig)
    bt_rm_client: ClientConfig = field(default_factory=ClientConfig)
    gen_rm_client: ClientConfig = field(default_factory=ClientConfig)
    kv_store_client: ClientConfig = field(default_factory=ClientConfig)
    fwd_only_mbs: int = field(default=1, metadata={"help": "Forward only micro batch size."})
