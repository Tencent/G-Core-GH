import collections
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class DistConfig(MappingProtocol):
    """Distributed training topology configuration.

    Attributes
    ----------
    num_gpus_per_node : int
    nnodes : int
        ``-1`` → use all available nodes.
    num_cpu_nodes : int
        CPU-only nodes (no GPU allocation); for rule-only reward models.
    torch_dist_timeout_minutes : int
    use_tp_pp_dp_mapping : bool
    expert_model_parallel_size : int
    expert_tensor_parallel_size : int
    context_parallel_size : int
    tensor_model_parallel_size : int
    pipeline_model_parallel_size : int
    num_layers_in_first_pipeline_stage : int or None
        For uneven splits.
    num_layers_in_last_pipeline_stage : int or None
        For uneven splits.
    virtual_pipeline_model_parallel_size : int or None
        Interleaved scheduling.
    pipeline_model_parallel_split_rank : int or None
        Rank where encoder and decoder split.
    sequence_parallel : bool
        Requires TP > 1.
    enable_custom_device_mesh : bool
        FSDP2 only.
    custom_device_mesh : list of int or None
    fsdp2_num_to_forward_prefetch : int
    fsdp2_num_to_backward_prefetch : int
    """
    num_gpus_per_node: int = field(
        default=8,
        metadata={"help": "Number of gpus per node."},
    )
    nnodes: int = field(
        default=-1,
        metadata={"help": "Number of nodes. The default is -1, meaning all nodes are used."},
    )
    num_cpu_nodes: int = field(
        default=0,
        metadata={
            "help":
                "Number of CPU-only nodes (no GPU allocation). "
                "Used for rule-only reward models that don't need GPUs."
        },
    )
    torch_dist_timeout_minutes: int = field(
        default=30,
        metadata={"help": "Timeout for torch distributed. Minutes"},
    )
    use_tp_pp_dp_mapping: bool = field(
        default=False,
        metadata={"help": "Whether to use tp pp dp mapping."},
    )

    # dist config commonly used by mlm and fsdp2
    expert_model_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert model parallel size."},
    )
    expert_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert tensor parallel size."},
    )
    context_parallel_size: int = field(
        default=1,
        metadata={"help": "Context parallel size."},
    )
    dynamic_context_parallel: bool = field(
        default=False,
        metadata={
            "help": "Enable dynamic context parallel for variable-length sequence balancing."
        },
    )
    max_seqlen_per_dp_cp_rank: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "Max tokens per DPxCP rank for Dynamic CP scheduling. Required when dynamic_context_parallel=True."
        },
    )
    min_dynamic_context_parallel_size: int = field(
        default=1,
        metadata={"help": "Minimum CP group size for dynamic context parallel."},
    )
    dynamic_cp_scheduler_type: str = field(
        default="default",
        metadata={
            "help":
                "Dynamic CP scheduler type. "
                "'default': global length-based sorting + all-to-all redistribution. "
                "'smart_padding': local scheduling with zero data communication, "
                "requires smart padding enabled in the dataset."
        },
    )

    # megatron args
    tensor_model_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor model parallel size."},
    )
    pipeline_model_parallel_size: int = field(
        default=1,
        metadata={"help": "Pipeline model parallel size."},
    )
    num_layers_in_first_pipeline_stage: Optional[int] = field(
        default=None,
        metadata={"help": "Number of layers in first pipeline stage."},
    )
    num_layers_in_last_pipeline_stage: Optional[int] = field(
        default=None,
        metadata={"help": "Number of layers in last pipeline stage."},
    )
    virtual_pipeline_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Virtual pipeline model parallel size."},
    )
    pipeline_model_parallel_split_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Rank where encoder and decoder should be split."},
    )
    sequence_parallel: bool = field(
        default=True,
        metadata={"help": "Whether to use sequence parallel."},
    )

    # fsdp2 args
    enable_custom_device_mesh: bool = field(
        default=False,
        metadata={"help": "Whether to use custom device mesh."},
    )
    custom_device_mesh: Optional[list[int]] = field(
        default=None,
        metadata={"help": "Custom device mesh."},
    )
    fsdp2_num_to_forward_prefetch: int = field(
        default=1,
        metadata={"help": "Number of forward prefetch for fsdp2."},
    )
    fsdp2_num_to_backward_prefetch: int = field(
        default=1,
        metadata={"help": "Number of backward prefetch for fsdp2."},
    )

    def __post_init__(self):
        self.sequence_parallel = self.sequence_parallel and self.tensor_model_parallel_size > 1
        if self.dynamic_context_parallel:
            assert self.max_seqlen_per_dp_cp_rank is not None, (
                "max_seqlen_per_dp_cp_rank must be set when dynamic_context_parallel=True"
            )
            assert self.context_parallel_size >= 2, (
                "dynamic_context_parallel requires context_parallel_size >= 2, "
                f"got context_parallel_size={self.context_parallel_size}"
            )
            assert 1 <= self.min_dynamic_context_parallel_size <= self.context_parallel_size, (
                "min_dynamic_context_parallel_size must be in "
                f"[1, context_parallel_size={self.context_parallel_size}], "
                f"got {self.min_dynamic_context_parallel_size}"
            )
            assert self.virtual_pipeline_model_parallel_size in (None, 1), (
                "dynamic_context_parallel does not support virtual_pipeline_model_parallel_size > 1 yet"
            )
        _valid_dcp_types = ("default", "smart_padding")
        assert self.dynamic_cp_scheduler_type in _valid_dcp_types, (
            f"dynamic_cp_scheduler_type must be one of {_valid_dcp_types}, "
            f"got '{self.dynamic_cp_scheduler_type}'"
        )
