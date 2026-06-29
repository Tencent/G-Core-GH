from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Union

from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class BaseInferEngineConfig(MappingProtocol):
    """Base inference engine configuration.

    Attributes
    ----------
    dist_config : DistConfig
    """
    dist_config: DistConfig = field(default_factory=DistConfig)


@dataclass
class InferEngineConfig(BaseInferEngineConfig):
    """Inference engine configuration (sglang / vllm).

    Attributes
    ----------
    dp_size : int
        DP size; sglang overlaps DP with TP (unlike Megatron).
    enable_deepep_moe : bool
    use_fast_tokenizer : bool
    dtype : str
    gpu_memory_utilization : float
    max_running_requests : int or None
    temperature : float
    top_k : int
        ``-1`` disables top-k.
    top_p : float
    min_p : float
        ``0.0`` disables min-p.
    generate_max_tokens : int
    engine_seed: int
    seed : int
    allow_auto_truncate : bool
        Let sglang auto-truncate long prompts.
    repetition_penalty : float
    frequency_penalty : float
    presence_penalty : float
    load_format : str
        ``"auto"`` or ``"dummy"``.
    enable_weights_cpu_backup : bool
        Keep CPU backup of inference weights for fast updates.
    mm_per_request_timeout : int or None
        Multi-modal processor timeout (seconds).
    sgl_chunked_prefill_size : int or None
        ``None`` → SGLang default.
    sgl_schedule_conservativeness : float or None
        ``None`` → SGLang default.
    sgl_sleep_on_idle : bool
    sgl_mamba_full_memory_ratio : float or None
        Mamba state vs full KV cache memory ratio (hybrid mamba models only).
        ``None`` → SGLang default (0.9).
    sgl_mamba_scheduler_strategy : str or None
        ``None`` → SGLang default.
    sgl_enable_spec_v2 : bool
        Set ``SGLANG_ENABLE_SPEC_V2=1`` before SGLang start.
    custom_rm_actor_impl : str or None
        Registered custom RM actor implementation name.
    allocated_gpus : int or None
        Exact GPU allocation for a gen-rm inference engine in Ray placement.
    """
    dp_size: int = field(
        default=1,
        metadata={
            'help':
                '''DP size of sglang / vllm. Different from fsdp2 and megatron, the DP size of
            sglang is overlapped with TP.

            Megatron:
                dp0: [tp0, tp1]
                dp1: [tp0, tp1]

            Sglang:
                [dp0 and tp0, dp1 and tp1]
            ''',
        },
    )
    enable_deepep_moe: bool = field(default=False, metadata={"help": "enable deepep moe."})
    use_fast_tokenizer: bool = field(default=False, metadata={"help": "use fast tokenizer."})

    dtype: str = field(default='bfloat16', metadata={"help": "gpu memory frac."})
    gpu_memory_utilization: float = field(default=0.7, metadata={"help": "gpu memory frac."})
    max_running_requests: Optional[int] = field(
        default=None, metadata={"help": "max running request"}
    )
    temperature: float = field(default=1.0, metadata={"help": "ppo temperature"})
    top_k: int = field(default=-1, metadata={"help": "ppo top-k"})
    top_p: float = field(default=1.0, metadata={"help": "ppo top-p"})
    min_p: float = field(default=0.0, metadata={"help": "min-p sampling, 0 disables"})
    generate_max_tokens: int = field(default=128, metadata={"help": "generate max tokens"})
    seed: Optional[int] = field(default=42, metadata={"help": "prompt seed"})
    engine_seed: Optional[int] = field(default=42, metadata={"help": "engine seed"})
    allow_auto_truncate: bool = field(
        default=False, metadata={"help": "sglang allow auto truncate"}
    )
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty"})
    frequency_penalty: float = field(default=0.0, metadata={"help": "frequency penalty"})
    presence_penalty: float = field(default=0.0, metadata={"help": "presence penalty"})
    load_format: str = field(default='auto', metadata={"help": "load format"})
    enable_weights_cpu_backup: bool = field(
        default=True, metadata={"help": "enable weights cpu backup"}
    )
    mm_per_request_timeout: Optional[int] = field(
        default=None, metadata={"help": "multi modal processor timeout second"}
    )

    custom_rm_actor_impl: Optional[str] = field(
        default=None, metadata={"help": "the registered custom_rm_actor_impl"}
    )

    allocated_gpus: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "Exact GPU allocation for this gen-rm engine in Ray placement. "
                "None keeps the default round-robin allocator. When set, it must be a "
                "positive multiple of tensor_model_parallel_size * pipeline_model_parallel_size."
        },
    )

    disable_cuda_graph: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable CUDA graph (enforce eager mode). "},
    )

    attention_backend: Optional[str] = field(
        default="flashinfer",
        metadata={
            "help":
                "Attention backend name. Uses sglang naming convention as the "
                "canonical format (e.g. 'flashinfer', 'triton', 'fa3'). "
                "For vllm, the name is automatically mapped to the vllm "
                "equivalent via infer_engine.py:InferEngine._map_sgl_attention_backend_to_vllm_attention_backend. "
                "None lets each backend auto-select."
        },
    )

    sgl_crash_dump_folder: Optional[str] = field(
        default=None, metadata={"help": "The folder to save the crash dump files."}
    )

    sgl_chunked_prefill_size: Optional[int] = field(
        default=None, metadata={"help": "Chunked prefill size for sglang."}
    )

    sgl_schedule_conservativeness: Optional[float] = field(
        default=None, metadata={"help": "Schedule conservativeness for sglang."}
    )

    sgl_sleep_on_idle: bool = field(default=False, metadata={"help": "Let sglang sleep when idle."})

    sgl_mamba_full_memory_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help":
                "The ratio of mamba state memory to full kv cache memory for sglang "
                "(only meaningful for hybrid mamba models). None uses sglang's default (0.9)."
        },
    )

    sgl_mamba_scheduler_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Sglang mamba scheduler strategy. None uses sglang's default. "
                "Supported values: no_buffer, extra_buffer."
        },
    )
    sgl_enable_spec_v2: bool = field(
        default=False,
        metadata={"help": "Set SGLANG_ENABLE_SPEC_V2=1 before starting sglang."},
    )
    override_infer_engine_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.load_format in [
            'auto', 'dummy'
        ], f"load format {self.load_format} not supported"
        assert self.sgl_mamba_scheduler_strategy in [
            None,
            "no_buffer",
            "extra_buffer",
        ], f"sgl_mamba_scheduler_strategy {self.sgl_mamba_scheduler_strategy} not supported"
