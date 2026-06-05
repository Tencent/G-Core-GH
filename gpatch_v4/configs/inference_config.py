from dataclasses import dataclass, field
from typing import Optional

from gpatch_v4.configs.infer_engine_config import InferEngineConfig
from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class InferResultConfig(MappingProtocol):
    output_dir: str = field(
        default="./inference_output", metadata={"help": "Output directory for inference results"}
    )
    output_prefix: Optional[str] = field(default=None, metadata={"help": "Output file prefix"})
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

    constant_seed: Optional[int] = field(
        default=None, metadata={"help": "Constant seed for inference"}
    )
    enable_think_mode: bool = field(
        default=False, metadata={"help": "Enable think mode for DeepSeek-R1 style reasoning"}
    )
    sampling_repeat: int = field(default=1, metadata={"help": "Number of times to repeat sampling"})
