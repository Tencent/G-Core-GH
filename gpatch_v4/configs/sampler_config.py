from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Union

from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.infer_engine_config import InferEngineConfig
from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class ModelInfo(MappingProtocol):
    """Information about a single inference model used by the sampler.

    Attributes
    ----------
    model_arch : str or None
        See ``gpatch_v4.core.constants.MODEL_ARCH``.
    hf_model_path : str or None
    gen_rollout_py_path : str or None
        Python file containing the generation rollout implementation.
    gen_rollout_fn_name : str or None
    """
    model_arch: Optional[str] = field(
        default=None,
        metadata={"help": "model arch, see gpatch_v4.core.constants.MODEL_ARCH for enums"},
    )
    hf_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "hf model path"},
    )
    gen_rollout_py_path: Optional[str] = field(
        default=None,
        metadata={"help": "gen rollout impl file"},
    )
    gen_rollout_fn_name: Optional[str] = field(
        default=None,
        metadata={"help": "gen rollout fn name"},
    )


@dataclass
class SamplerConfig(MappingProtocol):
    """Configuration for the inference sampler (rollout engine).

    Attributes
    ----------
    dist_config : DistConfig
    backend : str
        E.g. ``"sglang"``.
    sampler_type : str
        ``"sampler"`` (standard) or ``"off-policy-sampler"``.
    model_info : list of ModelInfo or None
    infer_engine_configs : list of InferEngineConfig or None
    """
    dist_config: DistConfig = field(default_factory=DistConfig)
    backend: str = field(default="sglang", metadata={"help": "infer engine impl"})
    sampler_type: str = field(default="sampler", metadata={"help": "sampler type"})

    model_info: Optional[List[ModelInfo]] = field(default=None)
    infer_engine_configs: Optional[List[InferEngineConfig]] = field(default=None)

    def __post_init__(self):
        assert self.sampler_type in ["sampler", "off-policy-sampler"]
        if self.infer_engine_configs is not None:
            for ie_cfg in self.infer_engine_configs:
                ie_cfg.ensure_generate_params()
