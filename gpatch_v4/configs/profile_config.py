import collections
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class ProfileConfig(MappingProtocol):
    """Configuration for PyTorch profiler integration.

    Attributes
    ----------
    enable_profile : bool
    profile_start_step : int
    profile_end_step : int
    profile_save_dir : str
        Directory for trace files.
    """
    enable_profile: bool = field(
        default=False,
        metadata={"help": "Whether to enable profiling."},
    )
    profile_start_step: int = field(default=2, metadata={"help": "The step to start profiling."})
    profile_end_step: int = field(default=3, metadata={"help": "The step to end profiling."})
    profile_save_dir: str = field(
        default="profile_result", metadata={"help": "The directory to save profiling results."}
    )
