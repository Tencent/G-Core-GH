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
        Inclusive start global train step (profiler ``start()``).
    profile_end_step : int
        Inclusive end global train step (profiler ``stop()`` + export).
        Set equal to ``profile_start_step`` for a single step.
    profile_save_dir : str
        Directory for trace files.
    record_shapes : bool
        Record tensor shapes (increases trace size).
    with_stack : bool
        Record Python stacks (largest size multiplier; keep off for browsing).
    profile_memory : bool
        Record allocator events (can grow quickly).
    with_modules : bool
        Record NN module hierarchy.
    export_ranks : list of int
        Ranks that write Chrome traces. Empty list means all ranks.
    gzip_trace : bool
        Write ``.json.gz`` after export (Perfetto opens gzip; Chrome may not).
    use_nsys : bool
        Wrap Ray training actors with Nsight Systems (``nsys``) via
        ``ray.remote(runtime_env={"nsight": ...})``. Sampling is bounded by
        ``torch.cuda.profiler.start/stop`` around ``profile_start_step`` /
        ``profile_end_step``. All ranks are profiled. Independent of
        ``enable_profile`` (PyTorch profiler).
    """
    enable_profile: bool = field(
        default=False,
        metadata={"help": "Whether to enable profiling."},
    )
    profile_start_step: int = field(
        default=2,
        metadata={"help": "Inclusive global step to start profiling."},
    )
    profile_end_step: int = field(
        default=3,
        metadata={
            "help":
                "Inclusive global step to stop profiling and export. "
                "Use the same value as profile_start_step for one step."
        },
    )
    profile_save_dir: str = field(
        default="profile_result",
        metadata={"help": "The directory to save profiling results."},
    )
    record_shapes: bool = field(
        default=True,
        metadata={"help": "Record op input shapes (increases JSON size)."},
    )
    with_stack: bool = field(
        default=True,
        metadata={
            "help":
                "Record Python stacks. Main Chrome-trace size killer; keep False "
                "unless debugging CPU Python overhead."
        },
    )
    profile_memory: bool = field(
        default=False,
        metadata={"help": "Record CUDA allocator events."},
    )
    with_modules: bool = field(
        default=False,
        metadata={"help": "Record nn.Module hierarchy."},
    )
    export_ranks: List[int] = field(
        default_factory=list,
        metadata={"help": "Ranks that export Chrome traces. Empty list exports all ranks."},
    )
    gzip_trace: bool = field(
        default=True,
        metadata={"help": "Gzip the Chrome trace after export (Perfetto-friendly)."},
    )
    use_nsys: bool = field(
        default=False,
        metadata={
            "help":
                "Wrap Ray training actors with Nsight Systems (nsys) via "
                "ray runtime_env. All ranks are profiled. Sampling is bounded "
                "by torch.cuda.profiler.start/stop around profile_start_step / "
                "profile_end_step."
        },
    )

    def __post_init__(self):
        if self.enable_profile or self.use_nsys:
            if self.profile_start_step > self.profile_end_step:
                raise ValueError(
                    "profile_start_step must be less than or equal to profile_end_step"
                )
            if self.profile_start_step < 0:
                raise ValueError("profile_start_step must be greater than or equal to 0")
            if self.profile_end_step < 0:
                raise ValueError("profile_end_step must be greater than or equal to 0")
