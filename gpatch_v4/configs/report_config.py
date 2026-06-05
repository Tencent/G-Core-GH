import collections
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.profile_config import ProfileConfig
from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class ReportConfig(MappingProtocol):
    """Configuration for experiment tracking and reporting.

    Attributes
    ----------
    report_to : str
        ``"none"`` / ``"wandb"`` / ``"tensorboard"``.
    wandb_key : str or None
    wandb_host : str or None
    wandb_project : str
    wandb_save_dir : str
    wandb_exp_name : str or None
    tensorboard_dir : str
    log_dir : str or None
        Defaults to ``checkpoint.save_ckpt_path/logs`` when unset.
    timing_log_option : str
        ``"max"`` / ``"minmax"`` / ``"all"``.
    timing_log_level : int
        ``0`` iteration only / ``1`` major ops / ``2`` all ops.
    log_level : str
        ``"info"`` or ``"debug"``.
    debug_log_to_file : bool
        Write ``log_debug()`` to per-rank debug shard logs.
    capture_infer_engine_log : bool
        Capture inference engine stdout/stderr to per-engine files under
        ``infer_engine_log``.
    profile : ProfileConfig
    """
    report_to: str = field(
        default="none",
        metadata={"help": "Where to report the results."},
    )
    wandb_key: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb API key."},
    )
    wandb_host: Optional[str] = field(
        default="http://localhost:8080",
        metadata={"help": "Wandb host."},
    )
    wandb_project: str = field(
        default="grpo_train",
        metadata={"help": "Name of the wandb project."},
    )
    wandb_save_dir: str = field(
        default="wandb_local",
        metadata={"help": "Path of the wandb save directory."},
    )
    wandb_exp_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the wandb experiment."},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the wandb run id"},
    )
    tensorboard_dir: str = field(
        default="tensorboard_local",
        metadata={"help": "Path of the tensorboard directory."},
    )
    log_to_driver: bool = field(
        default=True,
        metadata={"help": "Whether to log to driver."},
    )
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Base directory for GPatch task logs."},
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "GPatch task log verbosity. Use 'debug' to include log_debug() in training.log."
        },
    )
    log_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path of the log file."},
    )
    debug_log_to_file: bool = field(
        default=False,
        metadata={"help": "Write log_debug() records to per-rank DEBUG-only logs."},
    )
    capture_infer_engine_log: bool = field(
        default=True,
        metadata={"help": "Write sglang/vLLM stdout and stderr to infer_engine_log files."},
    )

    timing_log_option: str = field(
        default="minmax",
        metadata={
            "help":
                'Options for logging timing:'
                '  max: report the max timing across all ranks'
                '  minmax: report min and max timings across all ranks'
                '  all: report timings of all ranks.'
        }
    )
    timing_log_level: int = field(
        default=0,
        metadata={
            "help":
                'Granularity level to measure and report timing. '
                '   0: report only iteration time and make sure timing '
                '      does not introduce extra overhead.'
                '   1: report timing for operations that are executed '
                '      very limited times (basically once) during '
                '      each iteration (such as gradient all-reduce) '
                '   2: report timing for operations that migh be '
                '      executed numerous times during each iteration. '
        }
    )
    profile: ProfileConfig = field(default_factory=ProfileConfig)

    def __post_init__(self):
        assert self.timing_log_level in [
            0, 1, 2
        ], f"Invalid timing_log_level: {self.timing_log_level}"
        assert self.timing_log_option in [
            "max", "minmax", "all"
        ], f"Invalid timing_log_option: {self.timing_log_option}"
        assert self.log_level.lower() in ["debug", "info"], f"Invalid log_level: {self.log_level}"


@dataclass
class MonitorConfig(MappingProtocol):
    """Configuration for the training monitor server.

    Attributes
    ----------
    do_monitor : bool
    monitor_server_ip : str or None
    monitor_port : int or None
    """
    do_monitor: bool = field(default=False, metadata={"help": "Whether to do monitor."})
    monitor_server_ip: Optional[str] = field(default=None, metadata={"help": "Monitor server ips"})
    monitor_port: Optional[int] = field(default=60000, metadata={"help": "Monitor port"})
