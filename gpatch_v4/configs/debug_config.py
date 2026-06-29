import collections
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class DebugConfig(MappingProtocol):
    """Configuration for debugging and diagnostic tools.

    Attributes
    ----------
    debug_engine_update_weight : bool
    debug_engine_save_path : str
    save_first_rollout_data : bool
    save_every_rollout_data : bool
    skip_rollout_load_from_disk : bool
        DEBUG: skip RolloutController fire/collect AND actor process_rollout;
        load saved batches/metrics from ``load_rollout_path`` instead.
    load_rollout_path : str
        DEBUG: directory for loading saved rollout batches/metrics. Defaults
        to ``"debug-tmp"`` (hardcoded save directory).
    load_rollout_step : int
        DEBUG: which saved ppo_step's rollout to reuse for every step.
    save_images : bool
    save_images_dir : str or None
    images_attr_name : str
        Pipeline-returned images attribute name.
    test_ray_rpc : bool
    trainer_return_ppo_step_metrics : bool
    debug_no_optim : bool
        Disable optimizer updates entirely.
    debug_dump_first_n_ppo_step_moe_token_dist : int
        DEBUG: dump per-layer MoE per-expert token counts. ``0`` disables;
        ``-1`` every PPO step; ``N>0`` only first ``N`` PPO steps.
    debug_moe_dist_dump_dir : str
        DEBUG: root for ``step_{n}/pp{pp_rank}.jsonl`` dumps. Created on demand.
    """
    debug_engine_update_weight: bool = field(
        default=False, metadata={"help": "Whether to do debug."}
    )
    debug_engine_save_path: str = field(
        default='/root/engine_ckpt', metadata={'help': 'engine checkpoint save path'}
    )
    save_first_rollout_data: bool = field(
        default=False, metadata={"help": "Whether to save the first rollout data."}
    )
    save_every_rollout_data: bool = field(
        default=False, metadata={"help": "Whether to save every rollout data."}
    )
    skip_rollout_load_from_disk: bool = field(
        default=False,
        metadata={
            "help":
                (
                    "DEBUG: skip RolloutController fire/collect AND actor process_rollout, "
                    "load saved batches from `load_rollout_path` instead."
                )
        },
    )
    load_rollout_path: str = field(
        default="debug-tmp",
        metadata={
            "help":
                (
                    "DEBUG: directory to load saved rollout batches/metrics from when "
                    "`skip_rollout_load_from_disk` is enabled. Defaults to the same "
                    "hardcoded save directory."
                )
        },
    )
    load_rollout_step: int = field(
        default=0,
        metadata={"help": "DEBUG: which saved ppo_step's rollout to reuse for every step."},
    )
    save_images: bool = field(default=False, metadata={"help": "Whether to save the images."})
    save_images_dir: Optional[str] = field(default=None, metadata={"help": "Save images dir"})
    images_attr_name: str = field(
        default='images', metadata={'help': 'name of images attr return by pipeline'}
    )
    test_ray_rpc: bool = field(default=False, metadata={"help": "Whether to test ray rpc."})
    trainer_return_ppo_step_metrics: bool = field(
        default=False, metadata={"help": "returns ppo step metrics for testing purpose"}
    )
    debug_no_optim: bool = field(
        default=False, metadata={"help": "disable optim for debugging purpose"}
    )
    debug_dump_first_n_ppo_step_moe_token_dist: int = field(
        default=0,
        metadata={
            "help":
                (
                    "DEBUG: dump per-layer MoE per-expert token counts. "
                    "0 = disabled, -1 = every step, N>0 = only first N PPO steps."
                )
        },
    )
    debug_moe_dist_dump_dir: str = field(
        default="debug-tmp/moe_dist",
        metadata={
            "help":
                (
                    "DEBUG: root dir for MoE distribution dumps; per-step subdir "
                    "step_{n}/ is auto-created on demand."
                )
        },
    )
    experimental_pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help":
                (
                    "DEBUG: pad to max length instead of pad to multiple of. "
                    "This is for experimental purpose."
                )
        },
    )
    debug_truncate_num_hidden_layers: Optional[int] = field(
        default=None,
        metadata={
            "help":
                (
                    "DEBUG: if set, truncate the model config to this number of "
                    "decoder layers before constructing the model. Currently only "
                    "honored by the HpModule (DSV4 / Qwen3.5-MoE) construction "
                    "path in fsdp2 backend; intended for OOM smoke runs of large "
                    "MoE models. None (default) = use the upstream config as-is."
                )
        },
    )
    disable_save_checkpoint: bool = field(
        default=False,
        metadata={"help": "DEBUG: skip all checkpoint saves (including final save on exit)."},
    )
    ignore_global_retention_ratio: bool = field(
        default=False,
        metadata={
            "help":
                (
                    "DEBUG: if True, force ``loss_input.global_retention_ratio`` to be "
                    "None in the policy loss function so that the bwd_loss is NOT "
                    "scaled by 1/global_retention_ratio. Used for grad-scaling "
                    "diagnostic experiments."
                )
        },
    )
