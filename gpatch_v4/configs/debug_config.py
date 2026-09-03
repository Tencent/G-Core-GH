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
    save_decode_jsonl : bool
        Also save decoded rollout trajectories as JSONL when rollout data is saved.
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
    synthetic_generation_prompt_expansion : bool
        Repeat each SFT generation prompt by a deterministic per-sample factor.
    debug_dump_first_n_ppo_step_moe_token_dist : int
        DEBUG: dump per-layer MoE per-expert token counts. ``0`` disables;
        ``-1`` every PPO step; ``N>0`` only first ``N`` PPO steps.
    debug_moe_dist_dump_dir : str
        DEBUG: root for ``step_{n}/pp{pp_rank}.jsonl`` dumps. Created on demand.
    debug_dump_expert_token_counts: bool
        DEBUG: dump one global-batch per-expert token counts then sys.exit(0).
        Forces freeze_router_correction_bias=False to activate the counting hooks.
        Feeds the offline expert re-permutation pipeline.
    debug_dump_expert_token_counts_path: str
        DEBUG: path to dump expert token counts.
    log_thd_pack_layout : bool
        Rank-0 first microbatch of each step. Dumps per-segment
        ``raw_lens`` / ``pad_lens``, which grow with pack occupancy.
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
    save_decode_jsonl: bool = field(
        default=False,
        metadata={"help": "Whether to additionally save decoded rollout trajectories as JSONL."},
    )
    dump_agentic_trajectory: bool = field(
        default=False,
        metadata={
            "help":
                (
                    "DEBUG: dump decoded agentic per-turn prompt/response and final "
                    "concatenated trajectory sequence to JSONL files with special tokens kept."
                )
        },
    )
    agentic_trajectory_dump_dir: str = field(
        default="debug-tmp/agentic_trajectory",
        metadata={"help": "DEBUG: root dir for per-trajectory agentic JSONL dumps."},
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
    synthetic_generation_prompt_expansion: bool = field(
        default=False,
        metadata={
            "help":
                (
                    "DEBUG: repeat each SFT generation prompt by a deterministic "
                    "per-sample factor in [40, 100], capped at training.seq_length."
                )
        },
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
    save_first_post_rollout_batch: bool = field(
        default=False,
        metadata={
            "help":
                (
                    "DEBUG: save rollout_batches/metrics immediately after rollout "
                    "and optional post-filter. Used to align loss with/without "
                    "dynamic context parallel while avoiding sampler randomness."
                )
        },
    )
    load_first_post_rollout_batch: bool = field(
        default=False,
        metadata={
            "help":
                (
                    "DEBUG: replace rollout_batches/metrics immediately after rollout "
                    "and optional post-filter with saved post-rollout data."
                )
        },
    )
    post_rollout_batch_debug_dir: str = field(
        default="debug-tmp",
        metadata={"help": "DEBUG: directory for post-rollout batch dump/load."},
    )
    debug_align_mode: bool = field(
        default=False,
        metadata={"help": "DEBUG: enable alignment debug mode."},
    )
    debug_dump_expert_token_counts: bool = field(
        default=False,
        metadata={
            "help":
                (
                    "DEBUG: dump one global-batch per-expert token counts "
                    "([num_topk_layers, num_experts]) then sys.exit(0). Forces "
                    "freeze_router_correction_bias=False to activate the counting hooks. "
                    "Feeds the offline expert re-permutation pipeline."
                )
        },
    )
    debug_dump_expert_token_counts_path: str = field(
        default="debug-tmp/debug_moe_router/counts.pt",
        metadata={"help": ("DEBUG: path to dump expert token counts.")},
    )
    log_thd_pack_layout: bool = field(
        default=False,
        metadata={
            "help":
                (
                    "DEBUG: rank-0 first microbatch dumps THD pack/CP layout "
                    "(raw_lens / pad_lens / n_cross). Off by default; lists "
                    "grow with pack occupancy."
                )
        },
    )
    ppo_padding_check_mode: str = field(
        default="off",
        metadata={
            "help":
                (
                    "GDEBUG: check the PPO policy/critic config at actor init. "
                    "off | warn | abort."
                )
        },
    )

    def __post_init__(self):
        # keep the modes literal here: loading a config must not import gdebug
        assert self.ppo_padding_check_mode in ["off", "warn", "abort"], (
            f"debug.ppo_padding_check_mode={self.ppo_padding_check_mode!r}, "
            "expected one of ('off', 'warn', 'abort')"
        )
