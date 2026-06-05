from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class CheckpointConfig(MappingProtocol):
    """Configuration for model checkpoint saving and loading.

    Retention policy combines ``save_interval``, ``save_retain_interval``, and
    ``save_total_limit``. Example: 10200 steps with ``save_interval=100``,
    ``save_retain_interval=1000``, ``save_total_limit=3`` keeps only steps
    9000, 10000, 10200.

    Attributes
    ----------
    load_ckpt_path : str or None
    save_ckpt_path : str or None
    no_load_optim : bool
        Skip loading optimizer state.
    no_save_optim : bool
        Skip saving optimizer state.
    load_ref_ckpt_path : str or None
    save_ref_ckpt_path : str or None
    save_total_limit : int or None
        Max number of checkpoints kept on disk.
    save_retain_interval : int or None
        Retain one checkpoint every ``n`` steps (never auto-deleted).
    use_dist_checkpointing : bool or None
    async_save : bool or None
    convert_mcore_to_hf_online : bool
        Export HF model whenever a checkpoint is saved.
    export_hf_save_path : str or None
    mbridge_distributed_filesystem : bool or None
    mbridge_save_every_n_ranks : int
        Save from every ``n`` ranks (reduces I/O).
    convert_mcore_to_hf_offline : bool
    convert_target_step : int or None
    override_tokenizer_special_token : dict
        Override tokenizer special tokens (e.g. ``<|im_end|>`` as EOS for
        base-model fine-tuning).
    """
    load_ckpt_path: Optional[str] = field(default=None, metadata={"help": "Load model path"})
    save_ckpt_path: Optional[str] = field(default=None, metadata={"help": "Save model path"})
    no_load_optim: bool = field(default=False, metadata={"help": "Whether to load optimizer"})
    no_save_optim: bool = field(default=False, metadata={"help": "Whether to save optimizer"})

    load_ref_ckpt_path: Optional[str] = field(
        default=None, metadata={"help": "Reference ckpt load path"}
    )
    save_ref_ckpt_path: Optional[str] = field(
        default=None, metadata={"help": "Reference ckpt save path"}
    )
    # save_interval: 每 n step 保存一次 ckpt, 每次保存最新的 ckpt 后, 当 save_retain_interval 不为 None 时会自动删除上次的 ckpt
    # save_retain_interval: 每 m step 保存一次 ckpt, 不会自动删除
    # save_total_limit: 类似 hf 中的参数, 在上述两个参数基础上限制存保存的 ckpt 的总数量

    # e.g. save_interval=100 & save_retain_interval = 1000
    # 每 100 次迭代保存一次，但只保留每 1000 次迭代的 checkpoint + 最新的 checkpoint
    # save_total_limit=3 会在 上述两个开关的基础上，限制总的 checkpoint 数量, 删除旧的 ckpt 只剩下最新的 3 个

    # 训练到 10200 step 时的 ckpt 保留策略举例:
    # 1. save_interval=100                                                    保留 102 个 ckpts (100, 200, ..., 10100, 10200)
    # 2. save_interval=100 & save_total_limit=3                               保留   3 个 ckpts (10000, 10100, 10200)
    # 3. save_interval=100 & save_retain_interval=1000                        保留  11 个 ckpts (1000, 2000, ..., 10000, 10200)
    # 4. save_interval=100 & save_retain_interval=1000 & save_total_limit=3   保留   3 个 ckpts (9000, 10000, 10200)

    save_total_limit: Optional[int] = field(
        default=None, metadata={"help": "Max number of total checkpoints to save"}
    )
    save_retain_interval: Optional[int] = field(
        default=None, metadata={"help": "Retain checkpoints every n steps"}
    )

    use_dist_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "use torch distributed checkpoint"}
    )
    async_save: Optional[bool] = field(
        default=False, metadata={"help": "use torch distributed checkpoint"}
    )

    convert_mcore_to_hf_online: bool = field(
        default=False, metadata={"help": "Export hf model when save checkpoint"}
    )
    export_hf_save_path: Optional[str] = field(
        default=None, metadata={"help": "Export hf model path"}
    )
    mbridge_distributed_filesystem: Optional[bool] = field(
        default=True, metadata={"help": "use mbridge distributed filesystem"}
    )
    mbridge_save_every_n_ranks: int = field(default=1, metadata={"help": "save every n rank"})

    convert_mcore_to_hf_offline: bool = field(
        default=False, metadata={"help": "Convert mcore to hf offline"}
    )
    convert_target_step: Optional[int] = field(
        default=None, metadata={"help": "Convert target step"}
    )
    override_tokenizer_special_token: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help":
                "Override tokenizer special token. 正常模型训练不用，从 base model finetune 且 base model 的 eos_token 是 <|endoftext|> 的才需要配置，一般是配置成对话结束符 <|im_end|>"
        }
    )
    strict_export: bool = field(
        default=True,
        metadata={
            "help": "Strict compare the keys of mcore and hf checkpoint when save hf checkpoint"
        }
    )
    skip_save_mcore_model: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to skip saving model weights to the checkpoint. "
                "When enabled, model weights will be loaded from HF format via bridge during load_checkpoint."
                "mcore_v0.13.1 not support skip_save_mcore_model=True, case metadata['distrib_optim_sharding_type'] != 'dp_reshardable' "
        }
    )

    def __post_init__(self):
        if self.skip_save_mcore_model:
            assert self.convert_mcore_to_hf_online, (
                "convert_mcore_to_hf_online must be True when skip_save_mcore_model=True, "
                "so that model weights can be saved to HF format."
            )
