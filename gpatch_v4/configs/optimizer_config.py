from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class OptimizerConfig(MappingProtocol):
    """Configuration for the optimizer and learning rate scheduler.

    Attributes
    ----------
    lr : float
    min_lr : float
        Minimum LR after decay.
    vit_lr : float or None
        ``None`` → use base ``lr``.
    vit_min_lr : float or None
    vit_proj_lr : float or None
    vit_proj_min_lr : float or None
    lr_warmup_steps : int
    lr_warmup_step_frac : float
        Fraction of total steps for warmup (alternative to ``lr_warmup_steps``).
    lr_warmup_init : float
    lr_decay_steps : int or None
        ``None`` → use total training steps.
    lr_wsd_decay_style : str
        ``"constant"`` / ``"exponential"`` / ``"cosine"``.
    lr_wsd_decay_steps : int or None
    lr_decay_style : str
        ``"constant"`` / ``"linear"`` / ``"cosine"`` / ``"inverse_square_root"``.
    lr_num_cycles : int
        For the cosine scheduler.
    lr_power : float
        For the polynomial scheduler.
    weight_decay : float
    weight_decay_incr_style : str
    optimizer_type : str
        E.g. ``"adam"``.
    adam_beta1 : float
    adam_beta2 : float
    adam_epsilon : float
    max_grad_norm : float
    update_lr_by_train_step : bool
        *True* → update LR every optimizer step in ``rl_train_actor`` /
        ``rl_train_value`` (fine-grained); *False* → once per PPO step
        (V4 coarse-grained, default).
    use_checkpoint_opt_param_scheduler : bool or None
        Restore the optimizer param scheduler from a checkpoint.
    override_optimizer_config : dict or None
    """
    lr: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"},
    )
    min_lr: float = field(
        default=0.0,
        metadata={"help": "Minimum learning rate"},
    )
    vit_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Vision model learning rate"},
    )
    vit_min_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Vision model minimum learning rate"},
    )
    vit_proj_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Vision projection model learning rate"},
    )
    vit_proj_min_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Vision projection model minimum learning rate"},
    )
    und_lr_scale: float = field(
        default=1.0,
        metadata={
            "help":
                (
                    "Scale factor for WGOv3Beta understanding-path parameter groups relative "
                    "to the base optimizer lr. 1.0 disables the split."
                )
        },
    )
    lr_warmup_steps: int = field(
        default=0,
        metadata={"help": "Learning rate warmup step"},
    )
    lr_warmup_step_frac: float = field(
        default=0.0,
        metadata={"help": "Learning rate warmup step fraction"},
    )
    lr_warmup_init: float = field(
        default=0.0,
        metadata={"help": "Learning rate warmup init"},
    )
    lr_decay_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Learning rate decay steps"},
    )
    lr_wsd_decay_style: str = field(
        default="exponential",
        metadata={
            "help": "Weight-standard-deviation decay style: 'constant', 'exponential', or 'cosine'"
        },
    )
    lr_wsd_decay_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps for weight-standard-deviation decay"},
    )
    lr_decay_style: str = field(
        default="linear",
        metadata={
            "help":
                "Learning rate decay style: 'constant', 'linear', 'cosine', or 'inverse_square_root'"
        },
    )
    lr_num_cycles: int = field(
        default=1,
        metadata={"help": "Number of cycles for the cosine scheduler."},
    )
    lr_power: float = field(
        default=1.0,
        metadata={"help": "Power for the polynomial scheduler."},
    )

    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"},
    )
    weight_decay_incr_style: str = field(
        default="constant",
        metadata={"help": "Weight decay increment style"},
    )
    optimizer_type: str = field(
        default="adam",
        metadata={"help": "Optimizer type"},
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Adam beta1"},
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Adam beta2"},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Adam epsilon"},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "max_grad_norm"},
    )
    update_lr_by_train_step: bool = field(
        default=False,
        metadata={
            "help":
                "If True, update LR scheduler per optimizer step (fine-grained). "
                "If False, update once per PPO step (coarse-grained)."
        },
    )
    use_checkpoint_opt_param_scheduler: Optional[bool] = field(
        default=False,
        metadata={"help": "Use checkpoint opt param scheduler"},
    )
    override_optimizer_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Override optimizer config"},
    )
    muon_momentum: float = field(
        default=0.95,
        metadata={"help": "Muon SGD momentum coefficient"},
    )
    muon_nesterov: bool = field(
        default=True,
        metadata={"help": "Use Nesterov momentum in Muon"},
    )
    muon_num_ns_steps: int = field(
        default=5,
        metadata={"help": "Newton-Schulz iteration steps for Muon"},
    )
    muon_extra_scale_factor: float = field(
        default=0.2,
        metadata={"help": "Extra scale factor applied to Muon per-parameter lr, "
                 "validated in Moonlight paper Table 8"},
    )
    report_post_clip_grad_norm: bool = field(
        default=False,
        metadata={
            "help":
                "Debug switch (mcore_engine only). If True, additionally "
                "report the L2 grad norm *after* clipping to verify that "
                "max_grad_norm has taken effect. Adds one all-reduce per "
                "optimizer step."
        },
    )
