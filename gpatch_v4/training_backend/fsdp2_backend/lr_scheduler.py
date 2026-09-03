# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""Learning rate scheduler for FSDP training."""

import math

import torch
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import override

from gpatch_v4.utils import log


class FSDPLRScheduler(LRScheduler):
    """Learning rate scheduler for FSDP training.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        init_lr (float): Initial learning rate.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        lr_warmup_steps (int): Number of warmup steps.
        lr_decay_steps (int): Number of decay steps.
        lr_decay_style (str): Decay style for learning rate.
        use_checkpoint_lr_scheduler (bool, optional): Whether to use the checkpoint values
            for the lr scheduler.
        override_lr_scheduler (bool, optional): Whether to override the lr scheduler values
            with the class values.
        wsd_decay_steps (int, optional): Number of weight decay decay steps.
        lr_wsd_decay_style (str, optional): Decay style for learning rate during weight decay decay
            steps.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        use_checkpoint_lr_scheduler: bool | None = True,
        override_lr_scheduler: bool | None = False,
        wsd_decay_steps: int | None = None,
        lr_wsd_decay_style: str | None = None,
        last_epoch: int = -1,
    ) -> None:
        # Store our custom parameters
        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style

        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_style = lr_decay_style
        if self.lr_decay_style == "WSD":
            assert self.wsd_decay_steps is not None

        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler

        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, "both override and use-checkpoint are set."

        # Initialize parent class
        super().__init__(optimizer, last_epoch)

        log(f"> learning rate decay style: {self.lr_decay_style}")

    def _get_lr_for_group(self, param_group: dict) -> float:
        """Compute learning rate for a specific parameter group.

        Args:
            param_group (dict): parameter group from the optimizer.

        Returns:
            float: learning rate for this parameter group.
        """
        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.last_epoch <= self.lr_warmup_steps:
            return self.init_lr + (
                (max_lr - self.init_lr) * float(self.last_epoch) / float(self.lr_warmup_steps)
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == "constant":
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.last_epoch > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == "inverse-square-root":
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.last_epoch, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        num_steps_ = self.last_epoch - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0

        delta_lr = max_lr - min_lr
        coeff = None

        if self.lr_decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.lr_decay_style == "WSD":
            wsd_anneal_start_ = self.lr_decay_steps - self.wsd_decay_steps
            if self.last_epoch <= wsd_anneal_start_:
                coeff = 1.0
            else:
                wsd_steps = self.last_epoch - wsd_anneal_start_
                wsd_decay_ratio = float(wsd_steps) / float(self.wsd_decay_steps)
                if self.lr_wsd_decay_style == "linear":
                    coeff = 1.0 - wsd_decay_ratio
                elif self.lr_wsd_decay_style == "cosine":
                    coeff = 0.5 * (math.cos(math.pi * wsd_decay_ratio) + 1.0)
                elif self.lr_wsd_decay_style == "exponential":
                    coeff = (2.0 * math.pow(0.5, wsd_decay_ratio)) - 1.0
                elif self.lr_wsd_decay_style == "minus_sqrt":
                    coeff = 1.0 - math.sqrt(wsd_decay_ratio)
        else:
            raise Exception(f"{self.lr_decay_style} decay style is not supported.")

        assert coeff is not None
        return min_lr + coeff * delta_lr

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rates for each parameter group.

        Returns:
            list[float]: A list of learning rates, one for each parameter group.
        """
        return [
            self._get_lr_for_group(group) * group.get("lr_mult", 1.0)
            for group in self.optimizer.param_groups
        ]


def get_lr_scheduler(config, optimizer: torch.optim.Optimizer) -> FSDPLRScheduler:
    optimizer_config = config.optimizer
    lr_decay_steps = optimizer_config.lr_decay_steps
    lr_warmup_steps = optimizer_config.lr_warmup_steps

    if config.training.get("total_ppo_step") is not None:
        wd_incr_steps = config.training.total_ppo_step
    else:
        wd_incr_steps = config.training.total_training_step

    if optimizer_config.get("lr_decay_steps") is None:
        lr_decay_steps = wd_incr_steps

    wsd_decay_steps = None
    if optimizer_config.get("lr_wsd_decay_steps") is not None:
        wsd_decay_steps = optimizer_config.lr_wsd_decay_steps

    if optimizer_config.get("lr_warmup_step_frac") is not None and (
        optimizer_config.get("lr_warmup_steps") is None or optimizer_config.lr_warmup_steps <= 0
    ):
        lr_warmup_steps = int(optimizer_config.lr_warmup_step_frac * lr_decay_steps)

    lr_scheduler = FSDPLRScheduler(
        optimizer,
        init_lr=optimizer_config.lr_warmup_init,
        max_lr=optimizer_config.lr,
        min_lr=optimizer_config.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=optimizer_config.lr_decay_style,
        use_checkpoint_lr_scheduler=optimizer_config.use_checkpoint_opt_param_scheduler,
        override_lr_scheduler=(not optimizer_config.use_checkpoint_opt_param_scheduler),
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=optimizer_config.lr_wsd_decay_style,
    )

    return lr_scheduler
