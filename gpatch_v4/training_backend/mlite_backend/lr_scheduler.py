import math
from typing import Any, Optional

from megatron.lite.runtime.contracts.config import OptimizerConfig


class _MegatronLiteLRScheduler:
    def __init__(
        self,
        optimizer,
        *,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        start_wd: float,
        end_wd: float,
        wd_incr_steps: int,
        wd_incr_style: str,
        wsd_decay_steps: Optional[int],
        lr_wsd_decay_style: str,
    ):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.lr_warmup_steps = max(lr_warmup_steps, 0)
        self.lr_decay_steps = max(lr_decay_steps, self.lr_warmup_steps + 1)
        self.lr_decay_style = lr_decay_style.lower()
        self.start_wd = start_wd
        self.end_wd = end_wd
        self.wd_incr_steps = max(wd_incr_steps, 1)
        self.wd_incr_style = wd_incr_style.lower()
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style.lower()
        self.num_steps = 0
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {"num_steps": self.num_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.num_steps = int(state.get("num_steps", state.get("step", 0)))
        self._apply()

    def step(self, increment: int = 1) -> None:
        self.num_steps += increment
        self._apply()

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def _apply(self) -> None:
        lr = self._get_lr()
        wd = self._get_wd()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            if param_group.get("weight_decay") is not None:
                param_group["weight_decay"] = wd

    def _get_lr(self) -> float:
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            ratio = self.num_steps / self.lr_warmup_steps
            return self.init_lr + (self.max_lr - self.init_lr) * ratio
        if self.lr_decay_style == "constant":
            return self.max_lr
        if self.lr_decay_style == "inverse-square-root":
            warmup = max(self.lr_warmup_steps, 1)
            step = max(self.num_steps, 1)
            return max(self.min_lr, self.max_lr * math.sqrt(warmup) / math.sqrt(step))
        if self.lr_decay_style == "wsd":
            return self._get_wsd_lr()
        decay_span = max(self.lr_decay_steps - self.lr_warmup_steps, 1)
        ratio = min(max((self.num_steps - self.lr_warmup_steps) / decay_span, 0.0), 1.0)
        return self._decay(self.max_lr, self.min_lr, ratio, self.lr_decay_style)

    def _get_wsd_lr(self) -> float:
        decay_steps = self.wsd_decay_steps or 0
        decay_start = max(self.lr_decay_steps - decay_steps, self.lr_warmup_steps)
        if decay_steps <= 0 or self.num_steps <= decay_start:
            return self.max_lr
        ratio = min((self.num_steps - decay_start) / max(decay_steps, 1), 1.0)
        return self._decay(self.max_lr, self.min_lr, ratio, self.lr_wsd_decay_style)

    def _get_wd(self) -> float:
        if self.wd_incr_style == "constant":
            return self.end_wd
        ratio = min(max(self.num_steps / self.wd_incr_steps, 0.0), 1.0)
        return self._decay(self.start_wd, self.end_wd, ratio, self.wd_incr_style)

    @staticmethod
    def _decay(start: float, end: float, ratio: float, style: str) -> float:
        if style == "linear":
            return start + (end - start) * ratio
        if style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * ratio) + 1.0)
            return end + (start - end) * coeff
        if style == "exponential":
            if start == 0.0:
                return 0.0
            if end == 0.0:
                return start * (1.0 - ratio)
            return start * ((end / start)**ratio)
        if style == "constant":
            return start
        raise ValueError(f"Unsupported scheduler decay style: {style!r}")


def build_lr_scheduler(optimizer, opt: OptimizerConfig):
    if opt.total_training_steps <= 0:
        return None
    warmup_steps = opt.lr_warmup_steps
    if warmup_steps <= 0 and opt.lr_warmup_steps_ratio > 0:
        warmup_steps = int(opt.lr_warmup_steps_ratio * opt.total_training_steps)
    warmup_steps = max(warmup_steps, 0)
    decay_steps = (
        opt.lr_decay_steps if opt.lr_decay_steps is not None else opt.total_training_steps
    )
    min_lr = opt.min_lr if opt.min_lr is not None else 0.0
    for param_group in optimizer.param_groups:
        if param_group.get("min_lr") is None:
            param_group["min_lr"] = min_lr
    return _MegatronLiteLRScheduler(
        optimizer,
        init_lr=opt.lr_warmup_init,
        max_lr=opt.lr,
        min_lr=min_lr,
        lr_warmup_steps=warmup_steps,
        lr_decay_steps=decay_steps,
        lr_decay_style=opt.lr_decay_style,
        start_wd=opt.weight_decay,
        end_wd=opt.weight_decay,
        wd_incr_steps=opt.total_training_steps,
        wd_incr_style=opt.weight_decay_incr_style,
        wsd_decay_steps=opt.lr_wsd_decay_steps,
        lr_wsd_decay_style=opt.lr_wsd_decay_style,
    )
