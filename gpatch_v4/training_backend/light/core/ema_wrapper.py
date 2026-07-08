"""FSDP2-friendly EMA wrapper for a sharded shadow model."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Union

import torch
import torch.nn as nn

from .util import sync_weights


class EmaWrapper:
    """Maintain an FSDP2-sharded EMA copy of a training model.

    Unlike ``DistrubuteEMAModel`` (flat ``shadow_params`` list), this wrapper
    owns a full parallelized ``model`` that can be saved/loaded via DCP
    (:func:`save_model` / :func:`load_model`). Each ``step`` updates the EMA
    model in-place from the training model's sharded parameters.
    """
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
    ) -> None:
        self.model = model
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value: Optional[float] = None
        self._dtype_checked = False

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma)**-self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)
        return max(min(cur_decay_value, self.decay), self.min_decay)

    @torch.no_grad()
    def sync_from(self, source: Union[nn.Module, Iterable[torch.nn.Parameter]]) -> None:
        """Copy all parameters and buffers from *source* into the EMA model."""
        if isinstance(source, nn.Module):
            sync_weights(source, self.model)
        else:
            for src_p, ema_p in zip(source, self.model.parameters()):
                ema_p.data.copy_(src_p.data)

    @torch.no_grad()
    def step(self, source: Union[nn.Module, Iterable[torch.nn.Parameter]]) -> None:
        """Apply one EMA update from *source* into ``self.model``."""
        if isinstance(source, nn.Module):
            src_params = list(source.parameters())
        else:
            src_params = list(source)

        ema_params = list(self.model.parameters())
        if len(src_params) != len(ema_params):
            raise ValueError(
                f"EMA step param count mismatch: source={len(src_params)}, "
                f"ema={len(ema_params)}"
            )

        # EMA 必须在 fp32 下累加：bf16 下 decay≈0.9999 时每步更新量 (1-decay)*(ema-src)≈1e-4*Δ
        # 远小于 bf16 ULP，会被舍入成 0 → EMA 静默冻结在训练早期权重。这里一次性校验 shadow 是
        # fp32（FSDP2 MixedPrecisionPolicy 下 master 参数应为 fp32；建 EMA 前已 .to(torch.float32)）。
        if not self._dtype_checked:
            for ema_p in ema_params:
                if ema_p.is_floating_point() and ema_p.dtype != torch.float32:
                    raise AssertionError(
                        f"EMA shadow 必须是 fp32，但发现 {ema_p.dtype}。bf16 下 EMA 更新会下溢冻结；"
                        f"请确保建立 EMA 前已 model.to(torch.float32) 且 FSDP2 master 参数为 fp32。"
                    )
            self._dtype_checked = True

        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1.0 - decay

        for ema_p, src_p in zip(ema_params, src_params):
            if src_p.requires_grad:
                ema_p.sub_(one_minus_decay * (ema_p - src_p))
            else:
                ema_p.copy_(src_p)

    def meta_dict(self) -> Dict[str, Any]:
        return {
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
        }

    def load_meta(self, meta: Dict[str, Any]) -> None:
        self.optimization_step = int(meta.get("optimization_step", self.optimization_step))
        self.update_after_step = int(meta.get("update_after_step", self.update_after_step))
        self.use_ema_warmup = bool(meta.get("use_ema_warmup", self.use_ema_warmup))
        self.inv_gamma = meta.get("inv_gamma", self.inv_gamma)
        self.power = meta.get("power", self.power)
