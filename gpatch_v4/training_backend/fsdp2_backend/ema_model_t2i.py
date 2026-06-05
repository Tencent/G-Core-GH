# pylint: disable = missing-module-docstring,missing-function-docstring,line-too-long
import contextlib
import copy
from typing import Any, Dict, Iterable, Optional, Union

import torch
import transformers
from diffusers.utils import deprecate
from torch.distributed.tensor import DTensor

try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

from megatron.core import mpu


class T2iEmaModel:
    """Exponential Moving Average of model weights."""
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float,
        model_cls: Optional[Any],
        model_config: Optional[Any],
    ):
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

        parameters = list(parameters)
        self.rank = mpu.get_data_parallel_rank()
        self.world_size = mpu.get_data_parallel_world_size()
        self.model_cls = model_cls

        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.model_config = model_config
        with no_init_weights():
            self.model = self.model_cls.from_config(self.model_config)

    def save_pretrained(self, path):
        self.copy_to(self.model.parameters())
        if torch.distributed.get_rank() != 0:
            return

        self.model.save_pretrained(path, safe_serialization=False, max_shard_size="200GB")

    def prepare_tensor(self, tensor):
        if isinstance(tensor, DTensor):
            return tensor.full_tensor()
        return tensor

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage.step` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage.step`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

        parameters = list(parameters)
        assert len(parameters) == len(self.shadow_params)

        context_manager = contextlib.nullcontext

        for s_param, param in zip(self.shadow_params, parameters):
            with context_manager():
                s_param.copy_(self.decay * s_param + (1 - self.decay) * param)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Copy current ema parameters into the given parameters."""
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            full_tensor = self.prepare_tensor(s_param).to(param.device)
            if torch.distributed.get_rank() == 0:
                param.data.copy_(full_tensor.data)

    def to(self, device=None, dtype=None, non_blocking=False) -> None:
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if p.is_floating_point() else p.to(device=device, non_blocking=non_blocking)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow_params": self.shadow_params,
        }
