import torch
from typing_extensions import override

from gpatch_v4.training_backend.base_engine import BaseEngine
from gpatch_v4.training_backend.common.swap_mixin import EngineSwapMixin
from gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_lm import Fsdp2EngineLm


class Fsdp2EngineVlm(Fsdp2EngineLm, EngineSwapMixin):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

    @override
    def setup_model_and_get_optimizer(self, ):
        raise NotImplementedError(
            f"{self.__class__.__name__} setup_model_and_get_optimizer method not implemented"
        )
