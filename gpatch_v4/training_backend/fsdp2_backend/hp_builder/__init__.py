# coding=utf-8
"""Model-arch-specific hybrid-parallel (HpModule) construction for FSDP2."""

from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.training_backend.fsdp2_backend.hp_builder.base import HpModelBuilder
from gpatch_v4.training_backend.fsdp2_backend.hp_builder.deepseek_v4 import (
    DefaultHpBuilder,
)
from gpatch_v4.training_backend.fsdp2_backend.hp_builder.qwen4_exp import (
    Qwen4ExpHpBuilder,
)

_DEFAULT_HP_BUILDER = DefaultHpBuilder()

HP_BUILDER_MAP: dict[str, HpModelBuilder] = {
    MODEL_ARCH.DEEPSEEK_V4: _DEFAULT_HP_BUILDER,
    MODEL_ARCH.QWEN4_EXP: Qwen4ExpHpBuilder(),
}


def get_hp_builder(model_arch: str) -> HpModelBuilder:
    """Return the HP builder for ``model_arch``; unknown arches use DefaultHpBuilder."""
    return HP_BUILDER_MAP.get(model_arch, _DEFAULT_HP_BUILDER)


__all__ = [
    "DefaultHpBuilder",
    "HP_BUILDER_MAP",
    "HpModelBuilder",
    "Qwen4ExpHpBuilder",
    "get_hp_builder",
]
