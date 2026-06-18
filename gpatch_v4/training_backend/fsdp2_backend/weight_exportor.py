# coding=utf-8
"""Model-arch-specific weight export hooks for ``Fsdp2EngineLm.export_weights``."""

from collections.abc import Callable, Iterator

import torch
from torch.nn import Module

from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.models.deepseek_v4.weight_export import (
    export_deepseek_v4_bf16_weights,
    export_deepseek_v4_quantized_weights,
)

WeightExportor = Callable[[Module], Iterator[tuple[str, torch.Tensor]]]

WEIGHT_EXPORTOR_MAP: dict[str, WeightExportor] = {
    MODEL_ARCH.DEEPSEEK_V4: export_deepseek_v4_bf16_weights,
}


def get_weight_exportor(model_arch: str) -> WeightExportor | None:
    """Return the export callable for ``model_arch``, or ``None`` for default."""
    return WEIGHT_EXPORTOR_MAP.get(model_arch)
