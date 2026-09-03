# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Freeze CSA Lightning Indexer parameters for DeepSeek-V4 training.

gcore FSDP2 CSA indexer only produces discrete top-k indices (no KL loss),
so indexer params get no task gradient. Leaving them in AdamW/Muon with
``weight_decay > 0`` still applies decoupled decay and silently shrinks them.
Default training freezes the indexer before optimizer construction.
"""
from __future__ import annotations

from typing import Iterator

import torch.nn as nn

from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer


def _iter_csa_indexers(model: nn.Module) -> Iterator[DeepseekV4Indexer]:
    for module in model.modules():
        if isinstance(module, DeepseekV4Indexer):
            yield module


def freeze_csa_indexer_params(model: nn.Module) -> None:
    """Set ``requires_grad=False`` on every CSA Lightning Indexer parameter.

    Must run before optimizer construction so frozen weights are excluded
    from the optimizer parameter list (and thus from weight decay).
    """
    for indexer in _iter_csa_indexers(model):
        for param in indexer.parameters():
            param.requires_grad_(False)
