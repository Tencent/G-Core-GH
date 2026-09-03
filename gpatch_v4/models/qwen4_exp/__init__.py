# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen3.8-Flash-Next language-only modeling for the FSDP2 backend.

The configuration and modeling files are vendored from transformers. CP branches live
in the modeling classes; QSA, EP experts, and Engram are swapped after construction.
"""
from gpatch_v4.models.qwen4_exp.configuration_qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)
from gpatch_v4.models.qwen4_exp.engram import (
    OwnerShardedNGramEmbedding,
    Qwen4ExpEngramEmbedding,
)
from gpatch_v4.models.qwen4_exp.hp import (
    Qwen4ExpHpForCausalLM,
    apply_hp,
    set_activation_checkpointing,
    swap_parallel_modules,
)
from gpatch_v4.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForCausalLM,
    Qwen4ExpPreTrainedModel,
    Qwen4ExpTextModel,
)
from gpatch_v4.models.qwen4_exp.moe import Qwen4ExpEPExperts
from gpatch_v4.models.qwen4_exp.qsa import Qwen4ExpQSAAttention, Qwen4ExpQSAIndexer

__all__ = [
    "OwnerShardedNGramEmbedding",
    "Qwen4ExpConfig",
    "Qwen4ExpEPExperts",
    "Qwen4ExpEngramEmbedding",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpHpForCausalLM",
    "Qwen4ExpPreTrainedModel",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAIndexer",
    "Qwen4ExpTextConfig",
    "Qwen4ExpTextModel",
    "apply_hp",
    "set_activation_checkpointing",
    "swap_parallel_modules",
]
