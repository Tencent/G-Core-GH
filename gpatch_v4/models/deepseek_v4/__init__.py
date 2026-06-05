# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""DeepSeek-V4 modeling + EP (Expert Parallelism) helpers.

gcore fork of HF ``transformers.models.deepseek_v4`` (v5.8.1):

* :class:`DeepseekV4Experts` patched in :mod:`.modeling_deepseek_v4` to use
  explicit all-to-all + grouped-MM EP path (``@use_experts_implementation``
  removed).
* :class:`DeepseekV4TopKRouter` gains a ``router_replay`` field for pinning
  routing decisions to recorded baselines (used by EP correctness tests to
  isolate routing drift from numeric residuals).
* :func:`apply_hp` (in :mod:`.hp`) wraps a meta-device model with FSDP2 + EP
  and binds ``clip_grad_norm_`` / ``load_checkpoint_hp`` /
  ``save_checkpoint_hp``.
* :mod:`.router_replay`: :func:`capture_routing_decisions` (any model, via
  forward hooks) and :func:`router_replay_ctx` (scoped enable on our fork's
  ``DeepseekV4TopKRouter`` instances).

Config is re-exported from upstream — not patched.
"""
try:
    from transformers import DeepseekV4Config
except ImportError:
    DeepseekV4Config = None

from .hp import apply_hp
from .modeling_deepseek_v4 import DeepseekV4ForCausalLM
from .router_replay import (
    RouterReplay,
    capture_routing_decisions,
    disable_router_replay,
    enable_router_replay,
    router_replay_ctx,
)

__all__ = [
    "DeepseekV4Config",
    "DeepseekV4ForCausalLM",
    "RouterReplay",
    "apply_hp",
    "capture_routing_decisions",
    "disable_router_replay",
    "enable_router_replay",
    "router_replay_ctx",
]
