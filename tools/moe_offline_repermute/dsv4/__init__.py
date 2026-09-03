# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""DeepSeek-V4-Flash offline expert re-permutation.

Reuses algorithms from the parent Qwen tool.
Everything under this subpackage is DSV4-specific: it uses the ``counts.pt`` produced by
the Stage-1 counts-dump run and rewrites the native checkpoint by
relocating per-expert (weight, scale) key groups (bit-exact) plus gathering the
router gate weight/bias.
"""
