"""Triton kernel implementations (NVIDIA CUDA)."""

from .linear_cross_entropy import linear_cross_entropy, set_linear_ce_backend

__all__ = [
    "linear_cross_entropy",
    "set_linear_ce_backend",
]
