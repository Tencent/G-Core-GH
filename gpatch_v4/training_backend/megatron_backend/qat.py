# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Megatron QAT: in-place weight fake-quant + DSv4 attn activation patches.

``qat_type``: ``"fp8"`` (w=128x128) / ``"fp4"`` (32) / ``None``.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpatch_v4.kernel.quantize.qat import _fp4_simulate, _fp8_simulate_128x128
from gpatch_v4.utils.common_utils import logging_rank0

_QAT_PARAMS_ATTR = "_gcore_qat_selected_params"


def _ceil_align(n: int, align: int) -> int:
    return ((n + align - 1) // align) * align


def _pad_simulate(
    simulate_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    block_m: int,
    block_n: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Pad trailing dims to block multiples (0-fill), simulate, then crop.
    Matches TE / SGLang block-FP8 semantics for partial tiles: padded zeros do
    not change amax, and the returned tensor keeps the original shape.
    """

    def wrapped(x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            return simulate_fn(x)
        m, n = x.shape[-2], x.shape[-1]
        pad_m = _ceil_align(m, block_m) - m
        pad_n = _ceil_align(n, block_n) - n
        if pad_m == 0 and pad_n == 0:
            return simulate_fn(x)
        x_padded = F.pad(x, (0, pad_n, 0, pad_m), mode="constant", value=0)
        return simulate_fn(x_padded)[..., :m, :n].contiguous()

    return wrapped


def get_qat_simulate_fn(qat_type: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if qat_type == "fp8":
        return _pad_simulate(
            _fp8_simulate_128x128, block_m=128, block_n=128
        )
    if qat_type == "fp4":
        return _pad_simulate(
            partial(_fp4_simulate, block_size=32),
            block_m=1,
            block_n=32,
        )
    raise ValueError(f"Unsupported qat_type={qat_type!r}; expected 'fp8' or 'fp4'")


def _qat_module_types() -> Tuple[type, ...]:
    types: List[type] = []
    try:
        import megatron.core.extensions.transformer_engine as te

        for name in (
            "TELinear",
            "TEColumnParallelLinear",
            "TERowParallelLinear",
            "TELayerNormColumnParallelLinear",
            "TEGroupedLinear",
            "TEColumnParallelGroupedLinear",
            "TERowParallelGroupedLinear",
        ):
            cls = getattr(te, name, None)
            if cls is not None:
                types.append(cls)
    except Exception:
        pass
    return tuple(types)


def select_qat_parameters(
    model: Any,
    *,
    qat_type: Optional[str] = None,
) -> List[Tuple[str, nn.Parameter]]:
    host = model[0] if isinstance(model, (list, tuple)) else model
    cached = getattr(host, _QAT_PARAMS_ATTR, None)
    if cached is not None:
        return cached

    selected: List[Tuple[str, nn.Parameter]] = []
    linear_types = _qat_module_types()
    if linear_types:
        chunks = list(model) if isinstance(model, (list, tuple)) else [model]
        multi = isinstance(model, (list, tuple))
        seen: set[int] = set()
        for i, chunk in enumerate(chunks):
            prefix = f"chunk{i}." if multi else ""
            for mod_name, module in chunk.named_modules():
                if not isinstance(module, linear_types):
                    continue
                for pname, param in module.named_parameters(recurse=False):
                    if "weight" not in pname:
                        continue
                    if id(param) in seen or (param.ndim != 2):
                        continue
                    seen.add(id(param))
                    name = f"{prefix}{mod_name}.{pname}" if mod_name else f"{prefix}{pname}"
                    selected.append((name, param))

    setattr(host, _QAT_PARAMS_ATTR, selected)
    logging_rank0(
        f"[qat] selected {len(selected)} params (qat_type={qat_type})"
        + (f"; eg={[(n,t.shape) for n, t in selected[:]]}" if selected else "")
    )
    return selected


@contextmanager
def qat_parameters_context(model: Any, *, qat_type: Optional[str] = None) -> Iterator[None]:
    if qat_type is None:
        yield
        return

    simulate_fn = get_qat_simulate_fn(qat_type)
    for _, param in select_qat_parameters(model, qat_type=qat_type):
        param.data.copy_(simulate_fn(param.data),non_blocking=True)
    yield
