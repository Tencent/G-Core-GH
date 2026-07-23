# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# xiaotaoliu@tencent.com
"""FP8 block-wise quantization for DSV4 weight update to sglang.

Applies 128×128 block-wise FP8 (float8_e4m3fn + float32/ue8m0 scale)
quantization to DSV4 bf16 disk-format weights before sending them to sglang
running with ``--quantization fp8``.

The quantization scope follows Miles's ``quantizer_fp8.py``:
* Expert weights (``w1/w2/w3``)
* Shared-expert weights (``w1/w2/w3``)
* Dense attention projections (``wq_a/wq_b/wkv/wo_b``, indexer ``wq_b``)
* MTP projections (``e_proj/h_proj``)

Weights that are NOT quantized (norms, biases, embeddings, gate weights,
``wo_a`` [bf16 by default in sglang]) pass through unchanged.

The scale format (float32 vs ue8m0) depends on whether sglang is using
DeepGEMM at runtime.  We support both by attempting to import sglang's
ue8m0 utilities; when unavailable we fall back to float32 scales via
``per_block_cast_to_fp8``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from gpatch_v4.models.deepseek_v4.kernel.quantize_kernels import fp4_qat_to_sgl_fp8
from gpatch_v4.utils import log

# ---------------------------------------------------------------------------
# Classification patterns (aligned with Miles quantizer_fp8.py)
# ---------------------------------------------------------------------------

_EXPERT_RE = re.compile(r"\.experts\.\d+\.w[123]\.weight$")

_FP8_DENSE_PATTERNS: tuple[str, ...] = (
    # attention projections
    r"^(?:.*\.)?layers\.\d+\.attn\.(?:wq_a|wq_b|wkv|wo_b)\.weight$",
    r"^(?:.*\.)?layers\.\d+\.attn\.indexer\.wq_b\.weight$",
    # shared experts
    r"^(?:.*\.)?layers\.\d+\.ffn\.shared_experts\.w[123]\.weight$",
    # MTP layers
    r"^mtp\.\d+\.attn\.(?:wq_a|wq_b|wkv|wo_b)\.weight$",
    r"^mtp\.\d+\.attn\.indexer\.wq_b\.weight$",
    r"^mtp\.\d+\.ffn\.shared_experts\.w[123]\.weight$",
    r"^mtp\.\d+\.(?:e_proj|h_proj)\.weight$",
)
_FP8_DENSE_RE = re.compile("|".join(_FP8_DENSE_PATTERNS))

_WEIGHT_BLOCK_SIZE = [128, 128]

# ---------------------------------------------------------------------------
# Atomic update groups (aligned with Miles deepseekv4 AtomicUpdateGroup)
#
# sglang ``load_weights`` fuses wq_a+wkv (and compressor wkv+wgate) inside a
# *single* call.  Bucket flushes that split a pair leave ``wqkv_a`` / ``wkv_gate``
# unloaded.  Keep each pair in one atomic unit; chunk by size without splitting.
# Scales are separate pairs (same dtype) so IPC per-dtype packing still works.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomicUpdateGroup:
    key: str
    suffixes: tuple[str, ...]


def get_dsv4_sglang_atomic_update_groups() -> list[AtomicUpdateGroup]:
    """Disk-key atomic pairs that must share one ``load_weights`` call."""
    return [
        AtomicUpdateGroup(
            "wqkv_a_weight",
            (".attn.wq_a.weight", ".attn.wkv.weight"),
        ),
        AtomicUpdateGroup(
            "wqkv_a_scale",
            (".attn.wq_a.weight_scale_inv", ".attn.wkv.weight_scale_inv"),
        ),
        AtomicUpdateGroup(
            "compressor_wkv_gate",
            (".attn.compressor.wkv.weight", ".attn.compressor.wgate.weight"),
        ),
        AtomicUpdateGroup(
            "indexer_compressor_wkv_gate",
            (
                ".attn.indexer.compressor.wkv.weight",
                ".attn.indexer.compressor.wgate.weight",
            ),
        ),
    ]


def stream_atomic_units(
    items: Iterator[tuple[str, torch.Tensor]],
    atomic_update_groups: Sequence[AtomicUpdateGroup],
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Buffer fuse-paired tensors until the full group arrives, then yield together.

    Mirrors Miles ``_stream_atomic_units`` / ``get_named_update_units``.
    """
    pending: dict[tuple[str, str], list] = {}
    for name, tensor in items:
        match = next(
            (
                (group, idx, suffix) for group in atomic_update_groups
                for idx, suffix in enumerate(group.suffixes) if name.endswith(suffix)
            ),
            None,
        )
        if match is None:
            yield [(name, tensor)]
            continue
        group, idx, suffix = match
        prefix = name[:-len(suffix)]
        slots = pending.setdefault((prefix, group.key), [None] * len(group.suffixes))
        if slots[idx] is not None:
            raise RuntimeError(
                f"Duplicate atomic update member {name!r} for group "
                f"{prefix}:{group.key}"
            )
        slots[idx] = (name, tensor)
        if None not in slots:
            yield list(slots)
            del pending[(prefix, group.key)]
    if pending:
        raise RuntimeError(f"Incomplete atomic update groups at end of stream: {sorted(pending)}")


def chunk_atomic_units_by_size(
    units: Iterator[list[tuple[str, torch.Tensor]]],
    chunk_size: int,
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Pack atomic units into buckets of ``chunk_size`` bytes; never split a unit.

    Mirrors Miles ``_chunk_atomic_units_by_size``.
    """
    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_size = 0
    for unit in units:
        unit_size = sum(t.numel() * t.element_size() for _, t in unit)
        if bucket and bucket_size + unit_size >= chunk_size:
            yield bucket
            bucket = []
            bucket_size = 0
        bucket.extend(unit)
        bucket_size += unit_size
    if bucket:
        yield bucket


def iter_sglang_dsv4_weight_buckets(
    weights: Iterator[tuple[str, torch.Tensor]],
    max_bucket_bytes: int,
    *,
    moe_deepgemm: bool = False,
    fp4_qat: bool = False,
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Quantize DSV4 weights then emit size-bounded buckets that respect fuse pairs.

    When ``fp4_qat`` is enabled, routed experts are first projected onto the
    QAT FP4 grid and then represented in SGLang's FP8 expert format. This
    preserves the FP4 deployment error learned during training.
    """
    quantized = iter_fp8_quantized_weights(weights, moe_deepgemm=moe_deepgemm, fp4_qat=fp4_qat)
    units = stream_atomic_units(quantized, get_dsv4_sglang_atomic_update_groups())
    yield from chunk_atomic_units_by_size(units, max_bucket_bytes)


# ---------------------------------------------------------------------------
# Lazy-loaded FP8 quantizer
# ---------------------------------------------------------------------------

_per_block_cast_fn = None
_quant_weight_ue8m0_fn = None
_transform_scale_ue8m0_fn = None
_should_ue8m0_fn = None
_fp8_ready = False


def _init_fp8_quantizers():
    """Import FP8 quantization functions from sglang (or Triton fallback)."""
    global _per_block_cast_fn, _quant_weight_ue8m0_fn
    global _transform_scale_ue8m0_fn, _should_ue8m0_fn, _fp8_ready

    if _fp8_ready:
        return

    try:
        from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8
        _per_block_cast_fn = per_block_cast_to_fp8
    except ImportError:
        _per_block_cast_fn = None

    try:
        from sglang.srt.layers.quantization.fp8_utils import (
            quant_weight_ue8m0,
            transform_scale_ue8m0,
        )
        from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
        _quant_weight_ue8m0_fn = quant_weight_ue8m0
        _transform_scale_ue8m0_fn = transform_scale_ue8m0
        _should_ue8m0_fn = should_deepgemm_weight_requant_ue8m0
    except ImportError:
        _quant_weight_ue8m0_fn = None
        _transform_scale_ue8m0_fn = None
        _should_ue8m0_fn = None

    if _per_block_cast_fn is None and _quant_weight_ue8m0_fn is None:
        raise ImportError(
            "Neither sglang per_block_cast_to_fp8 nor quant_weight_ue8m0 "
            "is available. Cannot perform FP8 quantization."
        )

    log(
        f"FP8 quantizer ready per_block_cast={_per_block_cast_fn is not None}, ue8m0={_quant_weight_ue8m0_fn is not None}"
    )
    _fp8_ready = True


def _use_ue8m0_scale(name: str, *, moe_deepgemm: bool = False) -> bool:
    """Decide if a weight should use ue8m0 scale format (for DeepGEMM).

    Follows the same logic as Miles ``quantizer_fp8._get_scale_format``:

    * Non-expert weights use ue8m0 whenever DeepGEMM is detected by sglang.
    * Expert weights (``".experts."`` in the name) additionally require the
      MoE runner to be DeepGEMM (``moe_deepgemm=True``). When uncertain,
      callers should pass ``False`` to fall back to float32 scales.
    """
    if _should_ue8m0_fn is None:
        return False
    if not _should_ue8m0_fn(weight_block_size=_WEIGHT_BLOCK_SIZE):
        return False

    if ".experts." in name:
        return moe_deepgemm
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def should_fp8_quantize(name: str) -> bool:
    """Return True if the disk-format key should receive FP8 quantization."""
    return _EXPERT_RE.search(name) is not None or _FP8_DENSE_RE.search(name) is not None


def quantize_fp8(
    name: str,
    weight: torch.Tensor,
    *,
    moe_deepgemm: bool = False,
) -> list[tuple[str, torch.Tensor]]:
    """Quantize one weight tensor to FP8 (128×128 block, float8_e4m3fn + scale).

    Parameters
    ----------
    name : str
        Disk-format key (must end with ``.weight``).
    weight : torch.Tensor
        bf16 / fp32 weight; both dims must be divisible by 128.
    moe_deepgemm : bool
        Whether the sglang MoE runner uses DeepGEMM. When ``True``,
        expert weights also get ue8m0 scales; otherwise they use
        float32 scales (safe default).

    Returns
    -------
    list of (name, tensor)
        ``[(weight_name, qweight), (scale_name, scale)]``.
    """
    _init_fp8_quantizers()

    assert name.endswith(".weight"), f"Expected .weight suffix, got {name!r}"
    weight = weight.contiguous()

    if weight.dim() == 1:
        raise ValueError(f"Cannot FP8-quantize 1D weight {name!r}")

    weight_2d = weight.view(-1, weight.shape[-1])

    if _use_ue8m0_scale(name, moe_deepgemm=moe_deepgemm) and _quant_weight_ue8m0_fn is not None:
        qweight, scale = _quant_weight_ue8m0_fn(
            weight_2d,
            weight_block_size=_WEIGHT_BLOCK_SIZE,
        )
        scale = _transform_scale_ue8m0_fn(scale, mn=qweight.shape[-2])
    elif _per_block_cast_fn is not None:
        qweight, scale = _per_block_cast_fn(weight_2d)
    else:
        qweight, scale = _quant_weight_ue8m0_fn(
            weight_2d,
            weight_block_size=_WEIGHT_BLOCK_SIZE,
        )
        scale = _transform_scale_ue8m0_fn(scale, mn=qweight.shape[-2])

    qweight = qweight.view_as(weight).contiguous()
    scale_name = name[:-len(".weight")] + ".weight_scale_inv"
    return [(name, qweight), (scale_name, scale)]


def quantize_fp4_qat_expert(
    name: str,
    weight: torch.Tensor,
    *,
    moe_deepgemm: bool = False,
) -> list[tuple[str, torch.Tensor]]:
    """Deploy an FP4-QAT routed expert through SGLang's FP8 loader.

    Quantize the FP32 master parameter directly to E2M1 1×32 and rebase it
    into SGLang's E4M3 128×128 representation with one fused TileLang kernel.
    """
    assert _EXPERT_RE.search(name) is not None, f"Expected routed expert key, got {name!r}"
    assert name.endswith(".weight"), f"Expected .weight suffix, got {name!r}"

    qweight, fp8_scale = fp4_qat_to_sgl_fp8(weight)

    # The regular SGLang path uses float32 scales unless its DeepGEMM path
    # requires the transformed ue8m0 layout. Reuse exactly that decision and
    # layout conversion so this branch is load-compatible with both workers.
    scale: torch.Tensor = fp8_scale.float()
    if moe_deepgemm:
        _init_fp8_quantizers()
        if _use_ue8m0_scale(name, moe_deepgemm=True):
            assert _transform_scale_ue8m0_fn is not None
            scale = _transform_scale_ue8m0_fn(fp8_scale, mn=qweight.shape[-2])

    scale_name = name[:-len(".weight")] + ".weight_scale_inv"
    return [(name, qweight), (scale_name, scale.contiguous())]


def iter_fp8_quantized_weights(
    weights: Iterator[tuple[str, torch.Tensor]],
    *,
    moe_deepgemm: bool = False,
    fp4_qat: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Wrap a bf16 weight iterator with FP8 block-wise quantization for sglang.

    Expert weights, shared-expert weights, and dense linear weights are
    quantized to FP8 (128×128 block); all other weights (norms, biases,
    embeddings, gate weights) pass through unchanged.

    Parameters
    ----------
    weights : Iterator[tuple[str, torch.Tensor]]
        bf16 disk-format weight stream from
        :func:`~gpatch_v4.models.deepseek_v4.weight_export.export_deepseek_v4_bf16_weights`.
    moe_deepgemm : bool
        Whether the sglang MoE runner uses DeepGEMM. Controls ue8m0
        scale format for expert weights; see :func:`quantize_fp8`.
    fp4_qat : bool
        Recreate FP4 QAT's E2M1 1×32 quantization for routed experts before
        converting them into SGLang's FP8 expert representation.

    Yields
    ------
    tuple[str, torch.Tensor]
        Quantized or passthrough ``(name, tensor)`` pairs.
    """
    for name, tensor in weights:
        if name.endswith(".scale"):
            raise RuntimeError(
                f"FP8 sglang quantization expects bf16 weight keys, "
                f"got scale key {name!r}"
            )

        if fp4_qat and _EXPERT_RE.search(name) is not None:
            yield from quantize_fp4_qat_expert(name, tensor, moe_deepgemm=moe_deepgemm)
        elif should_fp8_quantize(name):
            yield from quantize_fp8(name, tensor, moe_deepgemm=moe_deepgemm)
        else:
            yield name, tensor
