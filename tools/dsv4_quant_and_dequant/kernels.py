# coding=utf-8
"""Low-level DSV4 official <-> SGLang FP8 format kernels.

Official ``deepseek-ai/DeepSeek-V4-Flash``:
  * MoE experts: int8-packed FP4 e2m1 + ``float8_e8m0fnu`` scale, block (1, 32)
  * Dense linears: ``float8_e4m3fn`` + ``float8_e8m0fnu`` scale, block (128, 128)
  * ``attn.wo_a``: FP8 + E8M0 scale

SGLang ``sgl-project/DeepSeek-V4-Flash-FP8``:
  * MoE experts: ``float8_e4m3fn`` + ``float32`` scale, block (128, 128)
    (lossless rebase of FP4 via DeepSeek ``cast_e2m1fn_to_e4m3fn``)
  * Dense linears: same FP8 weights, scales cast to ``float32``
  * ``attn.wo_a``: dequantized BF16 (no real ``.scale`` in the shard)
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys

import torch


def _load_fp_quantize():
    """Load ``fp_quantize`` by file path to avoid ``gpatch_v4`` package side effects."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(repo, "gpatch_v4", "models", "deepseek_v4", "fp_quantize.py")
    name = "gpatch_v4_dsv4_fp_quantize_standalone"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_fpq = _load_fp_quantize()
dequant_fp4_e2m1_fp8_scale_e8m0_packed = _fpq.dequant_fp4_e2m1_fp8_scale_e8m0_packed
quant_fp4_e2m1_scale_e8m0_packed = _fpq.quant_fp4_e2m1_scale_e8m0_packed
quant_fp8_e4m3_scale_e8m0 = _fpq.quant_fp8_e4m3_scale_e8m0

FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

_EXPERT_WEIGHT_RE = re.compile(r"(?:^|\.)experts\.\d+\.w[123]\.weight$")
_SHARED_EXPERT_RE = re.compile(r"shared_experts")
_WO_A_WEIGHT_RE = re.compile(r"(?:^|\.)attn\.wo_a\.weight$")


def is_expert_weight(name: str) -> bool:
    return _EXPERT_WEIGHT_RE.search(name) is not None and _SHARED_EXPERT_RE.search(name) is None


def is_wo_a_weight(name: str) -> bool:
    return _WO_A_WEIGHT_RE.search(name) is not None


def scale_key(weight_key: str) -> str:
    assert weight_key.endswith(".weight"), weight_key
    return weight_key[:-len(".weight")] + ".scale"


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 (or FP4-packed) weight with its companion block scale."""
    return dequant_fp4_e2m1_fp8_scale_e8m0_packed(weight, scale)


def cast_e2m1fn_to_e4m3fn(x: torch.Tensor,
                          scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lossless FP4 (int8-packed) -> FP8 e4m3 with 128x128 E8M0 outer scale.

    Copied from DeepSeek-V4-Flash ``inference/convert.py``. The per-32 E8M0
    residual is folded into the FP8 mantissa; the returned scale is the
    per-128 outer ``scale_max_offset_bits``.
    """
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    lut = FP4_TABLE.to(device=x.device)
    x = torch.stack([lut[low.long()], lut[high.long()]], dim=-1).flatten(2)

    # max_fp4 (6.0) * MAX_OFFSET must fit in e4m3fn (max 448)
    # 6.0 * 2^6 = 384 < 448; 6.0 * 2^7 = 768 > 448; so MAX_OFFSET_BITS = 6
    max_offset_bits = 6

    bout = out_dim // fp8_block_size
    bin_ = in_dim // fp8_block_size
    x = x.view(bout, fp8_block_size, bin_, fp8_block_size).transpose(1, 2)
    scale = scale.float().view(bout, fp8_block_size, bin_, -1).transpose(1, 2).flatten(2)
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**max_offset_bits)
    offset = scale / scale_max_offset_bits
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu)


def official_expert_to_sgl(weight: torch.Tensor,
                           scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Official FP4 expert -> SGL FP8 expert (float32 128x128 scale)."""
    q, s = cast_e2m1fn_to_e4m3fn(weight, scale)
    return q, s.float().contiguous()


def sgl_expert_to_official(weight: torch.Tensor,
                           scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """SGL FP8 expert -> official FP4 packed expert (E8M0 1x32 scale)."""
    full = dequant_fp8_block(weight, scale).float()
    return quant_fp4_e2m1_scale_e8m0_packed(full, block_size=(1, 32))


def official_dense_to_sgl(weight: torch.Tensor,
                          scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense FP8: keep weight bytes, cast E8M0 scale -> float32."""
    return weight, scale.float().contiguous()


def sgl_dense_to_official(weight: torch.Tensor,
                          scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense FP8: keep weight bytes, cast float32 scale -> E8M0."""
    return weight, scale.float().to(torch.float8_e8m0fnu).contiguous()


def official_wo_a_to_sgl(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Official FP8 wo_a -> SGL BF16 wo_a."""
    return dequant_fp8_block(weight, scale).to(torch.bfloat16).contiguous()


def sgl_wo_a_to_official(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """SGL BF16 wo_a -> official FP8 + E8M0 scale."""
    return quant_fp8_e4m3_scale_e8m0(weight.float(), block_size=(128, 128))
