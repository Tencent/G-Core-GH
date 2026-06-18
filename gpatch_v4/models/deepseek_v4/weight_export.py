# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com
"""Shared DSV4 model-key -> disk checkpoint export helpers."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from typing import Literal

import torch
from torch import distributed as dist
from torch.distributed.tensor import DTensor, Shard
from torch.nn import Module

from .fp_quantize import quant_fp4_e2m1_scale_e8m0_packed, quant_fp8_e4m3_scale_e8m0


def _weight_export_debug_log(msg: str) -> None:
    if not os.getenv("GPATCH_DEBUG_WEIGHT_EXPORT_PROGRESS", "0") == "1":
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    print(f"[weight_export] {msg}", flush=True)


# DSV4-Flash on-disk dtype layout (derived from upstream safetensors shard survey).
#
# FP8 (float8_e4m3fn + E8M0 scale, 128×128 block):
#   - attn.{wq_a,wq_b,wkv,wo_a,wo_b}  directly under self_attn.*
#     (NOT under compressor.* or indexer.compressor.* — those are BF16)
#   - attn.indexer.wq_b                (the indexer's only FP8 leaf)
#   - ffn.shared_experts.w{1,2,3}
#
# FP4 (e2m1 packed int8 + E8M0 scale, 1×32 block):
#   - experts.{i}.w{1,2,3}.weight      (routed MoE experts only)
#
# BF16: everything else (embed/head, all compressor/indexer linears, ffn.gate.weight)
# FP32: norms, sinks, hc_*, position_bias, gate_bias (see _F32_DISK_KEY_PATTERNS)
#
# We use an explicit FP8 whitelist rather than a shape heuristic because the
# heuristic over-quantizes embed.weight, head.weight, and all compressor.* linears —
# vanilla DSV4 expects those as BF16 and fails to load FP8 versions.
#
# All patterns operate on the final DISK-FORM key (after reverse WeightRenaming pass).
_FP8_DISK_KEY_PATTERNS: tuple[str, ...] = (
    r"^(?:.*\.)?layers\.\d+\.attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)\.weight$",
    r"^(?:.*\.)?layers\.\d+\.attn\.indexer\.wq_b\.weight$",
    r"^(?:.*\.)?layers\.\d+\.ffn\.shared_experts\.w[123]\.weight$",
    r"^mtp\.\d+\.attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)\.weight$",
    r"^mtp\.\d+\.attn\.indexer\.wq_b\.weight$",
    r"^mtp\.\d+\.ffn\.shared_experts\.w[123]\.weight$",
    r"^mtp\.\d+\.(?:e_proj|h_proj)\.weight$",
)
_FP8_DISK_KEY_RE = re.compile("|".join(_FP8_DISK_KEY_PATTERNS))

_F32_DISK_KEY_PATTERNS: tuple[str, ...] = (
    r"^(?:.*\.)?hc_(?:attn|ffn|head)_(?:fn|base|scale)$",
    r"^(?:.*\.)?layers\.\d+\.attn\.attn_sink$",
    r"^(?:.*\.)?layers\.\d+\.attn\.(?:.*\.)?compressor\.ape$",
    r"^(?:.*\.)?layers\.\d+\.ffn\.gate\.bias$",
    r"^mtp\.\d+\.attn\.attn_sink$",
    r"^mtp\.\d+\.attn\.(?:.*\.)?compressor\.ape$",
    r"^mtp\.\d+\.ffn\.gate\.bias$",
)
_F32_DISK_KEY_RE = re.compile("|".join(_F32_DISK_KEY_PATTERNS))


def _classify_for_save(name: str, t: torch.Tensor) -> str:
    if not t.dtype.is_floating_point:
        return "int_passthrough"
    if re.search(r"\.experts\.\d+\.w[123]\.weight$", name) is not None:
        return "fp4_expert"
    if _FP8_DISK_KEY_RE.search(name) is not None:
        return "fp8_e4m3"
    if _F32_DISK_KEY_RE.search(name) is not None:
        return "f32_passthrough"
    return "bf16_passthrough"


def _scale_key(weight_disk_key: str) -> str:
    assert weight_disk_key.endswith(".weight"), weight_disk_key
    return weight_disk_key[:-len(".weight")] + ".scale"


_MODEL_TO_DISK_RENAMES: tuple[tuple[str, str], ...] = (
    (r"^embed_tokens\.weight$", "embed.weight"),
    (r"^lm_head\.weight$", "head.weight"),
    (r"^hc_head\.hc_fn$", "hc_head_fn"),
    (r"^hc_head\.hc_base$", "hc_head_base"),
    (r"^hc_head\.hc_scale$", "hc_head_scale"),
    (r"^layers\.(\d+)\.attn_hc\.fn$", r"layers.\1.hc_attn_fn"),
    (r"^layers\.(\d+)\.attn_hc\.base$", r"layers.\1.hc_attn_base"),
    (r"^layers\.(\d+)\.attn_hc\.scale$", r"layers.\1.hc_attn_scale"),
    (r"^layers\.(\d+)\.ffn_hc\.fn$", r"layers.\1.hc_ffn_fn"),
    (r"^layers\.(\d+)\.ffn_hc\.base$", r"layers.\1.hc_ffn_base"),
    (r"^layers\.(\d+)\.ffn_hc\.scale$", r"layers.\1.hc_ffn_scale"),
    (r"^layers\.(\d+)\.hc_head\.hc_(fn|base|scale)$", r"layers.\1.hc_head_\2"),
    (r"^layers\.(\d+)\.input_layernorm\.", r"layers.\1.attn_norm."),
    (r"^layers\.(\d+)\.post_attention_layernorm\.", r"layers.\1.ffn_norm."),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.kv_proj\.",
        r"layers.\1.self_attn.indexer.compressor.wkv.",
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.gate_proj\.",
        r"layers.\1.self_attn.indexer.compressor.wgate.",
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.kv_norm\.",
        r"layers.\1.self_attn.indexer.compressor.norm.",
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\._position_bias_holder\.weight$",
        r"layers.\1.self_attn.compressor.indexer.position_bias",
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.position_bias$",
        r"layers.\1.self_attn.indexer.compressor.ape",
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.q_b_proj\.",
        r"layers.\1.self_attn.indexer.wq_b.",
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.weights_proj\.",
        r"layers.\1.self_attn.indexer.weights_proj.",
    ),
    (r"^layers\.(\d+)\.self_attn\.compressor\.kv_proj\.", r"layers.\1.self_attn.compressor.wkv."),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.gate_proj\.",
        r"layers.\1.self_attn.compressor.wgate.",
    ),
    (r"^layers\.(\d+)\.self_attn\.compressor\.kv_norm\.", r"layers.\1.self_attn.compressor.norm."),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\._position_bias_holder\.weight$",
        r"layers.\1.self_attn.compressor.position_bias",
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.position_bias$",
        r"layers.\1.self_attn.compressor.ape",
    ),
    (r"^layers\.(\d+)\.self_attn\.q_a_proj\.", r"layers.\1.self_attn.wq_a."),
    (r"^layers\.(\d+)\.self_attn\.q_b_proj\.", r"layers.\1.self_attn.wq_b."),
    (r"^layers\.(\d+)\.self_attn\.kv_proj\.", r"layers.\1.self_attn.wkv."),
    (r"^layers\.(\d+)\.self_attn\.o_a_proj\.", r"layers.\1.self_attn.wo_a."),
    (r"^layers\.(\d+)\.self_attn\.o_b_proj\.", r"layers.\1.self_attn.wo_b."),
    (r"^layers\.(\d+)\.self_attn\.q_a_norm\.", r"layers.\1.self_attn.q_norm."),
    (r"^layers\.(\d+)\.self_attn\._sink_holder\.weight$", r"layers.\1.self_attn.sinks"),
    (r"^layers\.(\d+)\.self_attn\.sinks$", r"layers.\1.self_attn.attn_sink"),
    (r"^layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$", r"layers.\1.mlp.gate.bias"),
    (r"^layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.", r"layers.\1.mlp.shared_experts.w1."),
    (r"^layers\.(\d+)\.mlp\.shared_experts\.down_proj\.", r"layers.\1.mlp.shared_experts.w2."),
    (r"^layers\.(\d+)\.mlp\.shared_experts\.up_proj\.", r"layers.\1.mlp.shared_experts.w3."),
    (r"^layers\.(\d+)\.self_attn\.", r"layers.\1.attn."),
    (r"^layers\.(\d+)\.mlp\.", r"layers.\1.ffn."),
)
_MODEL_TO_DISK_RENAMES_COMPILED: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(src), tgt) for src, tgt in _MODEL_TO_DISK_RENAMES
)

_EXPERT_STACKED_KEY_RE = re.compile(
    r"^(layers|mtp\.layers)\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
_MTP_LAYER_KEY_RE = re.compile(r"^mtp\.layers\.(\d+)\.(.+)$")


def _model_key_to_disk_key(model_key: str) -> str:
    for _ in range(8):
        for rgx, repl in _MODEL_TO_DISK_RENAMES_COMPILED:
            new_key, n = rgx.subn(repl, model_key, count=1)
            if n > 0:
                model_key = new_key
                break
        else:
            return model_key
    raise AssertionError(
        f"_model_key_to_disk_key did not converge on {model_key!r}; "
        "likely a self-retriggering rule in _MODEL_TO_DISK_RENAMES."
    )


def _split_per_expert(model_key: str, t: torch.Tensor) -> dict[str, torch.Tensor]:
    m = _EXPERT_STACKED_KEY_RE.match(model_key)
    if m is None:
        assert ".experts." not in model_key or ".shared_experts." in model_key, (
            f"unexpected expert key not handled by _split_per_expert: {model_key!r}"
        )
        return {model_key: t}

    key_prefix, layer, w_kind = m.group(1), m.group(2), m.group(3)
    n = t.shape[0]
    if w_kind == "down_proj":
        return {
            f"{key_prefix}.{layer}.mlp.experts.{i}.w2.weight": t[i].contiguous()
            for i in range(n)
        }

    assert t.shape[1] % 2 == 0, (
        f"{model_key} shape {tuple(t.shape)} not divisible by 2 on dim=1 "
        "(cannot split into w1/w3)"
    )
    w1_stack, w3_stack = t.chunk(2, dim=1)
    out: dict[str, torch.Tensor] = {}
    for i in range(n):
        out[f"{key_prefix}.{layer}.mlp.experts.{i}.w1.weight"] = w1_stack[i].contiguous()
        out[f"{key_prefix}.{layer}.mlp.experts.{i}.w3.weight"] = w3_stack[i].contiguous()
    return out


def _model_to_disk_key(model_key: str) -> str:
    mtp_m = _MTP_LAYER_KEY_RE.match(model_key)
    if mtp_m is None:
        return _model_key_to_disk_key(model_key)
    mtp_depth = int(mtp_m.group(1))
    logical_layer_key = f"layers.{mtp_depth}.{mtp_m.group(2)}"
    disk_key = _model_key_to_disk_key(logical_layer_key)
    return disk_key.replace(f"layers.{mtp_depth}.", f"mtp.{mtp_depth}.", 1)


def iter_disk_checkpoint_tensors(
    model_key: str,
    tensor: torch.Tensor,
    *,
    dtype_format: Literal["quantized", "bf16"] = "quantized",
    expert_dtype: Literal["fp4", "fp8"] = "fp4",
    include_mtp: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Convert one model tensor into one-or-more disk-form checkpoint tensors."""
    for mk, mv in _split_per_expert(model_key, tensor).items():
        if not include_mtp and mk.startswith("mtp."):
            continue
        disk_key = _model_to_disk_key(mk)
        cls = _classify_for_save(disk_key, mv)
        if cls == "fp4_expert" and expert_dtype == "fp8":
            cls = "fp8_e4m3"

        if dtype_format == "quantized":
            if cls == "fp4_expert":
                packed, scale = quant_fp4_e2m1_scale_e8m0_packed(mv)
                yield disk_key, packed
                yield _scale_key(disk_key), scale
            elif cls == "fp8_e4m3":
                qfp8, scale = quant_fp8_e4m3_scale_e8m0(mv)
                yield disk_key, qfp8
                yield _scale_key(disk_key), scale
            elif cls == "f32_passthrough":
                yield disk_key, mv.to(torch.float32)
            elif cls == "bf16_passthrough":
                yield disk_key, mv.to(torch.bfloat16)
            elif cls == "int_passthrough":
                yield disk_key, mv
            else:
                raise RuntimeError(f"unhandled save class {cls!r} for {disk_key}")
        else:
            if cls in ("fp4_expert", "fp8_e4m3", "bf16_passthrough"):
                yield disk_key, mv.to(torch.bfloat16)
            elif cls == "f32_passthrough":
                yield disk_key, mv.to(torch.float32)
            elif cls == "int_passthrough":
                yield disk_key, mv
            else:
                raise RuntimeError(f"unhandled save class {cls!r} for {disk_key}")


def _iter_deepseek_v4_gathered_state_dict(model: Module, ) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield model-form keys with full tensors after DTensor / EP gather."""
    assert hasattr(model, "_ep_size"), "HpModule model missing _ep_size."
    ep_size = int(model._ep_size)
    ep_group = model._ep_group if hasattr(model, "_ep_group") else None
    ep_fsdp_mesh = model._ep_fsdp_mesh if hasattr(model, "_ep_fsdp_mesh") else None

    for idx, (name, tensor) in enumerate(model.state_dict().items()):
        if name.startswith("mtp."):
            continue

        if isinstance(tensor, DTensor):
            is_expert_weight = (
                ".mlp.experts.gate_up_proj" in name or ".mlp.experts.down_proj" in name
            )

            if is_expert_weight and ep_size > 1:
                # Keep the same expert gather semantics as checkpoint.py:
                # gather ep_fsdp shards first, then all_gather over ep ranks
                # to recover experts in model order.
                assert ep_group is not None and ep_fsdp_mesh is not None, (
                    "DSV4 EP gather requires _ep_group and _ep_fsdp_mesh"
                )
                local = tensor.to_local().contiguous()
                if not local.is_cuda:
                    local = local.cuda()
                    torch.cuda.synchronize()
                dt_fsdp = DTensor.from_local(
                    local,
                    device_mesh=ep_fsdp_mesh,
                    placements=[Shard(0)],
                )
                ep_local_full = dt_fsdp.full_tensor()
                # Use a single preallocated receive buffer to avoid
                # list-of-tensors all_gather + cat peak duplication.
                gathered_shape = (ep_size, ) + tuple(ep_local_full.shape)
                ep_gathered = torch.empty(
                    gathered_shape, dtype=ep_local_full.dtype, device=ep_local_full.device
                )
                dist.all_gather_into_tensor(ep_gathered, ep_local_full, group=ep_group)
                full_tensor = ep_gathered.reshape(-1, *ep_local_full.shape[1:])
                del ep_local_full, ep_gathered
            else:
                # Non-expert tensors: prefer moving the native DTensor to CUDA
                # instead of rebuilding from local shard, so sharding metadata
                # (especially empty-shard layouts) stays identical to the
                # original parameter object.
                local = tensor.to_local().contiguous()
                if local.is_cuda:
                    dt_cuda = tensor
                else:
                    dt_cuda = tensor.cuda()
                    torch.cuda.synchronize()
                full_tensor = dt_cuda.full_tensor()
                if not local.is_cuda:
                    del dt_cuda
        else:
            full_tensor = tensor.detach()

        yield name.removeprefix("model."), full_tensor
        del full_tensor


def export_deepseek_v4_weights_for_vllm(
    model: Module,
    *,
    dtype_format: Literal["quantized", "bf16"] = "bf16",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield DSV4 disk-format tensors for vLLM weight reload.

    ``dtype_format="bf16"`` is the online update path: trainer sends
    checkpoint-form bf16 tensors and vLLM workers quantize before loading.
    ``dtype_format="quantized"`` preserves the older trainer-side quantized
    export path.
    """
    config = model.config
    assert hasattr(config, "expert_dtype"
                  ), ("DeepSeek-V4 config must define expert_dtype for vLLM weight export.")
    expert_dtype = config.expert_dtype
    assert expert_dtype in ("fp4",
                            "fp8"), (f"Unsupported DeepSeek-V4 expert_dtype={expert_dtype!r}")

    for model_key, full_tensor in _iter_deepseek_v4_gathered_state_dict(model):
        for disk_key, disk_tensor in iter_disk_checkpoint_tensors(
            model_key,
            full_tensor,
            dtype_format=dtype_format,
            expert_dtype=expert_dtype,
            include_mtp=False,
        ):
            if not disk_tensor.is_cuda:
                disk_tensor = disk_tensor.cuda()
            yield disk_key, disk_tensor.contiguous()
        del full_tensor


def export_deepseek_v4_bf16_weights(model: Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield bf16 disk-format tensors for worker-side quantization."""
    yield from export_deepseek_v4_weights_for_vllm(model, dtype_format="bf16")


def export_deepseek_v4_quantized_weights(model: Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield pre-quantized disk-format tensors."""
    yield from export_deepseek_v4_weights_for_vllm(model, dtype_format="quantized")
