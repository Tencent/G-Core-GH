"""Append Qwen3 MTP weights after HF FineGrained FP8 conversion.

Transformers Qwen3.5/3.6 model classes currently ignore top-level ``mtp.*``
weights during ``from_pretrained``. This module restores those weights by
reading them directly from the BF16 checkpoint, quantizing the official FP8
parts, and writing a separate ``mtp.safetensors`` shard.
"""

from __future__ import annotations

import json
import math
import os
import re
from glob import glob
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_MTP_SHARD_NAME = "mtp.safetensors"
_MTP_SHARED_REQUIRED_KEYS = (
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)


def _load_config(input_hf_path: str) -> dict[str, Any]:
    with open(os.path.join(input_hf_path, "config.json")) as f:
        return json.load(f)


def _get_text_config(config: dict[str, Any]) -> dict[str, Any]:
    if "text_config" in config:
        return config["text_config"]
    return config


def _get_mtp_num_hidden_layers(input_hf_path: str) -> int:
    config = _load_config(input_hf_path)
    text_config = _get_text_config(config)
    return int(text_config.get("mtp_num_hidden_layers", 0) or 0)


def _is_moe_config(input_hf_path: str) -> bool:
    config = _load_config(input_hf_path)
    text_config = _get_text_config(config)
    model_type = text_config.get("model_type", "")
    if "moe" in model_type:
        return True
    return int(text_config.get("num_experts", 0) or 0) > 1


def _required_moe_mtp_keys(num_mtp_layers: int) -> set[str]:
    required = set(_MTP_SHARED_REQUIRED_KEYS)
    for layer_idx in range(num_mtp_layers):
        layer = f"mtp.layers.{layer_idx}"
        required.update(
            {
                f"{layer}.input_layernorm.weight",
                f"{layer}.post_attention_layernorm.weight",
                f"{layer}.self_attn.q_proj.weight",
                f"{layer}.self_attn.k_proj.weight",
                f"{layer}.self_attn.v_proj.weight",
                f"{layer}.self_attn.o_proj.weight",
                f"{layer}.self_attn.q_norm.weight",
                f"{layer}.self_attn.k_norm.weight",
                f"{layer}.mlp.experts.gate_up_proj",
                f"{layer}.mlp.experts.down_proj",
                f"{layer}.mlp.gate.weight",
                f"{layer}.mlp.shared_expert.gate_proj.weight",
                f"{layer}.mlp.shared_expert.up_proj.weight",
                f"{layer}.mlp.shared_expert.down_proj.weight",
                f"{layer}.mlp.shared_expert_gate.weight",
            }
        )
    return required


def _should_convert_module(full_name: str, patterns: list[str] | None) -> bool:
    if patterns is None:
        return True
    should_not_convert = any(
        re.match(f"{key}\\.", full_name) or re.match(f"{key}", full_name) or
        full_name.endswith(key) for key in patterns
    )
    return not should_not_convert


def _weight_to_module_name(weight_name: str) -> str:
    if weight_name.endswith(".weight"):
        return weight_name[:-len(".weight")]
    return weight_name


def quantize_fp8_blockwise(
    weight: torch.Tensor,
    block_size: tuple[int, int],
    *,
    device: str | torch.device | None = None,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to HF fine-grained FP8 format.

    Parameters
    ----------
    weight:
        BF16/FP32 weight with shape ``(out_features, in_features)``.
    block_size:
        FP8 block size, usually ``(128, 128)``.
    device:
        Device used for temporary quantization. If ``None``, use CUDA when
        available, otherwise CPU.
    scale_dtype:
        On-disk dtype for ``weight_scale_inv``. Official Qwen FP8 checkpoints
        use BF16 scales.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight, got shape={tuple(weight.shape)}")

    block_m, block_n = block_size
    rows, cols = weight.shape
    if rows % block_m != 0 or cols % block_n != 0:
        raise ValueError(
            f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_m}, {block_n})"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    value = weight.to(device=device, dtype=torch.float32)
    rows_tiles = rows // block_m
    cols_tiles = cols // block_n
    reshaped = value.reshape(rows_tiles, block_m, cols_tiles, block_n)

    max_abs = reshaped.abs().amax(dim=(1, 3))
    safe_max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scales = _FP8_MAX / safe_max_abs
    scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))

    scaled = reshaped * scales[:, None, :, None]
    quantized = torch.clamp(scaled, min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE).reshape(rows, cols)
    scale_inv = (1.0 / scales).to(scale_dtype)

    return quantized.cpu(), scale_inv.cpu()


def _save_weight(
    output: dict[str, torch.Tensor],
    key: str,
    weight: torch.Tensor,
    *,
    block_size: tuple[int, int],
    modules_to_not_convert: list[str],
    device: str | torch.device | None,
    scale_dtype: torch.dtype,
) -> None:
    module_name = _weight_to_module_name(key)
    if _should_convert_module(module_name, modules_to_not_convert):
        quantized, scale_inv = quantize_fp8_blockwise(
            weight,
            block_size,
            device=device,
            scale_dtype=scale_dtype,
        )
        output[key] = quantized
        output[f"{module_name}.weight_scale_inv"] = scale_inv
    else:
        output[key] = weight.cpu()


def _load_mtp_tensors(input_hf_path: str) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(glob(os.path.join(input_hf_path, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    tensors[key] = f.get_tensor(key)
    return tensors


def _validate_mtp_tensors(input_hf_path: str, mtp_tensors: dict[str, torch.Tensor]) -> None:
    num_mtp_layers = _get_mtp_num_hidden_layers(input_hf_path)
    if num_mtp_layers == 0:
        return
    if not mtp_tensors:
        raise ValueError(
            f"{input_hf_path} config enables MTP (mtp_num_hidden_layers={num_mtp_layers}) "
            "but no top-level mtp.* tensors were found"
        )
    if num_mtp_layers != 1:
        raise NotImplementedError(
            f"Only one Qwen3 MTP layer is currently supported, got mtp_num_hidden_layers={num_mtp_layers}"
        )
    if not _is_moe_config(input_hf_path):
        return

    missing = sorted(_required_moe_mtp_keys(num_mtp_layers) - set(mtp_tensors))
    if missing:
        raise KeyError(
            "Missing required MoE MTP tensors: " + ", ".join(missing[:20]) +
            (" ..." if len(missing) > 20 else "")
        )


def _build_mtp_state_dict(
    mtp_tensors: dict[str, torch.Tensor],
    *,
    block_size: tuple[int, int],
    modules_to_not_convert: list[str],
    device: str | torch.device | None,
    scale_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}

    for key in sorted(mtp_tensors):
        weight = mtp_tensors[key]

        if key.endswith(".mlp.experts.gate_up_proj"):
            experts_base = key[:-len(".gate_up_proj")]
            for expert_idx, expert_weight in enumerate(weight):
                gate_proj, up_proj = expert_weight.chunk(2, dim=0)
                _save_weight(
                    output,
                    f"{experts_base}.{expert_idx}.gate_proj.weight",
                    gate_proj,
                    block_size=block_size,
                    modules_to_not_convert=modules_to_not_convert,
                    device=device,
                    scale_dtype=scale_dtype,
                )
                _save_weight(
                    output,
                    f"{experts_base}.{expert_idx}.up_proj.weight",
                    up_proj,
                    block_size=block_size,
                    modules_to_not_convert=modules_to_not_convert,
                    device=device,
                    scale_dtype=scale_dtype,
                )
            continue

        if key.endswith(".mlp.experts.down_proj"):
            experts_base = key[:-len(".down_proj")]
            for expert_idx, expert_weight in enumerate(weight):
                _save_weight(
                    output,
                    f"{experts_base}.{expert_idx}.down_proj.weight",
                    expert_weight,
                    block_size=block_size,
                    modules_to_not_convert=modules_to_not_convert,
                    device=device,
                    scale_dtype=scale_dtype,
                )
            continue

        if key.endswith(".weight"):
            _save_weight(
                output,
                key,
                weight,
                block_size=block_size,
                modules_to_not_convert=modules_to_not_convert,
                device=device,
                scale_dtype=scale_dtype,
            )
        else:
            output[key] = weight.cpu()

    return output


def _scan_weight_map(output_fp8_path: str) -> dict[str, str]:
    weight_map: dict[str, str] = {}
    for shard in sorted(glob(os.path.join(output_fp8_path, "*.safetensors"))):
        shard_name = os.path.basename(shard)
        if shard_name == _MTP_SHARD_NAME:
            continue
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = shard_name
    return weight_map


def _dtype_size(dtype: str) -> int:
    dtype_sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    if dtype not in dtype_sizes:
        raise ValueError(f"Unknown safetensors dtype {dtype!r}")
    return dtype_sizes[dtype]


def _compute_total_size(output_fp8_path: str) -> int:
    total_size = 0
    for shard in sorted(glob(os.path.join(output_fp8_path, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor_slice = f.get_slice(key)
                total_size += math.prod(tensor_slice.get_shape()
                                       ) * _dtype_size(tensor_slice.get_dtype())
    return total_size


def _load_or_create_index(output_fp8_path: str) -> tuple[dict[str, Any], dict[str, str]]:
    index_path = os.path.join(output_fp8_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        return dict(index.get("metadata", {})), dict(index["weight_map"])
    return {}, _scan_weight_map(output_fp8_path)


def _write_index(
    output_fp8_path: str, metadata: dict[str, Any], weight_map: dict[str, str]
) -> None:
    index_path = os.path.join(output_fp8_path, "model.safetensors.index.json")
    metadata = dict(metadata)
    metadata["total_size"] = _compute_total_size(output_fp8_path)
    with open(index_path, "w") as f:
        json.dump({"metadata": metadata, "weight_map": weight_map}, f, indent=2, sort_keys=True)
        f.write("\n")


def append_mtp_weights_to_fp8_checkpoint(
    input_hf_path: str,
    output_fp8_path: str,
    *,
    block_size: tuple[int, int],
    modules_to_not_convert: list[str],
    device: str | torch.device | None = None,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Append official-layout MTP weights to an FP8 checkpoint.

    Returns
    -------
    int
        Number of tensors written to ``mtp.safetensors``. Returns ``0`` when
        the input checkpoint has no top-level ``mtp.*`` tensors.
    """
    mtp_tensors = _load_mtp_tensors(input_hf_path)
    _validate_mtp_tensors(input_hf_path, mtp_tensors)
    if not mtp_tensors:
        return 0

    mtp_state_dict = _build_mtp_state_dict(
        mtp_tensors,
        block_size=block_size,
        modules_to_not_convert=modules_to_not_convert,
        device=device,
        scale_dtype=scale_dtype,
    )

    mtp_shard_path = os.path.join(output_fp8_path, _MTP_SHARD_NAME)
    save_file(mtp_state_dict, mtp_shard_path)

    metadata, weight_map = _load_or_create_index(output_fp8_path)
    weight_map = {
        key: shard
        for key, shard in weight_map.items()
        if not (key.startswith("mtp.") or shard == _MTP_SHARD_NAME)
    }
    for key in mtp_state_dict:
        weight_map[key] = _MTP_SHARD_NAME
    _write_index(output_fp8_path, metadata, weight_map)

    return len(mtp_state_dict)
