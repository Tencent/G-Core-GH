"""Build Qwen3.5 / Qwen3.6 official-style FineGrained FP8 skip lists.

Matches HuggingFace hub checkpoints such as ``Qwen/Qwen3.6-35B-A3B-FP8``:
quantize MoE experts and large linear/full-attention projections; keep routers,
norms, small GatedDeltaNet weights, vision tower, embed, and lm_head in BF16.
"""

from __future__ import annotations

import json
import os
from typing import Any

# GatedDeltaNet weights that are not block-128 aligned (official keeps BF16).
_LINEAR_ATTN_SKIP_SUFFIXES = (
    "A_log",
    "conv1d",
    "dt_bias",
    "in_proj_ba",
    "in_proj_b",
    "in_proj_a",
    "norm",
)

_MTP_SKIP_MODULES = (
    "mtp.layers.0.input_layernorm",
    "mtp.layers.0.post_attention_layernorm",
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.mlp.shared_expert_gate",
    "mtp.layers.0.self_attn.q_norm",
    "mtp.layers.0.self_attn.k_norm",
    "mtp.fc",
    "mtp.norm",
    "mtp.pre_fc_norm_embedding",
    "mtp.pre_fc_norm_hidden",
)


def load_config_dict(config_path: str) -> dict[str, Any]:
    """Load ``config.json`` from a checkpoint directory or file path."""
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    with open(config_path) as f:
        return json.load(f)


def load_modules_to_not_convert_from_reference(reference_path: str) -> list[str]:
    """Load explicit ``modules_to_not_convert`` from a reference FP8 checkpoint."""
    cfg = load_config_dict(reference_path)
    qcfg = cfg.get("quantization_config")
    if not qcfg or "modules_to_not_convert" not in qcfg:
        raise ValueError(f"No quantization_config.modules_to_not_convert in {reference_path}")
    modules = qcfg["modules_to_not_convert"]
    if not isinstance(modules, list) or not modules:
        raise ValueError("modules_to_not_convert must be a non-empty list")
    return list(modules)


def _as_dict(section: Any) -> dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return section
    if hasattr(section, "to_dict"):
        return section.to_dict()
    return dict(section)


def _get_text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if "text_config" in cfg:
        return _as_dict(cfg["text_config"])
    return cfg


def _get_vision_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    if "vision_config" not in cfg:
        return None
    return _as_dict(cfg["vision_config"])


def _is_moe_text_config(text_config: dict[str, Any]) -> bool:
    model_type = text_config.get("model_type", "")
    if "moe" in model_type:
        return True
    return int(text_config.get("num_experts", 0) or 0) > 1


def _language_model_prefix(cfg: dict[str, Any]) -> str:
    """Module prefix for decoder layers (VL vs text-only)."""
    architectures = cfg.get("architectures") or []
    arch = architectures[0] if architectures else ""
    arch_str = str(arch)
    if "ConditionalGeneration" in arch_str or "ImageTextToText" in arch_str:
        return "model.language_model"
    if "text_config" in cfg:
        return "model.language_model"
    return "model"


def _resolve_layer_types(text_config: dict[str, Any]) -> list[str]:
    layer_types = text_config.get("layer_types")
    if layer_types is not None:
        return list(layer_types)
    num_layers = int(text_config["num_hidden_layers"])
    interval = int(text_config.get("full_attention_interval", 4))
    return [
        "linear_attention" if bool((i + 1) % interval) else "full_attention"
        for i in range(num_layers)
    ]


def _vision_skip_modules(vision_config: dict[str, Any]) -> list[str]:
    depth = int(vision_config.get("depth", 0))
    modules: list[str] = ["visual", "model.visual"]

    deepstack_count = len(vision_config.get("deepstack_visual_indexes") or []) or 3

    for prefix in ("model.visual", "visual"):
        modules.extend(
            [
                f"{prefix}.patch_embed.proj",
                f"{prefix}.pos_embed",
                f"{prefix}.merger.linear_fc1",
                f"{prefix}.merger.linear_fc2",
                f"{prefix}.merger.norm",
            ]
        )
        for i in range(deepstack_count):
            modules.extend(
                [
                    f"{prefix}.deepstack_merger_list.{i}.linear_fc1",
                    f"{prefix}.deepstack_merger_list.{i}.linear_fc2",
                    f"{prefix}.deepstack_merger_list.{i}.norm",
                ]
            )

    for i in range(depth):
        modules.extend(
            [
                f"model.visual.blocks.{i}.attn.proj",
                f"model.visual.blocks.{i}.attn.qkv",
                f"model.visual.blocks.{i}.mlp.linear_fc1",
                f"model.visual.blocks.{i}.mlp.linear_fc2",
                f"visual.blocks.{i}.attn.proj",
                f"visual.blocks.{i}.attn.qkv_proj",
                f"visual.blocks.{i}.mlp.linear_fc1",
                f"visual.blocks.{i}.mlp.linear_fc2",
            ]
        )
    return modules


def build_qwen3_official_modules_to_not_convert(
    config: dict[str, Any],
    *,
    include_mtp: bool = True,
    include_vision: bool | None = None,
) -> list[str]:
    """Build hub-aligned explicit ``modules_to_not_convert`` from a model config.

    Parameters
    ----------
    config:
        Parsed ``config.json`` (VL MoE, VL dense, or text-only).
    include_mtp:
        If True, append MTP module names (harmless when MTP weights are absent).
    include_vision:
        If None, auto-enable when ``vision_config`` is present in *config*.
    """
    text_config = _get_text_config(config)
    num_layers = int(text_config["num_hidden_layers"])
    layer_types = _resolve_layer_types(text_config)
    lm_prefix = _language_model_prefix(config)
    is_moe = _is_moe_text_config(text_config)

    if include_vision is None:
        include_vision = _get_vision_config(config) is not None

    # Hub checkpoints list ``model.embed_tokens`` only (VL weights live under
    # ``model.language_model.embed_tokens``; Embedding is not FP8-quantized anyway).
    modules: list[str] = [
        "lm_head",
        "model.embed_tokens",
    ]

    for layer_idx in range(num_layers):
        layer = f"{lm_prefix}.layers.{layer_idx}"
        modules.append(f"{layer}.input_layernorm")
        modules.append(f"{layer}.post_attention_layernorm")
        if is_moe:
            modules.append(f"{layer}.mlp.gate")
            modules.append(f"{layer}.mlp.shared_expert_gate")

        layer_type = layer_types[layer_idx]
        if layer_type == "linear_attention":
            for suffix in _LINEAR_ATTN_SKIP_SUFFIXES:
                modules.append(f"{layer}.linear_attn.{suffix}")
        elif layer_type in ("full_attention", "sliding_attention"):
            modules.append(f"{layer}.self_attn.q_norm")
            modules.append(f"{layer}.self_attn.k_norm")
        else:
            raise ValueError(f"Unknown layer_type[{layer_idx}]={layer_type!r}")

    if include_mtp:
        modules.extend(_MTP_SKIP_MODULES)

    if include_vision:
        vision_config = _get_vision_config(config)
        if vision_config is not None:
            modules.extend(_vision_skip_modules(vision_config))

    return modules


def build_qwen3_official_modules_to_not_convert_from_path(config_path: str) -> list[str]:
    """Load config from path and build the official skip list."""
    return build_qwen3_official_modules_to_not_convert(load_config_dict(config_path))
