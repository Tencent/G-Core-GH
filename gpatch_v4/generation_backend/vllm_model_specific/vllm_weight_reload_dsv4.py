# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from __future__ import annotations

import inspect
import logging
import os
import re
from collections.abc import Iterable
from contextlib import contextmanager
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn

from gpatch_v4.generation_backend.vllm_moe_weight_loader_patch import (
    patch_vllm_moe_model_weight_loader,
)
from gpatch_v4.utils import logging_memory_usage_details

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = logging.getLogger(__name__)

_MEGA_MOE_CLS = "DeepseekV4MegaMoEExperts"
_MXFP4_MOE_METHODS = frozenset({"Mxfp4MoEMethod", "GptOssMxfp4MoEMethod"})
_MXFP4_WEIGHT_DTYPES = frozenset({"mxfp4", "gpt_oss_mxfp4"})

GCORE_RELOAD_STATE_ATTR = "_gcore_weight_reload_state"
GCORE_DENSE_FP8_SCALE_CACHE_ATTR = "_gcore_dense_fp8_scale_cache"

DSV4_VLLM_LOAD_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(src), tgt) for src, tgt in (
        (r"^embed\.weight$", "embed_tokens.weight"),
        (r"^head\.weight$", "lm_head.weight"),
        (
            r"^(layers\.\d+)\.attn\.(?:wq_a|wkv)\.weight$",
            r"\1.attn.fused_wqa_wkv.weight",
        ),
        (
            r"^(layers\.\d+)\.attn\.(?:wq_a|wkv)\.scale$",
            r"\1.attn.fused_wqa_wkv.weight_scale_inv",
        ),
        (
            r"^(layers\.\d+)\.attn\.compressor\.(?:wkv|wgate)\.weight$",
            r"\1.attn.mla_attn.compressor.fused_wkv_wgate.weight",
        ),
        (
            r"^(layers\.\d+)\.attn\.compressor\.norm\.weight$",
            r"\1.attn.mla_attn.compressor.norm.weight",
        ),
        (
            r"^(layers\.\d+)\.attn\.compressor\.ape$",
            r"\1.attn.mla_attn.compressor.ape",
        ),
        (
            r"^(layers\.\d+)\.attn\.indexer\.compressor\.(?:wkv|wgate)\.weight$",
            r"\1.attn.indexer.compressor.fused_wkv_wgate.weight",
        ),
        (
            r"^(layers\.\d+)\.ffn\.experts\.\d+\.w[13]\.weight$",
            r"\1.ffn.experts.w13_weight",
        ),
        (
            r"^(layers\.\d+)\.ffn\.experts\.\d+\.w2\.weight$",
            r"\1.ffn.experts.w2_weight",
        ),
    )
)
DSV4_EXPERT_WEIGHT_RE = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")


def _module_param_device(module: nn.Module) -> torch.device:
    for param_name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        param = getattr(module, param_name, None)
        if param is not None and hasattr(param, "device"):
            return param.device

    transformed = getattr(module, "_transformed_l1_weights", None)
    if transformed is not None:
        return transformed[0].device

    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _make_uint8_moe_param(
    shape: tuple[int, ...],
    device: torch.device,
    weight_loader,
    *,
    source_param: nn.Parameter | None = None,
    quant_method: str | None = None,
) -> nn.Parameter:
    from vllm.model_executor.utils import set_weight_attrs

    param = nn.Parameter(
        torch.empty(shape, dtype=torch.uint8, device=device),
        requires_grad=False,
    )
    if source_param is not None:
        _copy_param_attributes(
            param,
            source_param,
            exclude_attrs={"weight_loader", "_weight_loader"},
        )
    set_weight_attrs(param, {"weight_loader": weight_loader})
    if quant_method is not None:
        param.quant_method = quant_method
    return param


def _restore_moe_module_params(
    module: nn.Module,
    load_shape: tuple[int, int, int],
) -> None:
    num_experts, intermediate_size, hidden_size = load_shape
    device = _module_param_device(module)
    weight_loader = getattr(module, "weight_loader", None)
    assert weight_loader is not None, (
        f"MoE module {type(module).__name__} missing weight_loader during reload prepare"
    )

    old_w13 = getattr(module, "w13_weight", None)
    old_w2 = getattr(module, "w2_weight", None)
    old_w13_scale = getattr(module, "w13_weight_scale", None)
    old_w2_scale = getattr(module, "w2_weight_scale", None)

    module.w13_weight = _make_uint8_moe_param(
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device,
        weight_loader,
        source_param=old_w13,
    )
    module.w2_weight = _make_uint8_moe_param(
        (num_experts, hidden_size, intermediate_size // 2),
        device,
        weight_loader,
        source_param=old_w2,
    )
    module.w13_weight_scale = _make_uint8_moe_param(
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        device,
        weight_loader,
        source_param=old_w13_scale,
        quant_method="block",
    )
    module.w2_weight_scale = _make_uint8_moe_param(
        (num_experts, hidden_size, intermediate_size // 32),
        device,
        weight_loader,
        source_param=old_w2_scale,
        quant_method="block",
    )


def hook_cpu_mapped_fused_moe_weight_loader(module: nn.Module) -> None:
    expert_map = getattr(module, "_expert_map", None)
    if expert_map is None:
        return

    cpu_expert_map = [int(value) for value in expert_map.detach().cpu().tolist()]
    original_weight_loader = module.weight_loader

    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if expert_id < 0 or expert_id >= len(cpu_expert_map):
            local_expert_id = -1
        else:
            local_expert_id = cpu_expert_map[expert_id]

        use_global_sf = (
            getattr(module.quant_method, "use_global_sf", False) and "input_scale" in weight_name
        )
        if local_expert_id == -1 and not use_global_sf:
            return False if return_success else None

        saved_expert_map = module._expert_map
        module._expert_map = None
        try:
            return original_weight_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id=shard_id,
                expert_id=local_expert_id,
                return_success=return_success,
            )
        finally:
            module._expert_map = saved_expert_map

    for param_name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
    ):
        param = getattr(module, param_name, None)
        if param is not None:
            param.weight_loader = weight_loader


def _mega_moe_load_shape(module: nn.Module) -> tuple[int, int, int] | None:
    shape = (
        getattr(module, "num_local_experts", None),
        getattr(module, "intermediate_size", None),
        getattr(module, "hidden_size", None),
    )
    if any(value is None for value in shape):
        return None
    return tuple(int(value) for value in shape)


def _mxfp4_fused_moe_load_shape(module: nn.Module) -> tuple[int, int, int] | None:
    quant_method = getattr(module, "quant_method", None)
    shape = (
        getattr(quant_method, "num_experts", getattr(module, "local_num_experts", None)),
        getattr(
            quant_method,
            "intermediate_size",
            getattr(module, "intermediate_size_per_partition", None),
        ),
        getattr(quant_method, "hidden_size", getattr(module, "hidden_size", None)),
    )
    if any(value is None for value in shape):
        return None
    return tuple(int(value) for value in shape)


def _is_mxfp4_fused_moe_module(module: nn.Module) -> bool:
    try:
        from vllm.model_executor.layers.fused_moe import FusedMoE
    except ImportError:
        return False

    if not isinstance(module, FusedMoE):
        return False

    quant_method = getattr(module, "quant_method", None)
    if quant_method is None:
        return False
    quant_method_name = type(quant_method).__name__
    return (
        quant_method_name in _MXFP4_MOE_METHODS or
        getattr(quant_method, "weight_dtype", None) in _MXFP4_WEIGHT_DTYPES
    )


def _restore_mega_moe_module(module: nn.Module) -> None:
    module._transformed_l1_weights = None
    module._transformed_l2_weights = None

    if getattr(module, "w13_weight", None) is not None:
        return

    load_shape = _mega_moe_load_shape(module)
    assert load_shape is not None, (
        f"DeepseekV4MegaMoEExperts missing shape metadata: "
        f"num_local_experts={getattr(module, 'num_local_experts', None)!r} "
        f"intermediate_size={getattr(module, 'intermediate_size', None)!r} "
        f"hidden_size={getattr(module, 'hidden_size', None)!r}"
    )
    _restore_moe_module_params(module, load_shape)


def _restore_mxfp4_fused_moe_module(module: nn.Module) -> None:
    load_shape = _mxfp4_fused_moe_load_shape(module)
    assert load_shape is not None, (
        f"MXFP4 FusedMoE missing shape metadata on {type(module).__name__}"
    )

    # ``process_weights_after_loading`` may leave backend-specific runtime
    # tensors on the module. Recreate checkpoint-load-layout parameters before
    # each RL weight update so vLLM's FusedMoE loader can shard w1/w2/w3 normally.
    _restore_moe_module_params(module, load_shape)
    hook_cpu_mapped_fused_moe_weight_loader(module)
    quant_method = getattr(module, "quant_method", None)
    if quant_method is not None:
        for attr in (
            "moe_kernel",
            "moe_quant_config",
            "w13_precision_config",
            "w2_precision_config",
        ):
            if hasattr(quant_method, attr):
                setattr(quant_method, attr, None)


def _normalize_dim(dim: int | None, ndim: int) -> int:
    if dim is None:
        return 0
    if dim < 0:
        dim += ndim
    return dim


def _model_type(model: nn.Module) -> str | None:
    for obj in (model, getattr(model, "config", None), getattr(model, "hf_config", None)):
        if obj is None:
            continue
        model_type = getattr(obj, "model_type", None)
        if model_type is not None:
            return str(model_type)
    config = getattr(model, "config", None)
    if config is not None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            text_type = getattr(text_config, "model_type", None)
            if text_type is not None:
                return str(text_type)
    return None


def apply_shared_expert_alias(name: str, *, include_gate_up: bool) -> str:
    if include_gate_up and ".shared_experts.w1" in name:
        name = name.replace(".shared_experts.w1", ".shared_experts.gate_up_proj", 1)
    elif include_gate_up and ".shared_experts.w3" in name:
        name = name.replace(".shared_experts.w3", ".shared_experts.gate_up_proj", 1)
    elif ".shared_experts.w2" in name:
        name = name.replace(".shared_experts.w2", ".shared_experts.down_proj", 1)

    if name.endswith(".scale"):
        name = name[:-len(".scale")] + ".weight_scale_inv"
    return name


def map_checkpoint_key_to_vllm_param_name(model: nn.Module, name: str) -> str:
    """Map a GCore checkpoint export key to a vLLM parameter name."""
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    map_name = getattr(mapper, "_map_name", None)
    if callable(map_name):
        mapped = map_name(name)
        if mapped is not None:
            return str(mapped)

    mapped = apply_shared_expert_alias(name.removeprefix("model."), include_gate_up=False)
    if mapped.startswith("layers."):
        mapped = f"model.{mapped}"
    elif mapped.startswith("embed."):
        mapped = f"model.{mapped}"
    elif mapped == "head.weight":
        mapped = "lm_head.weight"
    return mapped


def cache_deepseek_v4_dense_fp8_scales(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> None:
    """Cache dense FP8 scale tensors (verl-style) for post-bucket TP copy."""
    if _model_type(model) != "deepseek_v4":
        return

    scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scale_dtype is None:
        return

    cache = getattr(model, GCORE_DENSE_FP8_SCALE_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(model, GCORE_DENSE_FP8_SCALE_CACHE_ATTR, cache)

    for name, tensor in weights:
        should_skip = (
            not name.endswith(".scale") or ".ffn.experts." in name or tensor.dtype != scale_dtype
        )
        if should_skip:
            continue
        mapped = map_checkpoint_key_to_vllm_param_name(model, name)
        cache[mapped] = tensor.detach().clone()


def _copy_scale_shard(param: nn.Parameter, loaded_scale: torch.Tensor) -> bool:
    target = param.data
    loaded = loaded_scale.to(device=target.device, dtype=target.dtype)
    if target.shape == loaded.shape:
        target.copy_(loaded)
        return True

    tp_rank = int(getattr(param, "tp_rank", 0))
    tp_size = int(getattr(param, "tp_size", 1))

    # Resolve ndim mismatch by trying to add leading dims to `loaded`.
    # Example: loaded (64, 32) vs target (1, 8, 32) with tp_size=8
    # → reshape to (8, 8, 32) → narrow dim 0 by tp_rank.
    if target.ndim != loaded.ndim and loaded.numel() % target.numel() == 0:
        mult = loaded.numel() // target.numel()
        if mult == tp_size and target.ndim > 0:
            try:
                # loaded (64, 32) * tp=8, target (1, 8, 32)
                # → reshaped (8, 8, 32) → narrow dim 0 → (1, 8, 32)
                extra_dims = target.shape[1:]
                reshaped = loaded.reshape((mult, ) + extra_dims)
                shard = reshaped.narrow(0, tp_rank, 1)
                if shard.shape == target.shape:
                    target.copy_(shard)
                    return True
            except RuntimeError:
                pass

    if target.ndim != loaded.ndim:
        return False

    candidate_dims: list[int] = []
    for attr in ("input_dim", "_input_dim", "output_dim", "_output_dim"):
        if hasattr(param, attr):
            dim = _normalize_dim(int(getattr(param, attr)), target.ndim)
            if dim not in candidate_dims:
                candidate_dims.append(dim)

    for dim in candidate_dims:
        if (
            loaded.shape[:dim] != target.shape[:dim] or
            loaded.shape[dim + 1:] != target.shape[dim + 1:]
        ):
            continue
        if loaded.shape[dim] != target.shape[dim] * tp_size:
            continue
        start = tp_rank * target.shape[dim]
        target.copy_(loaded.narrow(dim, start, target.shape[dim]))
        return True

    return False


def _resolve_param_via_module_tree(model: nn.Module, vllm_name: str) -> nn.Parameter | None:
    """Resolve a ``named_parameters`` key to the actual param via PyTorch module tree.

    Bypasses ``dict(named_parameters())`` which may return stale refs when
    ``_ensure_model_params_reloadable`` replaced ``LinearBase`` weights.
    """
    parts = vllm_name.split(".")
    param_name = parts[-1]
    module_path = parts[:-1]
    try:
        if not module_path:
            current: object = model
        else:
            current = model.get_submodule(".".join(module_path))
        result = getattr(current, param_name, None)
        if isinstance(result, nn.Parameter):
            return result
    except (AttributeError, IndexError, ValueError, TypeError):
        pass
    return None


def _build_scale_map_via_modules(model: nn.Module) -> dict[str, nn.Parameter]:
    """Build param-name → active-scale-param map via module tree traversal.

    Stores each scale under BOTH the canonical ``named_modules()`` key AND
    the simplified ``named_parameters()``-style key (stripping ``mla_attn.``
    infix that ``map_checkpoint_key_to_vllm_param_name`` discards).
    """
    scale_map: dict[str, nn.Parameter] = {}
    for mod_name, module in model.named_modules():
        for scale_attr in ("weight_scale_inv", "weight_scale"):
            scale = getattr(module, scale_attr, None)
            if not isinstance(scale, nn.Parameter):
                continue
            canonical = f"{mod_name}.{scale_attr}" if mod_name else scale_attr
            scale_map[canonical] = scale
            # Also register under the simplified key used by the cache
            # (map_checkpoint_key_to_vllm_param_name strips mla_attn. infix).
            simplified = canonical.replace(".mla_attn.", ".")
            if simplified != canonical:
                scale_map[simplified] = scale
    return scale_map


def reload_cached_deepseek_v4_dense_fp8_scales(model: nn.Module) -> None:
    cache = getattr(model, GCORE_DENSE_FP8_SCALE_CACHE_ATTR, None)
    if not cache:
        return

    scale_map = _build_scale_map_via_modules(model)
    for name, scale in cache.items():
        param = scale_map.get(name)
        if param is None:
            continue
        _copy_scale_shard(param, scale)


def _copy_param_attributes(
    dst_param: nn.Parameter,
    src_param: nn.Parameter,
    *,
    exclude_attrs: set[str] | None = None,
) -> None:
    base_param_dir = dir(nn.Parameter)
    excluded = exclude_attrs or set()
    for attr in dir(src_param):
        if attr in base_param_dir or attr.startswith("__") or attr in excluded:
            continue
        try:
            setattr(dst_param, attr, getattr(src_param, attr))
        except Exception:
            pass


def _create_param_from_subclass_attributes(
    custom_param: nn.Parameter,
    source_param: nn.Parameter | None = None,
) -> nn.Parameter:
    param = nn.Parameter(custom_param.data, requires_grad=False)
    if source_param is not None:
        _copy_param_attributes(param, source_param)
    _copy_param_attributes(param, custom_param)
    param.subclass_type = type(custom_param)
    return param


def _get_param_weight_loader(param: nn.Parameter):
    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None:
        weight_loader = getattr(param, "_weight_loader", None)
    return weight_loader


def _param_parallel_dim(
    param: nn.Parameter,
    public_name: str,
    private_name: str,
    default: int,
) -> int:
    if hasattr(param, private_name):
        return int(getattr(param, private_name))
    if hasattr(param, public_name):
        return int(getattr(param, public_name))
    return default


def _copy_loaded_weight(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight.to(device=param.data.device, dtype=param.data.dtype))


def _try_load_grouped_column_weight(param: nn.Parameter, loaded_weight: torch.Tensor) -> bool:
    tp_size = int(getattr(param, "tp_size", 1))
    tp_rank = int(getattr(param, "tp_rank", 0))
    if tp_size <= 1:
        return False

    data = param.data
    if loaded_weight.ndim == data.ndim + 1 and loaded_weight.shape == torch.Size(
        (tp_size, *data.shape)
    ):
        _copy_loaded_weight(param, loaded_weight.select(0, tp_rank))
        return True

    if data.ndim >= 2 and loaded_weight.ndim == data.ndim - 1:
        expected_flat_shape = (tp_size * data.shape[0] * data.shape[1], *data.shape[2:])
        if loaded_weight.shape == torch.Size(expected_flat_shape):
            global_shape = (tp_size, data.shape[0], data.shape[1], *data.shape[2:])
            _copy_loaded_weight(param, loaded_weight.reshape(global_shape).select(0, tp_rank))
            return True

    if data.ndim == loaded_weight.ndim and data.ndim > 0:
        expected_dim0 = tp_size * data.shape[0]
        if loaded_weight.shape[0] == expected_dim0 and loaded_weight.shape[1:] == data.shape[1:]:
            start = tp_rank * data.shape[0]
            _copy_loaded_weight(param, loaded_weight.narrow(0, start, data.shape[0]))
            return True

    return False


def _merged_column_offsets(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_offset: int | None,
    shard_size: int | None,
    loaded_shard_id: int | None,
):
    dim = _normalize_dim(
        getattr(param, "output_dim", getattr(param, "_output_dim", 0)), param.data.ndim
    )
    loaded_dim = loaded_weight.shape[dim]
    offsets: list[int] = []
    if isinstance(shard_offset, int):
        offsets.append(shard_offset)
        if isinstance(shard_size, int) and shard_size > 0:
            scaled_offset = shard_offset * loaded_dim // shard_size
            offsets.append(scaled_offset)
    if isinstance(loaded_shard_id, int):
        offsets.append(loaded_shard_id * loaded_dim)

    seen: set[int] = set()
    for offset in offsets:
        if offset in seen:
            continue
        seen.add(offset)
        if offset < 0 or offset + loaded_dim > param.data.shape[dim]:
            continue
        yield dim, offset


def _try_load_merged_column_weight(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_offset: int | None,
    shard_size: int | None,
    loaded_shard_id: int | None,
) -> bool:
    for dim, offset in _merged_column_offsets(
        param,
        loaded_weight,
        shard_offset,
        shard_size,
        loaded_shard_id,
    ):
        target = param.data.narrow(dim, offset, loaded_weight.shape[dim])
        if target.shape == loaded_weight.shape:
            target.copy_(loaded_weight.to(device=target.device, dtype=target.dtype))
            return True
    return False


def _attach_fp8_reload_fallbacks(param: nn.Parameter) -> None:
    subclass_type = getattr(param, "subclass_type", None)
    if subclass_type is None or getattr(param, "_gcore_fp8_reload_fallbacks", False):
        return

    original_column_loader = getattr(subclass_type, "load_column_parallel_weight", None)
    original_merged_loader = getattr(subclass_type, "load_merged_column_weight", None)

    def _parse_common_args(args, kwargs):
        loaded_weight = kwargs.get("loaded_weight")
        if loaded_weight is None and args:
            loaded_weight = args[0]
        loaded_shard_id = kwargs.get("loaded_shard_id", kwargs.get("shard_id"))
        if loaded_shard_id is None and len(args) > 1:
            loaded_shard_id = args[1]
        shard_offset = kwargs.get("shard_offset")
        if shard_offset is None and len(args) > 2:
            shard_offset = args[2]
        shard_size = kwargs.get("shard_size")
        if shard_size is None and len(args) > 3:
            shard_size = args[3]
        return loaded_weight, loaded_shard_id, shard_offset, shard_size

    def load_column_parallel_weight(self, *args, **kwargs):
        loaded_weight, _, _, _ = _parse_common_args(args, kwargs)
        if loaded_weight is not None:
            if self.data.shape == loaded_weight.shape:
                _copy_loaded_weight(self, loaded_weight)
                return
            if _try_load_grouped_column_weight(self, loaded_weight):
                return
        if original_column_loader is not None:
            return original_column_loader(self, *args, **kwargs)
        raise RuntimeError(
            "GCore FP8 reload fallback failed in load_column_parallel_weight: "
            f"param_shape={tuple(self.data.shape)} "
            f"loaded_shape={tuple(loaded_weight.shape) if loaded_weight is not None else None}"
        )

    def load_merged_column_weight(self, *args, **kwargs):
        loaded_weight, loaded_shard_id, shard_offset, shard_size = _parse_common_args(args, kwargs)
        if loaded_weight is not None:
            if _try_load_merged_column_weight(
                self,
                loaded_weight,
                shard_offset,
                shard_size,
                loaded_shard_id,
            ):
                return
            if self.data.shape == loaded_weight.shape:
                _copy_loaded_weight(self, loaded_weight)
                return
            if _try_load_grouped_column_weight(self, loaded_weight):
                return
        if original_merged_loader is not None:
            return original_merged_loader(self, *args, **kwargs)
        raise RuntimeError(
            "GCore FP8 reload fallback failed in load_merged_column_weight: "
            f"param_shape={tuple(self.data.shape)} "
            f"loaded_shape={tuple(loaded_weight.shape) if loaded_weight is not None else None} "
            f"loaded_shard_id={loaded_shard_id} shard_offset={shard_offset} shard_size={shard_size}"
        )

    param.load_column_parallel_weight = MethodType(load_column_parallel_weight, param)
    param.load_merged_column_weight = MethodType(load_merged_column_weight, param)
    param._gcore_fp8_reload_fallbacks = True


def _ensure_linear_params_reloadable(layer: nn.Module) -> None:
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )

    if hasattr(layer, "weight") and not hasattr(layer.weight, "subclass_type"):
        source_weight = layer.weight
        weight_loader = _get_param_weight_loader(source_weight)
        if weight_loader is not None:
            layer.weight = _create_param_from_subclass_attributes(
                ModelWeightParameter(
                    data=source_weight.data,
                    output_dim=_param_parallel_dim(source_weight, "output_dim", "_output_dim", 0),
                    input_dim=_param_parallel_dim(source_weight, "input_dim", "_input_dim", 1),
                    weight_loader=weight_loader,
                ),
                source_weight,
            )

    for scale_name in ("weight_scale_inv", "weight_scale"):
        if not hasattr(layer, scale_name):
            continue
        scale = getattr(layer, scale_name)
        if hasattr(scale, "subclass_type"):
            continue
        weight_loader = _get_param_weight_loader(scale)
        if weight_loader is not None:
            setattr(
                layer,
                scale_name,
                _create_param_from_subclass_attributes(
                    BlockQuantScaleParameter(
                        data=scale.data,
                        output_dim=_param_parallel_dim(scale, "output_dim", "_output_dim", 0),
                        input_dim=_param_parallel_dim(scale, "input_dim", "_input_dim", 1),
                        weight_loader=weight_loader,
                    ),
                    scale,
                ),
            )

    update_param_tp_status = getattr(layer, "update_param_tp_status", None)
    if callable(update_param_tp_status):
        update_param_tp_status()

    for param_name in ("weight", "weight_scale_inv", "weight_scale"):
        current = getattr(layer, param_name, None)
        if current is not None:
            _attach_fp8_reload_fallbacks(current)


def _ensure_model_params_reloadable(model: nn.Module) -> None:
    try:
        from vllm.model_executor.layers.linear import LinearBase
    except ImportError:
        return

    for module in model.modules():
        if isinstance(module, LinearBase):
            _ensure_linear_params_reloadable(module)


@contextmanager
def param_subclass_load_context(model: nn.Module):
    """Temporarily use vLLM param subclasses during ``load_weights`` (verl-style)."""
    patched: list[nn.Parameter] = []
    for _, param in model.named_parameters():
        subclass_type = getattr(param, "subclass_type", None)
        if subclass_type is None or subclass_type is type(param):
            continue
        param.orig_type = param.__class__
        param.__class__ = subclass_type
        patched.append(param)
    try:
        yield
    finally:
        for param in patched:
            param.__class__ = param.orig_type
            del param.orig_type


def _detect_mxfp4_moe_present(model: nn.Module) -> bool:
    return any(_is_mxfp4_fused_moe_module(module) for module in model.modules())


def _process_mxfp4_moe_weights_after_loading(model: nn.Module) -> None:
    """Run per-expert ``process_weights_after_loading`` (verl-style)."""
    for name, module in model.named_modules():
        if not _is_mxfp4_fused_moe_module(module):
            continue
        quant_method = getattr(module, "quant_method", None)
        process_weights = getattr(quant_method, "process_weights_after_loading", None)
        if callable(process_weights):
            process_weights(module)


def restore_quantized_moe_params_for_loading(model: nn.Module) -> bool:
    """Recreate checkpoint-layout MoE params destroyed by prior finalize.

    Returns
    -------
    bool
        ``True`` if at least one MoE module was restored.
    """
    restored = False
    for module in model.modules():
        if type(module).__name__ == _MEGA_MOE_CLS:
            _restore_mega_moe_module(module)
            restored = True
            continue
        if _is_mxfp4_fused_moe_module(module):
            _restore_mxfp4_fused_moe_module(module)
            restored = True
    return restored


def append_unique_name(names: list[str], seen: set[str], name: str) -> None:
    if name not in seen:
        names.append(name)
        seen.add(name)


def base_reload_name_variants(name: str) -> list[str]:
    raw_name = name.removeprefix("model.")
    variants: list[str] = []
    seen: set[str] = set()
    append_unique_name(variants, seen, name)
    append_unique_name(variants, seen, raw_name)
    for rgx, repl in DSV4_VLLM_LOAD_RENAMES:
        renamed, count = rgx.subn(repl, raw_name, count=1)
        if count > 0:
            append_unique_name(variants, seen, renamed)
    return variants


def with_model_prefix_variants(names: Iterable[str]) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()
    for name in names:
        append_unique_name(variants, seen, name)
        if not name.startswith("model."):
            append_unique_name(variants, seen, f"model.{name}")
    return variants


def loaded_name_candidates(name: str) -> set[str]:
    """Return vLLM ``load_weights`` names that can satisfy one checkpoint key."""
    base_names = base_reload_name_variants(name)
    candidates = set(with_model_prefix_variants(base_names))

    for base_name in base_names:
        if not base_name.endswith(".scale"):
            continue
        prefix = base_name[:-len(".scale")]
        scale_names = {
            f"{prefix}.weight_scale",
            f"{prefix}.weight_scale_inv",
        }
        candidates.update(scale_names)
        candidates.update(
            f"model.{scale_name}"
            for scale_name in scale_names if not scale_name.startswith("model.")
        )
    return candidates


def _is_strict_reload_key(name: str) -> bool:
    """Keys that should map one-to-one, unlike fused MoE/shared-expert keys."""
    if name.startswith("mtp."):
        return False
    if ".ffn.experts." in name or ".ffn.shared_experts." in name:
        return False
    return (
        ".attn." in name or ".attn_norm." in name or ".ffn_norm." in name or
        name.endswith((".attn_norm.weight", ".ffn_norm.weight"))
    )


def _record_reload_observation(
    model: nn.Module,
    bucket_names: list[str],
    loaded_names: set[str],
    strict_missing: list[str],
) -> None:
    state = getattr(model, "_gcore_weight_reload_debug", None)
    if state is None:
        state = {
            "buckets": 0,
            "input_keys": 0,
            "loaded_keys": 0,
            "strict_missing": [],
        }
        setattr(model, "_gcore_weight_reload_debug", state)

    state["buckets"] += 1
    state["input_keys"] += len(bucket_names)
    state["loaded_keys"] += len(loaded_names)
    state["strict_missing"].extend(strict_missing)


def validate_loaded_weights(
    model: nn.Module,
    bucket_names: list[str],
    loaded: object,
) -> set[str]:
    if not bucket_names:
        return set()
    if loaded is None:
        raise RuntimeError(
            "vLLM model.load_weights returned None for a non-empty GCore "
            f"weight bucket. first_keys={bucket_names[:8]}"
        )

    loaded_names = {str(name) for name in loaded}
    strict_bucket_names = [name for name in bucket_names if _is_strict_reload_key(name)]
    if not loaded_names and strict_bucket_names:
        raise RuntimeError(
            "vLLM model.load_weights loaded zero keys for a non-empty GCore "
            f"weight bucket. first_keys={bucket_names[:8]}"
        )

    strict_missing = [
        name
        for name in strict_bucket_names if loaded_names.isdisjoint(loaded_name_candidates(name))
    ]
    _record_reload_observation(model, bucket_names, loaded_names, strict_missing)
    if strict_missing:
        raise RuntimeError(
            "vLLM model.load_weights did not acknowledge strict GCore reload "
            f"keys. missing_count={len(strict_missing)} first_missing={strict_missing[:16]} "
            f"loaded_sample={sorted(loaded_names)[:16]}"
        )

    return loaded_names


_VALUE_VERIFIED_WEIGHT_PATTERNS = (
    ".ffn.shared_experts.w2.",  # trainer export key (w2=down_proj)
    ".ffn.shared_experts.down_proj.",  # vLLM/internal key alias
    ".attn.wo_b.",  # trainer export key (no mla_attn prefix)
)


def _is_value_verified_weight_key(name: str) -> bool:
    return name.endswith(".weight") and any(
        pattern in name for pattern in _VALUE_VERIFIED_WEIGHT_PATTERNS
    )


def _expected_tp_shard(param: nn.Parameter, loaded_weight: torch.Tensor) -> torch.Tensor:
    if param.data.shape == loaded_weight.shape:
        return loaded_weight

    input_dim = getattr(param, "input_dim", getattr(param, "_input_dim", None))
    if input_dim is None:
        raise RuntimeError(
            "GCore DSV4 reload verifier needs input_dim for row-parallel "
            f"param_shape={tuple(param.shape)} loaded_shape={tuple(loaded_weight.shape)}"
        )
    dim = _normalize_dim(int(input_dim), loaded_weight.ndim)
    shard_size = param.data.shape[dim]
    tp_rank = int(getattr(param, "tp_rank", 0))
    start = tp_rank * shard_size
    expected = loaded_weight.narrow(dim, start, shard_size)
    if expected.shape != param.data.shape:
        raise RuntimeError(
            "GCore DSV4 reload verifier shape mismatch: "
            f"param_shape={tuple(param.shape)} expected_shape={tuple(expected.shape)} "
            f"loaded_shape={tuple(loaded_weight.shape)} input_dim={dim} tp_rank={tp_rank}"
        )
    return expected


def _verify_value_verified_weights_loaded(
    model: nn.Module,
    bucket: list[tuple[str, torch.Tensor]],
    loaded: set[str],
) -> None:
    for name, tensor in bucket:
        if not _is_value_verified_weight_key(name):
            continue
        vllm_name = map_checkpoint_key_to_vllm_param_name(model, name)

        param = _resolve_param_via_module_tree(model, vllm_name)
        if param is None:
            raise RuntimeError(
                "GCore DSV4 reload verifier cannot resolve active module param: "
                f"input={name!r} mapped={vllm_name!r}"
            )

        expected = _expected_tp_shard(param, tensor).to(
            device=param.data.device,
            dtype=param.data.dtype,
        )
        if torch.equal(param.data, expected):
            continue

        actual_f32 = param.data.float()
        expected_f32 = expected.float()
        raise RuntimeError(
            "GCore DSV4 reload verifier failed: active param does not match "
            "the tensor loaded from this bucket. "
            f"input={name!r} mapped={vllm_name!r} param_ptr={param.data.data_ptr()} "
            f"param_shape={tuple(param.shape)} loaded_shape={tuple(tensor.shape)} "
            f"param_dtype={param.dtype} loaded_dtype={tensor.dtype} "
            f"actual_min={actual_f32.min().item():.6f} "
            f"actual_max={actual_f32.max().item():.6f} "
            f"expected_min={expected_f32.min().item():.6f} "
            f"expected_max={expected_f32.max().item():.6f} "
            f"max_abs_diff={(actual_f32 - expected_f32).abs().max().item():.6f}"
        )


def load_checkpoint_weights_for_update(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load one checkpoint-format bucket without premature MegaMoE finalize."""
    bucket = list(weights)
    bucket_names = [name for name, _ in bucket]
    cache_deepseek_v4_dense_fp8_scales(model, bucket)

    load_fn = model.load_weights
    sig = inspect.signature(load_fn)

    with param_subclass_load_context(model):
        more_args = {}
        if "defer_mega_moe_finalize" in sig.parameters:
            more_args["defer_mega_moe_finalize"] = True
        loaded = load_fn(bucket, **more_args)
        _verify_value_verified_weights_loaded(model, bucket, loaded)
        reload_cached_deepseek_v4_dense_fp8_scales(model)

    loaded_names = validate_loaded_weights(model, bucket_names, loaded)
    return loaded_names


def is_deepseek_v4_model(model: nn.Module, model_runner) -> bool:
    hf_config = model_runner.vllm_config.model_config.hf_config
    runner_model_type = getattr(hf_config, "model_type", None)
    if runner_model_type is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            runner_model_type = getattr(text_config, "model_type", None)
    runner_model_type = str(runner_model_type) if runner_model_type is not None else None
    model_model_type = _model_type(model)
    return (
        runner_model_type == "deepseek_v4" or model_model_type == "deepseek_v4" or
        type(model).__name__ == "DeepseekV4ForCausalLM"
    )


def _is_moe_module_for_reload(module: nn.Module | None) -> bool:
    if module is None:
        return False
    if type(module).__name__ == _MEGA_MOE_CLS:
        return True
    try:
        from vllm.model_executor.layers.fused_moe import FusedMoE
    except ImportError:
        return False
    return isinstance(module, FusedMoE)


def _is_vllm_linear_module(module: nn.Module | None) -> bool:
    if module is None:
        return False
    try:
        from vllm.model_executor.layers.linear import LinearBase
    except ImportError:
        return False
    return isinstance(module, LinearBase)


def _module_weight_dtype(module: nn.Module | None) -> torch.dtype | None:
    if module is None:
        return None
    for attr in ("weight", "w13_weight", "w2_weight"):
        weight = getattr(module, attr, None)
        if weight is not None and hasattr(weight, "dtype"):
            return weight.dtype
    return None


def candidate_param_names_for_reflection(model: nn.Module, name: str) -> list[str]:
    raw_name = name.removeprefix("model.")
    candidates: list[str] = [raw_name, map_checkpoint_key_to_vllm_param_name(model, raw_name)]
    reflection_alias = apply_shared_expert_alias(raw_name, include_gate_up=True)
    if reflection_alias != raw_name:
        candidates.append(reflection_alias)
        candidates.append(map_checkpoint_key_to_vllm_param_name(model, reflection_alias))

    for renamed in base_reload_name_variants(raw_name):
        if renamed == raw_name:
            continue
        candidates.append(renamed)
        candidates.append(map_checkpoint_key_to_vllm_param_name(model, renamed))

    deduped_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        append_unique_name(deduped_candidates, seen, candidate)
    return with_model_prefix_variants(deduped_candidates)


def module_from_param_name(model: nn.Module, param_name: str) -> nn.Module | None:
    parts = param_name.split(".")[:-1]
    current: object = model
    for part in parts:
        if _is_moe_module_for_reload(current):
            return current
        try:
            if isinstance(current, nn.ModuleList):
                current = current[int(part)]
            else:
                current = getattr(current, part)
        except (AttributeError, IndexError, ValueError):
            return None
    return current if isinstance(current, nn.Module) else None


def get_module_from_param_name(model: nn.Module, name: str) -> nn.Module | None:
    if hasattr(model, "runnable") and "ACLGraphWrapper" in type(model).__name__:
        model = model.runnable
    for candidate in candidate_param_names_for_reflection(model, name):
        module = module_from_param_name(model, candidate)
        if module is not None:
            return module
    return None


def iter_model_aware_quantized_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    expert_dtype: str,
) -> list[tuple[str, torch.Tensor]]:
    """Quantize DSV4 bf16 disk/HF keys by reflecting the live vLLM model."""
    from gpatch_v4.models.deepseek_v4.fp_quantize import (
        quant_fp4_e2m1_scale_e8m0_packed,
        quant_fp8_e4m3_scale_e8m0,
    )

    quantized: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        if name.endswith(".scale"):
            raise RuntimeError(
                f"DSV4 worker-side quantization expects bf16 weight keys, got scale key {name!r}"
            )

        # 先匹配 expert 权重，走自己的量化逻辑
        if DSV4_EXPERT_WEIGHT_RE.match(name) is not None:
            assert name.endswith(".weight"), name
            scale_name = name[:-len(".weight")] + ".scale"
            if expert_dtype == "fp4":
                packed, scale = quant_fp4_e2m1_scale_e8m0_packed(tensor)
                quantized.append((name, packed.contiguous()))
                quantized.append((scale_name, scale.contiguous()))
            else:
                qfp8, scale = quant_fp8_e4m3_scale_e8m0(tensor)
                quantized.append((name, qfp8.contiguous()))
                quantized.append((scale_name, scale.contiguous()))
            continue

        # 其他比如 linear 权重根据 vllm 实际权重类型决定要不要量化
        module = get_module_from_param_name(model, name)
        if (
            name.endswith(".weight") and _is_vllm_linear_module(module) and
            _module_weight_dtype(module) == torch.float8_e4m3fn
        ):
            scale_name = name[:-len(".weight")] + ".scale"
            qfp8, scale = quant_fp8_e4m3_scale_e8m0(tensor)
            quantized.append((name, qfp8.contiguous()))
            quantized.append((scale_name, scale.contiguous()))
            continue

        quantized.append((name, tensor.contiguous()))

    return quantized


def _validate_fp8_attention_modules_after_reload(model: nn.Module) -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        return

    errors: list[str] = []
    checked: list[str] = []
    for name, module in model.named_modules():
        if "wq_b" not in name:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype != torch.float8_e4m3fn:
            continue

        scale = getattr(module, "weight_scale_inv", None)
        if scale is None:
            scale = getattr(module, "weight_scale", None)
        if scale is None:
            errors.append(f"{name}: missing weight_scale_inv/weight_scale")
            continue
        if weight.device != scale.device:
            errors.append(f"{name}: weight device {weight.device} != scale device {scale.device}")
        if weight.dim() < 2 or scale.dim() < 2:
            errors.append(
                f"{name}: invalid dims weight_shape={tuple(weight.shape)} "
                f"scale_shape={tuple(scale.shape)}"
            )
            continue

        rows, cols = weight.shape[-2:]
        scale_rows, scale_cols = scale.shape[-2:]
        if scale_rows == 0 or scale_cols == 0 or rows % scale_rows or cols % scale_cols:
            errors.append(
                f"{name}: incompatible block scale grid "
                f"weight_shape={tuple(weight.shape)} scale_shape={tuple(scale.shape)}"
            )
        checked.append(
            f"{name}: weight_shape={tuple(weight.shape)} weight_dtype={weight.dtype} "
            f"weight_contiguous={weight.is_contiguous()} scale_shape={tuple(scale.shape)} "
            f"scale_dtype={scale.dtype} scale_contiguous={scale.is_contiguous()}"
        )

    if errors:
        raise RuntimeError(
            "GCore vLLM FP8 attention reload validation failed: " + "; ".join(errors[:16])
        )


def _finalize_mega_moe_weights(model: nn.Module) -> None:
    inner = getattr(model, "model", None)
    if inner is None:
        return
    finalize = getattr(inner, "finalize_mega_moe_weights", None)
    if callable(finalize):
        finalize()


def prepare_weights_for_reload(model: nn.Module, device: torch.device) -> dict:
    with torch.device(device):
        restore_quantized_moe_params_for_loading(model)
        patch_vllm_moe_model_weight_loader(model)
        _ensure_model_params_reloadable(model)
        reload_state = {"is_mxfp4_moe": _detect_mxfp4_moe_present(model)}
        setattr(model, GCORE_RELOAD_STATE_ATTR, reload_state)
        torch.cuda.empty_cache()
        return reload_state


def load_quanted_weights_for_update_dsv4(
    model: nn.Module,
    model_runner,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Quantize one DSV4 bf16 bucket on the worker, then load it."""
    hf_config = model_runner.vllm_config.model_config.hf_config
    expert_dtype = getattr(hf_config, "expert_dtype", None)
    assert expert_dtype in ("fp4", "fp8"), (
        f"DeepSeek-V4 vLLM reload requires expert_dtype='fp4' or 'fp8', got {expert_dtype!r}"
    )
    quantized = iter_model_aware_quantized_weights(
        model,
        weights,
        expert_dtype=expert_dtype,
    )
    return load_checkpoint_weights_for_update(model, quantized)


def finalize_weights_after_reload(
    model: nn.Module,
    model_config: ModelConfig,
    device: torch.device,
) -> None:
    del model_config  # kept for RPC signature compatibility

    reload_state = getattr(model, GCORE_RELOAD_STATE_ATTR, None) or {}

    with torch.device(device):
        _finalize_mega_moe_weights(model)
        if reload_state.get("is_mxfp4_moe"):
            _process_mxfp4_moe_weights_after_loading(model)
        reload_cached_deepseek_v4_dense_fp8_scales(model)
        _validate_fp8_attention_modules_after_reload(model)

    if hasattr(model, GCORE_RELOAD_STATE_ATTR):
        delattr(model, GCORE_RELOAD_STATE_ATTR)
    if hasattr(model, GCORE_DENSE_FP8_SCALE_CACHE_ATTR):
        delattr(model, GCORE_DENSE_FP8_SCALE_CACHE_ATTR)


def save_moe_for_sleep(model: nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    """Save all model tensors not tracked by CuMemAllocator to CPU.

    After weight update, ``finalize_weights`` /
    ``process_weights_after_loading`` may allocate new tensor storage
    outside the CuMem pool -- for MegaMoE ``_transformed_*_weights``
    created by DeepGEMM and for FusedMoE expert params reshuffled by
    FP8 kernel-format conversion.  We walk the entire model, check every
    CUDA tensor's ``data_ptr()`` against
    ``CuMemAllocator.pointer_to_data``, copy the untracked tensors to
    CPU, and release the GPU storage.

    Returns
    -------
    dict[str, torch.Tensor]
        Stash of saved CPU tensors, keyed by ``p:{param_name}`` or
        ``a:{id(module)}:{attr_name}``.
    """
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.get_instance()
    cumem_ptrs: set[int] = set(allocator.pointer_to_data.keys())

    stash: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if not param.is_cuda:
            continue
        if param.data_ptr() in cumem_ptrs:
            continue
        cpu = param.data.cpu()
        param.data = torch.empty(0, device=device)
        key = f"p:{name}"
        stash[key] = cpu

    for module in model.modules():
        for attr_name in list(module.__dict__.keys()):
            if not attr_name.startswith("_"):
                continue
            tensor = getattr(module, attr_name, None)
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                continue
            if tensor.data_ptr() in cumem_ptrs:
                continue
            cpu = tensor.cpu()
            setattr(module, attr_name, None)
            key = f"a:{id(module)}:{attr_name}"
            stash[key] = cpu

    torch.cuda.empty_cache()
    return stash


def restore_moe_after_wakeup(
    model: nn.Module,
    device: torch.device,
    stash: dict[str, torch.Tensor],
) -> None:
    """Restore model tensors previously saved by :func:`save_moe_for_sleep`."""
    if not stash:
        return

    for name, param in model.named_parameters():
        key = f"p:{name}"
        cpu = stash.pop(key, None)
        if cpu is None:
            continue
        param.data = cpu.to(device=device, non_blocking=False)

    for module in model.modules():
        for attr_name, _ in list(module.__dict__.items()):
            key = f"a:{id(module)}:{attr_name}"
            cpu = stash.pop(key, None)
            if cpu is None:
                continue
            setattr(module, attr_name, cpu.to(device=device, non_blocking=False))
