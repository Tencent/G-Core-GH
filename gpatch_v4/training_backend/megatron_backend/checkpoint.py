import contextlib
import copy
import gc
import inspect
import json
import os
import random
import re
import shutil
import threading
import traceback
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed
import yaml
from packaging.version import Version
from safetensors.torch import load_file, save_file
from torch.distributed.tensor import DTensor

from megatron.core import dist_checkpointing, mpu, package_info, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    HAVE_NVRX,
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

try:
    from mbridge.peft.canonical_lora import (
        LoRALinearSplitFC1UpGate,
        LoRALinearSplitGDNInProj,
        LoRALinearSplitQKV,
        get_gdn_in_proj_split_sizes,
    )
    from mbridge.peft.grouped_expert_adapter import GroupedExpertLinearAdapter
    from mbridge.peft.lora import (
        gather_lora_state_dict,
        infer_hf_target_modules,
        interleave_gdn_qkv_lora_b,
        lora_merged,
        mcore_adapter_name_to_hf,
    )
    from mbridge.peft.lora_layers import (
        LinearAdapter,
        LoRAGroupedLinear,
        LoRALinear,
        LoRATopKRouter,
    )

except ImportError:
    lora_merged = None
    gather_lora_state_dict = None
    infer_hf_target_modules = None
    mcore_adapter_name_to_hf = None
    LinearAdapter = None
    LoRAGroupedLinear = None
    LoRALinear = None
    LoRATopKRouter = None
    LoRALinearSplitFC1UpGate = None
    LoRALinearSplitGDNInProj = None
    LoRALinearSplitQKV = None
    GroupedExpertLinearAdapter = None
    get_gdn_in_proj_split_sizes = None
    interleave_gdn_qkv_lora_b = None

try:
    from megatron.bridge.training.checkpointing import (
        apply_peft_adapter_filter_to_state_dict,
    )
except ImportError:
    apply_peft_adapter_filter_to_state_dict = None

from gpatch_v4.configs.checkpoint_config import CheckpointConfig
from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.training_backend.megatron_backend.mcore_peft import (
    peft_to_run_config_dict,
)
from gpatch_v4.training_backend.megatron_backend.megatron_utils import (
    get_model_config,
    unwrap_model,
)
from gpatch_v4.utils.common_utils import (
    assert_hf_metadata_cache_exists,
    copy_cached_hf_metadata_files,
    log,
    logging_rank0,
    save_args_json,
    sync_cuda_and_get_time,
)


def get_dataloader_save_path(checkpoint_config, iteration, dp_rank):
    save_ckpt_path = checkpoint_config.save_ckpt_path
    if not save_ckpt_path:
        return None, None
    out_dir = os.path.join(save_ckpt_path, f"dataloader/iter_{iteration:07d}")
    state_path = os.path.join(out_dir, f"dp_rank_{dp_rank:03d}.pt")
    return out_dir, state_path


def get_dynamic_sampling_save_path(ckpt_path, iteration, dp_rank):
    if not ckpt_path:
        return None, None
    out_dir = os.path.join(ckpt_path, f"dynamic_sampling/iter_{iteration:07d}")
    state_path = os.path.join(out_dir, f"dp_rank_{dp_rank:03d}.pt")
    return out_dir, state_path


@contextlib.contextmanager
def _lora_structure_unwrapped(models):
    """Context manager that temporarily swaps LoRA wrappers with their inner
    ``to_wrap`` modules so that ``named_parameters()`` yields clean names
    (without ``.to_wrap.``), but does NOT merge LoRA deltas into base weights.

    Use this when saving unmerged base weights via the bridge, which only
    recognizes the original (unwrapped) parameter names.
    """
    _ADAPTER_WRAPPER_TYPES = (
        LoRALinear,
        LoRAGroupedLinear,
        LoRALinearSplitQKV,
        LoRALinearSplitFC1UpGate,
        LoRALinearSplitGDNInProj,
        LoRATopKRouter,
    )

    module_swaps = []
    linear_adapter_backups = []

    for model_chunk in models:
        all_modules = dict(model_chunk.named_modules())
        for name, module in all_modules.items():
            if isinstance(module, _ADAPTER_WRAPPER_TYPES):
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = all_modules[parent_name]
                else:
                    parent = model_chunk
                    attr_name = parts[0]
                setattr(parent, attr_name, module.to_wrap)
                module_swaps.append((parent, attr_name, module))
            elif isinstance(module, LinearAdapter):
                saved_in = module._modules.pop('linear_in')
                saved_out = module._modules.pop('linear_out')
                linear_adapter_backups.append((module, saved_in, saved_out))

    try:
        yield
    finally:
        for parent, attr_name, lora_module in module_swaps:
            setattr(parent, attr_name, lora_module)
        for la_module, saved_in, saved_out in linear_adapter_backups:
            la_module._modules['linear_in'] = saved_in
            la_module._modules['linear_out'] = saved_out


def _remap_peft_sharded_keys(loaded_state_dict, model):
    """Remap keys from sharded_state_dict convention to model.state_dict() convention.

    mbridge's AdapterWrapper.sharded_state_dict() flattens ``to_wrap`` keys
    (e.g., ``linear_proj.weight`` instead of ``linear_proj.to_wrap.weight``),
    but ``model.state_dict()`` uses the full hierarchy.  Build a mapping from
    the sharded naming to the model naming so that ``load_state_dict`` works.
    """
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(loaded_state_dict.keys())
    if loaded_keys == model_keys:
        return loaded_state_dict

    remap = {}
    for mk in model_keys:
        if mk in loaded_keys:
            continue
        candidate = mk.replace(".to_wrap.", ".")
        if candidate in loaded_keys:
            remap[candidate] = mk

    if not remap:
        return loaded_state_dict

    remapped = {}
    for k, v in loaded_state_dict.items():
        remapped[remap.get(k, k)] = v
    return remapped


def _save_peft_run_config(dist_checkpoint_path: str, peft) -> None:
    """Write ``peft_config.yaml`` with PEFT metadata for adapter export."""
    run_cfg = {"peft": peft_to_run_config_dict(peft)}
    os.makedirs(dist_checkpoint_path, exist_ok=True)
    with open(os.path.join(dist_checkpoint_path, "peft_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)


def _get_hf_save_path(checkpoint_config, iteration):
    if checkpoint_config.export_hf_save_path is not None:
        export_dir = Path(checkpoint_config.export_hf_save_path) / f"{iteration}"
    else:
        export_dir = Path(checkpoint_config.save_ckpt_path) / f"hf/{iteration}"
    return export_dir


def _mbridge_save_hf(config, iteration, model, bridge, peft):
    cpu_barrier()
    logging_rank0(f"begin to mbridge_save_hf")
    checkpoint_config = config.checkpoint
    start_eport_hf = sync_cuda_and_get_time()

    export_dir = _get_hf_save_path(checkpoint_config, iteration)

    if os.path.exists(export_dir):
        logging_rank0(f"WARNING: may override the huggingface checkpoint at {export_dir}")
    else:
        os.makedirs(export_dir, exist_ok=True)
    if torch.distributed.get_rank() == 0:
        assert_hf_metadata_cache_exists(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
        )
    cpu_barrier()
    unwrapped_model = unwrap_model(model)

    assert hasattr(bridge, "safetensor_io") and bridge.safetensor_io is not None, (
        "bridge.safetensor_io must be initialized when building from mbridge"
    )

    # 兼容不同版本的 mbridge
    save_func_sig = inspect.signature(bridge.save_weights)
    save_weights_kwargs = {}
    if "distributed_filesystem" in save_func_sig.parameters:
        save_weights_kwargs["distributed_filesystem"
                           ] = checkpoint_config.mbridge_distributed_filesystem
    if "strict" in save_func_sig.parameters:
        save_weights_kwargs["strict"] = checkpoint_config.strict_export

    # Save base weights to export_dir.
    # When LoRA is active, unwrap the module structure (without merging deltas)
    # so that the bridge sees clean parameter names.
    save_ctx = (_lora_structure_unwrapped(unwrapped_model) if peft is not None else nullcontext())
    with save_ctx:
        bridge.save_weights(
            unwrapped_model,
            export_dir,
            memory_efficient=True,
            **save_weights_kwargs,
        )

    cpu_barrier()
    if torch.distributed.get_rank() == 0:
        copy_cached_hf_metadata_files(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
            export_dir,
        )
    save_args_json(config, export_dir)

    if peft is not None:
        # Save adapter-only weights (HF PEFT format) for resume and export
        adapter_export_dir = _get_hf_save_path(checkpoint_config, f"{iteration}_adapter")
        _mbridge_save_adapter_only(config, model, adapter_export_dir, bridge=bridge)

        # Optionally save merged (base+adapter) weights for deployment
        if checkpoint_config.save_merged_lora_weights:
            assert lora_merged is not None, "lora_merged is not available"
            merge_export_dir = _get_hf_save_path(checkpoint_config, f"{iteration}_merge")
            os.makedirs(merge_export_dir, exist_ok=True)
            with lora_merged(unwrapped_model):
                bridge.save_weights(
                    unwrapped_model,
                    merge_export_dir,
                    memory_efficient=True,
                    **save_weights_kwargs,
                )
            cpu_barrier()
            if torch.distributed.get_rank() == 0:
                copy_cached_hf_metadata_files(
                    config.policy.hf_model_path,
                    checkpoint_config.save_ckpt_path,
                    merge_export_dir,
                )
            save_args_json(config, merge_export_dir)
            logging_rank0(f"Saved merged LoRA weights to {merge_export_dir}")

    cpu_barrier()
    end_eport_hf = sync_cuda_and_get_time()
    logging_rank0(
        f"Finish to mbridge_save_hf {export_dir}, time cost:{end_eport_hf - start_eport_hf}"
    )
    return export_dir


def _megatron_bridge_save_hf(config, iteration, model, bridge, peft):
    cpu_barrier()
    logging_rank0(f"begin to megatron_bridge_save_hf")
    checkpoint_config = config.checkpoint
    start_eport_hf = sync_cuda_and_get_time()

    export_dir = _get_hf_save_path(checkpoint_config, iteration)

    if os.path.exists(export_dir):
        logging_rank0(f"WARNING: may override the huggingface checkpoint at {export_dir}")
    else:
        os.makedirs(export_dir, exist_ok=True)
    if torch.distributed.get_rank() == 0:
        assert_hf_metadata_cache_exists(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
        )
    cpu_barrier()

    save_func_sig = inspect.signature(bridge.save_hf_weights)
    extra_kwargs = {}
    if "distributed_save" in save_func_sig.parameters:
        extra_kwargs["distributed_save"] = checkpoint_config.mbridge_distributed_filesystem
        extra_kwargs["save_every_n_ranks"] = checkpoint_config.mbridge_save_every_n_ranks
    if "strict" in save_func_sig.parameters:
        extra_kwargs["strict"] = checkpoint_config.strict_export
    bridge.save_hf_weights(model, export_dir, **extra_kwargs)
    cpu_barrier()
    if torch.distributed.get_rank() == 0:
        copy_cached_hf_metadata_files(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
            export_dir,
        )

    save_args_json(config, export_dir)
    cpu_barrier()
    end_eport_hf = sync_cuda_and_get_time()
    if peft is not None:
        adapter_export_dir = _get_hf_save_path(checkpoint_config, f"{iteration}_adapter")
        _megatron_bridge_save_hf_adapter(config, model, bridge, peft, adapter_export_dir)
    logging_rank0(
        f"Finish to megatron_bridge_save_hf {export_dir}, time cost:{end_eport_hf - start_eport_hf}"
    )
    return export_dir


def _update_tokenizer_configs(
    ckpt_dir: str, override_tokenizer_special_token: dict, debug_dry_run: bool = False
) -> dict:
    """Update config.json and tokenizer_config.json in `ckpt_dir`.

    Args:
        ckpt_dir:
        override_tokenizer_special_token: e.g. ``{"eos_token": [id, str]}``.
        debug_dry_run: If True, print changes without writing.

    Returns:
        Dict of changes applied.
    """
    changes = {"config.json": {}, "tokenizer_config.json": {}}

    def load_json_file(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config

    for k, v in override_tokenizer_special_token.items():
        assert len(v) == 2, f"len(v) must be 2, but got {len(v)} items {k=} {v=}"

    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        config = load_json_file(config_path)

        for k, v in override_tokenizer_special_token.items():
            _k = f"{k}_id"
            if _k in config.keys():
                original_token_id = config.get(_k)
                if original_token_id == v[0]:
                    continue

                config[_k] = v[0]
                changes["config.json"][_k] = {"old": original_token_id, "new": v[0]}

        if not debug_dry_run:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            log(f"update_tokenizer_configs {config_path}", rank=0)
        else:
            log(f"[DRY RUN] Would update {config_path}", rank=0)
    else:
        log(f"warning {config_path} not found", rank=0)

    tokenizer_config_path = os.path.join(ckpt_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        tokenizer_config = load_json_file(tokenizer_config_path)
        target_tokens = ["eos_token"]

        for target_token in target_tokens:
            if target_token in override_tokenizer_special_token.keys():
                original_token = tokenizer_config.get(target_token)
                new_token = override_tokenizer_special_token[target_token][1]
                if new_token == original_token:
                    continue

                tokenizer_config[target_token] = new_token
                changes["tokenizer_config.json"][target_token] = {
                    "old": original_token,
                    "new": new_token
                }

        if not debug_dry_run:
            with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
            log(f"update_tokenizer_configs {tokenizer_config_path}", rank=0)
        else:
            log(f"[DRY RUN] Would update {tokenizer_config_path}", rank=0)
    else:
        log(f"warning {tokenizer_config_path} not found", rank=0)

    log(f"total update changes {changes}", rank=0)
    return changes


def _mbridge_save_adapter_only(config, model, output_dir, bridge=None):
    """Export LoRA adapter weights in HF PEFT format.

    Produces ``adapter_config.json`` + ``adapter_model.safetensors`` in
    *output_dir*.  Uses ``gather_lora_state_dict`` which handles TP > 1.
    Also saves extra trainable state (e.g. out_norm) that would lose
    precision through bridge's format conversion.
    """
    cpu_barrier()
    logging_rank0(f"begin to mbridge_save_adapter_only -> {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    unwrapped_model = unwrap_model(model)
    adapter_state = gather_lora_state_dict(unwrapped_model, bridge=bridge)

    if torch.distributed.get_rank() == 0:
        save_file(adapter_state, os.path.join(output_dir, "adapter_model.safetensors"))

        lora_cfg = config.policy.lora
        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": config.policy.hf_model_path,
            "r": lora_cfg.rank,
            "lora_alpha": lora_cfg.alpha,
            "lora_dropout": lora_cfg.dropout,
            "target_modules": infer_hf_target_modules(adapter_state),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)

    cpu_barrier()
    logging_rank0(f"Finish mbridge_save_adapter_only {output_dir}")


def _scatter_weight_to_local_shard(full_weight, module, device):
    """Scatter a full (un-sharded) weight tensor to the local TP shard.

    Reverse of ``_gather_parallel_weight`` in mbridge:
    - ColumnParallelLinear: split along dim 0, take local rank's shard.
    - RowParallelLinear: split along dim 1, take local rank's shard.
    - Regular nn.Linear or expert-parallel modules (ETP != TP): no split.

    Compares shapes to avoid splitting modules that use a different
    parallel group (e.g. expert_tensor_parallel with ETP=1).
    """
    local_shape = module.weight.shape
    if full_weight.shape == local_shape:
        return full_weight.to(device)

    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return full_weight.to(device)

    tp_rank = mpu.get_tensor_model_parallel_rank()
    if isinstance(module, RowParallelLinear):
        chunks = full_weight.chunk(tp_size, dim=1)
    elif isinstance(module, ColumnParallelLinear):
        chunks = full_weight.chunk(tp_size, dim=0)
    else:
        return full_weight.to(device)
    return chunks[tp_rank].contiguous().to(device)


def _interleave_lora_b_for_scatter(full_b: torch.Tensor, stride: int, tp_size: int) -> torch.Tensor:
    """Re-interleave a sequential HF LoRA-B weight back for TP scatter.

    Inverse of ``_deinterleave_gathered_lora_b``:
    Input (sequential):  [gate_all, up_all]
    Output (interleaved per-rank): [rank0_gate, rank0_up, rank1_gate, rank1_up, ...]
    """
    if stride <= 1 or tp_size <= 1:
        return full_b

    total_rows = full_b.shape[0]
    per_stride = total_rows // stride
    per_stride_per_rank = per_stride // tp_size

    parts = []
    for r in range(tp_size):
        for s in range(stride):
            start = s * per_stride + r * per_stride_per_rank
            end = start + per_stride_per_rank
            parts.append(full_b[start:end])
    return torch.cat(parts, dim=0)


def _load_adapter_from_hf_peft(models, adapter_dir, bridge=None):
    """Load adapter weights from HF PEFT format into model's adapter modules.

    This is the inverse of ``_mbridge_save_adapter_only`` /
    ``gather_lora_state_dict``. It loads ``adapter_model.safetensors``,
    maps HF keys back to mcore module names, scatters full tensors to
    local TP shards, and assigns to ``linear_in.weight`` / ``linear_out.weight``.

    Parameters
    ----------
    models : list[nn.Module]
        Unwrapped model chunks.
    adapter_dir : str
        Directory containing ``adapter_model.safetensors``.
    bridge : optional
        mbridge Bridge instance for HF↔mcore name mapping.
    """
    adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter file not found for resume: {adapter_path}")

    adapter_state = load_file(adapter_path, device="cpu")
    log(f"Loaded adapter state dict from {adapter_path} ({len(adapter_state)} keys)", rank=0)

    # Unwrap DDP/FSDP wrappers so named_modules() yields the same names
    # as during save (which uses unwrap_model before gather_lora_state_dict).
    unwrapped_models = unwrap_model(models)

    # Build reverse mapping: hf_key -> (adapter_module, part, parent, expert_local_idx|None)
    # expert_local_idx is set for packed GroupedExpertLinearAdapter / legacy LoRAGroupedLinear
    # per-expert tensors; None for a single 2D adapter weight.
    hf_to_mcore = {}
    for model_chunk in unwrapped_models:
        for name, module in model_chunk.named_modules():
            if LoRALinearSplitQKV is not None and isinstance(
                module,
                (
                    LoRALinearSplitQKV,
                    LoRALinearSplitFC1UpGate,
                    LoRALinearSplitGDNInProj,
                ),
            ):
                for sub_name, sub_adapter in module.adapter.items():
                    if sub_adapter is None:
                        continue
                    for part in ("linear_in", "linear_out"):
                        mcore_key = f"{name}.adapter.{sub_name}.{part}.weight"
                        hf_key = mcore_adapter_name_to_hf(mcore_key, bridge=bridge)
                        hf_to_mcore[hf_key] = (sub_adapter, part, module, None)

            elif LoRATopKRouter is not None and isinstance(module, LoRATopKRouter):
                adapter = module.adapter
                for part in ("linear_in", "linear_out"):
                    mcore_key = f"{name}.adapter.{part}.weight"
                    hf_key = mcore_adapter_name_to_hf(mcore_key, bridge=bridge)
                    hf_to_mcore[hf_key] = (adapter, part, module, None)

            elif LoRAGroupedLinear is not None and isinstance(module, LoRAGroupedLinear):
                ep_rank = mpu.get_expert_model_parallel_rank()
                num_local_experts = len(module.adapter)
                for i in range(num_local_experts):
                    global_expert_id = ep_rank * num_local_experts + i
                    adapter_i = module.adapter[i]
                    for part in ("linear_in", "linear_out"):
                        mcore_key = f"{name}.adapter.{global_expert_id}.{part}.weight"
                        hf_key = mcore_adapter_name_to_hf(mcore_key, bridge=bridge)
                        hf_to_mcore[hf_key] = (adapter_i, part, module, None)

            elif LinearAdapter is not None and isinstance(module, LinearAdapter):
                for part in ("linear_in", "linear_out"):
                    mcore_key = f"{name}.{part}.weight"
                    hf_key = mcore_adapter_name_to_hf(mcore_key, bridge=bridge)
                    hf_to_mcore[hf_key] = (module, part, module, None)

            elif LoRALinear is not None and isinstance(module, LoRALinear):
                adapter = module.adapter
                # Packed per-expert adapters: HF keys use global expert id (same as gather).
                if GroupedExpertLinearAdapter is not None and isinstance(
                    adapter, GroupedExpertLinearAdapter
                ):
                    ep_rank = mpu.get_expert_model_parallel_rank()
                    num_local_experts = adapter.num_local_experts
                    for i in range(num_local_experts):
                        global_expert_id = ep_rank * num_local_experts + i
                        for part in ("linear_in", "linear_out"):
                            mcore_key = (f"{name}.adapter.{global_expert_id}.{part}.weight")
                            hf_key = mcore_adapter_name_to_hf(mcore_key, bridge=bridge)
                            hf_to_mcore[hf_key] = (adapter, part, module, i)
                else:
                    for part in ("linear_in", "linear_out"):
                        mcore_key = f"{name}.adapter.{part}.weight"
                        hf_key = mcore_adapter_name_to_hf(mcore_key, bridge=bridge)
                        hf_to_mcore[hf_key] = (adapter, part, module, None)

    loaded_count = 0
    for hf_key, full_weight in adapter_state.items():
        if hf_key not in hf_to_mcore:
            log(f"Warning: adapter key {hf_key!r} not matched to any model module", rank=0)
            continue

        adapter_module, part, parent_lora_module, expert_local_idx = hf_to_mcore[hf_key]
        weight_container = getattr(adapter_module, part)
        device = weight_container.weight.device

        # For standard LoRA linear_out on strided layers (SwiGLU), re-interleave
        if (
            expert_local_idx is None and part == "linear_out" and LoRALinear is not None and
            isinstance(parent_lora_module, LoRALinear)
        ):
            stride = getattr(parent_lora_module.to_wrap, 'stride', 1)
            tp_size = mpu.get_tensor_model_parallel_world_size()
            if stride > 1 and tp_size > 1:
                full_weight = _interleave_lora_b_for_scatter(full_weight, stride, tp_size)

        if (
            expert_local_idx is None and part == "linear_out" and
            LoRALinearSplitGDNInProj is not None and
            isinstance(parent_lora_module, LoRALinearSplitGDNInProj) and
            adapter_module is parent_lora_module.adapter.adapter_qkv
        ):
            config = parent_lora_module.to_wrap.config
            qkv_size, _, _, _ = get_gdn_in_proj_split_sizes(config)
            qk_size = config.linear_num_key_heads * config.linear_key_head_dim
            value_size = (config.linear_num_value_heads * config.linear_value_head_dim)
            component_sizes = (qk_size, qk_size, value_size)
            assert qkv_size == sum(component_sizes)
            full_weight = interleave_gdn_qkv_lora_b(
                full_weight,
                component_sizes,
                mpu.get_tensor_model_parallel_world_size(),
            )

        if expert_local_idx is not None:
            # GroupedExpertLinearAdapter: copy into weight[local_expert] slice.
            param = weight_container.weight
            slice_holder = SimpleNamespace(weight=param[expert_local_idx])
            local_weight = _scatter_weight_to_local_shard(full_weight, slice_holder, device)
            param.data[expert_local_idx].copy_(local_weight)
        else:
            local_weight = _scatter_weight_to_local_shard(full_weight, weight_container, device)
            weight_container.weight.data.copy_(local_weight)
        loaded_count += 1

    log(f"Loaded {loaded_count} adapter weight tensors from {adapter_dir}", rank=0)


def _megatron_bridge_save_hf_adapter(config, model, bridge, peft, export_dir):
    cpu_barrier()
    logging_rank0("begin to megatron_bridge_save_hf_adapter")
    checkpoint_config = config.checkpoint
    start_export = sync_cuda_and_get_time()

    if os.path.exists(export_dir):
        logging_rank0(f"WARNING: may override the huggingface adapter at {export_dir}")
    else:
        os.makedirs(export_dir, exist_ok=True)

    bridge.save_hf_adapter(
        model,
        export_dir,
        peft_config=peft,
        base_model_name_or_path=config.policy.hf_model_path,
    )
    cpu_barrier()
    if torch.distributed.get_rank() == 0:
        assert_hf_metadata_cache_exists(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
        )
        copy_cached_hf_metadata_files(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
            export_dir,
        )
    save_args_json(config, export_dir)
    cpu_barrier()
    end_export = sync_cuda_and_get_time()
    logging_rank0(
        f"Finish megatron_bridge_save_hf_adapter {export_dir}, "
        f"time cost:{end_export - start_export}"
    )
    return export_dir


def bridge_save_hf(
    config,
    iteration,
    model,
    bridge,
    override_tokenizer_special_token: dict = None,
    peft=None,
):
    try:
        if config.training.build_from_mbridge:
            save_path = _mbridge_save_hf(config, iteration, model, bridge, peft)
        else:
            save_path = _megatron_bridge_save_hf(config, iteration, model, bridge, peft)

        if torch.distributed.get_rank() == 0:
            if override_tokenizer_special_token is not None:
                assert isinstance(
                    override_tokenizer_special_token, dict
                ), f"{type(override_tokenizer_special_token)}"
                _update_tokenizer_configs(save_path, override_tokenizer_special_token)

    except Exception as e:
        log(f"bridge_save_hf failed: {e}")
        traceback.print_exc()
        raise e


def get_latest_checkpoint_folder(path):
    if path is None:
        return None

    if torch.distributed.get_rank() == 0:
        latest_checkpoint_file = os.path.join(path, "latest_checkpointed_iteration.txt")

        if os.path.exists(latest_checkpoint_file):
            with open(latest_checkpoint_file, "r") as f:
                step = int(f.read().strip())
        else:
            step = -1
    else:
        step = 0
    step_tensor = torch.tensor(step, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(step_tensor, op=torch.distributed.ReduceOp.SUM)
    latest_ckpt_step = step_tensor.item()
    if latest_ckpt_step == -1:
        return None
    return latest_ckpt_step


def get_rng_state(
    use_dist_ckpt: bool = True,
    data_parallel_random_init: bool = False,
    use_megatron_fsdp: bool = False,
):
    """collect rng state across data parallel ranks"""
    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }

    if torch.cuda.is_available():
        rng_state["cuda_rng_state"] = torch.cuda.get_rng_state()

    rng_state_list = None
    if torch.distributed.is_initialized() and mpu.get_data_parallel_world_size(
    ) > 1 and data_parallel_random_init:
        rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            rng_state_list, rng_state, group=mpu.get_data_parallel_group()
        )
    else:
        rng_state_list = [rng_state]

    if use_megatron_fsdp:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        return {f"({pp_rank}, {tp_rank})": rng_state_list}

    if use_dist_ckpt:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        rng_state_list = ShardedObject(
            "rng_state",
            rng_state_list,
            (pp_size, tp_size),
            (pp_rank, tp_rank),
            replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
        )

    return rng_state_list


def load_rng_states(
    rng_states,
    data_parallel_random_init=False,
    use_dist_ckpt=True,
    use_megatron_fsdp: bool = False,
):
    if use_megatron_fsdp:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        key = f"({pp_rank}, {tp_rank})"
        if key in rng_states:
            rng_states = rng_states[key]
        else:
            log(
                f"RNG state for PP/TP key {key} not found; falling back to the first saved RNG state.",
                rank=0,
            )
            rng_states = next(iter(rng_states.values()))

    # access rng_state for data parallel rank
    if data_parallel_random_init:
        rng_states = rng_states[mpu.get_data_parallel_rank()]
    else:
        rng_states = rng_states[0]
    random.setstate(rng_states["random_rng_state"])
    np.random.set_state(rng_states["np_rng_state"])
    torch.set_rng_state(rng_states["torch_rng_state"])

    if torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_states["cuda_rng_state"])

    # Check for empty states array
    if not rng_states["rng_tracker_states"]:
        raise KeyError
    tensor_parallel.get_cuda_rng_tracker().set_states(rng_states["rng_tracker_states"])


def _write_no_fork(transform_list, use_msc, rank, write_buckets, global_results_queue):
    """In-process checkpoint writer, drop-in for
    ``FileSystemWriterAsync.write_preloaded_data_multiproc``.

    Why
    ---
    Megatron 的 ``write_preloaded_data_multiproc`` 通过
    ``mp.get_context("fork")`` 写 checkpoint。POSIX 下多线程进程 fork 是
    UB：子进程只保留调用线程，其他线程持有的锁/资源残留，导致内存分配器、
    CRC 库访问到损坏状态，最终在 ``PyTorchStreamWriter::writeRecord →
    crc32_16bytes`` segfault。Megatron-LM 独立训练 (torchrun) 后台线程少，
    撞概率低；Ray Actor 进程额外有 gRPC / object store / uvloop /
    health-check 等线程，fork 时几乎必然撞锁，因此稳定复现。

    本函数在调用者进程内直接做相同的 I/O，避免 fork。
    """
    from torch.distributed.checkpoint.filesystem import _write_item

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        write_results = {}
        extra_kwargs = {}
        if "serialization_format" in inspect.signature(_write_item).parameters:
            from torch.distributed.checkpoint.filesystem import SerializationFormat
            extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE

        open_fn = open
        if use_msc:
            import multistorageclient as msc
            open_fn = msc.open

        for i, write_bucket in enumerate(write_buckets):
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            local_results = []
            with open_fn(file_name, "wb") as stream:
                for write_item, data in bytes_data:
                    local_results.append(
                        _write_item(
                            *transform_list,
                            stream,
                            data,
                            write_item,
                            storage_key,
                            **extra_kwargs,
                        )
                    )
                for write_item, tensor in tensor_data:
                    assert tensor.is_cpu
                    local_results.append(
                        _write_item(
                            *transform_list,
                            stream,
                            tensor,
                            write_item,
                            storage_key,
                            **extra_kwargs,
                        )
                    )
                if use_msc:
                    stream.fsync()
                else:
                    os.fsync(stream.fileno())
            write_results[i] = local_results

        global_results_queue.put(write_results)
    except Exception as e:
        global_results_queue.put(RuntimeError(f"In-process write failure: {e}"))
    finally:
        if gc_was_enabled:
            gc.enable()


def save_dist_checkpointing(sharded_state_dict, ckpt_path, async_save=False):
    """Save sharded checkpoint via Megatron dist_checkpointing.

    通过 ``async_sharded_save=True`` 取 ``AsyncRequest`` 并把 ``async_fn``
    （默认 fork 子进程的 ``write_preloaded_data_multiproc``）替换为
    ``_write_no_fork``，避免 Ray Actor 多线程 fork segfault。详见
    ``_write_no_fork``。
    """
    validate_sharding_integrity = True
    save_strategy = TorchDistSaveShardedStrategy("torch_dist", 1, thread_count=1)
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    async_request = dist_checkpointing.save(
        sharded_state_dict,
        ckpt_path,
        sharded_strategy=save_strategy,
        async_sharded_save=True,
        validate_access_integrity=validate_sharding_integrity,
        async_strategy="nvrx" if HAVE_NVRX else "mcore",
    )

    if async_request.async_fn is not None:
        orig = async_request.async_fn
        transform_list = orig.args[0] if hasattr(orig, "args") and len(orig.args) >= 1 else []
        use_msc = orig.args[1] if hasattr(orig, "args") and len(orig.args) >= 2 else False
        async_request = async_request._replace(
            async_fn=partial(_write_no_fork, transform_list, use_msc)
        )

    if async_save:
        return async_request

    async_request.execute_sync()
    return None


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    # Get checkpointing strategies
    load_strategy = TorchDistLoadShardedStrategy()
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Fix torch.load weights only error
    try:
        import transformer_engine as te

        torch.serialization.add_safe_globals([torch.optim.AdamW])
        torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])
    except Exception:
        pass

    # Load model sharded state dicts
    state_dict = dist_checkpointing.load(
        sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy
    )

    return state_dict


def _build_sharded_state_dict_metadata(
    dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    use_megatron_fsdp: bool = False,
) -> dict:
    """Builds metadata used for sharded_state_dict versioning.

    The metadata is passed into model/optimizer ``sharded_state_dict``
    methods. Keep it minimal, semantically named (e.g.
    ``distrib_optim_sharding_type``); a plain integer/SemVer version
    flag is discouraged because metadata is shared by all
    models/optimizers and a single linear version can't cover them.

    Args:
        args: Arguments namespace
        dp_cp_group: DP+CP group; falls back to mpu API if None.
    """
    metadata = {}
    if Version(package_info.__version__) <= Version("0.13.1"):
        return metadata

    if use_megatron_fsdp:
        metadata['distrib_optim_sharding_type'] = 'fsdp_dtensor'
    else:
        metadata['distrib_optim_sharding_type'] = 'dp_reshardable'

    metadata['singleton_local_shards'] = False
    metadata['chained_optim_avoid_prefix'] = True
    # Add dp_cp_group to metadata. If not provided, fallback to global parallel state.
    if dp_cp_group is None:
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
    metadata['dp_cp_group'] = dp_cp_group
    return metadata


def generate_state_dict(
    models,
    optimizer,
    lr_scheduler,
    generate_optimizer: bool = True,
    generate_model: bool = True,
    is_loading: bool = False,
    use_megatron_fsdp: bool = False,
):
    torch.cuda.synchronize()
    state_dict = {}

    sharded_sd_metadata = dict(
        metadata=_build_sharded_state_dict_metadata(use_megatron_fsdp=use_megatron_fsdp)
    )

    # Should always generate model state dict
    # All ranks Save Model to reduce memory pressure
    # Get sharded state dict, notice that state_dict will collect among dp groups, causing memory pressure
    # NOTE: optimizer needs model sharded state as sharding metadata even when generate_model=False
    need_model_sharded_state = generate_model or generate_optimizer
    model_keys_to_remove = []

    if need_model_sharded_state:
        for vpp_rank, model in enumerate(models):
            if len(models) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                key = f"model{vpp_rank}" if len(models) > 1 else "model"
            else:
                key = "model"
            if use_megatron_fsdp:
                state_dict[key] = model.state_dict_for_save_checkpoint()
            else:
                unwrapped = model.module if hasattr(model, "module") else model
                state_dict[key] = unwrapped.sharded_state_dict(**sharded_sd_metadata)
            if not generate_model:
                model_keys_to_remove.append(key)

    # Optimizer State Dict
    if generate_optimizer:
        torch.distributed.barrier()
        optimizer_sharded_states = optimizer.sharded_state_dict(
            state_dict, is_loading=is_loading, **sharded_sd_metadata
        )
        # Remove model keys that were only needed as optimizer metadata
        for key in model_keys_to_remove:
            state_dict.pop(key, None)
        state_dict["optimizer"] = optimizer_sharded_states

        if lr_scheduler is not None:
            lr_state_dict = lr_scheduler.state_dict()
            state_dict["lr_scheduler"] = lr_state_dict

        # RNG States State Dict
        torch.distributed.barrier()
        rng_state = get_rng_state(use_megatron_fsdp=use_megatron_fsdp)
        state_dict["rng_state"] = rng_state

    return state_dict


def get_dist_checkpoint_path(checkpoint_path, global_step):
    dist_checkpoint_path = os.path.join(checkpoint_path, 'iter_{:07d}'.format(global_step))
    return dist_checkpoint_path


def _assert_megatron_fsdp_checkpoint_supported(config, models, *, is_save: bool):
    checkpoint_config = config.checkpoint
    assert not config.training.build_from_mbridge, (
        "Megatron-FSDP checkpointing requires training.build_from_mbridge=False."
    )
    assert len(
        models
    ) == 1, "Megatron-FSDP checkpointing with virtual pipeline chunks is not verified yet."
    assert not checkpoint_config.async_save, "Megatron-FSDP DTensor checkpoints do not support async_save yet."
    assert not checkpoint_config.skip_save_mcore_model, (
        "Megatron-FSDP DTensor checkpoints require saving/loading the mcore model state."
    )
    assert not checkpoint_config.convert_mcore_to_hf_online, (
        "Online HF export for Megatron-FSDP checkpoints is not verified yet."
    )
    if is_save:
        assert not checkpoint_config.no_save_optim, (
            "Megatron-FSDP optimizer-only or model-only checkpoint layouts are not verified yet."
        )
    else:
        assert not checkpoint_config.no_load_optim, (
            "Loading Megatron-FSDP checkpoints without optimizer state is not verified yet."
        )


def _make_megatron_bridge_fsdp_checkpoint_config(config, *, is_save: bool):
    checkpoint_config = config.checkpoint
    return SimpleNamespace(
        save=checkpoint_config.save_ckpt_path,
        load=checkpoint_config.load_ckpt_path,
        save_optim=not checkpoint_config.no_save_optim,
        save_rng=True,
        load_optim=not checkpoint_config.no_load_optim,
        load_rng=True,
        ckpt_format="fsdp_dtensor",
        async_save=False,
        strict_fsdp_dtensor_load=False,
        fully_parallel_save=True,
        fully_parallel_load=False,
        finetune=False,
        use_checkpoint_args=False,
        ckpt_step=None,
    )


def resolve_megatron_fsdp_dist_param(model, key: str) -> torch.nn.Parameter:
    candidates = [key, f"module.{key}"]
    if key.startswith("module."):
        candidates.append(key[len("module."):])
    seen = set()
    ordered = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    for name in ordered:
        try:
            return model.get_parameter(name)
        except AttributeError:
            continue

    name_map = dict(model.named_parameters())
    for name in ordered:
        if name in name_map:
            return name_map[name]
    raise KeyError(
        f"Cannot resolve Megatron-FSDP parameter for optimizer state key {key!r}; "
        f"tried {ordered}"
    )


def megatron_fsdp_param_local_tensor(dist_param: torch.nn.Parameter) -> torch.Tensor:
    try:
        from torch.distributed._tensor import DTensor
    except ImportError:
        from torch.distributed.tensor import DTensor

    if isinstance(dist_param, DTensor):
        return dist_param.to_local()
    data = dist_param.data
    if isinstance(data, DTensor):
        return data.to_local()
    if hasattr(dist_param, "_local_tensor"):
        return dist_param._local_tensor
    raise TypeError(
        "Megatron-FSDP parameter is missing a local tensor view "
        f"(type={type(dist_param)!r})"
    )


def plain_tensor_to_fsdp_local_shard(
    tensor: torch.Tensor, dist_param: torch.nn.Parameter
) -> torch.Tensor:
    """Slice a plain Adam moment into this rank's FSDP-local shard.

    TE FusedAdam may allocate moments with global numel for ``nn.Parameter(DTensor)``.
    tip LM SwiGLU/GDN preprocess expects DTensor or plain FSDP-local shards.
    """
    if not hasattr(dist_param, "megatron_fsdp_slice"):
        raise AttributeError(
            "Megatron-FSDP parameter is missing megatron_fsdp_slice; "
            "cannot normalize plain optimizer tensor."
        )
    fsdp_slice = dist_param.megatron_fsdp_slice
    local_expected = fsdp_slice.stop - fsdp_slice.start
    flat = tensor.detach().view(-1)
    if flat.numel() == dist_param.numel():
        local = flat[fsdp_slice].contiguous()
    elif flat.numel() == local_expected:
        local = flat.contiguous()
    else:
        raise ValueError(
            "Plain optimizer tensor numel does not match FSDP global or local shard: "
            f"got {flat.numel()}, global={dist_param.numel()}, "
            f"local_expected={local_expected}, fsdp_slice={fsdp_slice}"
        )
    local_ref = megatron_fsdp_param_local_tensor(dist_param)
    if local.numel() != local_ref.numel():
        raise ValueError(
            "FSDP-local optimizer shard numel mismatch after slice: "
            f"got {local.numel()}, param_local={local_ref.numel()}, "
            f"fsdp_slice={fsdp_slice}"
        )
    return local.reshape(local_ref.shape)


def normalize_megatron_fsdp_optimizer_state_tensors(model, optimizer_state_dict: dict) -> dict:
    """Rewrite plain full/local Adam tensors into Megatron-FSDP DTensors for DCP save."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
        make_fsdp_dtensor,
    )

    if "state" not in optimizer_state_dict:
        raise KeyError(
            "Megatron-FSDP optimizer state dict missing 'state' "
            f"(keys={list(optimizer_state_dict.keys())})"
        )

    # Weight-sized optimizer tensors only; skip scalars like step.
    weight_state_keys = frozenset({"exp_avg", "exp_avg_sq", "master_param", "param", "fp32_param"})

    opt_state = optimizer_state_dict["state"]
    for key, param_state in opt_state.items():
        if not isinstance(param_state, dict):
            continue
        dist_param = resolve_megatron_fsdp_dist_param(model, key)
        if not hasattr(dist_param, "megatron_fsdp_dist_index"):
            raise AttributeError(
                f"Parameter for optimizer key {key!r} is missing megatron_fsdp_dist_index"
            )
        is_expert_param = "mlp.experts" in key
        for subkey, value in list(param_state.items()):
            if subkey not in weight_state_keys:
                continue
            if not isinstance(value, torch.Tensor) or isinstance(value, DTensor):
                continue
            local = plain_tensor_to_fsdp_local_shard(value, dist_param)
            param_state[subkey] = make_fsdp_dtensor(
                local_tensor=local,
                param=dist_param,
                dist_index=dist_param.megatron_fsdp_dist_index,
                is_sharded_param=True,
                is_expert_param=is_expert_param,
                run_check=True,
                update_uneven_dtensor_chunk_meta=True,
            )
    return optimizer_state_dict


def save_megatron_fsdp_checkpoint(config, models, optimizer, lr_scheduler, dist_checkpoint_path):
    import torch.distributed.checkpoint as torch_dist_checkpoint
    from torch.distributed.checkpoint import FileSystemWriter

    from megatron.bridge.training.checkpointing import (
        preprocess_fsdp_dtensor_state_dict,
    )

    state_dict = generate_state_dict(
        models,
        optimizer,
        lr_scheduler,
        generate_optimizer=True,
        generate_model=True,
        use_megatron_fsdp=True,
    )
    checkpoint_model = getattr(models[0], "module", models[0])
    if "optimizer" in state_dict:
        state_dict["optimizer"] = normalize_megatron_fsdp_optimizer_state_tensors(
            checkpoint_model, state_dict["optimizer"]
        )
    state_dict = preprocess_fsdp_dtensor_state_dict(
        get_model_config(checkpoint_model),
        state_dict,
        checkpoint_model,
    )
    storage_writer = FileSystemWriter(dist_checkpoint_path)
    torch_dist_checkpoint.save(state_dict=state_dict, storage_writer=storage_writer)
    torch.distributed.barrier()
    return None


def load_megatron_fsdp_checkpoint(
    config,
    models,
    optimizer,
    lr_scheduler,
    global_step: int,
    bridge=None,
):
    from megatron.bridge.training.checkpointing import load_fsdp_dtensor_checkpoint

    checkpoint_config = config.checkpoint
    sharded_state_dict = generate_state_dict(
        models,
        optimizer,
        lr_scheduler,
        generate_optimizer=True,
        generate_model=True,
        is_loading=True,
        use_megatron_fsdp=True,
    )
    checkpoint_model = getattr(models[0], "module", models[0])
    sharded_state_dict["_model"] = [checkpoint_model]

    bridge_ckpt_config = _make_megatron_bridge_fsdp_checkpoint_config(config, is_save=False)
    load_sig = inspect.signature(load_fsdp_dtensor_checkpoint)
    required_params = {
        "load_dir",
        "ckpt_cfg",
        "rank0",
        "sharded_state_dict",
        "iteration",
        "release",
        "cfg",
    }
    assert required_params.issubset(
        load_sig.parameters
    ), ("Megatron-Bridge load_fsdp_dtensor_checkpoint signature is unsupported: "
        f"{load_sig}")
    state_dict, _, _, _ = load_fsdp_dtensor_checkpoint(
        load_dir=checkpoint_config.load_ckpt_path,
        ckpt_cfg=bridge_ckpt_config,
        rank0=False,
        sharded_state_dict=sharded_state_dict,
        iteration=global_step,
        release=False,
        cfg=get_model_config(checkpoint_model),
    )

    assert "model" in state_dict, (
        f"Model state dict not found in {state_dict.keys()}. Please check the checkpoint file "
        f"{checkpoint_config.load_ckpt_path}."
    )
    models[0].load_state_dict(state_dict["model"], strict=True)
    log(f"Loaded Megatron-FSDP model checkpoint from {checkpoint_config.load_ckpt_path}", rank=0)

    assert "optimizer" in state_dict, (
        f"Optimizer state dict not found in {state_dict.keys()}. Please check the checkpoint file "
        f"{checkpoint_config.load_ckpt_path}."
    )
    optimizer.load_state_dict(state_dict["optimizer"])
    log(
        f"Loaded Megatron-FSDP optimizer checkpoint from {checkpoint_config.load_ckpt_path}",
        rank=0
    )

    if "lr_scheduler" in state_dict and lr_scheduler is not None:
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        log(
            f"Loaded Megatron-FSDP LR scheduler checkpoint from {checkpoint_config.load_ckpt_path}",
            rank=0
        )

    assert "rng_state" in state_dict, (
        f"RNG state dict not found in {state_dict.keys()}. Please check the checkpoint file "
        f"{checkpoint_config.load_ckpt_path}."
    )
    load_rng_states(state_dict["rng_state"], use_megatron_fsdp=True)


def save_checkpoint(
    config,
    models,
    optimizer: MegatronOptimizer = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    global_step: int = 0,
    bridge=None,
    override_tokenizer_special_token: dict = None,
    peft=None,
    use_megatron_fsdp: bool = False,
):
    tracker_filename = "latest_checkpointed_iteration.txt"
    checkpoint_config = config.checkpoint
    # record the previous global step
    dist_checkpoint_path = get_dist_checkpoint_path(checkpoint_config.save_ckpt_path, global_step)
    os.makedirs(dist_checkpoint_path, exist_ok=True)
    log(f"saving checkpoint to {dist_checkpoint_path}", rank=0)
    if not isinstance(models, list):
        models = [models]

    if use_megatron_fsdp:
        _assert_megatron_fsdp_checkpoint_supported(config, models, is_save=True)

    if checkpoint_config.convert_mcore_to_hf_online and checkpoint_config.save_ckpt_path is not None:
        assert bridge is not None
        bridge_save_hf(
            config,
            global_step,
            models,
            bridge,
            override_tokenizer_special_token=override_tokenizer_special_token,
            peft=peft,
        )

    # Note that model weights, optimizer states, and extra states are generated
    # together in a state dict, we save them in one time
    if use_megatron_fsdp:
        async_save_request = save_megatron_fsdp_checkpoint(
            config,
            models,
            optimizer,
            lr_scheduler,
            dist_checkpoint_path,
        )
    elif checkpoint_config.use_dist_checkpointing:
        # Generate state dict for saving
        state_dict = generate_state_dict(
            models,
            optimizer,
            lr_scheduler,
            generate_optimizer=not checkpoint_config.no_save_optim,
            generate_model=not checkpoint_config.skip_save_mcore_model,
        )
        log(f"Generated state dict for saving: {state_dict.keys()}", rank=0)
        if not checkpoint_config.skip_save_mcore_model:
            for vpp_rank, _model in enumerate(models):
                if len(models) > 1:
                    model_i_keys = state_dict[f"model{vpp_rank}"].keys()
                    log(f"Generated state dict for model saving: {model_i_keys}", rank=0)
                else:
                    log(
                        f"Generated state dict for model saving: {state_dict['model'].keys()}",
                        rank=0
                    )
        else:
            log(f"NOT Generated state dict for model saving", rank=0)
        if peft is not None and not config.training.build_from_mbridge:
            state_dict = apply_peft_adapter_filter_to_state_dict(state_dict, peft)
            log("Filtered state dict to PEFT adapter parameters only", rank=0)
        # Start Async save if enabled
        async_save_request = save_dist_checkpointing(
            sharded_state_dict=state_dict,
            ckpt_path=dist_checkpoint_path,
            async_save=checkpoint_config.async_save,
        )

        # Synchronize all async save requests
        if not checkpoint_config.async_save:
            assert async_save_request is None, "Async save request should be None when not using async save."
            torch.distributed.barrier()
    else:
        assert checkpoint_config.use_hf_checkpoint, "When not using distributed checkpointing, use_hf_checkpoint should be True."
        raise NotImplementedError("hf format checkpoint has not been implemented")
        # TODO (@yeazhao) hf format checkpoint

    prev_iteration = 0
    save_retain_interval = checkpoint_config.save_retain_interval
    # 这里不应该多 rank 去读，ceph 可能有偶现读到\x00的问题
    if save_retain_interval is not None and 0 == torch.distributed.get_rank():
        tracker_path = os.path.join(checkpoint_config.save_ckpt_path, tracker_filename)
        if os.path.exists(tracker_path):
            with open(tracker_path, 'r') as f:
                content = f.read().strip().strip('\x00')
                if content:
                    try:
                        prev_iteration = int(content)
                    except ValueError:
                        log(
                            f"Warning: corrupted tracker file '{tracker_path}', "
                            f"content={content!r}. Treating as iteration 0.",
                            rank=0
                        )
                        prev_iteration = 0
                else:
                    prev_iteration = 0
        else:
            prev_iteration = 0

    if 0 == torch.distributed.get_rank():
        latest_checkpoint_file = os.path.join(checkpoint_config.save_ckpt_path, tracker_filename)
        with open(latest_checkpoint_file, "w") as f:
            f.write(str(global_step))
    cpu_barrier()
    if peft is not None:
        _save_peft_run_config(dist_checkpoint_path, peft)

    log(f"successfully saved checkpoint to {dist_checkpoint_path}", rank=0)

    def delete_checkpoint(iteration_to_delete):
        directory = 'iter_{:07d}'.format(iteration_to_delete)
        checkpoint_name = os.path.join(checkpoint_config.save_ckpt_path, directory)
        hf_dir = format(iteration_to_delete)
        hf_checkpoint_name = os.path.join(checkpoint_config.save_ckpt_path, "hf", hf_dir)

        hf_checkpoint_name = _get_hf_save_path(checkpoint_config, iteration_to_delete)
        adapter_hf_checkpoint_name = _get_hf_save_path(
            checkpoint_config, f"{iteration_to_delete}_adapter"
        )
        merge_hf_checkpoint_name = _get_hf_save_path(
            checkpoint_config, f"{iteration_to_delete}_merge"
        )
        dataloader_save_path, _ = get_dataloader_save_path(
            checkpoint_config, iteration_to_delete, 0
        )
        ds_save_path, _ = get_dynamic_sampling_save_path(
            checkpoint_config.save_ckpt_path, iteration_to_delete, 0
        )
        try:
            shutil.rmtree(checkpoint_name)
            log(
                f"successfully deleted checkpoint from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}"
            )

        except Exception as e:
            log(
                f'encountered exception "{e}" when trying to delete checkpoint from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}'
            )
            # Any exception encountered in checkpoint deletion can be ignored and is not fatal.
            pass
        try:
            shutil.rmtree(hf_checkpoint_name)
            log(
                f"successfully deleted hf checkpoint from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}"
            )
            if peft is not None:
                shutil.rmtree(adapter_hf_checkpoint_name)
                log(
                    f"successfully deleted adapter adapter_only hf checkpoint from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}"
                )
                if checkpoint_config.save_merged_lora_weights:
                    shutil.rmtree(merge_hf_checkpoint_name)
                    log(
                        f"successfully deleted merge hf checkpoint from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}"
                    )
        except Exception as e:
            log(
                f'encountered exception "{e}" when trying to delete hf checkpoint from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}'
            )
            # Any exception encountered in checkpoint deletion can be ignored and is not fatal.
            pass
        if dataloader_save_path is not None and os.path.exists(dataloader_save_path):
            try:
                shutil.rmtree(dataloader_save_path)
                log(
                    f"successfully deleted dataloader state from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}"
                )
            except Exception as e:
                log(
                    f'encountered exception "{e}" when trying to delete dataloader state from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}'
                )
                pass
        if ds_save_path is not None and os.path.exists(ds_save_path):
            try:
                shutil.rmtree(ds_save_path)
                log(
                    f"successfully deleted dynamic sampling state from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}"
                )
            except Exception as e:
                log(
                    f'encountered exception "{e}" when trying to delete dynamic sampling state from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}'
                )
                pass

    def sorted_checkpoints(output_dir=None, checkpoint_prefix="iter_", use_mtime=True):
        ordering_and_checkpoint_path = []
        glob_checkpoints = [
            str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}*") if os.path.isdir(x)
        ]
        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        # mtime is not reliable on all filesystems, especially on some fuse fs in cloud environments
        # so we check if the mtime is fake and fallback to numerical ordering if needed
        if use_mtime and len(ordering_and_checkpoint_path) > 1:
            mtime_diff = checkpoints_sorted[-1][0] - checkpoints_sorted[0][0]
            if mtime_diff < 1.0:  # less than 1 second, which is almost impossible when mtime works fine
                warnings.warn(
                    "mtime may not be reliable on this filesystem, falling back to numerical ordering"
                )
                return sorted_checkpoints(
                    use_mtime=False, output_dir=output_dir, checkpoint_prefix=checkpoint_prefix
                )
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def rotate_checkpoints(output_dir=None):
        save_total_limit = checkpoint_config.save_total_limit
        if save_total_limit is None or save_total_limit <= 0:
            return
        # Decide which iterations to drop from mcore ``iter_*`` dirs, then delete
        # all related artifacts (mcore / hf / peft / dataloader) via delete_checkpoint.
        checkpoints_sorted = sorted_checkpoints(
            output_dir=output_dir, checkpoint_prefix="iter_", use_mtime=True
        )
        if len(checkpoints_sorted) <= save_total_limit:
            return

        num_of_ckpt_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:num_of_ckpt_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            regex_match = re.match(r".*iter_([0-9]+)", checkpoint)
            if regex_match is None:
                log(f"skipping unexpected checkpoint path [{checkpoint}] during rotate")
                continue
            iteration_to_delete = int(regex_match.group(1))
            log(
                f"deleting older checkpoint [{checkpoint}] due to --gcore-save-total-limit={save_total_limit}."
            )
            delete_checkpoint(iteration_to_delete)

    def async_cleanup(prev_iteration):
        delete_checkpoint(prev_iteration)
        rotate_checkpoints(checkpoint_config.save_ckpt_path)

    if 0 == torch.distributed.get_rank():
        if save_retain_interval is not None:
            if prev_iteration > 0 and prev_iteration != global_step and prev_iteration % save_retain_interval != 0:
                directory = 'iter_{:07d}'.format(prev_iteration)
                checkpoint_name = os.path.join(checkpoint_config.save_ckpt_path, directory)

                # Don't delete if `checkpoint_name` is a symbolic link.
                if os.path.islink(checkpoint_name):
                    log(
                        f'skipping deleting checkpoint from iteration {prev_iteration:7d} at {checkpoint_config.save_ckpt_path} since it is a symbolic link'
                    )
                else:
                    # Asynchronous version of async_cleanup(prev_iteration).
                    threading.Thread(target=async_cleanup, args=(prev_iteration, )).start()
        else:
            threading.Thread(target=rotate_checkpoints,
                             args=(checkpoint_config.save_ckpt_path, )).start()


def load_checkpoint(
    config,
    models,
    optimizer: MegatronOptimizer = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    global_step: int = 0,
    bridge=None,
    peft=None,
    use_megatron_fsdp: bool = False,
):
    checkpoint_config = config.checkpoint
    assert global_step is not None, \
        f"fail to read training step from dir {checkpoint_config.load_ckpt_path}/latest_checkpointed_iteration.txt"
    dist_checkpoint_path = get_dist_checkpoint_path(checkpoint_config.load_ckpt_path, global_step)
    assert os.path.exists(
        dist_checkpoint_path
    ), f"Checkpoint path {dist_checkpoint_path} does not exist."
    log(f"loading checkpoint from {dist_checkpoint_path}", rank=0)

    if not isinstance(models, list):
        models = [models]

    if use_megatron_fsdp:
        _assert_megatron_fsdp_checkpoint_supported(config, models, is_save=False)
        load_megatron_fsdp_checkpoint(
            config,
            models,
            optimizer,
            lr_scheduler,
            global_step,
            bridge=bridge,
        )
        cpu_barrier()
        log(f"successfully loaded Megatron-FSDP checkpoint from {dist_checkpoint_path}", rank=0)
        return global_step

    # 当 skip_save_mcore_model 为 True 时，模型权重需要从 HF 格式通过 bridge 加载
    load_model_from_hf = checkpoint_config.skip_save_mcore_model

    # Get State Dict for loading
    sharded_state_dict = generate_state_dict(
        models,
        optimizer,
        lr_scheduler,
        generate_optimizer=not checkpoint_config.no_load_optim,
        generate_model=not load_model_from_hf,
        is_loading=True,
    )
    log(f"Generated state dict for loading: {sharded_state_dict.keys()}", rank=0)

    if peft is not None and not load_model_from_hf and not config.training.build_from_mbridge:
        sharded_state_dict = apply_peft_adapter_filter_to_state_dict(sharded_state_dict, peft)
        log("Filtered sharded state dict for PEFT adapter resume", rank=0)

    # Load Dist Checkpointing
    state_dict = load_dist_checkpointing(
        sharded_state_dict=sharded_state_dict,
        ckpt_dir=dist_checkpoint_path,
    )

    if load_model_from_hf:
        # 从 HF 格式通过 bridge 加载模型权重
        assert bridge is not None, (
            "bridge must be provided when skip_save_mcore_model=True, "
            "so that model weights can be loaded from HF format."
        )
        hf_model_path = _get_hf_save_path(checkpoint_config, global_step)
        log(f"Loading model weights from HF format via bridge: {hf_model_path}", rank=0)
        # When LoRA is active, temporarily unwrap module structure so the
        # bridge sees clean parameter names (without .to_wrap.).
        load_ctx = (_lora_structure_unwrapped(models) if peft is not None else nullcontext())
        with load_ctx:
            if config.training.build_from_mbridge:
                bridge.load_weights(models, hf_model_path, memory_efficient=True)
            else:
                bridge.load_hf_weights(models, hf_model_path, allowed_mismatched_params=[])
        log(f"Loaded model weights from HF format: {hf_model_path}", rank=0)

        # Load adapter (linear_in / linear_out) + extra trainable state for LoRA resume
        if peft is not None:
            adapter_dir = _get_hf_save_path(checkpoint_config, f"{global_step}_adapter")
            log(f"Loading adapter weights for LoRA resume: {adapter_dir}", rank=0)
            _load_adapter_from_hf_peft(models, adapter_dir, bridge=bridge)
    elif checkpoint_config.use_dist_checkpointing:
        assert "model" in state_dict or any(
            f"model{vpp_rank}" in state_dict for vpp_rank in range(len(models))
        ), f"Model state dict not found in {state_dict.keys()}. Please check the checkpoint file {checkpoint_config.load_ckpt_path}."
        for vpp_rank, _model in enumerate(models):
            if len(models) == 1:
                model_state_dict = state_dict["model"]
            else:
                assert f"model{vpp_rank}" in state_dict, f"model{vpp_rank} not found in state_dict"
                model_state_dict = state_dict[f"model{vpp_rank}"]
            mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            if peft is not None and config.training.build_from_mbridge:
                model_state_dict = _remap_peft_sharded_keys(model_state_dict, models[vpp_rank])
            models[vpp_rank].load_state_dict(model_state_dict, strict=False)
        log(f"Loaded sharded model checkpoint from {checkpoint_config.load_ckpt_path}", rank=0)
    elif checkpoint_config.use_hf_checkpoint:
        raise NotImplementedError("hf format checkpoint has not been implemented")

    if not checkpoint_config.no_load_optim:
        assert "optimizer" in state_dict, (
            f"Optimizer state dict not found in {state_dict.keys()}. Please check the checkpoint file {checkpoint_config.load_ckpt_path}."
        )
        optimizer_state_dict = state_dict["optimizer"]
        optimizer.load_state_dict(optimizer_state_dict)
        log(f"Loaded optimizer checkpoint from {checkpoint_config.load_ckpt_path}", rank=0)

        # When model weights were loaded from HF format (skip_save_mcore_model),
        # force-sync model BF16 params with optimizer's FP32 main params to
        # eliminate any precision discrepancy from the bridge roundtrip.
        if load_model_from_hf and peft is not None:
            sub_optimizers = (
                optimizer.chained_optimizers
                if hasattr(optimizer, 'chained_optimizers') else [optimizer]
            )
            for sub_opt in sub_optimizers:
                if hasattr(sub_opt, '_copy_main_params_to_model_params'):
                    sub_opt._copy_main_params_to_model_params()
            for model_chunk in models:
                if hasattr(model_chunk, 'start_param_sync'):
                    model_chunk.start_param_sync(force_sync=True)
            log("Synced model params from optimizer FP32 main params", rank=0)

        if "lr_scheduler" in state_dict:
            lr_scheduler_state_dict = state_dict["lr_scheduler"]
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log(
                    f"Loaded LR scheduler checkpoint from {checkpoint_config.load_ckpt_path}",
                    rank=0
                )

        assert "rng_state" in state_dict, (
            f"RNG state dict not found in {state_dict.keys()}. Please check the checkpoint file {checkpoint_config.load_ckpt_path}."
        )
        rng_state = state_dict["rng_state"]
        load_rng_states(rng_state)
        log(f"Loaded RNG states from {checkpoint_config.load_ckpt_path}", rank=0)
    cpu_barrier()
    log(f"successfully loaded checkpoint from {dist_checkpoint_path}", rank=0)
    return global_step
