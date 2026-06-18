import inspect
from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import ModelConfig

from gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4 import (
    finalize_weights_after_reload as finalize_weights_after_reload_dsv4,
)
from gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4 import (
    is_deepseek_v4_model,
    load_quanted_weights_for_update_dsv4,
)
from gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4 import (
    prepare_weights_for_reload as prepare_weights_for_reload_dsv4,
)
from gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4 import (
    restore_moe_after_wakeup as restore_moe_after_wakeup_dsv4,
)
from gpatch_v4.generation_backend.vllm_model_specific.vllm_weight_reload_dsv4 import (
    save_moe_for_sleep as save_moe_for_sleep_dsv4,
)


def prepare_weights_for_reload(model: nn.Module, model_runner, device: torch.device) -> dict:
    """Prepare a vLLM model to receive checkpoint-format weight buckets.
    Must be called once before the first bucket of a multi-bucket update.

    Returns
    -------
    dict
        Reload state consumed by :func:`finalize_weights_after_reload`. or do nothing if not a Deepseek V4 model
    """
    if is_deepseek_v4_model(model, model_runner):
        return prepare_weights_for_reload_dsv4(model, device)
    return None


def load_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load one checkpoint-format bucket without premature MegaMoE finalize."""
    bucket = list(weights)
    # bucket_names = [name for name, _ in bucket]

    load_fn = model.load_weights
    sig = inspect.signature(load_fn)
    more_args = {}
    if "defer_mega_moe_finalize" in sig.parameters:
        more_args["defer_mega_moe_finalize"] = True
    loaded = load_fn(bucket, **more_args)
    return loaded


def load_weights_for_update(
    model: nn.Module,
    model_runner,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load one update bucket"""
    if is_deepseek_v4_model(model, model_runner):
        return load_quanted_weights_for_update_dsv4(model, model_runner, weights)
    return load_weights(model, weights)


def finalize_weights_after_reload(
    model: nn.Module,
    model_config: ModelConfig,
    model_runner,
    device: torch.device,
) -> None:
    """Finalize one full checkpoint-format weight update on the worker."""
    if is_deepseek_v4_model(model, model_runner):
        return finalize_weights_after_reload_dsv4(model, model_config, device)
    else:
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        with torch.device(device):
            process_weights_after_loading(model, model_config, device)


def save_moe_for_sleep(
    model: nn.Module,
    model_runner,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if is_deepseek_v4_model(model, model_runner):
        return save_moe_for_sleep_dsv4(model, device)
    return None


def restore_moe_after_wakeup(
    model: nn.Module,
    model_runner,
    device: torch.device,
    stash: dict[str, torch.Tensor] | None,
) -> None:
    if stash and is_deepseek_v4_model(model, model_runner):
        restore_moe_after_wakeup_dsv4(model, device, stash)
