"""FSDP2 distributed checkpoint (DCP) save/load helpers.

Reference: ``gpatch_v4/training_backend/fsdp2_backend/checkpoint.py``.
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import _init_optim_state

from .stateful import ModelState, OptimizerState

logger = logging.getLogger(__name__)


def ensure_dir_exist(*paths: os.PathLike[str] | str) -> None:
    """Create checkpoint directories on rank 0, then barrier all ranks.

    Matches gcore ``save_checkpoint`` mkdir + ``cpu_barrier`` pattern so DCP
    writers on every rank see existing parent dirs.
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        for path in paths:
            Path(path).expanduser().mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()


def fill_missing_optim_state(optimizer: torch.optim.Optimizer) -> None:
    """Fill zero AdamW-style entries for params that never received gradients.

    DCP requires every optimizer-managed parameter to have a checkpoint entry.
    Unused params otherwise cause ``Missing key in checkpoint state_dict`` on load.
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param in optimizer.state:
                continue
            optimizer.state[param] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
            }


def save_model(model: torch.nn.Module, model_dir: os.PathLike[str] | str) -> None:
    """Save FSDP2 model weights to a DCP checkpoint directory."""
    model_state = ModelState(model)
    dcp.save({"model_state": model_state}, checkpoint_id=str(model_dir))


def load_model(
    model: torch.nn.Module,
    model_dir: os.PathLike[str] | str,
    state_dict_preprocess=None,
) -> None:
    """Load FSDP2 model weights from a DCP checkpoint directory.

    ``state_dict_preprocess`` is an optional ``{fqn: tensor} -> {fqn: tensor}`` hook
    applied to the template state dict (see :class:`ModelState`); popping a key makes
    DCP skip it, enabling e.g. cross-resolution resume of runtime-regenerated
    pos-embeds. When provided, the load is non-strict.

    Returns the ``set_state_dict`` result
    (``_IncompatibleKeys(missing_keys, unexpected_keys)``, or ``None`` on older
    torch) so callers can validate which model params were left unfilled — e.g. the
    intentionally-popped runtime-only pos-embeds.
    """
    model_state = ModelState(model, state_dict_preprocess=state_dict_preprocess)
    state_dict = {"model_state": model_state}
    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
    except Exception:
        logger.error("Failed to load model from %s", model_dir)
        traceback.print_exc()
        raise
    return model_state.load_result


def save_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_dir: os.PathLike[str] | str,
) -> None:
    """Save FSDP2 optimizer state to a DCP checkpoint directory."""
    fill_missing_optim_state(optimizer)
    optimizer_state = OptimizerState(model, optimizer)
    dcp.save({"optim_state": optimizer_state}, checkpoint_id=str(optimizer_dir))


def load_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_dir: os.PathLike[str] | str,
    state_dict_preprocess=None,
) -> None:
    """Load FSDP2 optimizer state from a DCP checkpoint directory.

    ``state_dict_preprocess`` is an optional hook on the optimizer state dict (see
    :class:`OptimizerState`); dropping a param's ``state`` entry makes DCP skip it,
    enabling cross-resolution resume of runtime-regenerated params. When provided,
    the load is non-strict.
    """
    _init_optim_state(optimizer)
    fill_missing_optim_state(optimizer)
    optimizer_state = OptimizerState(model, optimizer, state_dict_preprocess=state_dict_preprocess)
    state_dict = {"optim_state": optimizer_state}
    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(optimizer_dir))
    except Exception:
        logger.error("Failed to load optimizer from %s", optimizer_dir)
        traceback.print_exc()
        raise
