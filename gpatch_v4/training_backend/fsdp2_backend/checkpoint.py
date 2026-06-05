import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    _init_optim_state,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.utils import copy_cached_hf_metadata_files, log


def _fill_missing_optim_state(optimizer):
    """Create zero state entries for parameters that never received gradients.

    DCP requires every optimizer-managed parameter to have a state entry;
    params with no grads (e.g. an unused audio tower) otherwise trigger
    ``Missing key in checkpoint state_dict`` on load. We fill zero
    AdamW-style state (``step``, ``exp_avg``, ``exp_avg_sq``) for those.
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


# 此文件 reference from slime: slime/backends/fsdp_utils/checkpoint.py
class ModelState(Stateful):
    """Wrapper for model state only."""
    def __init__(self, model):
        self.model = model

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None
        )


class OptimizerState(Stateful):
    """Wrapper for optimizer state only."""
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=self.optimizer)
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            optimizers=self.optimizer,
            model_state_dict=None,
            optim_state_dict=state_dict["optim"]
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""
    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


def load_checkpoint(config, model, optimizer=None, lr_scheduler=None, global_step: int = 0):
    checkpoint_config = config.checkpoint
    assert global_step is not None, \
        f"fail to read training step from dir {checkpoint_config.load_ckpt_path}/latest_checkpointed_iteration.txt"

    base_dir = Path(checkpoint_config.load_ckpt_path).expanduser()
    checkpoint_dir = base_dir / f"iter_{global_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        log(f"[FSDP] Model checkpoint {model_dir} not found; skipping load.", rank=0)
        return None

    log(f"loading checkpoint from {checkpoint_dir}", rank=0)

    model_state = ModelState(model)
    state_dict = {"model_state": model_state}
    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
    except Exception as e:
        log(f"[FSDP] Failed to load model from {model_dir}: {e}")
        traceback.print_exc()
        raise e

    if not checkpoint_config.no_load_optim:
        if not optimizer_dir.exists():
            log(f"[FSDP] Model optimizer {optimizer_dir} not found; skipping load.", rank=0)
            return None
        if not lr_scheduler_dir.exists():
            log(f"[FSDP] Model lr_scheduler {lr_scheduler_dir} not found; skipping load.", rank=0)
            return None

        if optimizer is not None:
            _init_optim_state(optimizer)
            _fill_missing_optim_state(optimizer)
            optimizer_state = OptimizerState(model, optimizer)
            optim_state_dict = {"optim_state": optimizer_state}
            try:
                dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            except Exception as e:
                log(f"[FSDP] Failed to load optimizer from {optimizer_dir}: {e}")
                traceback.print_exc()
                raise e

        if lr_scheduler is not None:
            lr_scheduler_state = LRSchedulerState(lr_scheduler)
            lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
            try:
                dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            except Exception as e:
                log(f"[FSDP] Failed to load LR scheduler from {lr_scheduler_dir}: {e}")
                traceback.print_exc()
                raise e

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")
    #TODO: rng state 看看要怎么复用还是
    return global_step


def save_checkpoint(config, model, optimizer=None, lr_scheduler=None, global_step: int = 0):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    checkpoint_config = config.checkpoint

    base_dir = Path(checkpoint_config.save_ckpt_path).expanduser()
    checkpoint_dir = base_dir / f"iter_{global_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"
    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
    cpu_barrier()
    log(f"saving checkpoint to {checkpoint_dir}", rank=0)

    model_state = ModelState(model)
    state_dict = {"model_state": model_state}
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    if not checkpoint_config.no_save_optim:
        if optimizer is not None:
            _fill_missing_optim_state(optimizer)
            optimizer_state = OptimizerState(model, optimizer)
            optim_state_dict = {"optim_state": optimizer_state}
            dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

        if lr_scheduler is not None:
            lr_scheduler_state = LRSchedulerState(lr_scheduler)
            lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
            dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, checkpoint_dir / "rng.pt")

    if 0 == torch.distributed.get_rank():
        latest_checkpoint_file = base_dir / "latest_checkpointed_iteration.txt"

        with open(latest_checkpoint_file, "w") as f:
            f.write(str(global_step))
    cpu_barrier()
    log(f"successfully saved checkpoint to {checkpoint_dir}", rank=0)
    return global_step


def _remove_tied_weight_keys(model, state_dict):
    """Remove tied weight keys to avoid duplicate storage.

    ``get_model_state_dict`` with ``full_state_dict=True`` materialises
    every parameter independently, breaking weight-tying.
    ``save_pretrained`` can't detect the sharing via ``id()`` so it saves
    duplicates; we drop ``_tied_weights_keys`` so the file size matches
    and ``from_pretrained`` re-ties on load.
    """
    tied_keys = getattr(model, "_tied_weights_keys", None) or []
    for key in tied_keys:
        if key in state_dict:
            del state_dict[key]
            log(f"removed tied weight key from HF state_dict: {key}", rank=0)


def save_hf_checkpoint(config, model, tokenizer, global_step: int = 0):
    """Export FSDP2 model to HuggingFace ``save_pretrained`` format.

    Gathers the full state dict on rank 0 (CPU), then saves model
    weights, config, and tokenizer for ``from_pretrained`` reload.
    """
    checkpoint_config = config.checkpoint
    if checkpoint_config.export_hf_save_path is not None:
        hf_dir = Path(checkpoint_config.export_hf_save_path) / str(global_step)
    else:
        hf_dir = Path(checkpoint_config.save_ckpt_path).expanduser() / f"hf/{global_step}"

    log(f"exporting HuggingFace checkpoint to {hf_dir}", rank=0)
    t0 = time.time()

    torch.cuda.empty_cache()
    full_state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    if dist.get_rank() == 0:
        _remove_tied_weight_keys(model, full_state)
        hf_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(hf_dir), state_dict=full_state)
        copy_cached_hf_metadata_files(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
            hf_dir,
        )

    cpu_barrier()
    elapsed = time.time() - t0
    log(f"HuggingFace checkpoint saved to {hf_dir} ({elapsed:.1f}s)", rank=0)
