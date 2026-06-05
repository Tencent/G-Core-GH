"""
Shared FSDP2 pipeline base class for Bagel and WGOv3.

Subclasses must implement:
  - build_model_and_optimizer()
  - forward_backward_step()
"""

import functools
import gc
import os
import time
from abc import abstractmethod
from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)
from typing_extensions import override

from gpatch_v4.core.device import get_device_module
from gpatch_v4.extended_pipeline.pipeline_base import ExtendedPipelineAbc
from gpatch_v4.training_backend.common.omni_training_utils import (
    LoggerAdaptor,
    count_parameters,
    get_latest_ckpt,
)
from gpatch_v4.training_backend.common.swap_mixin import EngineSwapMixin
from gpatch_v4.training_backend.fsdp2_backend.fsdp2_swap_impl import Fsdp2SwapImpl
from gpatch_v4.training_backend.fsdp2_backend.fsdp2_utils_bagel import (
    FSDPConfig,
    _init_device_mesh,
    fsdp_ema_update,
)


@contextmanager
def to_meta_context(model):
    """Context manager to move model to meta device and restore original data."""
    parameters = [p.detach() for p in list(model.parameters())]
    named_buffers = {name: p.detach() for name, p in model.named_buffers()}
    try:
        model.to_empty(device="meta")
        yield
    finally:
        for (p, meta_p) in zip(parameters, model.parameters()):
            meta_p._data = p.data
        for (name, meta_p) in model.named_buffers():
            meta_p._data = named_buffers[name].data

        del parameters
        del named_buffers

        def apply(t):
            data = t._data
            del t._data
            return data

        model._apply(apply)


def sync_ema(model, ema_model):
    for (p, ema_p) in zip(model.parameters(), ema_model.parameters()):
        ema_p.data.copy_(p.data)

    named_buffers = {
        name.replace("._checkpoint_wrapped_module", ""): p
        for name, p in model.named_buffers()
    }

    for (name, ema_p) in ema_model.named_buffers():
        try:
            ema_p.data.copy_(named_buffers[name].data)
        except Exception as e:
            print(f"Failed to sync buffer {name} {named_buffers.keys()}: {e}", flush=True)
            raise e


class ModelAdaptor(EngineSwapMixin):
    """Thin wrapper so the GRPO actor can reference engine.model."""
    def __init__(self, config, model, ema_model, optimizer, lr_scheduler, fsdp_checkpoint_cls):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ema_model = ema_model
        self.ref_model = None
        self.swap_impl = Fsdp2SwapImpl()
        self._fsdp_checkpoint_cls = fsdp_checkpoint_cls

    def set_logger(self, logger):
        self.logger = logger

    def save(self, step, data_status=None):
        get_device_module().empty_cache()
        get_device_module().synchronize()
        self._fsdp_checkpoint_cls.fsdp_save_ckpt(
            ckpt_dir=self.config.training.checkpoint_dir,
            train_steps=step,
            model=self.model,
            ema_model=self.ema_model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            logger=self.logger,
            data_status=data_status,
        )
        gc.collect()
        get_device_module().empty_cache()
        get_device_module().synchronize()


class FSDP2EngineBase(ExtendedPipelineAbc):
    """Shared FSDP2 engine logic: auto_resume, freeze, FSDP setup, optimizer/scheduler,
    optimize_step, save_ckpt, set_grad_sync_flag.
    """
    def __init__(self, config):
        self.config = config

    @property
    def training_args(self):
        return self.config.training

    @property
    def model_args(self):
        return self.config.model

    def set_logger(self, logger):
        self.logger = logger

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _resolve_resume(self):
        """Resolve auto_resume logic. Returns (resume_from, resume_model_only, finetune_from_ema)."""
        training_args = self.training_args
        if training_args.auto_resume:
            resume_from = get_latest_ckpt(training_args.checkpoint_dir)
            if resume_from is None:
                resume_from = training_args.resume_from
                resume_model_only = training_args.resume_model_only
                finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False
            else:
                resume_model_only = False
                finetune_from_ema = False
        else:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False
        return resume_from, resume_model_only, finetune_from_ema

    def _freeze_modules(self, model, vae_model=None, vit_model=None):
        """Apply optional module freezing based on training_args."""
        training_args = self.training_args
        if training_args.freeze_vae and training_args.visual_gen and vae_model is not None:
            for param in vae_model.parameters():
                param.requires_grad = False
        if training_args.freeze_llm:
            model.language_model.eval()
            for param in model.language_model.parameters():
                param.requires_grad = False
        if training_args.freeze_vit and training_args.visual_und and vit_model is not None:
            vit_attr = getattr(model, "vit_model", None)
            if vit_attr is not None:
                vit_attr.eval()
                for param in vit_attr.parameters():
                    param.requires_grad = False
        if getattr(training_args, "freeze_projector", False) and training_args.visual_und:
            projector_attr = getattr(model, "projector", None)
            if projector_attr is None:
                projector_attr = getattr(model, "connector", None)
            if projector_attr is not None:
                projector_attr.eval()
                for param in projector_attr.parameters():
                    param.requires_grad = False

    def _setup_fsdp(
        self, model, fsdp_wrapper_fn, grad_ckpt_check_fn, fsdp_ema_setup_fn, fsdp_checkpoint_cls,
        resume_from, finetune_from_ema
    ):
        """FSDP wrap model + EMA, load checkpoint if resuming. Returns (fsdp_model, ema_model)."""
        training_args = self.training_args
        use_ema = getattr(self.config.ema, "use_ema", True)
        fsdp_config = FSDPConfig(
            sharding_strategy=training_args.sharding_strategy,
            backward_prefetch=training_args.backward_prefetch,
            cpu_offload=training_args.cpu_offload,
            num_replicate=training_args.num_replicate,
            num_shard=training_args.num_shard,
            num_to_forward_prefetch=getattr(training_args, "fsdp2_num_to_forward_prefetch", 2),
            torch_dist_timeout_minutes=training_args.torch_dist_timeout_minutes,
        )
        stage_start = time.time()
        device_mesh = _init_device_mesh(fsdp_config)
        self.logger.info(
            f"[InitProfile] fsdp setup: init device mesh took {time.time() - stage_start:.1f}s"
        )
        stage_start = time.time()
        ema_model = None
        if use_ema:
            # only copy in meta device, to avoid OOM
            with to_meta_context(model):
                ema_model = deepcopy(model)
            self.logger.info(
                "[InitProfile] fsdp setup: prepare ema model took "
                f"{time.time() - stage_start:.1f}s "
            )
        else:
            self.logger.info(
                "[InitProfile] fsdp setup: prepare ema model skipped (ema.use_ema=False)"
            )

        stage_start = time.time()
        fsdp_model = fsdp_wrapper_fn(model, device_mesh, fsdp_config)
        gc.collect()
        self.logger.info(
            f"[InitProfile] fsdp setup: wrap train model took {time.time() - stage_start:.1f}s"
        )

        stage_start = time.time()
        if ema_model is not None:
            ema_model = ema_model.to_empty(device="cpu")
            ema_model = fsdp_ema_setup_fn(ema_model, device_mesh, fsdp_config)
            self.logger.info(
                f"[InitProfile] fsdp setup: wrap ema model took {time.time() - stage_start:.1f}s"
            )
            # sync weights after fsdp setup
            sync_ema(fsdp_model, ema_model)
        else:
            self.logger.info("[InitProfile] EMA disabled, skipping EMA model setup")

        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=functools.partial(grad_ckpt_check_fn, self.model_args),
        )
        self.logger.info("[InitProfile] fsdp setup: activation checkpointing configured")

        stage_start = time.time()
        model, ema_model = fsdp_checkpoint_cls.try_load_ckpt(
            resume_from, self.logger, model, ema_model, resume_from_ema=finetune_from_ema
        )
        self.logger.info(
            f"[InitProfile] fsdp setup: resume checkpoint hook took {time.time() - stage_start:.1f}s"
        )

        if dist.get_rank() == 0:
            for name, param in model.named_parameters():
                print(name, param.requires_grad)

        return fsdp_model, ema_model

    def _get_optimizer_param_groups(self, fsdp_model):
        """Return optimizer parameters or param groups for AdamW."""
        return fsdp_model.parameters()

    def _build_optimizer_scheduler(self, fsdp_model):
        """Create AdamW optimizer and LR scheduler."""
        optimizer = torch.optim.AdamW(
            self._get_optimizer_param_groups(fsdp_model),
            lr=self.config.optimizer.lr,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
            weight_decay=0,
        )
        if self.config.optimizer.lr_decay_style == "cosine":
            scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.config.optimizer.lr_warmup_steps,
                num_training_steps=self.config.training.total_steps,
                min_lr=self.config.optimizer.min_lr,
            )
        elif self.config.optimizer.lr_decay_style in ("constant", "constant_with_warmup"):
            scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.config.optimizer.lr_warmup_steps,
            )
        else:
            raise ValueError(f"Unknown lr_decay_style: {self.config.optimizer.lr_decay_style}")
        return optimizer, scheduler

    def _store_state(
        self, fsdp_model, ema_model, optimizer, scheduler, llm_config, resume_from,
        resume_model_only, fsdp_checkpoint_cls
    ):
        """Load training state (if resuming) and store all engine attributes."""
        if resume_model_only:
            train_step = 0
            data_status = None
        else:
            optimizer, scheduler, train_step, data_status = fsdp_checkpoint_cls.try_load_train_state(
                resume_from, fsdp_model, optimizer, scheduler
            )

        self.fsdp_model = fsdp_model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = ModelAdaptor(
            self.config,
            self.fsdp_model,
            self.ema_model,
            self.optimizer,
            self.scheduler,
            fsdp_checkpoint_cls,
        )
        self.model.set_logger(self.logger)
        self.data_status = data_status
        self.train_step = train_step
        self.llm_config = llm_config

    # ------------------------------------------------------------------
    # Shared training loop helpers
    # ------------------------------------------------------------------

    def prepare_for_train(self):
        self.fsdp_model.train()
        if self.ema_model is not None:
            self.ema_model.eval()
        self.optimizer.zero_grad()

    @abstractmethod
    def forward_backward_step(self, data, loss_scale=None):
        ...

    def optimize_step(self):
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.fsdp_model.parameters(), self.config.optimizer.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        if self.ema_model is not None:
            fsdp_ema_update(self.ema_model, self.fsdp_model, decay=self.config.ema.ema_decay)
        self.optimizer.zero_grad()
        return total_norm

    def save_ckpt(self):
        data_status = self.data_status
        curr_step = self.curr_step
        logger = self.logger

        get_device_module().empty_cache()
        get_device_module().synchronize()
        if dist.get_rank() == 0:
            gather_list = [None] * dist.get_world_size()
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
        except Exception as e:
            logger.error(f"Error during gather_object at step {curr_step}: {e}")
            gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()
        self.model.save(curr_step, data_status=gather_list)

    def set_grad_sync_flag(self, is_last_micro_batch):
        if self.training_args.sharding_strategy == "HYBRID_SHARD":
            self.fsdp_model.set_requires_all_reduce(is_last_micro_batch, recurse=True)

    @override
    def setup_pipeline(self):
        self.set_logger(LoggerAdaptor())
        self.build_model_and_optimizer()
        return self.train_step
