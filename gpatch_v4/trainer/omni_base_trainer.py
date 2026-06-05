"""Base trainer with shared default implementations.

Subclasses must implement:
  - arg_cls (class attribute): top-level config dataclass
  - build_engine() -> engine instance
  - get_flop_coefficients() -> (dense_token_factor, attn_factor)

Optionally override:
  - log_prefix (default "train")
  - build_train_valid_test_data_iter()
  - get_dataset_and_collate()
  - customize_dataset_config()
"""

import gc
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from datetime import timedelta
from time import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import set_seed
from typing_extensions import override

from gpatch_v4.core.device import (
    get_device_module,
    get_device_perf_activity,
    get_dist_backend,
    get_profiler_module,
)
from gpatch_v4.trainer.base_trainer import BaseTrainer
from gpatch_v4.training_backend.common.omni_training_utils import (
    DummyProfiler,
    LoggerAdaptor,
    detect_peak_tflops,
)
from gpatch_v4.utils import (
    TrainReporterSingleton,
    dataclass_from_args,
    init_train_reporter_singleton,
)


class OmniBaseTrainer(BaseTrainer):
    """Base trainer with shared train_loop template method."""

    arg_cls = None  # subclass must set
    log_prefix: str = "train"

    def __init__(self, args):
        self.args = args
        self._val_runner = None

    # ------------------------------------------------------------------
    # Properties (shared by all trainers)
    # ------------------------------------------------------------------

    @property
    def training_args(self):
        return self.args.training

    @property
    def model_args(self):
        return self.args.model

    @property
    def data_args(self):
        return self.args.data

    # ------------------------------------------------------------------
    # Abstract methods that subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def build_engine(self):
        """Create and return the training engine (FSDP2Engine*)."""
        ...

    @abstractmethod
    def get_flop_coefficients(self) -> tuple:
        """Return (dense_token_factor, attn_factor) for MFU computation."""
        ...

    # ------------------------------------------------------------------
    # Data loading: shared implementation with hooks for customization
    # ------------------------------------------------------------------

    def get_dataset_and_collate(self):
        """Return ``(DatasetClass, collate_fn)`` for training data.

        Default: Bagel PackedDataset + collate_wrapper. Override for a
        different dataset (e.g. WGOv3PackedDataset).
        """
        from tasks.omni.bagel.data.dataset_base import PackedDataset, collate_wrapper
        return PackedDataset, collate_wrapper()

    def customize_dataset_config(self, dataset_config):
        """Hook to add subclass-specific fields to DataConfig.

        Called after base fields are set.
        """
        pass

    def build_train_valid_test_data_iter(self):
        """Build training data iterator.

        Loads YAML config, creates DataConfig, instantiates dataset, wraps
        in DataLoader. Subclasses customize via ``get_dataset_and_collate``
        and ``customize_dataset_config``.
        """
        import yaml

        from tasks.omni.bagel.data.dataset_base import DataConfig
        training_args = self.training_args
        model_args = self.model_args
        data_args = self.data_args

        vae_config = self.engine.vae_config
        tokenizer = self.engine.tokenizer
        new_token_ids = self.engine.new_token_ids
        data_status = self.engine.data_status
        torch.multiprocessing.set_start_method(training_args.mp_start_method, force=True)
        with open(data_args.dataset_config_file, "r") as f:
            dataset_meta = yaml.safe_load(f)

        dataset_config = DataConfig(
            grouped_datasets=dataset_meta,
            debug_num_used_data=getattr(data_args, "debug_num_used_data", None),
        )
        if training_args.visual_und:
            dataset_config.vit_patch_size = getattr(
                self.engine, "vit_patch_size", model_args.vit_patch_size
            )
            dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        if training_args.visual_gen and vae_config is not None:
            vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
            dataset_config.vae_image_downsample = vae_image_downsample
            dataset_config.max_latent_size = model_args.max_latent_size
            dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
            dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
            dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

        # Hook for subclass-specific DataConfig customization
        self.customize_dataset_config(dataset_config)

        DatasetClass, collate_fn = self.get_dataset_and_collate()

        train_dataset = DatasetClass(
            dataset_config,
            tokenizer=tokenizer,
            special_tokens=new_token_ids,
            local_rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            num_workers=data_args.num_workers,
            expected_num_tokens=data_args.expected_num_tokens,
            max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
            max_num_tokens=data_args.max_num_tokens,
            max_buffer_size=data_args.max_buffer_size,
            prefer_buffer_before=data_args.prefer_buffer_before,
            interpolate_pos=model_args.interpolate_pos,
            use_flex=training_args.use_flex,
            data_status=data_status,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=data_args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
            prefetch_factor=data_args.prefetch_factor,
        )
        return iter(train_loader), None, None

    # ------------------------------------------------------------------
    # Default implementations (can be overridden)
    # ------------------------------------------------------------------

    def profile_ctx(self):
        if self.training_args.profile:

            def trace_handler(p):
                if p.step_num < 100:
                    profile_dir = getattr(self.training_args, "profile_dir", "./profile_traces")
                    os.makedirs(profile_dir, exist_ok=True)
                    p.export_chrome_trace(
                        f"{profile_dir}/trace_rank_{dist.get_rank()}_step_{p.step_num}.json"
                    )

            profiler_module = get_profiler_module()
            return profiler_module.profile(
                activities=[get_device_perf_activity(), torch.profiler.ProfilerActivity.CPU],
                schedule=profiler_module.schedule(wait=1, warmup=1, active=2),
                on_trace_ready=trace_handler,
            )
        return DummyProfiler()

    def record_ctx(self, name):
        return record_function(name) if self.training_args.profile else nullcontext()

    def update_data_status(self, curr_step, data_indexes):
        data_status = self.engine.data_status
        if data_status is None:
            data_status = {}
        for item in data_indexes:
            dataset_name = item["dataset_name"]
            if dataset_name not in data_status:
                data_status[dataset_name] = {}
            data_status[dataset_name][item["worker_id"]] = item["data_indexes"]
        self.engine.data_status = data_status
        self.engine.curr_step = curr_step

    @override
    def init_distributed(self):
        training_args = self.training_args
        assert get_device_module().is_available()
        print(f"torch_dist_timeout_minutes: {training_args.torch_dist_timeout_minutes}")
        dist.init_process_group(
            get_dist_backend(), timeout=timedelta(minutes=training_args.torch_dist_timeout_minutes)
        )
        device = dist.get_rank() % get_device_module().device_count()
        get_device_module().set_device(device)
        seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
        set_seed(seed)
        if training_args.peak_device_tflops <= 0:
            auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
            if auto_tflops > 0:
                training_args.peak_device_tflops = auto_tflops

        # init gloo
        timeout = timedelta(minutes=90)
        world_size = torch.distributed.get_world_size()
        ranks = np.arange(world_size)
        self.gloo_group = torch.distributed.new_group(ranks=ranks, timeout=timeout, backend='gloo')

    def cpu_barrier(self):
        get_device_module().synchronize()
        torch.distributed.barrier(group=self.gloo_group)

    def init_log(self):
        training_args = self.training_args
        logger = LoggerAdaptor()
        if dist.get_rank() == 0:
            os.makedirs(training_args.checkpoint_dir, exist_ok=True)
            init_train_reporter_singleton(self.args.report, self.args)
            if training_args.peak_device_tflops > 0:
                logger.info(
                    f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU)."
                )
        self.cpu_barrier()
        dist.barrier()
        logger.info(f"Training arguments: {training_args}")
        logger.info(f"Model arguments: {self.model_args}")
        logger.info(f"Data arguments: {self.data_args}")
        self.logger = logger
        self.engine.set_logger(logger)

    @override
    def build_model_and_optimizer(self):
        self.engine.build_model_and_optimizer()

    @override
    def save_ckpt(self):
        self.engine.save_ckpt()

    @override
    def load_ckpt(self):
        pass

    def post_step(self, curr_step):
        if self.training_args.manual_gc and self.training_args.manual_gc_interval > 0:
            if (curr_step + 1) % self.training_args.manual_gc_interval == 0:
                gc.collect()

    def maybe_validate(self, curr_step: int, force: bool = False):
        """Hook for in-training validation. Return True when validation actually ran."""
        return False

    def finalize(self):
        logger = self.logger
        logger.info("Done!")
        if dist.get_rank() == 0:
            TrainReporterSingleton.finish()
        dist.destroy_process_group()

    # ------------------------------------------------------------------
    # train_loop: shared template with hooks for subclass customization
    # ------------------------------------------------------------------

    @override
    def train_loop(self):
        training_args = self.training_args
        logger = self.logger
        device = get_device_module().current_device()
        train_step = self.engine.train_step
        gbs = training_args.global_batch_size
        world_size = dist.get_world_size()
        assert gbs % world_size == 0
        gradient_accumulation_steps = gbs // world_size

        self.engine.prepare_for_train()

        dense_token_factor, attn_factor = self.get_flop_coefficients()
        profile_ctx = self.profile_ctx()

        start_time = time()
        logger.info(
            f"Training {self.log_prefix} for {training_args.total_steps} steps, starting at {train_step}..."
        )
        total_norm = torch.tensor(0.0, device=device)
        token_window = 0.0
        seqlen_square_window = 0.0
        total_ce_token_window = 0.0
        total_mse_token_window = 0.0

        if training_args.log_time_breakdown:
            io_time_window = 0.0
            forward_backward_time_window = 0.0
            optimizer_time_window = 0.0

        def reset_perf_windows():
            nonlocal start_time
            nonlocal token_window, seqlen_square_window
            nonlocal total_ce_token_window, total_mse_token_window
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0
            total_ce_token_window = 0.0
            total_mse_token_window = 0.0
            if training_args.log_time_breakdown:
                nonlocal io_time_window, forward_backward_time_window, optimizer_time_window
                io_time_window = 0.0
                forward_backward_time_window = 0.0
                optimizer_time_window = 0.0

        train_iter, _, _ = self.build_train_valid_test_data_iter()
        self.cpu_barrier()
        if training_args.manual_gc and training_args.manual_gc_interval > 0:
            gc.disable()
            gc.collect()

        curr_step = train_step

        if getattr(training_args, "validation_at_start", False):
            if self.maybe_validate(curr_step, force=True):
                reset_perf_windows()

        with profile_ctx:
            if training_args.log_time_breakdown:
                iter_start_time = time()

            for micro_step, data in enumerate(train_iter):
                curr_step = train_step + micro_step // gradient_accumulation_steps
                if curr_step >= training_args.total_steps:
                    logger.info(f"Reached total_steps={training_args.total_steps}, stopping.")
                    break

                logger.info(
                    f"[Rank {dist.get_rank()}] Step {curr_step} micro_step {micro_step}: fetched batch from train_iter"
                )
                data = data.cuda(device).to_dict()
                data_indexes = data.pop("batch_data_indexes", None)
                tokens_tensor = torch.tensor(float(data["sequence_length"]), device=device)
                dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
                token_window += tokens_tensor.item()

                if data["sample_lens"]:
                    sample_lens_tensor = torch.tensor(
                        data["sample_lens"], dtype=torch.float32, device=device
                    )
                    sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
                    dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
                    seqlen_square_window += sample_square.item()

                is_last_micro_batch = (micro_step + 1) % gradient_accumulation_steps == 0

                if training_args.skip_steps > 0 and curr_step < training_args.skip_steps:
                    if (micro_step % gradient_accumulation_steps) == 0:
                        print(
                            f"[Rank {dist.get_rank()}] [SKIP] Step {curr_step} — skipping compute",
                            flush=True,
                        )
                    self.update_data_status(curr_step, data_indexes)
                    if training_args.log_time_breakdown:
                        iter_start_time = time()
                    continue

                self.engine.set_grad_sync_flag(is_last_micro_batch)

                if training_args.log_time_breakdown:
                    get_device_module().synchronize()
                    io_end_time = time()
                    io_time_window += io_end_time - iter_start_time

                with self.record_ctx("forward_backward_step"):
                    if training_args.log_time_breakdown:
                        forward_backward_start = time()

                    logger.info(
                        f"[Rank {dist.get_rank()}] Step {curr_step} micro_step {micro_step}: entering forward_backward_step"
                    )
                    loss, loss_dict, total_mse_tokens, total_ce_tokens = (
                        self.engine.forward_backward_step(
                            data, loss_scale=1.0 / gradient_accumulation_steps
                        )
                    )
                    logger.info(
                        f"[Rank {dist.get_rank()}] Step {curr_step} micro_step {micro_step}: finished forward_backward_step"
                    )
                    total_ce_token_window += total_ce_tokens.item()
                    total_mse_token_window += total_mse_tokens.item()

                    if training_args.log_time_breakdown:
                        get_device_module().synchronize()
                        forward_backward_time_window += time() - forward_backward_start

                if is_last_micro_batch:
                    with self.record_ctx("optimize_step"):
                        if training_args.log_time_breakdown:
                            optimizer_start = time()
                        total_norm = self.engine.optimize_step()
                        if training_args.log_time_breakdown:
                            get_device_module().synchronize()
                            optimizer_time_window += time() - optimizer_start
                    self.post_step(curr_step)

                # Logging
                if curr_step % training_args.log_every == 0:
                    total_samples = torch.tensor(len(data["sample_lens"]), device=device)
                    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

                    get_device_module().synchronize()
                    end_time = time()
                    elapsed = max(end_time - start_time, 1e-6)
                    steps_per_sec = training_args.log_every / elapsed
                    tokens_per_sec = token_window / elapsed
                    tokens_per_step = token_window / training_args.log_every
                    flops_all = dense_token_factor * token_window + attn_factor * seqlen_square_window
                    actual_tflops = flops_all / elapsed / 1e12
                    peak_total_tflops = training_args.peak_device_tflops * world_size
                    mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0

                    current_rank = dist.get_rank()
                    message = f"[Rank {current_rank}/{world_size}] (step={curr_step:07d}) "
                    wandb_log = {}
                    for key, value in loss_dict.items():
                        avg_loss = torch.tensor(value.item(), device=device)
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                        avg_loss = avg_loss.item() / world_size
                        message += f"Loss/{key}: {avg_loss:.4f}, "
                        wandb_log[key] = avg_loss

                    optimizer_lr_items = []
                    for group_idx, group in enumerate(self.engine.optimizer.param_groups):
                        group_name = group.get("group_name", f"group{group_idx}")
                        lr_value = float(group["lr"])
                        optimizer_lr_items.append((group_name, lr_value))
                        wandb_log[f"lr_{group_name}"] = lr_value
                    if optimizer_lr_items:
                        wandb_log["lr"] = optimizer_lr_items[0][1]

                    message += (
                        f"Steps/Sec: {steps_per_sec:.2f}, "
                        f"Tokens/Sec: {tokens_per_sec/1000:.2f}k, "
                        f"MFU: {mfu_value*100:.1f}%"
                    )
                    if optimizer_lr_items:
                        lr_message = " ".join(
                            f"{group_name}:{lr_value:.3e}"
                            for group_name, lr_value in optimizer_lr_items
                        )
                        message += f", LR[{lr_message}]"

                    if training_args.log_time_breakdown:
                        avg_io = io_time_window / training_args.log_every
                        avg_fwd = forward_backward_time_window / training_args.log_every
                        avg_opt = optimizer_time_window / training_args.log_every
                        avg_total = elapsed / training_args.log_every
                        message += (
                            f", Time[IO:{avg_io:.3f}s FwdBwd:{avg_fwd:.3f}s Opt:{avg_opt:.3f}s]"
                        )
                        wandb_log.update(
                            {
                                "time_io": avg_io,
                                "time_fwd_bwd": avg_fwd,
                                "time_opt": avg_opt,
                                "time_total": avg_total,
                            }
                        )

                    logger.info(message)
                    wandb_log["total_mse_tokens"] = total_mse_token_window
                    wandb_log["total_ce_tokens"] = total_ce_token_window
                    wandb_log["total_norm"] = total_norm.item()
                    wandb_log["total_samples"] = total_samples.item()
                    wandb_log["steps_per_sec"] = steps_per_sec
                    wandb_log["tokens_per_sec"] = tokens_per_sec
                    wandb_log["tokens_per_sec_per_device"] = tokens_per_sec / world_size
                    wandb_log["tokens_per_step"] = tokens_per_step
                    wandb_log["actual_tflops"] = actual_tflops
                    wandb_log["mfu"] = mfu_value

                    mem_allocated = torch.tensor(
                        get_device_module().max_memory_allocated() / 1024**2, device=device
                    )
                    dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
                    wandb_log["mem_allocated"] = mem_allocated
                    mem_cache = torch.tensor(
                        get_device_module().max_memory_reserved() / 1024**2, device=device
                    )
                    dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
                    wandb_log["mem_cache"] = mem_cache

                    if dist.get_rank() == 0:
                        TrainReporterSingleton.log_and_report(
                            wandb_log, curr_step, log_prefix=self.log_prefix
                        )

                    start_time = time()
                    token_window = 0.0
                    seqlen_square_window = 0.0
                    total_mse_token_window = 0.0
                    total_ce_token_window = 0.0
                    if training_args.log_time_breakdown:
                        io_time_window = 0.0
                        forward_backward_time_window = 0.0
                        optimizer_time_window = 0.0

                self.update_data_status(curr_step, data_indexes)

                if (curr_step > 0 and curr_step % training_args.save_every == 0) or curr_step == 50:
                    self.save_ckpt()
                    self.cpu_barrier()

                if self.maybe_validate(curr_step):
                    reset_perf_windows()
                    if training_args.log_time_breakdown:
                        iter_start_time = time()

                profile_ctx.step()

                if training_args.log_time_breakdown:
                    iter_start_time = time()

        # Final checkpoint
        if curr_step > 0 and curr_step % training_args.save_every:
            logger.info(f"Saving final checkpoint at step {curr_step}...")
            self.save_ckpt()
