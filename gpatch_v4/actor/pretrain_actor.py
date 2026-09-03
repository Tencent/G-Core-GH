"""Lean pretrain actor: dataloader → train (H2D per microbatch in forward)."""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Dict, List

import torch
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.actor.finetune_actor import FinetuneActor
from gpatch_v4.core.parallel_state import cpu_barrier, cpu_group, is_last_rank
from gpatch_v4.utils import (
    TimerSingleton,
    TrainReporterSingleton,
    log,
    record_time_to_metrics,
)
from gpatch_v4.utils.test_utils import save_data

try:
    from megatron.core.gcore_utils import clear_gathered_routing_info
except ImportError:
    clear_gathered_routing_info = None


class PretrainActor(FinetuneActor):
    """High-throughput packed pretrain actor.

    Hot path: dataloader → ``pretrain_packed`` (per mb) → train.
    Skips dyn-CP / ``expand_rollout_batches`` / ``sft_train``.
    """

    train_log_tag = "PRETRAIN"

    @override
    def validated_config(self):
        super().validated_config()
        assert self.config.training.training_backend == "mcore", (
            "PretrainActor requires training.training_backend='mcore' "
            f"(got {self.config.training.training_backend!r})"
        )
        assert not self.config.policy.dist_config.dynamic_context_parallel, (
            "PretrainActor is incompatible with dynamic_context_parallel"
        )
        assert not self.config.policy.smart_pad_train, (
            "PretrainActor does not support smart_pad_train"
        )
        assert self.config.training.train_mbs == 1, ("PretrainActor requires training.train_mbs=1")

    @override
    def auto_calc_train_step(self):
        """Estimate optimizer steps while actual stopping follows sample count."""
        training_config = self.config.training
        dp_size = mpu.get_data_parallel_world_size()
        assert training_config.train_gbs is not None, "train_gbs must be configured"
        gas = training_config.train_gbs // (dp_size * training_config.train_mbs)
        assert gas > 0, f"gradient_accumulation_steps must be positive, got {gas}"
        train_step_per_epoch = len(self.train_dataloader) // gas
        assert train_step_per_epoch > 0, (
            f"packed dataloader is too short: len={len(self.train_dataloader)}, gas={gas}"
        )
        total_training_step = math.ceil(train_step_per_epoch * training_config.num_train_epoches)

        training_config.total_training_step = total_training_step
        training_config.train_step_per_epoch = train_step_per_epoch
        training_config.gradient_accumulation_steps = gas
        log(
            f"train_dataset length: {len(self.train_dataloader)=} {len(self.train_dataset)=} "
            f"{dp_size=} estimated {training_config.total_training_step=}"
        )

    def _load_sample_progress(self) -> tuple[int, int, int, int]:
        """Load and validate global sample progress from the dataloader."""
        assert hasattr(self.train_dataloader, "get_checkpoint_metadata"
                      ), ("PretrainActor requires dataloader checkpoint metadata")
        assert hasattr(self.train_dataloader, "update_checkpoint_metadata"
                      ), ("PretrainActor requires mutable dataloader checkpoint metadata")
        metadata = self.train_dataloader.get_checkpoint_metadata()
        raw_num_samples = int(metadata["raw_num_samples"])
        consumed_samples = int(metadata["consumed_samples"])
        skipped_samples = int(metadata["skipped_samples"])
        assert raw_num_samples > 0, (
            f"invalid raw_num_samples in dataloader metadata: {raw_num_samples}"
        )
        assert consumed_samples >= 0, (
            f"invalid consumed_samples in dataloader metadata: {consumed_samples}"
        )
        assert 0 <= skipped_samples <= consumed_samples, (
            f"invalid skipped_samples={skipped_samples}, "
            f"consumed_samples={consumed_samples}"
        )

        self._assert_global_sample_counts(
            raw_num_samples=raw_num_samples,
            consumed_samples=consumed_samples,
            skipped_samples=skipped_samples,
        )

        target_samples = math.ceil(raw_num_samples * self.config.training.num_train_epoches)
        return raw_num_samples, consumed_samples, skipped_samples, target_samples

    def _assert_global_sample_counts(self, **counts: int) -> None:
        values = torch.tensor(list(counts.values()), dtype=torch.int64)
        bounds = torch.cat((values, -values))
        torch.distributed.all_reduce(
            bounds,
            op=torch.distributed.ReduceOp.MIN,
            group=cpu_group(),
        )
        min_values = bounds[:len(values)]
        max_values = -bounds[len(values):]
        assert torch.equal(min_values, max_values), (
            f"sample counts differ across ranks: names={list(counts)}, "
            f"min={min_values.tolist()}, max={max_values.tolist()}"
        )

    @override
    async def _train_loop(self):
        self.setup_profile()
        timers = TimerSingleton.get_timer()
        training_config = self.config.training
        assert training_config.gradient_accumulation_steps is not None
        assert training_config.total_training_step is not None
        num_microbatches = int(training_config.gradient_accumulation_steps)
        assert self.train_step is not None
        train_step = int(self.train_step)
        collected_metrics = []

        raw_num_samples, consumed_samples, skipped_samples, target_samples = (
            self._load_sample_progress()
        )

        cpu_barrier()
        self.train_iter = iter(self.train_dataloader)
        exit_step = training_config.exit_step
        scheduler_estimate_warning_logged = False
        while consumed_samples < target_samples and not (
            exit_step is not None and exit_step >= 1 and train_step >= exit_step
        ):
            if (
                not scheduler_estimate_warning_logged and
                train_step >= training_config.total_training_step
            ):
                if is_last_rank():
                    log(
                        "WARNING: actual packed-pretraining steps reached the estimated "
                        f"total_training_step={training_config.total_training_step}; "
                        "the configured LR/WD schedule may finish before sample progress."
                    )
                scheduler_estimate_warning_logged = True
            await asyncio.sleep(0)
            timers("train_step_total", log_level=0).start(barrier=True)
            self.last_progress_time = time.time()

            timers("get_batched_data", log_level=0).start(barrier=True)
            # Flat packed dicts on CPU; H2D happens per-mb in pretrain_packed.
            microbatches: List[Dict[str, Any]] = [
                next(self.train_iter) for _ in range(num_microbatches)
            ]
            timers("get_batched_data").stop()

            if self.config.debug.save_every_rollout_data or (
                self.config.debug.save_first_rollout_data and train_step == 0
            ):
                save_data(
                    microbatches,
                    "debug-tmp",
                    f"{self.train_log_tag.lower()}_batches_{train_step}_"
                    f"{torch.distributed.get_rank()}.pt",
                )

            timers("train_step", log_level=0).start(barrier=True)
            self.profile_start(train_step)

            metric: Dict[str, Any] = {
                "pretrain/num_skipped_samples_sum":
                    sum(int(microbatch["num_skipped_samples"]) for microbatch in microbatches),
            }
            if not training_config.skip_train_step:
                metric.update(
                    self.model_engine.pretrain_step(microbatches, num_microbatches, train_step)
                )
            else:
                metric["pretrain/num_samples_sum"] = sum(
                    int(microbatch["cu_seqlens_padded"].shape[0] - 1) for microbatch in microbatches
                )
            self.profile_end(train_step)
            timers("train_step").stop()

            if clear_gathered_routing_info is not None:
                clear_gathered_routing_info()

            timers("train_step_total").stop()
            time_log_keys = ["get_batched_data", "train_step", "train_step_total"]
            metric = record_time_to_metrics(timers, time_log_keys, metric, reset=True)
            mfu, avg_mfu = self.flops_counter_calc(
                train_step,
                microbatches,
                metric["time_perf/train_step"],
                metric.get("pretrain/seq_length", training_config.seq_length),
            )
            if mfu is not None:
                metric["pretrain/mfu"] = mfu
                metric["pretrain/avg_mfu"] = avg_mfu
            metric = self.model_engine.reduce_metrics_across_data_parallel_group(metric)

            step_trained_samples = int(metric["pretrain/num_samples_sum"])
            step_skipped_samples = int(metric["pretrain/num_skipped_samples_sum"])
            step_consumed_samples = step_trained_samples + step_skipped_samples
            consumed_samples += step_consumed_samples
            skipped_samples += step_skipped_samples
            consumed_epochs = consumed_samples / raw_num_samples
            self.train_dataloader.update_checkpoint_metadata(
                raw_num_samples=raw_num_samples,
                consumed_samples=consumed_samples,
                skipped_samples=skipped_samples,
            )
            metric["pretrain/step_consumed_samples"] = step_consumed_samples
            metric["pretrain/total_consumed_samples"] = consumed_samples
            metric["pretrain/total_skipped_samples"] = skipped_samples

            if is_last_rank():
                log_prefix = (
                    f"[{self.train_log_tag}] training train_step "
                    f"{train_step}/{training_config.total_training_step} "
                    f"epoch {consumed_epochs:.6f}/{float(training_config.num_train_epoches):g}"
                )
                TrainReporterSingleton.log_and_report(metric, train_step, log_prefix=log_prefix)
            if self.config.debug.trainer_return_ppo_step_metrics:
                collected_metrics.append(metric)
            cpu_barrier()

            train_step += 1
            if (
                train_step % training_config.save_interval == 0 and
                not self.config.debug.disable_save_checkpoint
            ):
                self._assert_global_sample_counts(
                    consumed_samples=consumed_samples,
                    skipped_samples=skipped_samples,
                )
                self.model_engine.save_checkpoint(train_step, dataloader=self.train_dataloader)

        if self.compact_thread is not None:
            self.compact_thread.join()

        self.train_step_finished = True
        cpu_barrier()
        if not self.config.debug.disable_save_checkpoint:
            if train_step % training_config.save_interval != 0:
                self.model_engine.save_checkpoint(train_step, dataloader=self.train_dataloader)

        if is_last_rank():
            TrainReporterSingleton.finish()

        return collected_metrics
