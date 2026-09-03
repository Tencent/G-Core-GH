import os
import time
from typing import Any, Dict, List

import ray
import torch
import torch.distributed as dist

from megatron.core import mpu

from gpatch_v4.core.dynamic_batch_advantage import (
    DYNAMIC_BATCH_ADVANTAGE_TYPES,
    compute_dynamic_batch_advantages,
    register_custom_dynamic_advantage,
)
from gpatch_v4.core.parallel_state import cpu_barrier, is_last_rank, is_mp_and_cp_head
from gpatch_v4.utils import (
    BroadcastUtils,
    TimerSingleton,
    TrainReporterSingleton,
    clear_memory,
    extend_value_to_dict,
    logging_rank0,
    record_time_to_metrics,
    reduce_metrics,
    save_data,
    sync_cuda_and_get_time,
)
from gpatch_v4.utils.common_utils import compress_ppo_save_train_data

from .grpo_async_train_actor import GrpoAsyncTrainActor
from .mixin import OnloadManager


class DynBatchGrpoAsyncTrainActor(GrpoAsyncTrainActor):
    """Single-controller GRPO actor consuming dynamic train-step batches.
    This actor support more flexible dynamic batch training like: agentic training,
    filtering out invalid samples, etc.
    TODO(@yeazhao): we will merge this actor into GrpoAsyncTrainActor gradually.
    """
    def validated_config(self):
        super().validated_config()
        assert self.config.training.dynamic_batch_train
        assert not self.config.ppo.use_legacy_loss
        # TODO(@yeazhao): Remove this guard after the dynamic-batch PPO path is covered by tests.
        assert self.config.ppo.advantage_type != "ppo", (
            "dynamic-batch PPO is not supported because this path has not been tested"
        )
        assert self.config.ppo.advantage_type not in {"on_policy_distill", "g_opd"}
        has_custom_advantage = (
            self.config.ppo.custom_advantage_py_path is not None and
            self.config.ppo.custom_advantage_py_name is not None
        )
        assert (
            self.config.ppo.advantage_type in DYNAMIC_BATCH_ADVANTAGE_TYPES or has_custom_advantage
        ), f"unsupported dynamic advantage_type: {self.config.ppo.advantage_type!r}"
        if has_custom_advantage:
            assert self.config.ppo.custom_post_advantage_py_name is None, (
                "dynamic_batch custom advantage does not support custom_post_advantage_py_name"
            )
            register_custom_dynamic_advantage(
                self.config.ppo.advantage_type,
                self.config.ppo.custom_advantage_py_path,
                self.config.ppo.custom_advantage_py_name,
            )
        if self.require_critic_model():
            assert not self.config.critic.dist_config.dynamic_context_parallel, (
                "dynamic-batch PPO critic training does not support dynamic context parallel"
            )
            if self.config.ppo.ppo_initial_policy_kl_penalty > 0:
                assert not self.config.policy.without_ref, (
                    "PPO initial policy KL penalty requires a reference model"
                )
        assert self.config.ppo.loss_func != "steer"
        if not self.config.policy.dist_config.dynamic_context_parallel:
            assert not self.config.policy.smart_pad_train

    def _debug_load_train_steps(self) -> tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
        load_path = self.config.debug.load_rollout_path
        load_step = self.config.debug.load_rollout_step
        rank = dist.get_rank()
        steps_path = os.path.join(load_path, f"sample_train_steps_{load_step}_{rank}.pt")
        metrics_path = os.path.join(load_path, f"sample_train_metrics_{load_step}_{rank}.pt")
        logging_rank0(f"[DEBUG] load train steps from {steps_path}, {metrics_path}")
        samples_by_train_step = torch.load(steps_path, weights_only=False)
        metrics = torch.load(metrics_path, weights_only=False)
        # Normally compute_log_probs() onloads the policy model; we skipped it.
        self.policy_engine.onload_model()
        cpu_barrier()
        return samples_by_train_step, metrics

    def _debug_maybe_save_train_steps(
        self,
        samples_by_train_step: List[List[Dict[str, Any]]],
        metrics: Dict[str, Any],
        ppo_step: int,
    ) -> None:
        if not (
            self.config.debug.save_every_rollout_data or
            (self.config.debug.save_first_rollout_data and ppo_step == 0)
        ):
            return
        save_data(
            samples_by_train_step,
            "debug-tmp",
            f"sample_train_steps_{ppo_step}_{dist.get_rank()}.pt",
        )
        save_data(
            metrics,
            "debug-tmp",
            f"sample_train_metrics_{ppo_step}_{dist.get_rank()}.pt",
        )

    def _process_rollout_from_ref(self, ppo_step, dp_refs):
        if self.config.debug.skip_rollout_load_from_disk:
            return self._debug_load_train_steps()

        timers = TimerSingleton.get_timer()
        dp_rank = mpu.get_data_parallel_rank()
        samples_by_train_step = (
            ray.get(dp_refs[dp_rank].inner_data) if is_mp_and_cp_head() else None
        )
        samples_by_train_step = BroadcastUtils.broadcast_rollout_batch(samples_by_train_step)
        assert isinstance(samples_by_train_step, list) and samples_by_train_step
        assert all(
            isinstance(step_samples, list) and step_samples
            for step_samples in samples_by_train_step
        )

        samples = [sample for step_samples in samples_by_train_step for sample in step_samples]
        # =============== compute prev_logprobs ===============
        skip_prev = self.config.ppo.skip_prev_logps
        timers("compute_logps", log_level=0).start(barrier=True)
        if self.config.policy.dist_config.dynamic_context_parallel:
            ref_logprobs, prev_logprobs = self.policy_engine.compute_log_probs_dynamic_cp(
                samples,
                compute_pre_logps=not skip_prev,
            )
        else:
            ref_logprobs, prev_logprobs = self.policy_engine.compute_log_probs(
                samples,
                compute_pre_logps=not skip_prev,
            )
        cpu_barrier()
        timers("compute_logps").stop()
        # =============== compute ref_logprobs ===============
        if not self.config.policy.without_ref:
            assert ref_logprobs is not None
            for sample, logprobs in zip(samples, ref_logprobs, strict=True):
                sample["ref_logprobs"] = logprobs
        if skip_prev:
            for sample in samples:
                sample["logprobs"] = torch.zeros(
                    len(sample["tokens"]) - 1,
                    dtype=torch.float32,
                )
        else:
            assert prev_logprobs is not None
            for sample, logprobs in zip(samples, prev_logprobs, strict=True):
                sample["logprobs"] = logprobs

        # =============== compute values ===============
        if self.require_critic_model():
            # TODO(@yeazhao): 测试PPO
            self.policy_engine.offload_model()
            values = self.critic_engine.compute_values(samples)
            for sample, value in zip(samples, values, strict=True):
                assert "values" not in sample or sample["values"] is None
                sample["values"] = value

        # =============== compute advantages ===============
        timers("generate_ppo_data", log_level=0).start(barrier=True)
        metrics = {}
        dp_group = mpu.get_data_parallel_group()
        for step_samples in samples_by_train_step:
            step_metrics = compute_dynamic_batch_advantages(
                self.config,
                step_samples,
                dp_group,
            )
            extend_value_to_dict(metrics, step_metrics)
        cpu_barrier()
        timers("generate_ppo_data").stop()
        # TODO @yeazhao: display_rollout_generation
        self._debug_maybe_save_train_steps(samples_by_train_step, metrics, ppo_step)
        clear_memory()
        return samples_by_train_step, metrics

    async def train_step(self, epoch, ppo_step, dp_refs, extra_metrics=None):
        timers = TimerSingleton.get_timer()
        begin_time = sync_cuda_and_get_time()
        training_config = self.config.training

        timers("process_rollout", log_level=0).start(barrier=True)
        samples_by_train_step, metrics = self._process_rollout_from_ref(ppo_step, dp_refs)
        cpu_barrier()
        timers("process_rollout").stop()

        timers("train_step", log_level=0).start(barrier=True)
        if self.require_critic_model():
            logging_rank0("train value model")
            with OnloadManager(self.critic_engine, True):
                for _ in range(training_config.ppo_max_epochs_2):
                    value_metrics = self.critic_engine.rl_train_value(iter(samples_by_train_step))
                    extend_value_to_dict(metrics, value_metrics)
                metrics["value/lr"] = self.critic_engine.step_and_get_lr()
            cpu_barrier()
            self.policy_engine.onload_model()

        self.policy_engine.onload_optimizer()
        logging_rank0("train policy model")
        should_dump = training_config.ppo_dump_metrics_interval > 0 and (
            ppo_step + 1
        ) % training_config.ppo_dump_metrics_interval == 0
        dump_samples = (
            [sample for step_samples in samples_by_train_step
             for sample in step_samples] if should_dump else None
        )
        for epoch_idx in range(training_config.ppo_max_epochs_2):
            if ppo_step >= self.get_critic_model_warmup_step():
                self.policy_engine.should_dump_metrics = should_dump
                policy_metrics = self.policy_engine.rl_train_actor(iter(samples_by_train_step))
                if should_dump:
                    self._save_dumped_metrics(policy_metrics, dump_samples, ppo_step, epoch_idx)
                extend_value_to_dict(metrics, policy_metrics)
        cpu_barrier()
        timers("train_step").stop()

        if should_dump and dist.get_rank() == 0:
            self.compact_thread = compress_ppo_save_train_data(
                self.compact_thread, training_config.ppo_dump_metrics_dir
            )

        metrics["policy/lr"] = self.policy_engine.step_and_get_lr()
        end_time = sync_cuda_and_get_time()
        metrics["time_perf/total_time"] = end_time - begin_time
        output_metrics = reduce_metrics(metrics)
        output_metrics = record_time_to_metrics(
            timers,
            [
                "process_rollout",
                "rollout",
                "compute_logps",
                "generate_ppo_data",
                "train_step",
                "sampler_generate",
                "gen_rm_generate",
                "bt_rm_generate",
            ],
            output_metrics,
            reset=True,
        )
        if extra_metrics:
            output_metrics.update(extra_metrics)
            if "time_perf/rollout" in extra_metrics:
                output_metrics["time_perf/total_time"] += extra_metrics["time_perf/rollout"]

        if is_last_rank():
            TrainReporterSingleton.log_and_report(
                output_metrics,
                ppo_step,
                log_prefix=(
                    f"training ppo_step {ppo_step}/{training_config.total_ppo_step} "
                    f"epoch {epoch}"
                ),
            )
        clear_memory()
        cpu_barrier()
        self._overlap_ts.setdefault(ppo_step, {})["train_done"] = time.monotonic()
        if self.config.debug.trainer_return_ppo_step_metrics:
            return output_metrics
        return {}
