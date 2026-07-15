import os
import time
from typing import Dict

import ray
import torch
import torch.distributed

from megatron.core import mpu

from gpatch_v4.core.parallel_state import cpu_barrier, is_last_rank, is_mp_and_cp_head
from gpatch_v4.core.smart_pad_helper import (
    DPBalanceHelper,
    smart_pad_train_get_reorder_rollout_batches,
)
from gpatch_v4.utils import (
    BroadcastUtils,
    FilterSamplingRegistry,
    TimerSingleton,
    TrainReporterSingleton,
    check_rollout_batches,
    clear_memory,
    display_rollout_generation,
    expand_rollout_batches,
    extend_value_to_dict,
    get_iterator_k_split_list,
    log,
    logging_memory_usage,
    logging_memory_usage_details,
    logging_rank0,
    record_time_to_metrics,
    reduce_metrics,
    save_data,
    sync_cuda_and_get_time,
)

from .grpo_train_actor import GrpoTrainActor
from .mixin import OnloadManager


class GrpoAsyncTrainActor(GrpoTrainActor):
    """Async rollout GRPO train actor (single-controller mode).

    Inherits ``GrpoTrainActor`` to reuse all init/setup/utility code.
    In single-controller mode, rollout I/O is centralized in the
    driver-side ``RolloutController``. This actor only handles GPU-bound
    processing on controller-provided rollout shards plus PPO training.
    """
    def validated_config(self):
        super().validated_config()

        # more check
        rollout_gbs = self.config.training.rollout_gbs
        keep_n = self.config.training.sampling_keep_n
        rb_m = getattr(self.config.training, "rb_multiplier", 1)
        assert rollout_gbs * keep_n * rb_m % self.config.training.train_gbs == 0, \
            f"{rollout_gbs=} * {keep_n=} * {rb_m=} % {self.config.training.train_gbs=} != 0"

    # ------------------------------------------------------------------ #
    #  State management
    # ------------------------------------------------------------------ #
    async def init_train_state(self):
        """Prepare per-run state and return training schedule info.

        Returns
        -------
        dict
            Training schedule: total steps, steps per epoch, etc.
        """
        self._overlap_ts: Dict[int, Dict[str, float]] = {}

        # Verify swap state after setup_model_and_optimizer:
        # policy model + optimizer on GPU, ref_model offloaded.
        ps = self.policy_engine.get_swap_state()
        assert ps.model, f"policy model should be onloaded, got {ps}"
        assert ps.optimizer, f"policy optimizer should be onloaded, got {ps}"
        assert not ps.ref_model, f"policy ref_model should be offloaded, got {ps}"
        if self.require_critic_model():
            cs = self.critic_engine.get_swap_state()
            assert not cs.model, f"critic model should be offloaded, got {cs}"
            assert not cs.optimizer, f"critic optimizer should be offloaded, got {cs}"

        return {
            "total_ppo_step": self.config.training.total_ppo_step,
            "ppo_step_per_epoch": self.config.training.ppo_step_per_epoch,
            "num_train_epoches": self.config.training.num_train_epoches,
            "prev_ppo_step": self.prev_ppo_step,
        }

    def get_overlap_stats(self):
        """Return recorded timestamps for overlap verification (testing purpose).

        Returns
        -------
        dict[int, dict[str, float]]
            ``{ppo_step: {"fire": t, "gen_done": t, "train_done": t}}``.
        """
        return dict(self._overlap_ts)

    # ------------------------------------------------------------------ #
    #  Process rollout from controller-provided ObjectRef
    # ------------------------------------------------------------------ #

    def _process_rollout_from_ref(self, ppo_step, dp_refs):
        """Process rollout data provided by the ``RolloutController``.

        Instead of reading from the dataloader / sampler / RM (which are
        now centralized in the controller), this method receives pre-split
        DP shard data via ``dp_refs`` and only performs GPU-bound work:
        broadcast to TP/PP non-head ranks, filter sampling, logps
        computation, and PPO data generation.

        Parameters
        ----------
        ppo_step : int
        dp_refs : list[ray.ObjectRef]
            One ``ObjectRef`` per DP rank, each containing a list of
            rollout-batch dicts for that rank.

        Returns
        -------
        tuple[list[dict], dict]
            ``(rollout_batches, metrics)``.
        """
        if self.config.debug.skip_rollout_load_from_disk:
            load_path = self.config.debug.load_rollout_path
            load_step = self.config.debug.load_rollout_step
            rank = torch.distributed.get_rank()
            rb_path = os.path.join(load_path, f"rollout_batches_{load_step}_{rank}.pt")
            m_path = os.path.join(load_path, f"rollout_metrics_{load_step}_{rank}.pt")
            logging_rank0(f"[DEBUG] load rollout from {rb_path}, {m_path}")
            rollout_batches = torch.load(rb_path, weights_only=False)
            metrics = torch.load(m_path, weights_only=False)
            # Normally compute_log_probs() onloads the policy model back to GPU
            # (and intentionally does not offload it because train follows).
            # We skipped compute_log_probs, so do the onload ourselves to keep
            # the same invariant — otherwise rl_train_actor will hit illegal
            # memory access on the still-offloaded model.
            self.policy_engine.onload_model()
            cpu_barrier()
            return rollout_batches, metrics

        timers = TimerSingleton.get_timer()
        training_config = self.config.training

        # Each DP rank picks its own shard
        dp_rank = mpu.get_data_parallel_rank()
        if is_mp_and_cp_head():
            rbs = ray.get(dp_refs[dp_rank].inner_data)
        else:
            num_microbatches = (
                self.get_num_rollout_micro_batches() * getattr(training_config, "rb_multiplier", 1)
            )
            rbs = [None for _ in range(num_microbatches)]

        # ---- broadcast to TP/PP non-head ranks ----
        logging_memory_usage_details("memory tracking before bcast data (from_ref)", rank=0)
        rollout_batches = BroadcastUtils.broadcast_rollout_batch(rbs)
        clear_memory()
        logging_memory_usage_details("memory tracking after bcast data (from_ref)", rank=0)

        # NOTE: add_back_rollout_attr_after_sampling is already done
        # inside RolloutController.generate(), so we skip it here.
        assert check_rollout_batches(rollout_batches), "rollout_batches fmt error"

        # ---- filter sampling ----
        if training_config.sampling_keep_n != training_config.sampling_repeat_n:
            strategy = training_config.sampling_keeping_strategy
            filter_fn = FilterSamplingRegistry.get(strategy)
            rollout_batches = filter_fn(
                self.config,
                rollout_batches,
                training_config.sampling_repeat_n,
                training_config.sampling_keep_n,
            )

        # ---- compute logps (requires GPU) ----
        samples_per_batch = training_config.rollout_mbs * training_config.sampling_keep_n
        for data in rollout_batches:
            ll = len(data["tokens"])
            data["src_dp"] = [torch.tensor(mpu.get_data_parallel_rank())] * ll

        restore_info = None
        if getattr(self.config.policy, 'balance_dp_seqlen', False):
            rebalanced_batches, restore_info = DPBalanceHelper.rebalance_for_compute_log_probs(
                rollout_batches,
                samples_per_batch,
                add_custom_keys=getattr(self.config.task, "add_custom_keys", None),
            )
            origin_rollout_batches = rollout_batches
            rollout_batches = rebalanced_batches

        skip_prev = self.config.ppo.skip_prev_logps
        timers("compute_logps", log_level=0).start(barrier=True)
        if self.config.policy.dist_config.dynamic_context_parallel:
            ref_logprobs, prev_logprobs = self.policy_engine.compute_log_probs_dynamic_cp(
                rollout_batches,
                compute_pre_logps=not skip_prev,
            )
        else:
            ref_logprobs, prev_logprobs = self.policy_engine.compute_log_probs(
                rollout_batches,
                compute_pre_logps=not skip_prev,
            )
        cpu_barrier()
        timers("compute_logps").stop()

        if not self.config.policy.without_ref:
            for rb, ref_logps in zip(rollout_batches, ref_logprobs):
                rb["ref_logprobs"] = ref_logps
        if skip_prev:
            for rb in rollout_batches:
                rb["logprobs"] = [
                    torch.zeros(len(t) - 1, dtype=torch.float32) for t in rb["tokens"]
                ]
        else:
            for rb, prev_logps in zip(rollout_batches, prev_logprobs, strict=True):
                rb["logprobs"] = prev_logps

        if getattr(self.config.policy, 'balance_dp_seqlen', False):
            rollout_batches = DPBalanceHelper.restore_log_probs_to_original_batches(
                origin_rollout_batches,
                rollout_batches,
                restore_info,
                without_ref=self.config.policy.without_ref,
            )

        clear_memory()

        # ---- PPO data (advantages, etc.) ----
        expected_mbs = training_config.rollout_mbs * training_config.sampling_keep_n
        for rb in rollout_batches:
            assert len(rb['tokens']) == expected_mbs
        assert check_rollout_batches(rollout_batches)

        rollout_metrics = self.compute_rollout_metrics(rollout_batches)
        cpu_barrier()
        self.maybe_calculate_values(rollout_batches)
        timers("generate_ppo_data", log_level=0).start(barrier=True)
        rollout_batches, ppo_metrics = self.generate_ppo_data(rollout_batches)
        cpu_barrier()
        timers("generate_ppo_data").stop()
        metrics = rollout_metrics | ppo_metrics
        display_rollout_generation(self.tokenizer, self.disp_rng, rollout_batches)

        if self.config.debug.save_every_rollout_data or (
            self.config.debug.save_first_rollout_data and ppo_step == 0
        ):
            save_data(
                rollout_batches, "debug-tmp",
                f"rollout_batches_{ppo_step}_{torch.distributed.get_rank()}.pt"
            )
            save_data(
                metrics, "debug-tmp",
                f"rollout_metrics_{ppo_step}_{torch.distributed.get_rank()}.pt"
            )

        return rollout_batches, metrics

    # ------------------------------------------------------------------ #
    #  PPO training
    # ------------------------------------------------------------------ #

    async def train_step(self, epoch, ppo_step, dp_refs, extra_metrics=None):
        """Process controller-provided rollout data and run PPO training.

        Parameters
        ----------
        epoch : int
        ppo_step : int
        dp_refs : list[ray.ObjectRef]
            One ``ObjectRef`` per DP rank, each containing a list of
            rollout-batch dicts for that rank.
        extra_metrics : dict or None
            Additional metrics from the driver (e.g. rollout timing)
            to include in wandb reporting.

        Returns
        -------
        dict
            Reduced output metrics for this step.
        """
        timers = TimerSingleton.get_timer()
        begin_time = sync_cuda_and_get_time()
        training_config = self.config.training

        timers("process_rollout", log_level=0).start(barrier=True)
        rollout_batches, metrics = self._process_rollout_from_ref(ppo_step, dp_refs)
        cpu_barrier()
        timers("process_rollout").stop()

        rollout_gbs = training_config.rollout_gbs
        keep_n = training_config.sampling_keep_n
        rb_m = getattr(training_config, "rb_multiplier", 1)

        expanded_rbs = expand_rollout_batches(rollout_batches)
        total_samples = rollout_gbs * keep_n * rb_m
        assert len(expanded_rbs) * mpu.get_data_parallel_world_size() == total_samples

        if getattr(self.config.policy, 'balance_dp_seqlen', False):
            expanded_rbs = DPBalanceHelper.rebalance_row_batches_for_train(
                expanded_rbs,
                add_custom_keys=getattr(self.config.task, "add_custom_keys", None),
            )

        if self.config.policy.smart_pad_train:
            num_global_batch = rollout_gbs * keep_n * rb_m // training_config.train_gbs
            expanded_rbs = smart_pad_train_get_reorder_rollout_batches(
                expanded_rbs,
                num_global_batch,
                training_config.train_gbs // mpu.get_data_parallel_world_size(),
                training_config.pad_to_mulitiple_of,
                reorder_seed=ppo_step,
            )

        timers("train_step", log_level=0).start(barrier=True)
        if self.require_critic_model():
            logging_rank0("train value model")
            with OnloadManager(self.critic_engine, True):
                for _ in range(training_config.ppo_max_epochs_2):
                    num_train_global_steps = rollout_gbs * keep_n * rb_m // training_config.train_gbs
                    ppo_step_iters = get_iterator_k_split_list(expanded_rbs, num_train_global_steps)
                    _metrics = self.critic_engine.rl_train_value(ppo_step_iters)
                    extend_value_to_dict(metrics, _metrics)
                lr = self.critic_engine.step_and_get_lr()
                metrics["value/lr"] = lr
            cpu_barrier()
            logging_rank0("train value model done")

        self.policy_engine.onload_optimizer()
        logging_rank0("train policy model")
        for _ in range(training_config.ppo_max_epochs_2):
            if ppo_step >= self.get_critic_model_warmup_step():
                num_train_global_steps = rollout_gbs * keep_n * rb_m // training_config.train_gbs
                ppo_step_iters = get_iterator_k_split_list(expanded_rbs, num_train_global_steps)
                _metrics = self.policy_engine.rl_train_actor(ppo_step_iters)
                extend_value_to_dict(metrics, _metrics)
        cpu_barrier()
        logging_rank0("train policy model done")
        timers("train_step").stop()

        lr = self.policy_engine.step_and_get_lr()
        metrics["policy/lr"] = lr
        end_time = sync_cuda_and_get_time()
        metrics["time_perf/total_time"] = end_time - begin_time
        output_metrics = reduce_metrics(metrics)

        time_log_keys = [
            "process_rollout",
            "rollout",
            "compute_logps",
            "generate_ppo_data",
            "train_step",
            "sampler_generate",
            "gen_rm_generate",
            "bt_rm_generate",
        ]
        output_metrics = record_time_to_metrics(timers, time_log_keys, output_metrics, reset=True)

        if extra_metrics:
            output_metrics.update(extra_metrics)
            if "time_perf/rollout" in extra_metrics:
                output_metrics["time_perf/total_time"] += extra_metrics["time_perf/rollout"]

        if is_last_rank():
            log_prefix = (
                f"training ppo_step {ppo_step}/{training_config.total_ppo_step} "
                f"epoch {epoch}"
            )
            TrainReporterSingleton.log_and_report(output_metrics, ppo_step, log_prefix=log_prefix)
        clear_memory()
        cpu_barrier()

        self._overlap_ts.setdefault(ppo_step, {})["train_done"] = time.monotonic()

        if self.config.debug.trainer_return_ppo_step_metrics:
            return output_metrics
        return {}
