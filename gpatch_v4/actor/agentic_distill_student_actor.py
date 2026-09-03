import time
from typing import Dict

import ray
import torch
import torch.distributed
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.actor.grpo_async_train_actor import GrpoAsyncTrainActor
from gpatch_v4.actor.mixin import OnloadManager
from gpatch_v4.client import TeacherClient
from gpatch_v4.core.parallel_state import cpu_barrier, is_last_rank, is_mp_and_cp_head
from gpatch_v4.core.smart_pad_helper import (
    DPBalanceHelper,
    smart_pad_train_get_reorder_rollout_batches,
)
from gpatch_v4.rollout_generator import RolloutGeneratorFactory
from gpatch_v4.utils import (
    BroadcastUtils,
    FilterSamplingRegistry,
    TimerSingleton,
    TrainReporterSingleton,
    check_rollout_batches,
    clear_memory,
    expand_rollout_batches,
    extend_value_to_dict,
    get_iterator_k_split_list,
    is_same_tokenizer,
    log,
    logging_memory_usage_details,
    logging_rank0,
    record_time_to_metrics,
    reduce_metrics,
    save_data,
    sync_cuda_and_get_time,
)


class AgenticDistillStudentActor(GrpoAsyncTrainActor):
    """Train actor for agentic on-policy distillation under single-controller rollout.

    Bridges two existing paths:
    - ``GrpoAsyncTrainActor`` (single-controller agentic rollout: env loop +
      RolloutController feeds DP shards in via ``dp_refs``).
    - ``DistillStudentActor`` (sets up teacher clients and computes
      teacher log-probs as ``teacher_logprobs_{name}`` on each rollout
      batch).

    Teacher log-prob fan-out happens AFTER the controller's rollout has
    produced sequences and AFTER the student forward pass, BEFORE
    ``generate_ppo_data`` runs advantage calculation — same order as
    ``DistillStudentActor.rollout``.
    """

    @override
    async def init(self, config):
        await super().init(config)
        is_same_tokenizer(self.tokenizer, self.teacher_tokenizer)
        is_same_tokenizer(self.tokenizer, self.sampler_tokenizers[0])
        self._teacher_sample_idx = 0

    @override
    def validated_config(self):
        super().validated_config()
        g_opd_use_base_model = getattr(self.config.ppo, "g_opd_use_base_model", False)
        if g_opd_use_base_model:
            self.config.policy.without_ref = False
            log(
                f"G-OPD enabled: ref model will be loaded as base model (π_base), "
                f"lambda={self.config.ppo.g_opd_lambda}",
                rank=0,
            )
        else:
            self.config.policy.without_ref = True
        if getattr(self.config.ppo, "g_opd_mix_reward_advantage", False):
            assert self.config.training.use_bt_rm_reward, (
                "g_opd_mix_reward_advantage is True, but use_bt_rm_reward is False"
            )
        # 3D top-K teacher logps require the DistillStudentActor non-async
        # code path; the agentic single-controller path only supports 2D
        # label-based OPD.
        assert self.config.ppo.log_prob_top_k == 0, (
            "AgenticDistillStudentActor currently only supports log_prob_top_k=0 "
            "(2D label-based OPD)."
        )

    @override
    async def setup_client(self):
        await super().setup_client()
        self.teacher_clients: Dict[str, TeacherClient] = {}
        for t_name, t_cfg in self.config.teachers.items():
            self.teacher_clients[t_name] = TeacherClient(
                self.config, teacher_name=t_name, teacher_config=t_cfg
            )
        log(f"Teacher clients created: {list(self.teacher_clients.keys())}", rank=0)

    @override
    async def setup_rollout_generator(self):
        # The controller drives rollout, but we still build a rollout
        # generator here so that ``calc_all_teacher_logps`` / sample_idx
        # bookkeeping live in one place.
        extra_kwargs = {"teacher_clients": self.teacher_clients}
        self.train_rollout_generator = RolloutGeneratorFactory.get_rollout_generator(
            self.config,
            self.sampler_client,
            self.gen_rm_client,
            self.bt_rm_client,
            **extra_kwargs,
        )

    def _normalize_metric_keys(self, rollout_batches):
        """Give every microbatch the same ``metrics_report`` keys.

        Multi-env rollout mixes microbatches from different envs, each emitting
        its own env-specific metric keys (e.g. moment's ``reward_judge`` vs
        frozenlake's ``format_penalty``). ``compute_rollout_metrics`` derives its
        reduced key set from ``rollout_batches[0]`` and then (a) asserts every
        batch carries those keys and (b) builds a fixed-length tensor for an
        ``all_reduce`` over the DP group — so ranks/batches with different key
        sets would assert or deadlock the collective. Normalising to the config
        ``metrics_report`` list (identical on every rank) makes the reduced set
        rank-deterministic; any listed key missing from a batch is filled with
        per-sample ``0.0`` so it reduces to a harmless mean.
        """
        report_keys = list(getattr(self.config.training, "metrics_report", None) or [])
        if not report_keys:
            return
        for rb in rollout_batches:
            n = len(rb["tokens"])
            for key in report_keys:
                present_key = self._metric_presence_key(key)
                key_was_present = key in rb
                if present_key not in rb:
                    rb[present_key] = [
                        torch.tensor(float(key_was_present), dtype=torch.float) for _ in range(n)
                    ]
                if not key_was_present:
                    rb[key] = [torch.tensor(0.0, dtype=torch.float) for _ in range(n)]

    @staticmethod
    def _metric_presence_key(metric_name):
        return f"__metric_present__{metric_name}"

    def _strip_metric_presence_keys(self, rollout_batches):
        for rb in rollout_batches:
            for key in list(rb.keys()):
                if key.startswith("__metric_present__"):
                    rb.pop(key, None)

    def _filter_global_missing_metrics(self, rollout_metrics, globally_reportable_keys):
        """Drop global metric series that exist only because of zero-fill."""
        report_keys = list(getattr(self.config.training, "metrics_report", None) or [])
        if not report_keys:
            return rollout_metrics
        filtered = dict(rollout_metrics)
        for key in report_keys:
            if key in globally_reportable_keys:
                continue
            prefix = "rollout-rewards" if "reward" in key else "rollout-metrics"
            filtered.pop(f"{prefix}/global_{key}", None)
        return filtered

    def _compute_per_env_metrics(self, rollout_batches):
        """Per-env breakdown of ``metrics_report`` keys, keyed by ``teacher_type``.

        ``compute_rollout_metrics`` all-reduces every metric over the whole DP
        group and divides by the *global* sample count, so in a multi-env run its
        numbers mix envs together — and env-specific keys (main_tool's
        ``is_positive_sample``, frozenlake's ``format_penalty``) get diluted by the
        0.0 fills that ``_normalize_metric_keys`` adds to the other env's samples.

        This method instead groups samples by their per-sample ``teacher_type``
        (stamped by the env manager, one env per sample) and reduces each metric
        within its env, dividing by the metric's real present count. Output keys are
        ``rollout-rewards/{env}/{metric}`` (keys containing ``reward``) or
        ``rollout-metrics/{env}/{metric}`` — mirroring the global prefixes.
        A metric is emitted for an env only when at least one sample from that
        env really carried the key before ``_normalize_metric_keys`` zero-filled
        it for distributed collective safety.

        The trajectory-level ``rewards`` (env ``step`` reward summed over turns,
        the SAME signal ``compute_rollout_metrics`` reports as global_rewards) is
        always included even though it is not in ``metrics_report`` — otherwise a
        env whose only reward signal is ``rewards`` (e.g. frozenlake, which emits
        no ``raw_reward`` metric) would have no per-env reward series at all.

        Collective safety: the env list is ``sorted(config.teachers)`` (identical
        on every rank) and the accumulator layout is fixed, so all DP ranks issue
        one all_reduce of the same length regardless of which envs they happened to
        sample. Returns ``{}`` for single-teacher / non-multi-env runs.
        """
        teachers = getattr(self.config, "teachers", None) or {}
        env_names = sorted(teachers.keys())
        if len(env_names) <= 1:
            return {}, set(getattr(self.config.training, "metrics_report", None) or [])
        # Always track trajectory ``rewards`` (the real env reward) first, then the
        # configured metrics_report keys. ``rewards`` is set on every rollout_batch
        # by traj_env_manager, so it is present for every env (unlike raw_reward,
        # which is main_tool-specific and 0.0-filled for frozenlake).
        report_keys = ["rewards"] if "rewards" in rollout_batches[0] else []
        for k in (getattr(self.config.training, "metrics_report", None) or []):
            if k in rollout_batches[0] and k not in report_keys:
                report_keys.append(k)
        if not report_keys:
            return {}, set()

        env_idx = {name: i for i, name in enumerate(env_names)}
        n_env = len(env_names)
        n_key = len(report_keys)
        # Layout per env:
        # [env_count, sum(key_0), present_count(key_0), sum(key_1), ...]
        stride = 1 + 2 * n_key
        acc = torch.zeros(n_env * stride, dtype=torch.float64)

        for rb in rollout_batches:
            tt = rb.get("teacher_type")
            n = len(rb["tokens"])
            # Per-sample teacher_type list; fall back to a single stamp broadcast
            # to all samples in the microbatch (single-env-per-worker guarantees
            # one env per batch, so a length-1 list still maps cleanly).
            if isinstance(tt, list) and len(tt) == n:
                sample_envs = tt
            elif isinstance(tt, list) and len(tt) >= 1:
                sample_envs = [tt[0]] * n
            else:
                sample_envs = [tt] * n

            stacked = {k: torch.stack(rb[k]).view(-1).to(torch.float64) for k in report_keys}
            present = {}
            for k in report_keys:
                if k == "rewards":
                    present[k] = torch.ones(n, dtype=torch.float64)
                    continue
                present_key = self._metric_presence_key(k)
                if present_key in rb:
                    present[k] = torch.stack(rb[present_key]).view(-1).to(torch.float64)
                else:
                    present[k] = torch.ones(n, dtype=torch.float64)
            for s in range(n):
                e = env_idx.get(sample_envs[s])
                if e is None:
                    continue
                base = e * stride
                acc[base] += 1.0
                for ki, k in enumerate(report_keys):
                    p = float(present[k][s].item())
                    if p <= 0.0:
                        continue
                    key_base = base + 1 + 2 * ki
                    acc[key_base] += stacked[k][s].item()
                    acc[key_base + 1] += p

        acc = acc.to(torch.cuda.current_device())
        torch.distributed.all_reduce(acc, group=mpu.get_data_parallel_group())
        acc = acc.tolist()

        metrics = {}
        globally_reportable_keys = set()
        for e, name in enumerate(env_names):
            base = e * stride
            count = acc[base]
            metrics[f"rollout-metrics/{name}/num_samples"] = count
            if count <= 0:
                continue
            for ki, k in enumerate(report_keys):
                key_base = base + 1 + 2 * ki
                present_count = acc[key_base + 1]
                if present_count <= 0:
                    continue
                prefix = "rollout-rewards" if "reward" in k else "rollout-metrics"
                metrics[f"{prefix}/{name}/{k}"] = acc[key_base] / present_count

        active_env_bases = [
            e * stride for e in range(n_env)
            if acc[e * stride] > 0
        ]
        for ki, k in enumerate(report_keys):
            if k == "rewards":
                continue
            present_in_all_active_envs = active_env_bases and all(
                acc[base + 1 + 2 * ki + 1] > 0 for base in active_env_bases
            )
            if present_in_all_active_envs:
                globally_reportable_keys.add(k)
        return metrics, globally_reportable_keys

    async def _process_rollout_from_ref_async(self, ppo_step, dp_refs):
        """Async sibling of the parent's sync ``_process_rollout_from_ref``.

        Identical to the parent except for an inserted teacher fan-out
        before ``generate_ppo_data``. Cannot reuse the parent verbatim
        because teacher RPCs are async.
        """
        if self.config.debug.skip_rollout_load_from_disk:
            return self._process_rollout_from_ref(ppo_step, dp_refs)

        timers = TimerSingleton.get_timer()
        training_config = self.config.training

        dp_rank = mpu.get_data_parallel_rank()
        if is_mp_and_cp_head():
            rbs = ray.get(dp_refs[dp_rank].inner_data)
        else:
            num_microbatches = (
                self.get_num_rollout_micro_batches() *
                getattr(training_config, "rb_multiplier", 1)
            )
            rbs = [None for _ in range(num_microbatches)]

        logging_memory_usage_details("memory tracking before bcast data (from_ref)", rank=0)
        rollout_batches = BroadcastUtils.broadcast_rollout_batch(rbs)
        clear_memory()
        logging_memory_usage_details("memory tracking after bcast data (from_ref)", rank=0)

        assert check_rollout_batches(rollout_batches), "rollout_batches fmt error"

        if training_config.sampling_keep_n != training_config.sampling_repeat_n:
            strategy = training_config.sampling_keeping_strategy
            filter_fn = FilterSamplingRegistry.get(strategy)
            rollout_batches = filter_fn(
                self.config,
                rollout_batches,
                training_config.sampling_repeat_n,
                training_config.sampling_keep_n,
            )

        samples_per_batch = training_config.rollout_mbs * training_config.sampling_keep_n
        for data in rollout_batches:
            ll = len(data["tokens"])
            data["src_dp"] = [torch.tensor(mpu.get_data_parallel_rank())] * ll

        # Multi-env: harmonize per-sample metric keys BEFORE the DP rebalance.
        # The rebalance exchanges samples across ranks and serializes a fixed
        # tensor-key set derived from rank0's first microbatch
        # (dp_balancing._classify_sample_keys); if an env-specific key such as
        # main_tool's ``is_positive_sample`` is in that set but absent from a
        # frozenlake sample being packed, ``_serialize_samples_to_buffer`` raises
        # KeyError. Filling every ``metrics_report`` key with a 0.0 tensor here
        # makes every microbatch's key set identical, so the cross-env exchange is
        # safe. (Also re-run after teacher fan-out below for the metrics path.)
        self._normalize_metric_keys(rollout_batches)

        restore_info = None
        if getattr(self.config.policy, 'balance_dp_seqlen', False):
            rebalanced_batches, restore_info = DPBalanceHelper.rebalance_for_compute_log_probs(
                rollout_batches,
                samples_per_batch,
                add_custom_keys=self.config.policy.dp_balance_extra_keys,
            )
            origin_rollout_batches = rollout_batches
            rollout_batches = rebalanced_batches

        skip_prev = self.config.ppo.skip_prev_logps
        timers("compute_logps", log_level=0).start(barrier=True)
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

        # ---- teacher logps fan-out (OPD-only addition) ----
        timers("compute_teacher_logps", log_level=0).start(barrier=True)
        num_microbatches = len(rollout_batches)
        rollout_batches = await self.train_rollout_generator.calc_all_teacher_logps(
            rollout_batches,
            num_microbatches,
            ppo_step,
            sample_idx_base=self._teacher_sample_idx,
        )
        self._teacher_sample_idx += num_microbatches
        cpu_barrier()
        timers("compute_teacher_logps").stop()

        rollout_batches = BroadcastUtils.broadcast_rollout_batch(rollout_batches)

        expected_mbs = training_config.rollout_mbs * training_config.sampling_keep_n
        for rb in rollout_batches:
            assert len(rb['tokens']) == expected_mbs
        assert check_rollout_batches(rollout_batches)

        self._normalize_metric_keys(rollout_batches)
        rollout_metrics = self.compute_rollout_metrics(rollout_batches)
        # Multi-env: add per-env (per-teacher) breakdown of the same metrics. Runs
        # on the same all-DP-ranks-have-data invariant as compute_rollout_metrics
        # (post-broadcast), so its all_reduce is collective-safe. No-op single-env.
        per_env_metrics, globally_reportable_keys = self._compute_per_env_metrics(rollout_batches)
        rollout_metrics = self._filter_global_missing_metrics(
            rollout_metrics, globally_reportable_keys
        )
        if per_env_metrics:
            rollout_metrics = rollout_metrics | per_env_metrics
        self._strip_metric_presence_keys(rollout_batches)
        cpu_barrier()
        self.maybe_calculate_values(rollout_batches)
        timers("generate_ppo_data", log_level=0).start(barrier=True)
        rollout_batches, ppo_metrics = self.generate_ppo_data(rollout_batches)
        cpu_barrier()
        timers("generate_ppo_data").stop()
        metrics = rollout_metrics | ppo_metrics

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

    @override
    async def train_step(self, epoch, ppo_step, dp_refs, extra_metrics=None):
        """Mirrors GrpoAsyncTrainActor.train_step, but invokes the async
        rollout-processing path so teacher RPCs can run via ``await``.
        """
        timers = TimerSingleton.get_timer()
        begin_time = sync_cuda_and_get_time()
        training_config = self.config.training

        timers("process_rollout", log_level=0).start(barrier=True)
        rollout_batches, metrics = await self._process_rollout_from_ref_async(
            ppo_step, dp_refs
        )
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
                add_custom_keys=self.config.policy.dp_balance_extra_keys,
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
                    num_train_global_steps = (
                        rollout_gbs * keep_n * rb_m // training_config.train_gbs
                    )
                    ppo_step_iters = get_iterator_k_split_list(
                        expanded_rbs, num_train_global_steps
                    )
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
                num_train_global_steps = (
                    rollout_gbs * keep_n * rb_m // training_config.train_gbs
                )
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
            "compute_teacher_logps",
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
            TrainReporterSingleton.log_and_report(
                output_metrics, ppo_step, log_prefix=log_prefix
            )
        clear_memory()
        cpu_barrier()

        self._overlap_ts.setdefault(ppo_step, {})["train_done"] = time.monotonic()

        if self.config.debug.trainer_return_ppo_step_metrics:
            return output_metrics
        return {}
