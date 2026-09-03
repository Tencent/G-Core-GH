import asyncio
import inspect
import os
import time
import traceback
from typing import Any, Dict, List

import torch
import torch.distributed
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.actor.grpo_train_actor import GrpoTrainActor
from gpatch_v4.client import TeacherClient
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    init_pg,
    initlize_parallel_state,
    is_last_rank,
    is_mp_and_cp_head,
)
from gpatch_v4.core.smart_pad_helper import (
    DPBalanceHelper,
    smart_pad_train_get_reorder_rollout_batches,
)
from gpatch_v4.extended_pipeline import ExtendPipelineFactory
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.rollout_generator import RolloutGeneratorFactory
from gpatch_v4.training_backend import TrainingEngineFactory
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
    import_fn_from_path,
    init_train_reporter_singleton,
    is_same_tokenizer,
    log,
    logging_meminfo_str,
    logging_memory_usage,
    logging_memory_usage_details,
    logging_rank0,
    record_time_to_metrics,
    reduce_metrics,
    sync_cuda_and_get_time,
    unbind_tensor_to_list,
)
from gpatch_v4.utils.test_utils import save_data
from gpatch_v4.utils.training_utils import compute_topk_overlap_masks, pad_3d_seq_dim

#TODO: 实际上是 on policy distill student actor, 后面找个时间 rename 一下
# 1. 实际上可以直接设置 without_ref, 可以省去一轮 compute logps 的时间，目前还没设置


class DistillStudentActor(GrpoTrainActor):
    async def init(self, config):
        await super().init(config)
        is_same_tokenizer(self.tokenizer, self.teacher_tokenizer)
        is_same_tokenizer(self.tokenizer, self.sampler_tokenizers[0])

    @override
    def validated_config(self):
        super().validated_config()
        g_opd_use_base_model = getattr(self.config.ppo, "g_opd_use_base_model", False)
        if g_opd_use_base_model:
            # G-OPD: keep ref model loaded — it serves as the base model (π_base)
            self.config.policy.without_ref = False
            log(
                f"G-OPD enabled: ref model will be loaded as base model (π_base), "
                f"lambda={self.config.ppo.g_opd_lambda}",
                rank=0,
            )
        else:
            self.config.policy.without_ref = True
        if getattr(self.config.ppo, "g_opd_mix_reward_advantage", False):
            assert self.config.training.use_bt_rm_reward, f"g_opd_mix_reward_advantage is True, but use_bt_rm_reward is False"

    @override
    async def setup_client(self):
        await super().setup_client()
        self.teacher_clients = {}
        for t_name, t_cfg in self.config.teachers.items():
            self.teacher_clients[t_name] = TeacherClient(
                self.config, teacher_name=t_name, teacher_config=t_cfg
            )
        log(f"Teacher clients created: {list(self.teacher_clients.keys())}", rank=0)

    @override
    async def setup_rollout_generator(self):
        extra_kwargs = {"teacher_clients": self.teacher_clients}
        self.train_rollout_generator = RolloutGeneratorFactory.get_rollout_generator(
            self.config,
            self.sampler_client,
            self.gen_rm_client,
            self.bt_rm_client,
            **extra_kwargs,
        )
        self.eval_rollout_generator = RolloutGeneratorFactory.get_rollout_generator(
            self.config,
            self.sampler_client,
            self.gen_rm_client,
            self.bt_rm_client,
            run_eval=True,
            **extra_kwargs,
        )

    @override
    async def _compute_log_probs(
        self,
        rollout_batches: List[Dict[str, Any]],
        training_config,
        effective_keep_n: int,
        timers: TimerSingleton,
        curr_ppo_step: int,
    ) -> List[Dict[str, Any]]:
        samples_per_batch = training_config.rollout_mbs * effective_keep_n
        restore_info = None
        for data in rollout_batches:
            ll = len(data["tokens"])
            src_dp = [torch.tensor(mpu.get_data_parallel_rank())] * ll
            data["src_dp"] = src_dp

        if getattr(self.config.policy, 'balance_dp_seqlen', False):
            rebalanced_batches, restore_info = DPBalanceHelper.rebalance_for_compute_log_probs(
                rollout_batches,
                samples_per_batch,
                add_custom_keys=self.config.policy.dp_balance_extra_keys,
            )
            origin_rollout_batches = rollout_batches
            rollout_batches = rebalanced_batches

        log_prob_top_k = self.config.ppo.log_prob_top_k
        strategy = self.config.ppo.opd_top_k_strategy
        has_ref = not self.config.policy.without_ref
        need_ref_topk_gather = (
            log_prob_top_k > 0 and self.config.ppo.advantage_type == "g_opd" and has_ref
        )

        # --- Determine compute_log_probs arguments based on strategy ---
        compute_logps_kwargs: Dict[str, Any] = {}
        if log_prob_top_k > 0:
            if strategy in ("only_stu", "intersection"):
                compute_logps_kwargs["policy_compute_topk"] = True
                if need_ref_topk_gather:
                    compute_logps_kwargs["ref_gather_ids_key"] = "stu_topk_ids"
            elif strategy == "only_tch":
                # Set canonical per-sample key from teacher's result (generator already ran).
                default_t_name = next(iter(self.train_rollout_generator.teacher_clients))
                for rb in rollout_batches:
                    rb["tch_topk_ids"] = rb[f"teacher_topk_ids_{default_t_name}"]
                compute_logps_kwargs["policy_gather_ids_key"] = "tch_topk_ids"
                if need_ref_topk_gather:
                    compute_logps_kwargs["ref_gather_ids_key"] = "tch_topk_ids"

        timers("compute_logps", log_level=0).start(barrier=True)
        ref_logprobs, prev_logprobs = self.policy_engine.compute_log_probs(
            rollout_batches, **compute_logps_kwargs
        )
        cpu_barrier()
        timers("compute_logps").stop()

        extra_restore_keys: List[str] = []

        if log_prob_top_k > 0:
            if strategy in ("only_stu", "intersection"):
                # Student produced its own topk.
                for rb, prev in zip(rollout_batches, prev_logprobs, strict=True):
                    rb["logprobs"] = [d["logprobs"] for d in prev]
                    rb["prev_topk_logprobs"] = [d["topk_logprobs"] for d in prev]
                    rb["stu_topk_ids"] = [d["topk_ids"] for d in prev]
                extra_restore_keys.extend(["prev_topk_logprobs", "stu_topk_ids"])

                if need_ref_topk_gather:
                    for rb, ref_list in zip(rollout_batches, ref_logprobs, strict=True):
                        rb["ref_logprobs"] = [d["logprobs"] for d in ref_list]
                        rb["base_on_topk_logprobs"] = [d["gather_logprobs"] for d in ref_list]
                    extra_restore_keys.append("base_on_topk_logprobs")

                # generator 跳过了 teacher 调用（需要先有 stu_topk_ids），在此补调。
                # sample_idx_base 回退到 generator 递增前的值以保证 EP 路由正确。
                self.policy_engine.offload_model()
                clear_memory()
                num_microbatches = len(rollout_batches)
                sample_idx_base = self.train_rollout_generator.sample_idx - num_microbatches
                timers("compute_teacher_logps", log_level=0).start(barrier=True)
                rollout_batches = await self.train_rollout_generator.calc_all_teacher_logps(
                    rollout_batches,
                    num_microbatches,
                    curr_ppo_step,
                    sample_idx_base=sample_idx_base,
                )
                cpu_barrier()
                timers("compute_teacher_logps").stop()
                self.policy_engine.onload_model()
                rollout_batches = BroadcastUtils.broadcast_rollout_batch(rollout_batches)
                for t_name in self.train_rollout_generator.teacher_clients:
                    extra_restore_keys.append(f"teacher_logprobs_{t_name}")
                    if strategy == "only_stu":
                        extra_restore_keys.append(f"teacher_on_stu_topk_logprobs_{t_name}")
                    elif strategy == "intersection":
                        extra_restore_keys.append(f"teacher_on_stu_topk_logprobs_{t_name}")
                        extra_restore_keys.append(f"teacher_topk_ids_{t_name}")

                # Compute overlap_mask for intersection.
                if strategy == "intersection":
                    for rb in rollout_batches:
                        for t_name in self.train_rollout_generator.teacher_clients:
                            tch_ids_key = f"teacher_topk_ids_{t_name}"
                            if tch_ids_key not in rb:
                                continue
                            overlap_masks = []
                            for s_ids, t_ids in zip(rb["stu_topk_ids"], rb[tch_ids_key]):
                                # Pad teacher ids (truncated) to match student ids (full length).
                                if t_ids.shape[0] < s_ids.shape[0]:
                                    t_ids = pad_3d_seq_dim(t_ids, s_ids.shape[0], value=-1)
                                stu_in_tch, _ = compute_topk_overlap_masks(s_ids, t_ids)
                                overlap_masks.append(stu_in_tch)
                            rb[f"overlap_mask_{t_name}"] = overlap_masks
                    for t_name in self.train_rollout_generator.teacher_clients:
                        extra_restore_keys.append(f"overlap_mask_{t_name}")

                # Set the canonical topk_ids key for training forward.
                for rb in rollout_batches:
                    rb["opd_topk_ids"] = rb["stu_topk_ids"]
                extra_restore_keys.append("opd_topk_ids")

            elif strategy == "only_tch":
                # Student gathered on teacher's topk_ids.
                for rb, prev in zip(rollout_batches, prev_logprobs, strict=True):
                    rb["logprobs"] = [d["logprobs"] for d in prev]
                    rb["prev_topk_logprobs"] = [d["gather_logprobs"] for d in prev]
                    rb["stu_on_tch_topk_logprobs"] = [d["gather_logprobs"] for d in prev]
                extra_restore_keys.extend(["prev_topk_logprobs", "stu_on_tch_topk_logprobs"])

                if need_ref_topk_gather:
                    for rb, ref_list in zip(rollout_batches, ref_logprobs, strict=True):
                        rb["ref_logprobs"] = [d["logprobs"] for d in ref_list]
                        rb["base_on_topk_logprobs"] = [d["gather_logprobs"] for d in ref_list]
                    extra_restore_keys.append("base_on_topk_logprobs")

                # Teacher already ran in generator; unpack tch_topk_ids → opd_topk_ids.
                for t_name in self.train_rollout_generator.teacher_clients:
                    extra_restore_keys.append(f"teacher_logprobs_{t_name}")
                    extra_restore_keys.append(f"teacher_topk_ids_{t_name}")
                    extra_restore_keys.append(f"teacher_topk_logprobs_{t_name}")

                # Set the canonical topk_ids key for training forward.
                # Use the first (or default) teacher's topk_ids.
                default_t_name = next(iter(self.train_rollout_generator.teacher_clients))
                for rb in rollout_batches:
                    rb["opd_topk_ids"] = rb[f"teacher_topk_ids_{default_t_name}"]
                extra_restore_keys.append("opd_topk_ids")

        else:
            # -- non-topk --
            for rb, prev_logps in zip(rollout_batches, prev_logprobs, strict=True):
                rb["logprobs"] = prev_logps
            if has_ref:
                for rb, ref_logps in zip(rollout_batches, ref_logprobs):
                    rb["ref_logprobs"] = ref_logps

        if getattr(self.config.policy, 'balance_dp_seqlen', False):
            rollout_batches = DPBalanceHelper.restore_log_probs_to_original_batches(
                origin_rollout_batches,
                rollout_batches,
                restore_info,
                without_ref=self.config.policy.without_ref,
                extra_keys=extra_restore_keys or None,
            )
        return rollout_batches

    @override
    async def rollout(
        self, epoch_i, curr_ppo_step, num_rollout_micro_batches, debug_disable_advantage=False
    ):
        timers = TimerSingleton.get_timer()
        rollout_batches = []

        timers("rollout", log_level=0).start(barrier=True)
        rollout_batches = await self.train_rollout_generator(
            self.train_iter,
            num_rollout_micro_batches,
            curr_ppo_step,
        )
        cpu_barrier()
        timers("rollout").stop()
        logging_memory_usage_details("memory tracking before bcast data", rank=0)
        rollout_batches = BroadcastUtils.broadcast_rollout_batch(rollout_batches)
        clear_memory()
        logging_memory_usage_details("memory tracking after bcast data", rank=0)

        rollout_batches = self.train_rollout_generator.add_back_rollout_attr_after_sampling(
            rollout_batches
        )
        assert check_rollout_batches(
            rollout_batches
        ), f"rollout_batches fmt error: {rollout_batches=}"
        logging_meminfo_str("CPU Memory: after get_rollout_batches: ")

        # filter sampling (pre-filter: before logprobs/advantage computation)
        training_config = self.config.training
        need_filter = training_config.sampling_keep_n != training_config.sampling_repeat_n
        filter_stage = training_config.filter_sampling_stage if need_filter else None

        if filter_stage == "pre":
            strategy = training_config.sampling_keeping_strategy
            samples_before = len(rollout_batches[0][next(iter(rollout_batches[0]))])
            filter_sampling_fn = FilterSamplingRegistry.get(strategy)
            rollout_batches = filter_sampling_fn(
                self.config,
                rollout_batches,
                training_config.sampling_repeat_n,
                training_config.sampling_keep_n,
            )
            samples_after = len(rollout_batches[0][next(iter(rollout_batches[0]))])
            logging_rank0(
                f"[filter] stage=pre, strategy={strategy}, "
                f"repeat_n={training_config.sampling_repeat_n}, "
                f"keep_n={training_config.sampling_keep_n}, "
                f"samples_per_batch: {samples_before} -> {samples_after}"
            )
        elif filter_stage == "post":
            logging_rank0(
                f"[filter] stage=post (deferred), strategy={training_config.sampling_keeping_strategy}, "
                f"repeat_n={training_config.sampling_repeat_n}, "
                f"keep_n={training_config.sampling_keep_n}, "
                f"processing all {training_config.sampling_repeat_n} samples through logprobs/advantage"
            )

        effective_keep_n = (
            training_config.sampling_repeat_n
            if filter_stage == "post" else training_config.sampling_keep_n
        )

        # compute logps
        if self.config.policy.dist_config.dynamic_context_parallel:
            timers("compute_logps", log_level=0).start(barrier=True)
            rollout_batches = self._compute_log_probs_dynamic_cp(
                rollout_batches,
                training_config,
                effective_keep_n,
            )
            timers("compute_logps").stop()
        else:
            rollout_batches = await self._compute_log_probs(
                rollout_batches,
                training_config,
                effective_keep_n,
                timers,
                curr_ppo_step,
            )
        clear_memory()
        logging_memory_usage_details("memory tracking after compute_log_probs", rank=0)
        logging_meminfo_str("CPU Memory: after get_rollout_batches: ")

        expected_mbs = training_config.rollout_mbs * effective_keep_n
        for rb in rollout_batches:
            assert len(
                rb['tokens']
            ) == expected_mbs, f"len(rb['tokens']) {len(rb['tokens'])} != {expected_mbs}"
        assert check_rollout_batches(rollout_batches), f"rbs fmt error: {rollout_batches=}"
        rollout_metrics = self.compute_rollout_metrics(rollout_batches)
        cpu_barrier()

        timers("generate_ppo_data", log_level=0).start(barrier=True)
        rollout_batches, ppo_metrics = self.generate_ppo_data(rollout_batches)
        cpu_barrier()
        timers("generate_ppo_data").stop()

        metrics = rollout_metrics | ppo_metrics
        display_rollout_generation(self.tokenizer, self.disp_rng, rollout_batches)

        self.train_rollout_generator.clear_data_cache()

        return rollout_batches, metrics

    @override
    async def train_one_ppo_step(
        self,
        epoch_i,
        ppo_step_i,
        cur_epoch_ppo_step,
    ):
        timers = TimerSingleton.get_timer()
        begint_time = sync_cuda_and_get_time()
        training_config = self.config.training
        rollout_nb = self.get_num_rollout_micro_batches()
        assert rollout_nb > 0, f"rollout_nb {rollout_nb}"
        rollout_gbs = training_config.rollout_gbs
        rollout_mbs = training_config.rollout_mbs
        repeat_n = training_config.sampling_repeat_n
        keep_n = training_config.sampling_keep_n
        assert keep_n <= repeat_n, f"keep_n {keep_n} > repeat_n {repeat_n}"

        # rollout_NB = rollout_GBS / rollout_MBS / DP_size
        # len(rollout_batches) == rollout_NB
        # v = rollout_batches[0][k]
        # len(v) == rollout_MBS * keep_n
        rollout_batches, metrics = await self.rollout(epoch_i, ppo_step_i, rollout_nb)

        # post-filter: filter after logprobs/advantage computation
        filter_stage = (training_config.filter_sampling_stage if keep_n != repeat_n else None)
        if filter_stage == "post":
            samples_before = len(rollout_batches[0][next(iter(rollout_batches[0]))])
            original_metrics = {k + "_original": v for k, v in metrics.items()}
            strategy = training_config.sampling_keeping_strategy
            filter_sampling_fn = FilterSamplingRegistry.get(strategy)
            rollout_batches = filter_sampling_fn(self.config, rollout_batches, repeat_n, keep_n)
            samples_after = len(rollout_batches[0][next(iter(rollout_batches[0]))])
            logging_rank0(
                f"[filter] stage=post applied, strategy={strategy}, "
                f"samples_per_batch: {samples_before} -> {samples_after}, "
                f"recomputing metrics on filtered set"
            )
            rollout_metrics = self.compute_rollout_metrics(rollout_batches)
            ppo_data_metrics = self.compute_ppo_global_statistics(rollout_batches)
            metrics = rollout_metrics | ppo_data_metrics | original_metrics

        if self.config.debug.save_every_rollout_data or (
            self.config.debug.save_first_rollout_data and ppo_step_i == 0
        ):
            save_data(
                rollout_batches, "debug-tmp",
                f"rollout_batches_{ppo_step_i}_{torch.distributed.get_rank()}.pt"
            )
            save_data(
                metrics, "debug-tmp",
                f"rollout_metrics_{ppo_step_i}_{torch.distributed.get_rank()}.pt"
            )

        assert len(rollout_batches) == rollout_nb, f'{len(rollout_batches)=} {rollout_nb=}'
        assert len(next(iter(rollout_batches[0].values()))) == rollout_mbs * keep_n

        expanded_rbs = expand_rollout_batches(rollout_batches)
        total_samples = training_config.rollout_gbs * keep_n
        assert len(expanded_rbs) * mpu.get_data_parallel_world_size(
        ) == total_samples, f"{len(expanded_rbs)} != {total_samples}"

        # ---- dp_balance for train: rebalance expanded samples across DP ranks ----
        if getattr(self.config.policy, 'balance_dp_seqlen', False):
            expanded_rbs = DPBalanceHelper.rebalance_row_batches_for_train(
                expanded_rbs,
                add_custom_keys=self.config.policy.dp_balance_extra_keys,
            )

        # ---- smart_pad_train: sort samples by seqlen for efficient padding ----
        if self.config.policy.smart_pad_train:
            num_global_batch = rollout_gbs * keep_n // training_config.train_gbs
            reorder_expanded_rbs = smart_pad_train_get_reorder_rollout_batches(
                expanded_rbs,
                num_global_batch,
                training_config.train_gbs // mpu.get_data_parallel_world_size(),
                training_config.pad_to_mulitiple_of,
                reorder_seed=ppo_step_i,
            )
            expanded_rbs = reorder_expanded_rbs

        self.policy_engine.onload_optimizer()
        timers("train_step", log_level=0).start(barrier=True)
        for _ in range(training_config.ppo_max_epochs_2):
            num_train_global_steps = rollout_gbs * keep_n // training_config.train_gbs
            ppo_step_iters = get_iterator_k_split_list(expanded_rbs, num_train_global_steps)

            _metrics = self.policy_engine.rl_train_actor(ppo_step_iters)
            extend_value_to_dict(metrics, _metrics)
        cpu_barrier()
        timers("train_step").stop()

        lr = self.policy_engine.step_and_get_lr()
        metrics["policy/lr"] = lr
        end_time = sync_cuda_and_get_time()
        metrics["time_perf/total_time"] = end_time - begint_time
        output_metrics = reduce_metrics(metrics)

        time_log_keys = [
            "rollout", "compute_logps", "generate_ppo_data", "train_step", "sampler_generate",
            "gen_rm_generate", "bt_rm_generate", "compute_teacher_logps"
        ]
        output_metrics = record_time_to_metrics(timers, time_log_keys, output_metrics, reset=True)

        if is_last_rank():
            log_prefix = f"training ppo_step {ppo_step_i}/{training_config.total_ppo_step} epoch {epoch_i}"
            TrainReporterSingleton.log_and_report(output_metrics, ppo_step_i, log_prefix=log_prefix)
        clear_memory()
        cpu_barrier()

        return output_metrics

    async def _train_loop(self):
        training_config = self.config.training
        ppo_step = self.prev_ppo_step
        init_ppo_step = self.prev_ppo_step
        init_epoch = init_ppo_step // training_config.ppo_step_per_epoch
        init_ppo_step = init_ppo_step % training_config.ppo_step_per_epoch
        eval_before_train_flag = self.config.training.eval_before_train

        await self.update_weights()
        cpu_barrier()
        exit_flag = False
        ret_metrics = []

        for epoch in range(init_epoch, training_config.num_train_epoches):
            if epoch == init_epoch and init_ppo_step > 0:
                reset_start_index = False
            else:
                reset_start_index = True
            self.maybe_set_epoch(epoch, reset_start_index)
            self.train_iter = iter(self.train_dataloader)
            if epoch == init_epoch:
                start_steps_per_epoch = init_ppo_step
            else:
                start_steps_per_epoch = 0
            for cur_epoch_ppo_step in range(
                start_steps_per_epoch, training_config.ppo_step_per_epoch
            ):
                if eval_before_train_flag and self.config.training.total_eval_step > 0:
                    await self._eval_loop(ppo_step)
                    eval_before_train_flag = False

                self.last_progress_time = time.time()
                metrics = await self.train_one_ppo_step(epoch, ppo_step, cur_epoch_ppo_step)
                if self.config.debug.trainer_return_ppo_step_metrics:
                    ret_metrics.append(metrics)

                ppo_step += 1
                if ppo_step % training_config.save_interval == 0:
                    await self.save_checkpoint(ppo_step)

                await self.update_weights()

                if self.config.training.total_eval_step > 0 and (
                    ppo_step + 1
                ) % training_config.eval_interval == 0:
                    await self._eval_loop(ppo_step)

                cpu_barrier()

                if ppo_step == training_config.exit_step:
                    exit_flag = True
                    break

            if exit_flag:
                break

        self.train_step_finished = True

        cpu_barrier()
        if ppo_step % training_config.save_interval != 0:
            if self.config.placement_type != "disaggregated":
                for sampler_idx in range(self.sampler_client.num_samplers):
                    await self.sampler_client.sleep(sampler_idx)
                cpu_barrier()
            self.policy_engine.onload_model()
            self.policy_engine.onload_optimizer()
            await self.save_checkpoint(ppo_step)

        if is_last_rank():
            TrainReporterSingleton.finish()

        return ret_metrics

    @override
    async def train_loop(self):
        try:
            return await self._train_loop()
        except Exception as e:
            log(f"train_loop error: {e}")
            traceback.print_exc()
            raise e

    @override
    def eval_logging(self, metrics, iteration):
        if is_last_rank():
            log_prefix = f"[DISTILL STUDENT EVAL] step {iteration}"
            TrainReporterSingleton.log_and_report(metrics, iteration, log_prefix=log_prefix)
