import asyncio
import inspect
import os
import time
import traceback
from typing import Any, Dict, List

import torch
import torch.distributed
import torch.nn.functional
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.actor.finetune_actor import FinetuneActor
from gpatch_v4.actor.mixin import ProfileMixin, TestActorMixin
from gpatch_v4.client import SamplerClient, TeacherClient
from gpatch_v4.core.mappings import all_gather_from_context_parallel_region
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    init_pg,
    initlize_parallel_state,
    is_last_rank,
    is_mp_and_cp_head,
)
from gpatch_v4.rollout_generator import RolloutGeneratorFactory
from gpatch_v4.training_backend import TrainingEngineFactory
from gpatch_v4.transfer import get_tq_connector, init_tq_connector
from gpatch_v4.utils import (
    TrainReporterSingleton,
    display_rollout_generation,
    expand_rollout_batches,
    get_k_split_list,
    is_same_tokenizer,
    log,
    reorder_samples_by_alpha_and_seqlen,
    save_data,
    sort_list_by_sequence_lengths,
    sync_cuda_and_get_time,
)


class OffloadStudentEngineManager:
    def __init__(self, model_engine):
        self.model_engine = model_engine

    def __enter__(self):
        self.model_engine.offload_optimizer()
        self.model_engine.offload_model()
        if self.model_engine.config.training.enable_teacher_kl_loss:
            self.model_engine.offload_teacher_output_weight()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class OnloadStudentEngineManager:
    def __init__(self, model_engine):
        self.model_engine = model_engine

    def __enter__(self):
        self.model_engine.onload_model()
        self.model_engine.onload_optimizer()
        if self.model_engine.config.training.enable_teacher_kl_loss:
            self.model_engine.onload_teacher_output_weight()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class OffPolicyDistillStudentActor(FinetuneActor, TestActorMixin, ProfileMixin):
    async def init(self, config):
        await super().init(config)

        if self.config.tq.enable:
            init_tq_connector(self.config.tq)

        if self.config.training.enable_teacher_kl_loss and not self.config.training.setup_teacher_in_independent_topo:
            assert self.config.training.prefetch_num_gb == 1, "to speed up prefetch_num_gb must be 1 when setup_teacher_in_same_topo"
            extra_args = {"policy_config": config.teacher, "tokenizer": self.teacher_tokenizer}
            self.teacher_engine = TrainingEngineFactory.get_training_engine(config, **extra_args)

    @override
    async def setup_model_and_optimizer(self):
        await super().setup_model_and_optimizer()
        if self.config.training.enable_teacher_kl_loss:
            self.model_engine.setup_teacher_output_weight()

    @override
    def validated_config(self):
        super().validated_config()
        self.config.teacher.without_ref = True
        self.config.teacher.without_optim = True
        self.config.policy.without_ref = True
        if self.config.training.enable_teacher_kl_loss:
            is_same_tokenizer(
                self.actor_tokenizer, self.teacher_tokenizer, skip_eos_token_judge=True
            )
            if self.config.training.enable_teacher_rollout:
                is_same_tokenizer(self.sampler_tokenizers[0], self.teacher_tokenizer)

    async def setup_client(self):
        self.gen_rm_client = None
        self.bt_rm_client = None
        self.sampler_client = None
        self.teacher_client = None

        if self.config.training.enable_teacher_rollout:
            self.sampler_client = SamplerClient(self.config)
            await self.sampler_client.maybe_init_distributed_weight_group_for_disagg()
        if self.config.training.enable_teacher_kl_loss:
            if self.config.training.setup_teacher_in_independent_topo:
                self.teacher_client = TeacherClient(
                    self.config,
                    teacher_name=None,
                    teacher_config=self.config.teacher,
                )
            else:
                self.teacher_engine.setup_model_and_get_optimizer()
                self.teacher_engine.offload_model()

    @override
    async def setup_rollout_generator(self):
        extra_kwargs = {"teacher_client": self.teacher_client}
        self.train_rollout_generator = RolloutGeneratorFactory.get_rollout_generator(
            self.config,
            self.sampler_client,
            self.gen_rm_client,
            self.bt_rm_client,
            **extra_kwargs,
        )

    @override
    async def rollout(
        self, epoch_i, curr_train_step, num_rollout_micro_batches, debug_disable_advantage=False
    ):
        rollout_batches = []

        rollout_batches = await self.train_rollout_generator(
            self.train_iter,
            num_rollout_micro_batches,
            curr_train_step,
        )

        display_rollout_generation(self.tokenizer, self.disp_rng, rollout_batches)
        metrics = {}
        return rollout_batches, metrics

    def compute_teacher_hidden_states_in_same_process(
        self, rollout_batches_list: List[List[Dict[str, torch.Tensor]]]
    ):
        self.teacher_engine.onload_model()
        for rollout_batches in rollout_batches_list:
            _, teacher_outputs = self.teacher_engine.compute_hidden_states(rollout_batches)
            if (mpu.is_pipeline_last_stage() and mpu.get_context_parallel_world_size() > 1):
                teacher_outputs = [
                    all_gather_from_context_parallel_region(teacher_output, gather_dim=0)
                    for teacher_output in teacher_outputs
                ]
            for rollout_batch, teacher_output in zip(rollout_batches, teacher_outputs):
                rollout_batch["teacher_hidden_states"] = teacher_output
        self.teacher_engine.offload_model()
        return rollout_batches_list

    def train_one_step(
        self,
        epoch_i,
        train_step_i,
        batched_data: List[Dict[str, Any]],
        avg_rollout_time=None,
    ):
        begint_time = sync_cuda_and_get_time()
        training_config = self.config.training
        num_microbatches = training_config.gradient_accumulation_steps
        metric = self.model_engine.finetune_step(batched_data, num_microbatches, train_step_i)
        end_time = sync_cuda_and_get_time()
        metric["time_perf/train_step"] = end_time - begint_time
        if avg_rollout_time is not None:
            metric["time_perf/avg_rollout_time"] = avg_rollout_time
        if is_last_rank():
            log_prefix = f"training train_step {train_step_i}/{training_config.total_training_step} epoch {epoch_i}"
            TrainReporterSingleton.log_and_report(metric, train_step_i, log_prefix=log_prefix)
        cpu_barrier()

        return metric

    @override
    async def _train_loop(self):
        self.setup_profile()
        training_config = self.config.training
        train_step = self.train_step
        init_step = self.train_step
        init_epoch = init_step // training_config.train_step_per_epoch
        init_step = init_step % training_config.train_step_per_epoch

        cpu_barrier()
        exit_flag = False
        ret_metrics = []

        for epoch in range(init_epoch, training_config.num_train_epoches):
            self.maybe_set_epoch(epoch)
            #TODO: 如果是热重启，要跳过消费状态
            self.train_iter = iter(self.train_dataloader)
            if epoch == init_epoch:
                last_steps = training_config.train_step_per_epoch - init_step
                steps_in_curr_epoch = init_step
            else:
                last_steps = training_config.train_step_per_epoch
                steps_in_curr_epoch = 0

            prefetch_times = (
                last_steps + training_config.prefetch_num_gb - 1
            ) // training_config.prefetch_num_gb
            for prefetch_i in range(prefetch_times):
                self.last_progress_time = time.time()

                cpu_barrier()
                _prefetch_num_gb = min(
                    training_config.prefetch_num_gb,
                    last_steps - prefetch_i * training_config.prefetch_num_gb
                )
                rollout_step = train_step
                with OffloadStudentEngineManager(self.model_engine):
                    rollout_begin_time = sync_cuda_and_get_time()
                    rollout_batches, metrics = await self.rollout(
                        epoch, train_step,
                        _prefetch_num_gb * training_config.gradient_accumulation_steps
                    )
                    cpu_barrier()
                    rollout_end_time = sync_cuda_and_get_time()
                    rollout_times = rollout_end_time - rollout_begin_time

                expanded_rbs = expand_rollout_batches(rollout_batches)
                total_samples = training_config.train_gbs * _prefetch_num_gb
                assert total_samples == len(expanded_rbs) * mpu.get_data_parallel_world_size(
                ), f"{len(expanded_rbs)} != {total_samples}"

                if self.config.debug.save_every_rollout_data or (
                    self.config.debug.save_first_rollout_data and train_step == 0
                ):
                    save_data(
                        rollout_batches, "debug-tmp",
                        f"distill_batches_{train_step}_{torch.distributed.get_rank()}.pt"
                    )

                if self.config.distill.enable_data_with_alpha:
                    sorted_rbs = reorder_samples_by_alpha_and_seqlen(
                        expanded_rbs, training_config.sort_batched
                    )
                elif training_config.sort_batched:
                    sorted_rbs = sort_list_by_sequence_lengths(expanded_rbs)
                else:
                    sorted_rbs = expanded_rbs
                cpu_barrier()

                rollout_batches_list = get_k_split_list(sorted_rbs, _prefetch_num_gb)
                if self.config.training.enable_teacher_kl_loss and not self.config.training.setup_teacher_in_independent_topo:
                    rollout_begin_time = sync_cuda_and_get_time()
                    rollout_batches_list = self.compute_teacher_hidden_states_in_same_process(
                        rollout_batches_list
                    )
                    rollout_end_time = sync_cuda_and_get_time()
                    rollout_times += (rollout_end_time - rollout_begin_time)

                log(f"training begin", rank=0)
                with OnloadStudentEngineManager(self.model_engine):
                    for batch in rollout_batches_list:
                        if train_step == training_config.exit_step:
                            exit_flag = True
                            break

                        self.profile_start(train_step)
                        metric = self.train_one_step(
                            epoch,
                            train_step,
                            batch,
                            avg_rollout_time=rollout_times / _prefetch_num_gb
                        )
                        if self.config.debug.trainer_return_ppo_step_metrics:
                            ret_metrics.append(metric)
                        self.profile_end(train_step)

                        train_step += 1
                        steps_in_curr_epoch += 1
                        if (
                            train_step % training_config.save_interval == 0 and
                            not self.config.debug.disable_save_checkpoint
                        ):
                            self.model_engine.save_checkpoint(train_step)
                    cpu_barrier()

                if self.config.tq.enable:
                    cpu_barrier()
                    if (is_mp_and_cp_head() and mpu.get_data_parallel_rank() == 0):
                        connector = get_tq_connector()
                        await connector.async_clear_step(rollout_step)
                    cpu_barrier()

                if exit_flag:
                    break

            if exit_flag:
                break

        self.train_step_finished = True

        cpu_barrier()
        if (
            train_step % training_config.save_interval != 0 and
            not self.config.debug.disable_save_checkpoint
        ):
            self.model_engine.save_checkpoint(train_step)

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
