import asyncio
import inspect
import os
import random
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.distributed
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from megatron.core import mpu

from gpatch_v4.actor.mixin import (
    CheckpointConverterMixin,
    MetricsMixin,
    ProfileMixin,
    RetryActorMixin,
    TokenizerMixin,
)
from gpatch_v4.client import KvStoreClient
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    init_pg,
    initlize_parallel_state,
    is_last_rank,
    is_mp_and_cp_head,
    is_tp_and_cp_head,
)
from gpatch_v4.extended_pipeline import ExtendPipelineFactory
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.rollout_generator import RolloutGeneratorFactory
from gpatch_v4.training_backend import TrainingEngineFactory
from gpatch_v4.utils import (
    TimerSingleton,
    TrainReporterSingleton,
    expand_rollout_batches,
    format_config,
    import_fn_from_path,
    init_timer_singleton,
    init_train_reporter_singleton,
    log,
    logging_rank0,
    record_time_to_metrics,
    reduce_metrics,
    safe_import_class,
)
from gpatch_v4.utils.common_utils import compress_ppo_save_train_data
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler
from gpatch_v4.utils.test_utils import save_data
from gpatch_v4.utils.training_utils import get_dump_moe_metrics

try:
    from megatron.core.gcore_utils import clear_gathered_routing_info  # only branch wxdev support
except ImportError:
    clear_gathered_routing_info = None


class T2iEditSftActor(BaseActor, MetricsMixin, RetryActorMixin, ProfileMixin):
    async def init(self, config):
        super().init(config)
        self.retry_actor_init()

        # init parallel group
        initlize_parallel_state(config, config.policy.dist_config)
        init_pg(config.policy.dist_config)
        self.kv_store_client = KvStoreClient(config)

        # 第一次调用，不设置 resume_step
        await self.build_dataset_and_dataloader()
        self.train_step = 0
        self.validated_config()
        logging_rank0(f"{self.__class__.__name__} config {format_config(self.config)}")

        init_timer_singleton(self.config.report)

        self.compact_thread = None

        if is_last_rank():
            init_train_reporter_singleton(self.config.report, self.config)
        await self.setup_pipeline()

    async def build_dataset_and_dataloader(self, resume_step=None):
        if self.config.data.py_path is None:
            fn = safe_import_class(self.config.data.fn_name)
        else:
            fn = import_fn_from_path(self.config.data.py_path, self.config.data.fn_name)
        fn_kwargs = inspect.signature(fn).parameters

        cond1 = all(
            [
                len(fn_kwargs) >= 3,
                'config' in fn_kwargs,
                'dp_rank' in fn_kwargs,
                'dp_size' in fn_kwargs,
            ]
        )
        if cond1:
            extra_args = {}
            if resume_step is not None and "meta_info" in fn_kwargs:
                extra_args['meta_info'] = {'resume_step': resume_step}
            if "kv_store_client" in fn_kwargs:
                extra_args['kv_store_client'] = self.kv_store_client
            fn_ret = fn(
                config=self.config,
                dp_rank=mpu.get_data_parallel_rank(),
                dp_size=mpu.get_data_parallel_world_size(),
                **extra_args,
            )
            if inspect.iscoroutine(fn_ret):
                fn_ret = await fn_ret
        else:
            raise ValueError(f'unexpected signature {fn_kwargs}')

        self.train_dataset = fn_ret.get('train_dataset')
        self.train_dataloader = fn_ret.get('train_dataloader')
        self.train_sampler = fn_ret.get('train_sampler', None)
        self.eval_dataset = fn_ret.get('eval_dataset', None)
        self.eval_dataloader = fn_ret.get('eval_dataloader', None)

        self.train_iter = None
        if resume_step is not None:
            # 此时说明是第二次调用了，就不重复算了
            return
        self.auto_calc_train_step()

    def auto_calc_train_step(self):
        training_config = self.config.training
        dp_rank = mpu.get_data_parallel_rank()
        dp_size = mpu.get_data_parallel_world_size()

        gas = training_config.train_gbs // (dp_size * training_config.train_mbs)
        assert gas > 0, f"gradient_accumulation_steps must be positive, got {gas}"
        # TODO: train_dataloader 允许用户自定义的话，len() 在 dp rank 之间会不会不 match，导致程序
        # hang，有待处理。
        train_step_per_epoch = (len(self.train_dataloader) // gas)
        total_training_step = train_step_per_epoch * training_config.num_train_epoches

        self.config.training.total_training_step = total_training_step
        self.config.training.train_step_per_epoch = train_step_per_epoch
        self.config.training.gradient_accumulation_steps = gas
        log(
            f"train_dataset length: {len(self.train_dataloader)=} {len(self.train_dataset)=} "
            f"{dp_size=} {self.config.training.total_training_step=}"
        )

        assert self.config.training.eval_interval == 0
        if self.config.training.eval_interval > 0:
            assert self.eval_dataset is not None, f"enable eval:{self.config.training.eval_interval} should have eval_dataset and eval_dataloader"
            assert self.eval_dataloader is not None, f"enable eval:{self.config.training.eval_interval} should have eval_dataset and eval_dataloader"
            eval_step = (len(self.eval_dataloader) // gas)
            self.config.training.total_eval_step = eval_step
            assert eval_step > 0, f"{len(self.eval_dataloader)=} > {gas=}"

    def validated_config(self):
        self.config.policy.without_ref = True
        assert self.config.policy.without_ref
        if self.config.training.auto_load_from_save_ckpt:
            if os.path.exists(
                os.path.join(
                    self.config.checkpoint.save_ckpt_path, 'latest_checkpointed_iteration.txt'
                )
            ):
                self.config.checkpoint.load_ckpt_path = self.config.checkpoint.save_ckpt_path
            self.config.checkpoint.no_load_optim = False

    async def setup_pipeline(self):
        logging_rank0(f"begin setup_pipeline...")
        self.extended_pipeline = ExtendPipelineFactory.get_pipeline(self.config)
        self.train_step = self.extended_pipeline.setup_pipeline()
        logging_rank0(f"finished setup_pipeline at {self.train_step} ...")
        if self.train_step > 0:
            # 如果是续训（train_step > 0），重建 dataloader 以跳过已训练的数据
            self.build_dataset_and_dataloader(self.train_step)
            logging_rank0(f"dataloader rebuilt from {self.train_step}")

    def maybe_set_epoch(self, epoch, reset_start_index=True):
        if self.train_sampler:
            if isinstance(self.train_sampler, ResumableDistributedSampler):
                self.train_sampler.set_epoch(epoch)
                if reset_start_index:
                    self.train_sampler.start_index = 0
            elif isinstance(self.train_sampler, DistributedSampler):
                self.train_sampler.set_epoch(epoch)
            else:
                raise ValueError(f"train_sampler {type(self.train_sampler)} is not supported")
        elif hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

    # TODO save and load checkpoint
    async def _train_loop(self):
        timers = TimerSingleton.get_timer()
        training_config = self.config.training
        num_microbatches = training_config.gradient_accumulation_steps
        train_step = self.train_step
        init_step = self.train_step
        init_epoch = init_step // training_config.train_step_per_epoch
        init_step = init_step % training_config.train_step_per_epoch
        eval_before_train_flag = self.config.training.eval_before_train

        for epoch in range(init_epoch, training_config.num_train_epoches):
            if epoch == init_epoch and init_step > 0:
                reset_start_index = False
            else:
                reset_start_index = True
            self.maybe_set_epoch(epoch, reset_start_index)

            self.train_iter = iter(self.train_dataloader)
            if epoch == init_epoch:
                start_steps_per_epoch = init_step
            else:
                start_steps_per_epoch = 0

            for cur_epoch_train_step in range(
                start_steps_per_epoch, training_config.train_step_per_epoch
            ):
                self.last_progress_time = time.time()
                if clear_gathered_routing_info is not None:
                    clear_gathered_routing_info()

                if eval_before_train_flag and self.config.training.total_eval_step > 0:
                    self._eval_loop(train_step)
                    eval_before_train_flag = False

                timers("get_batched_data", log_level=0).start(barrier=True)
                batched_data = []
                for _ in range(num_microbatches):
                    new_data = next(self.train_iter)
                    batched_data.append(new_data)

                # 将 batched_data 转为 List[Dict[str, List[Any]]]
                expanded_rbs = expand_rollout_batches(batched_data)

                timers("get_batched_data").stop()
                if self.config.debug.save_every_rollout_data or (
                    self.config.debug.save_first_rollout_data and train_step == 0
                ):
                    save_data(
                        expanded_rbs, "debug-tmp",
                        f"sft_batches_{train_step}_{torch.distributed.get_rank()}.pt"
                    )

                timers("train_step", log_level=0).start(barrier=True)
                if not training_config.skip_train_step:
                    metrics = self.extended_pipeline.sft_train_step(expanded_rbs)
                timers("train_step").stop()

                output_metrics = reduce_metrics(metrics)
                time_log_keys = ["get_batched_data", "train_step"]
                metrics = record_time_to_metrics(timers, time_log_keys, metrics, reset=True)

                if is_last_rank():
                    log_prefix = f"[SFT] training train_step {train_step + 1}/{training_config.total_training_step} epoch {epoch}"
                    TrainReporterSingleton.log_and_report(
                        metrics, train_step, log_prefix=log_prefix
                    )
                cpu_barrier()

                if self.config.training.total_eval_step is not None and self.config.training.total_eval_step > 0 and (
                    train_step + 1
                ) % training_config.eval_interval == 0:
                    self._eval_loop(train_step)

                train_step += 1
                if train_step % training_config.save_interval == 0:
                    self.extended_pipeline.model.save(train_step)

                if train_step == training_config.exit_step:
                    break

            if train_step == training_config.exit_step:
                break

        if self.compact_thread is not None:
            self.compact_thread.join()

        self.train_step_finished = True
        cpu_barrier()

        if train_step % training_config.save_interval != 0:
            self.extended_pipeline.model.save(train_step)

        if is_last_rank():
            TrainReporterSingleton.finish()

    async def train_loop(self):
        try:
            await self._train_loop()
        except Exception as e:
            log(f"train_loop error: {e}")
            traceback.print_exc()
            raise e
