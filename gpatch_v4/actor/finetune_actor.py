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

# isort: off
import gpatch_v4.core.device  # noqa: F401  # ensure device backend is initialized early
# isort: on

import torch.distributed
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from megatron.core import mpu

from gpatch_v4.actor.mixin import (
    CheckpointConverterMixin,
    FlopsCounterMixin,
    ProfileMixin,
    RetryActorMixin,
    TokenizerMixin,
    TrainingPltMixin,
)
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    init_pg,
    initlize_parallel_state,
    is_last_rank,
    is_mp_and_cp_head,
    is_tp_and_cp_head,
    preserve_rng_state,
)
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.rollout_generator import RolloutGeneratorFactory
from gpatch_v4.training_backend import (
    BUILDIN_LOSS_FUNC,
    TrainingEngineFactory,
    register_custom_loss_fn,
)
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
)
from gpatch_v4.utils.common_utils import compress_ppo_save_train_data
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler
from gpatch_v4.utils.test_utils import save_data
from gpatch_v4.utils.training_utils import (
    align_sampler_num_samples,
    get_dump_moe_metrics,
)

try:
    from megatron.core.gcore_utils import (
        clear_gathered_routing_info,  # only branch wxdev support
    )
except ImportError:
    clear_gathered_routing_info = None


class FinetuneActor(
    BaseActor, TokenizerMixin, CheckpointConverterMixin, RetryActorMixin, TrainingPltMixin,
    FlopsCounterMixin, ProfileMixin
):
    """Ray actor for supervised fine-tuning (SFT)."""

    # Log tag for train/eval prefixes; subclasses override to distinguish runs.
    train_log_tag = "SFT"

    async def init(self, config):
        """Initialize the actor: parallel state, tokenizer, dataset, and model.

        Parameters
        ----------
        config : FinetuneConfig
        """
        super().init(config)
        self.retry_actor_init()

        # init parallel group
        initlize_parallel_state(config, config.policy.dist_config)
        init_pg(config.policy.dist_config)

        self.build_tokenizer()
        self.tokenizer = self.actor_tokenizer

        # 第一次调用，不设置 resume_step
        self.build_dataset_and_dataloader()
        self.train_step = 0
        self.validated_config()
        logging_rank0(f"{self.__class__.__name__} config {format_config(self.config)}")

        extra_args = {"policy_config": config.policy, "tokenizer": self.tokenizer}
        self.model_engine = TrainingEngineFactory.get_training_engine(config, **extra_args)
        self.disp_rng = random.Random(self.config.training.seed)
        self.compact_thread = None

        if (not config.checkpoint.convert_mcore_to_hf_offline) and is_last_rank():
            init_train_reporter_singleton(self.config.report, self.config)
        init_timer_singleton(self.config.report)
        self.load_hf_config()
        self.training_plt_init()

        # init flops counter
        self.flops_counter_init(self.model_engine.hf_config)

    def _save_dumped_metrics(self, metric, expanded_rbs, train_step):
        """Collect base, loss_fn and MoE-topk data, save to disk, and compress."""
        training_config = self.config.training
        all_dumped_metrics = []

        # merge base metrics and loss_fn metrics
        dumped_loss_fn_metrics = metric.pop("dumped_loss_fn_metrics", None)
        if is_mp_and_cp_head():
            for rb in expanded_rbs:
                all_dumped_metrics.append(
                    {
                        "train_step": train_step,
                        "tokens": rb["tokens"].detach().to(torch.int32),
                    }
                )
            assert dumped_loss_fn_metrics is not None and len(all_dumped_metrics) == len(dumped_loss_fn_metrics), \
                f"all_dumped_metrics length {len(all_dumped_metrics)} != dumped_loss_fn_metrics length {len(dumped_loss_fn_metrics) if dumped_loss_fn_metrics else 0}"
            for s1, s2 in zip(all_dumped_metrics, dumped_loss_fn_metrics):
                s1.update(s2)

        # merge MoE topk metrics
        if is_tp_and_cp_head() and training_config.ppo_dump_moe_topk > 0:
            dumped_moe_topk_metrics = get_dump_moe_metrics()
            if mpu.is_pipeline_first_stage():
                assert len(all_dumped_metrics) <= len(dumped_moe_topk_metrics), \
                    f"all_dumped_metrics length {len(all_dumped_metrics)} > dumped_moe_topk_metrics length {len(dumped_moe_topk_metrics)}"
                # 由于dumped_moe_topk_metrics是training阶段采集的数据，当开启moe_layer_recompute或者recompute_granularity时，
                # dumped_moe_topk_metrics的长度可能会大于all_dumped_metrics的长度，取前len(all_dumped_metrics)个就好了。
                for s1, topk in zip(
                    all_dumped_metrics, dumped_moe_topk_metrics[:len(all_dumped_metrics)]
                ):
                    s1["moe_topk_info"] = topk

        # save to .pt file
        if len(all_dumped_metrics) > 0:
            save_path = os.path.join(
                training_config.ppo_dump_metrics_dir,
                f'tmp/TrainStep{train_step}_{datetime.now().strftime("%Y%m%d_%H%M")}'
            )
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                all_dumped_metrics,
                os.path.join(
                    save_path,
                    f'dp{mpu.get_data_parallel_rank()}_rank{torch.distributed.get_rank()}.pt'
                )
            )

        # compress to avoid too many .pt files
        if torch.distributed.get_rank() == 0:
            self.compact_thread = compress_ppo_save_train_data(
                self.compact_thread, training_config.ppo_dump_metrics_dir
            )

    def _training_plt_report(self, train_state: TrainingPltMixin.TrainState, data: dict):
        self.training_plt_report(
            "FinetuneActor", self.config.policy.model_arch, TrainingPltMixin.TrainType.SFT,
            train_state, data
        )

    def validated_config(self):
        """Validate fine-tuning config and auto-load settings."""
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
        if self.config.checkpoint.convert_mcore_to_hf_offline:
            self.config.checkpoint.no_load_optim = True

        if self.config.training.loss_func not in BUILDIN_LOSS_FUNC:
            assert self.config.training.loss_func == "custom", "Custom loss function must be provided"
            register_custom_loss_fn(
                self.config.training.loss_func,
                self.config.training.loss_func_py_path,
                self.config.training.loss_func_py_name,
            )

    async def setup_model_and_optimizer(self):
        """Build model, optimizer, and optionally rebuild dataloader for resume."""
        logging_rank0(f"begin setup_model_and_optimizer...")
        self.train_step = self.model_engine.setup_model_and_get_optimizer()
        logging_rank0(f"finished setup_model_and_optimizer at {self.train_step} ...")
        if self.train_step > 0:
            with preserve_rng_state():
                self.build_dataset_and_dataloader(self.train_step)
            logging_rank0(f"dataloader rebuilt from {self.train_step}.")

    def build_dataset_and_dataloader(self, resume_step=None):
        """Build training dataset and dataloader from a user-provided factory function.

        Parameters
        ----------
        resume_step : int, optional
            If set, skip consumed samples for resuming.
        """
        fn = import_fn_from_path(self.config.data.py_path, self.config.data.fn_name)
        fn_kwargs = inspect.signature(fn).parameters

        cond1 = all(
            [
                len(fn_kwargs) >= 4,
                'config' in fn_kwargs,
                'tokenizer' in fn_kwargs,
                'dp_rank' in fn_kwargs,
                'dp_size' in fn_kwargs,
            ]
        )
        if cond1:
            extra_args = {}
            if resume_step is not None and "meta_info" in fn_kwargs:
                extra_args['meta_info'] = {'resume_step': resume_step}
            fn_ret = fn(
                config=self.config,
                tokenizer=self.tokenizer,
                dp_rank=mpu.get_data_parallel_rank(),
                dp_size=mpu.get_data_parallel_world_size(),
                **extra_args,
            )

        else:
            raise ValueError(f'unexpected signature {fn_kwargs}')

        self.train_dataset = fn_ret.get('train_dataset')
        self.train_dataloader = fn_ret.get('train_dataloader')
        self.train_sampler = fn_ret.get('train_sampler', None)
        self.eval_dataset = fn_ret.get('eval_dataset', None)
        self.eval_dataloader = fn_ret.get('eval_dataloader', None)

        self.train_iter = None
        if resume_step is not None:
            self.align_and_resume_sampler(resume_step)
            return
        self.auto_calc_train_step()

    def align_and_resume_sampler(self, resume_step):
        if not isinstance(self.train_sampler, ResumableDistributedSampler):
            return
        dp_size = mpu.get_data_parallel_world_size()
        mbs = self.config.training.train_mbs
        gas = self.config.training.train_gbs // (dp_size * mbs)
        step_per_epoch = self.train_sampler.num_samples // (gas * mbs)
        assert self.config.training.train_step_per_epoch == step_per_epoch, f"train_step_per_epoch {self.config.training.train_step_per_epoch} != {step_per_epoch}"
        align_sampler_num_samples(self.train_sampler, step_per_epoch, mbs, gas)
        self.train_sampler.set_start_index(resume_step * gas, mbs)

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

        if self.config.training.eval_interval > 0:
            assert self.eval_dataset is not None, f"enable eval:{self.config.training.eval_interval} should have eval_dataset and eval_dataloader"
            assert self.eval_dataloader is not None, f"enable eval:{self.config.training.eval_interval} should have eval_dataset and eval_dataloader"
            eval_step = (len(self.eval_dataloader) // gas)
            self.config.training.total_eval_step = eval_step
            assert eval_step > 0, f"{len(self.eval_dataloader)=} > {gas=}"

    def maybe_set_epoch(self, epoch, reset_start_index=True):
        if self.train_sampler:
            if isinstance(self.train_sampler, ResumableDistributedSampler):
                self.train_sampler.set_epoch(epoch)
                if reset_start_index:
                    self.train_sampler.start_index = 0
            elif isinstance(self.train_sampler, DistributedSampler):
                self.train_sampler.set_epoch(epoch)
            else:
                print(f"train_sampler {type(self.train_sampler)} is not supported set_epoch")
        elif hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

    def process_batched_data(self, batched_data):
        #TODO: 如果是 llama-factory 等其他的 dataset 返回的 batched_data，
        # 将 batched_data 转为 List[Dict[str, List[Any, 这里是 mbs 的大小]]]
        expanded_rbs = expand_rollout_batches(batched_data)
        return expanded_rbs

    def _eval_loop(self, train_step):
        self._training_plt_report(TrainingPltMixin.TrainState.EVAL_START, {})
        timers = TimerSingleton.get_timer()
        timers("eval_loop", log_level=0).start(barrier=True)
        self.model_engine.set_model_eval()
        # eval_dataloader 每次都会重新开始
        eval_iter = iter(self.eval_dataloader)
        num_microbatches = self.config.training.gradient_accumulation_steps
        metric_list = []
        for _ in range(self.config.training.total_eval_step):
            batched_data = []
            for _ in range(num_microbatches):
                batched_data.append(next(eval_iter))
            expanded_rbs = self.process_batched_data(batched_data)
            metric = self.model_engine.eval_step(expanded_rbs, num_microbatches)
            metric_list.append(metric)

        self.model_engine.set_model_train()
        timers("eval_loop").stop()
        report_metric = defaultdict(list)
        for metric in metric_list:
            for k, v in metric.items():
                report_metric[k].append(v)
        for k in list(report_metric.keys()):
            report_metric[k] = sum(report_metric[k]) / len(metric_list)

        time_log_keys = ["eval_loop"]
        report_metric = record_time_to_metrics(timers, time_log_keys, report_metric, reset=True)
        if is_last_rank():
            log_prefix = f"[{self.train_log_tag}]eval "
            TrainReporterSingleton.log_and_report(report_metric, train_step, log_prefix=log_prefix)
        cpu_barrier()
        self._training_plt_report(TrainingPltMixin.TrainState.EVAL_END, {})

    async def _train_loop(self):
        self.setup_profile()
        timers = TimerSingleton.get_timer()
        training_config = self.config.training
        num_microbatches = training_config.gradient_accumulation_steps
        train_step = self.train_step
        init_step = self.train_step
        init_epoch = init_step // training_config.train_step_per_epoch
        init_step = init_step % training_config.train_step_per_epoch
        eval_before_train_flag = self.config.training.eval_before_train
        collected_metrics = []

        cpu_barrier()
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
                await asyncio.sleep(0.01)
                timers("train_step_total", log_level=0).start(barrier=True)
                self.last_progress_time = time.time()

                if eval_before_train_flag and self.config.training.total_eval_step > 0:
                    self._eval_loop(train_step)
                    eval_before_train_flag = False

                timers("get_batched_data", log_level=0).start(barrier=True)
                batched_data = []
                for _ in range(num_microbatches):
                    new_data = next(self.train_iter)
                    batched_data.append(new_data)
                expanded_rbs = self.process_batched_data(batched_data)
                timers("get_batched_data").stop()
                if self.config.debug.save_every_rollout_data or (
                    self.config.debug.save_first_rollout_data and train_step == 0
                ):
                    save_data(
                        expanded_rbs, "debug-tmp",
                        f"{self.train_log_tag.lower()}_batches_{train_step}_{torch.distributed.get_rank()}.pt"
                    )

                should_dump = training_config.ppo_dump_metrics_interval > 0 and (
                    train_step + 1
                ) % training_config.ppo_dump_metrics_interval == 0
                self.model_engine.should_dump_metrics = should_dump

                timers("train_step", log_level=0).start(barrier=True)
                self.profile_start(train_step)
                if not training_config.skip_train_step:
                    metric = self.model_engine.finetune_step(
                        expanded_rbs, num_microbatches, train_step
                    )
                self.profile_end(train_step)
                timers("train_step").stop()
                self._training_plt_report(
                    TrainingPltMixin.TrainState.TRAIN_STEP, dict(step=train_step)
                )

                if should_dump:
                    self._save_dumped_metrics(metric, expanded_rbs, train_step)

                if clear_gathered_routing_info is not None:
                    clear_gathered_routing_info()

                timers("train_step_total").stop()
                time_log_keys = ["get_batched_data", "train_step", "train_step_total"]
                metric = record_time_to_metrics(timers, time_log_keys, metric, reset=True)

                mfu, avg_mfu = self.flops_counter_calc(
                    train_step,
                    expanded_rbs,
                    metric['time_perf/train_step'],
                    metric['finetune/seq_length'],
                    seqlen_sum=metric.get('finetune/dyn_cp_seqlen_sum'),
                    seqlen_sq_sum=metric.get('finetune/dyn_cp_seqlen_sq_sum'),
                )
                if mfu is not None:
                    metric['finetune/mfu'] = mfu
                    metric['finetune/avg_mfu'] = avg_mfu

                if is_last_rank():
                    log_prefix = f"[{self.train_log_tag}] training train_step {train_step}/{training_config.total_training_step} epoch {epoch}"
                    TrainReporterSingleton.log_and_report(metric, train_step, log_prefix=log_prefix)
                if self.config.debug.trainer_return_ppo_step_metrics:
                    collected_metrics.append(metric)
                cpu_barrier()

                if self.config.training.total_eval_step is not None and self.config.training.total_eval_step > 0 and (
                    train_step + 1
                ) % training_config.eval_interval == 0:
                    self._eval_loop(train_step)

                train_step += 1
                if train_step % training_config.save_interval == 0 and not self.config.debug.disable_save_checkpoint:
                    self.model_engine.save_checkpoint(train_step)

                if train_step == training_config.exit_step:
                    break

            if train_step == training_config.exit_step:
                break

        if self.compact_thread is not None:
            self.compact_thread.join()

        self.train_step_finished = True

        cpu_barrier()
        if not self.config.debug.disable_save_checkpoint:
            if train_step % training_config.save_interval != 0:
                self.model_engine.save_checkpoint(train_step)

        if is_last_rank():
            TrainReporterSingleton.finish()

        return collected_metrics

    async def train_loop(self):
        try:
            self._training_plt_report(TrainingPltMixin.TrainState.TRAIN_START, {})
            ret = await self._train_loop()
            self._training_plt_report(TrainingPltMixin.TrainState.TRAIN_END, {})
            return ret
        except Exception as e:
            log(f"train_loop error: {e}")
            traceback.print_exc()
            raise e
