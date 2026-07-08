import asyncio
import inspect
import os
import time
import traceback
from functools import partial
from typing import Any, Dict, List

import torch
import torch.distributed
from typing_extensions import override

from gpatch_v4.actor.finetune_actor import FinetuneActor
from gpatch_v4.core.parallel_state import cpu_barrier, is_last_rank
from gpatch_v4.training_backend.loss_factory import get_policy_loss_fn
from gpatch_v4.training_backend.megatron_backend.megatron_utils import unwrap_model
from gpatch_v4.training_backend.megatron_backend.model_forward import (
    gptmodel_pack_foward,
)
from gpatch_v4.utils import (
    TimerSingleton,
    TrainReporterSingleton,
    log,
    record_time_to_metrics,
    save_data,
)

# TODO：整条链路没有任何显式标记（比如一个 is_chosen: bool 字段）来标识每个 sample 的身份。完全靠
#  "前半 chosen 后半 rejected" 的隐式约定。如果中间任何环节插了 shuffle、排序、或者 reorder，配
# 对就会悄悄错乱，模型在垃圾信号上训练，而且 loss 不会有明显异常（只是不再收敛到正确的偏好）。


class OffloadEngineManager:
    """Context manager that offloads a model engine on enter and onloads on exit.

    Parameters
    ----------
    model_engine : object
    """
    def __init__(self, model_engine):
        self.model_engine = model_engine

    def __enter__(self):
        self.model_engine.offload_optimizer()
        self.model_engine.offload_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model_engine.onload_model()
        self.model_engine.onload_optimizer()


class DpoActor(FinetuneActor):
    """Ray actor for Direct Preference Optimization (DPO) training."""
    async def init(self, config):
        await super().init(config)

    @override
    def validated_config(self):
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

    def compute_ref_logps(self, batched_data: List[List[Dict[str, torch.Tensor]]]):
        """Compute reference log probs with the policy model offloaded.

        Parameters
        ----------
        batched_data : list of dict

        Returns
        -------
        list of dict
            Updated data with ``'ref_logprobs'``.
        """
        with OffloadEngineManager(self.model_engine):
            ref_logprobs, _ = self.model_engine.compute_log_probs(
                batched_data, compute_pre_logps=False
            )
            assert len(batched_data) == len(ref_logprobs)
            for i, ref_logprob in enumerate(ref_logprobs):
                assert "ref_logprobs" not in batched_data[i]
                batched_data[i]["ref_logprobs"] = ref_logprob
        return batched_data

    @override
    async def _train_loop(self):
        timers = TimerSingleton.get_timer()
        training_config = self.config.training
        num_microbatches = training_config.gradient_accumulation_steps
        train_step = self.train_step
        init_step = self.train_step
        init_epoch = init_step // training_config.train_step_per_epoch
        init_step = init_step % training_config.train_step_per_epoch

        cpu_barrier()
        ret_metrics = []

        cpu_barrier()

        for epoch in range(init_epoch, training_config.num_train_epoches):
            self.maybe_set_epoch(epoch)
            #TODO: 如果是热重启，怎么跳过消费状态？
            self.train_iter = iter(self.train_dataloader)
            if epoch == init_epoch:
                start_steps_per_epoch = init_step
            else:
                start_steps_per_epoch = 0
            for cur_epoch_train_step in range(
                start_steps_per_epoch, training_config.train_step_per_epoch
            ):
                await asyncio.sleep(0.01)
                self.last_progress_time = time.time()
                # get batched data for dataloader
                timers("get_batched_data", log_level=0).start(barrier=True)
                batched_data = []
                for _ in range(num_microbatches):
                    batched_data.append(next(self.train_iter))
                timers("get_batched_data").stop()

                # TODO(guanyouhe): 这里频繁切换模型状态，性能会受影响
                # 可以考虑 prefetch_num_gb
                timers("compute_ref_logps", log_level=0).start(barrier=True)
                batched_data = self.compute_ref_logps(batched_data)
                expanded_rbs = self.process_batched_data(batched_data)
                timers("compute_ref_logps").stop()

                if self.config.debug.save_every_rollout_data or (
                    self.config.debug.save_first_rollout_data and train_step == 0
                ):
                    save_data(
                        expanded_rbs, "debug-tmp",
                        f"sft_batches_{train_step}_{torch.distributed.get_rank()}.pt"
                    )

                timers("train_step", log_level=0).start(barrier=True)
                metric = self.model_engine.finetune_step(expanded_rbs, num_microbatches, train_step)
                timers("train_step").stop()

                time_log_keys = ["get_batched_data", "train_step"]
                metric = record_time_to_metrics(timers, time_log_keys, metric, reset=True)
                if self.config.debug.trainer_return_ppo_step_metrics:
                    ret_metrics.append(metric)

                if is_last_rank():
                    log_prefix = f"[DPO] training train_step {train_step}/{training_config.total_training_step} epoch {epoch}"
                    TrainReporterSingleton.log_and_report(metric, train_step, log_prefix=log_prefix)
                cpu_barrier()

                train_step += 1
                if train_step % training_config.save_interval == 0:
                    self.model_engine.save_checkpoint(train_step)

                if train_step == training_config.exit_step:
                    break

            if train_step == training_config.exit_step:
                break

        self.train_step_finished = True

        cpu_barrier()
        if train_step % training_config.save_interval != 0:
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
