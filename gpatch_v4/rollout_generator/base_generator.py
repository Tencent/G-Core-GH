import asyncio
import os
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.parallel_state import cpu_barrier, is_mp_and_cp_head
from gpatch_v4.extended_model import ApplySamplingRolloutAttrFactory
from gpatch_v4.rollout_generator.generator_abc import RolloutGeneratorAbc
from gpatch_v4.rollout_generator.mixin import SendRequestMixin, build_stream_ready_view
from gpatch_v4.utils import (
    TimerSingleton,
    destroy_process_groups,
    reload_process_groups,
)
from gpatch_v4.utils.common_utils import (
    clear_memory,
    log,
    logging_memory_usage,
    logging_memory_usage_details,
)
from gpatch_v4.utils.training_utils import check_rollout_batches


class BaseRolloutGenerator(RolloutGeneratorAbc, SendRequestMixin):
    """Concrete rollout generator implementing sampling and reward pipelines.

    Parameters
    ----------
    config : RlConfig
    sampler_client : object
    gen_rm_client : object
    bt_rm_client : object
    run_eval : bool, optional
    """
    def __init__(
        self, config: RlConfig, sampler_client, gen_rm_client, bt_rm_client, run_eval=False
    ):
        super().__init__(config, sampler_client, gen_rm_client, bt_rm_client, run_eval)
        self.apply_sampling_rollout_attr = ApplySamplingRolloutAttrFactory.get_apply_sampling(
            config
        )

        self.is_mp_and_cp_head = is_mp_and_cp_head()
        self.handle_external_reward_in_generator = False

    def assign_unique_id_to_batches(self, batched_data: Dict[str, Any], dp_rank: int,
                                    rbi: int) -> Dict[str, Any]:
        if "unique_id" not in batched_data:
            first_key = list(batched_data.keys())[0]
            batch_size = len(batched_data[first_key])
            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_id_list = [
                f"dp_rank_{dp_rank}_rbi_{rbi}_batch_id_{bid}_{ts}" for bid in range(batch_size)
            ]
            batched_data["unique_id"] = unique_id_list
        return batched_data

    @override
    async def rollout_samples(
        self, data_iter, num_microbatches, curr_ppo_step, dp_rank=None, on_ready=None
    ):
        num_samplers = self.sampler_client.num_samplers
        assert num_samplers == 1, f"当前只能是一个 sampler, 但保留扩展异构 sampler 的能力，如果要扩展到异构 sampler 的话，注意 tokenizer 的使用"

        rbs = [None for _ in range(num_microbatches)]
        for sampler_idx in range(num_samplers):
            clear_memory()
            await self.sampler_client.mark_ppo_step_begin(sampler_idx, curr_ppo_step)
            cpu_barrier()
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details(
                    f"memory tracking after sampler {sampler_idx} wake_up", rank=0
                )
            if dp_rank is None:
                dp_rank = mpu.get_data_parallel_rank()

            sidx = self.sample_idx
            rollout_batches: List[Dict[str, List[Any]]] = [None for _ in range(num_microbatches)]
            if self.is_mp_and_cp_head:
                for rbi in range(num_microbatches):
                    batched_data = next(data_iter)
                    batched_data = self.assign_unique_id_to_batches(batched_data, dp_rank, rbi)

                    rollout_batches[rbi] = self.remove_rollout_attr_before_sampling(batched_data)

                stream_ready = None
                if on_ready is not None:
                    # on_ready fires before add_back_rollout_attr_after_sampling, so
                    # hand the consumer the stripped attrs from the handler's cache
                    def stream_ready(rbi, rb):
                        cached = self.apply_sampling_rollout_attr.cached_rollout_attrs()
                        on_ready(rbi, build_stream_ready_view(rb, cached))

                rbs = await self.sampler_gen_out(
                    rollout_batches,
                    sampler_idx,
                    curr_ppo_step,
                    sidx,
                    self.sampling_repeat,
                    on_ready=stream_ready
                )
            else:
                rbs = rollout_batches

            cpu_barrier()
            await self.sampler_client.infer_engine_flush_cache(sampler_idx)

            await self.sampler_client.mark_ppo_step_end(sampler_idx, curr_ppo_step)
            cpu_barrier()
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details(
                    f"memory tracking after sampler {sampler_idx} sleep", rank=0
                )

        return rbs

    @override
    async def generate_gen_rm_reward(
        self, rbs: List[Dict[str, List[Any]]], num_microbatches, curr_ppo_step
    ):
        assert len(rbs) == num_microbatches, f"{len(rbs)=} != {num_microbatches=}"

        num_rms = self.gen_rm_client.num_rms

        # Phase 1: mark_ppo_step_begin for all RMs concurrently
        await asyncio.gather(
            *[
                self.gen_rm_client.mark_ppo_step_begin(rm_idx, ppo_step=curr_ppo_step)
                for rm_idx in range(num_rms)
            ]
        )
        cpu_barrier()
        if self.config.placement_type != "disaggregated":
            logging_memory_usage_details("memory tracking after all gen_rm ppo step begin")

        # NOTE(astrachang): 这里加上try catche 吧，方便以后做重启
        try:
            # Phase 2: send reward requests to all RMs concurrently
            if self.is_mp_and_cp_head:
                rm_tasks = []
                for rm_idx in range(num_rms):
                    req_rbs = list(rbs)
                    req_indices = list(range(len(rbs)))
                    rm_tasks.append((rm_idx, req_rbs, req_indices))

                all_results = await asyncio.gather(
                    *[
                        self.get_reward_fn(
                            req_rbs, self.gen_rm_client, curr_ppo_step, self.sample_idx, rm_idx
                        ) for rm_idx, req_rbs, req_indices in rm_tasks
                    ]
                )

                for (_, _, req_indices), results in zip(rm_tasks, all_results):
                    for idx, result in zip(req_indices, results):
                        rbs[idx].update(result)
        finally:
            # Phase 3: mark_ppo_step_end for all RMs concurrently
            cpu_barrier()
            await asyncio.gather(
                *[
                    self.gen_rm_client.mark_ppo_step_end(rm_idx, ppo_step=curr_ppo_step)
                    for rm_idx in range(num_rms)
                ]
            )
            cpu_barrier()
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details("memory tracking after all gen_rm ppo step end")

        if self.is_mp_and_cp_head:
            assert check_rollout_batches(rbs), f"rbs format error after gen_rm: {rbs=}"

        return rbs

    @override
    async def calc_bt_rm_reward(
        self, rbs: List[Dict[str, List[Any]]], num_microbatches, curr_ppo_step
    ):
        assert len(rbs) == num_microbatches, f"{len(rbs)=} != {num_microbatches=}"
        num_rms = self.bt_rm_client.num_rms
        assert num_rms == 1, f"num_rms should be 1, but got {num_rms}"

        for rm_idx in range(num_rms):
            await self.bt_rm_client.mark_ppo_step_begin(rm_idx, curr_ppo_step)
            cpu_barrier()
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details(
                    "memory tracking actor after bt_reward wake_up", rank=0
                )

            if self.is_mp_and_cp_head:
                await self.issue_bt_rm(rbs, curr_ppo_step, rm_idx)
            cpu_barrier()

            if self.is_mp_and_cp_head:
                bt_rm_resp_dicts = await self.get_bt_rm_result(
                    curr_ppo_step, num_microbatches, rm_idx, self.sampling_repeat
                )
                for rbi, (rollout_batch, bt_rm_resp_dict) in enumerate(zip(rbs, bt_rm_resp_dicts)):
                    rollout_batch.update(bt_rm_resp_dict)
                assert check_rollout_batches(
                    rbs
                ), f"rbs format error, may need pop('ready'): {rbs=}"
            cpu_barrier()

            await self.bt_rm_client.mark_ppo_step_end(rm_idx, curr_ppo_step)
            cpu_barrier()
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details("memory tracking actor after bt_reward sleep", rank=0)
        return rbs

    @override
    async def __call__(self, data_iter, num_microbatches, curr_ppo_step, on_ready=None):
        #TODO: support timer record times per stage
        timers = TimerSingleton.get_timer()
        dp_rank = mpu.get_data_parallel_rank()
        offload_process_group = self.config.training.offload_process_group
        if offload_process_group:
            destroy_process_groups()
            clear_memory()

        timers("sampler_generate", log_level=0).start(barrier=True)
        rbs = await self.rollout_samples(
            data_iter, num_microbatches, curr_ppo_step, dp_rank=dp_rank, on_ready=on_ready
        )
        if self.is_mp_and_cp_head:
            rbs = self._hook_after_sampling(rbs, curr_ppo_step)
            assert check_rollout_batches(rbs), f"rbs format error, may need pop('ready'): {rbs=}"
        cpu_barrier()
        timers("sampler_generate").stop()

        timers("gen_rm_generate", log_level=0).start(barrier=True)
        if self.training_config.use_gen_rm_reward:
            rbs = await self.generate_gen_rm_reward(rbs, num_microbatches, curr_ppo_step)
        cpu_barrier()
        timers("gen_rm_generate").stop()

        timers("bt_rm_generate", log_level=0).start(barrier=True)
        if self.training_config.use_bt_rm_reward:
            rbs = await self.calc_bt_rm_reward(rbs, num_microbatches, curr_ppo_step)
        cpu_barrier()
        timers("bt_rm_generate").stop()

        if self.is_mp_and_cp_head:
            rbs = self._post_process_rm_rollout_batch(rbs)

        self.sample_idx += num_microbatches
        if offload_process_group:
            reload_process_groups()
        return rbs

    # Some data that is too large (such as pictures and videos) needs to
    # be removed first to avoid being sent directly to the sampler
    @override
    def remove_rollout_attr_before_sampling(self, rollout_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Remove attrs before sending prompts to samplers (avoid OOM).

        Parameters
        ----------
            rollout_batch : Dict[str, List[Any]]

        Returns
        -------
            Dict[str, List[Any]]
        """
        return self.apply_sampling_rollout_attr.remove_rollout_attr_before_sampling(rollout_batch)

    # Add the data removed from remove_rollout_attr_before_sampling back
    @override
    def add_back_rollout_attr_after_sampling(self, rollout_batches: List[Dict[str, List[Any]]]):
        return self.apply_sampling_rollout_attr.add_back_rollout_attr_after_sampling(
            rollout_batches
        )

    # TODO Abc interface 用 _ 命名不合理。
    @override
    def _hook_after_sampling(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        ppo_step: int,
    ) -> List[Dict[str, List[Any]]]:
        """Hook after sampling.

        Parameters
        ----------
            rollout_batches : List[Dict[str, List[Any]]]
            ppo_step : int

        Returns
        -------
            List[Dict[str, List[Any]]]
        """
        return rollout_batches

    @override
    def _post_process_rm_rollout_batch(self, rollout_batches: List[Dict[str, List[Any]]]):
        """Post-process RM rollout batch.

        Parameters
        ----------
            rollout_batches : List[Dict[str, List[Any]]]

        Returns
        -------
            List[Dict[str, List[Any]]]
        """
        return rollout_batches

    @override
    def clear_data_cache(self):
        return self.apply_sampling_rollout_attr.clear_data_cache()

    @override
    def set_external_reward(self, external_reward) -> None:
        return

    @override
    def setup_data_source(
        self,
        dataloader,
        reset_iter: Callable[..., Iterator],
        resume_step: int = 0,
    ) -> Tuple[Optional[Any], bool, bool]:
        return None, False, False

    @override
    def should_stop_for_consumed_data_epochs(self) -> bool:
        return False

    @override
    def save_resume_state(self, step: int) -> None:
        return

    @override
    def pop_step_metrics(self) -> Dict[str, float]:
        return {}
