import asyncio
import os
import time
from typing import Any, Dict, List

import torch
import torch.distributed
from typing_extensions import override

from gpatch_v4.agentic.env_worker import EnvironmentWorker
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.parallel_state import cpu_barrier, is_mp_and_cp_head
from gpatch_v4.rollout_generator.generator_abc import RolloutGeneratorAbc
from gpatch_v4.rollout_generator.mixin import SendRequestMixin
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
    logging_rank0,
)
from gpatch_v4.utils.training_utils import check_rollout_batches


# 几种 rollout 方式，老是 if 改来改去，很难整
# 直接抽成一个类出来，每种方式直接定义自己的 call function，否则太麻了
class AgenticRolloutGenerator(RolloutGeneratorAbc, SendRequestMixin):
    def __init__(
        self, config: RlConfig, sampler_client, gen_rm_client, bt_rm_client, run_eval=False
    ):
        super().__init__(config, sampler_client, gen_rm_client, bt_rm_client, run_eval)

        self.is_mp_and_cp_head = is_mp_and_cp_head()

    @override
    async def rollout_samples(self, data_iter, num_microbatches, curr_ppo_step):
        """
        data_iter, num_microbatches is useless for now
        """
        num_samplers = self.sampler_client.num_samplers
        assert num_samplers == 1, f"当前只能是一个 sampler, 但保留扩展异构 sampler 的能力，如果要扩展到异构 sampler 的话，注意 tokenizer 的使用"
        sampler_idx = 0
        await self.sampler_client.mark_ppo_step_begin(sampler_idx, curr_ppo_step)
        cpu_barrier()
        if self.config.placement_type != "disaggregated":
            logging_memory_usage_details(
                f"memory tracking after sampler {sampler_idx} wake_up", rank=0
            )
        res = None
        # 容易 barier 超时， cpu group 超时时间考虑设置地更长一些
        if self.is_mp_and_cp_head:
            await self.env_worker.run_rollout_loop(curr_ppo_step, curr_ppo_step)
            res = self.env_worker.get_output_data()

        cpu_barrier()
        await self.sampler_client.infer_engine_flush_cache(sampler_idx)
        await self.sampler_client.mark_ppo_step_end(sampler_idx, curr_ppo_step)
        cpu_barrier()
        if self.config.placement_type != "disaggregated":
            logging_memory_usage_details(
                f"memory tracking after sampler {sampler_idx} sleep", rank=0
            )

        return res

    @override
    async def generate_gen_rm_reward(
        self, rbs: List[Dict[str, List[Any]]], num_microbatches, curr_ppo_step
    ):
        assert False

    @override
    async def calc_bt_rm_reward(
        self, rbs: List[Dict[str, List[Any]]], num_microbatches, curr_ppo_step
    ):
        assert False

    @override
    async def __call__(self, data_iter, num_microbatches, curr_ppo_step):
        #TODO: support timer record times per stage
        timers = TimerSingleton.get_timer()
        offload_process_group = self.config.training.offload_process_group
        if offload_process_group:
            destroy_process_groups()
            clear_memory()

        timers("sampler_generate", log_level=0).start(barrier=True)
        rbs = await self.rollout_samples(data_iter, num_microbatches, curr_ppo_step)
        if self.is_mp_and_cp_head:
            rbs = self._hook_after_sampling(rbs, curr_ppo_step)
            # assert check_rollout_batches(rbs), f"rbs format error, may need pop('ready'): {rbs=}"
        cpu_barrier()
        timers("sampler_generate").stop()

        if self.is_mp_and_cp_head:
            rbs = self._post_process_rm_rollout_batch(rbs)
        if offload_process_group:
            reload_process_groups()
        return rbs

    # Some data that is too large (such as pictures and videos) needs to
    # be removed first to avoid being sent directly to the sampler
    @override
    def remove_rollout_attr_before_sampling(self, rollout_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        remove attrs from before send a prompts to samplers (to avoid OOM).

        清理字段，防止 oom。

        Parameters
        ----------
            rollout_batch : Dict[str, List[Any]]
                rollout batch containing prompts

        Returns
        -------
            Dict[str, List[Any]]
                processed rollout batch
        """
        return rollout_batch

    # Add the data removed from remove_rollout_attr_before_sampling back
    @override
    def add_back_rollout_attr_after_sampling(self, rollout_batches: List[Dict[str, List[Any]]]):
        return rollout_batches

    @override
    def _hook_after_sampling(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        ppo_step: int,
    ) -> List[Dict[str, List[Any]]]:
        """
        hook after sampling

        Parameters
        ----------
            rollout_batches : List[Dict[str, List[Any]]]
                rollout batches containing sampling results.
            ppo_step : int
                ppo step of train or eval.
        Returns
        -------
            List[Dict[str, List[Any]]]
                processed rollout batches.
        """
        return rollout_batches

    @override
    def _post_process_rm_rollout_batch(self, rollout_batches: List[Dict[str, List[Any]]]):
        """
        post_process_rm_rollout_batch

        Parameters
        ----------
            rollout_batches : List[Dict[str, List[Any]]]
                rollout batches containing sampling results.
        Returns
        -------
            List[Dict[str, List[Any]]]
                processed rollout batches.
        """
        return rollout_batches

    @override
    def clear_data_cache(self):
        pass

    async def init_env_worker(self):
        if self.is_mp_and_cp_head:
            self.env_worker = EnvironmentWorker(self.config)
            await self.env_worker.initialize(self.sampler_client, None)
