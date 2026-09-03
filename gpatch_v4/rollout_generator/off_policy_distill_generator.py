import asyncio
import os
from typing import Any, Dict, List

import ray
import torch
import torch.distributed
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.configs.config import OffPolicyDistillConfig
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    get_model_and_context_parallel_group,
    get_model_and_context_parallel_group_gloo,
    get_model_and_context_parallel_src_rank,
    is_mp_and_cp_head,
)
from gpatch_v4.rollout_generator.base_generator import BaseRolloutGenerator
from gpatch_v4.transfer.utils import (
    TqPayloadType,
    async_offload_to_tq,
    async_restore_from_tq,
)
from gpatch_v4.utils import (
    BroadcastUtils,
    clear_memory,
    destroy_process_groups,
    log,
    logging_memory_usage,
    logging_memory_usage_details,
    reload_process_groups,
    sync_cuda_and_get_time,
)
from gpatch_v4.utils.training_utils import check_rollout_batches


def build_cuda_mesh_from_group(group, mesh_dim_names):
    """Build a CUDA DeviceMesh from a process group.

    Parameters
    ----------
    group : ProcessGroup
    mesh_dim_names : tuple of str

    Returns
    -------
    tuple[DeviceMesh, list[int]]
        ``(mesh, group_ranks)``.
    """
    group_ranks = dist.get_process_group_ranks(group)
    mesh_tensor = torch.tensor(group_ranks, device="cuda")
    mesh = DeviceMesh(device_type="cuda", mesh=mesh_tensor, mesh_dim_names=mesh_dim_names)
    return mesh, group_ranks


class OffPolicyDistillRolloutGenerator(BaseRolloutGenerator):
    """Rollout generator for off-policy distillation.

    Supports both teacher-generated and data-driven rollouts, optionally
    computing teacher logits for KL loss.

    Parameters
    ----------
    config : OffPolicyDistillConfig
    sampler_client : object
    gen_rm_client : object
        Must be *None*.
    bt_rm_client : object
        Must be *None*.
    run_eval : bool, optional
    teacher_client : object, optional
    """
    def __init__(
        self,
        config: OffPolicyDistillConfig,
        sampler_client,
        gen_rm_client,
        bt_rm_client,
        run_eval=False,
        teacher_client=None,
    ):
        assert gen_rm_client is None
        assert bt_rm_client is None
        super().__init__(config, sampler_client, gen_rm_client, bt_rm_client, run_eval)
        self.teacher_client = teacher_client
        self.training_config = config.training
        self.sampling_repeat = 1
        self.process_prefix = 'eval_' if run_eval else ''
        self.sample_idx = 0
        self.is_mp_and_cp_head = is_mp_and_cp_head()

    @override
    def _init_sampling_repeat(self) -> int:
        return 1

    @override
    async def sampler_gen_out(
        self, rbs: List[Dict[str, List[Any]]], sampler_idx, curr_train_step, sidx, repeat_n
    ):
        cos = []
        for rbi, rollout_batch in enumerate(rbs):
            _sidx = sidx + rbi
            co = self.sampler_client.generate(
                sampler_idx,
                curr_train_step,
                _sidx,
                rollout_batch,
                repeat_n=repeat_n,
                load_aware=self.config.training.load_aware_sampler_routing,
                is_eval=self.run_eval,
            )
            cos.append(co)
        return await asyncio.gather(*cos)

    @override
    async def rollout_samples(self, data_iter, num_microbatches, curr_train_step, dp_rank=None):
        """Generate rollout samples using the sampler.

        Parameters
        ----------
        data_iter : iterator
        num_microbatches : int
        curr_train_step : int

        Returns
        -------
        list of dict or None
            Rollout batches (only on MP head).
        """
        num_samplers = self.sampler_client.num_samplers
        assert num_samplers == 1, f"当前只能是一个 sampler"

        #TODO:
        # 其实，现在先将 teacher 放到 sglang 上去生成，一边蒸馏一边训练，训练中 onload-rollout-offload,
        # 每次 offload 会清理 kvcache，这样有点低效。
        # 两个其他的写法
        # 1. 如果直接把 teacher 挂 sglang 放出去在外部离线生成好，效率是不是更高一些？
        # 2. 再不然就是+额外的卡，把 sglang 挂到另外的 cluster 上面，变成 train 和 sampler 两个 cluster
        rbs = [None for _ in range(num_microbatches)]
        for sampler_idx in range(num_samplers):
            await self.sampler_client.mark_ppo_step_begin(sampler_idx, curr_train_step)
            cpu_barrier()

            sidx = self.sample_idx
            rollout_batches: List[Dict[str, List[Any]]] = [None for _ in range(num_microbatches)]

            if self.is_mp_and_cp_head:
                for rbi in range(num_microbatches):
                    batched_data = next(data_iter)
                    rollout_batches[rbi] = batched_data

                rbs = await self.sampler_gen_out(
                    rollout_batches, sampler_idx, curr_train_step, sidx, self.sampling_repeat
                )
            else:
                rbs = rollout_batches

            cpu_barrier()
            await self.sampler_client.mark_ppo_step_end(sampler_idx, curr_train_step)
            cpu_barrier()
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details(
                    f"memory tracking after sampler {sampler_idx} sleep", rank=0
                )

        return rbs

    def gen_data_from_data_iter(self, data_iter, num_microbatches, curr_train_step):
        """Generate data directly from the data iterator (no sampler).

        Parameters
        ----------
        data_iter : iterator
        num_microbatches : int
        curr_train_step : int

        Returns
        -------
        list of dict
        """
        rbs = [None for _ in range(num_microbatches)]
        if self.is_mp_and_cp_head:
            for rbi in range(num_microbatches):
                batched_data = next(data_iter)
                rbs[rbi] = batched_data
        else:
            pass
        cpu_barrier()
        return rbs

    async def issue_teacher_hidden_states(self, rbs: List[Dict[str, List[Any]]], curr_ppo_step):
        """Submit Teacher hidden-state computation requests."""
        s_idx = self.sample_idx

        async def issue_one(batch, sample_idx):
            payload = await async_offload_to_tq(
                batch,
                curr_ppo_step,
                TqPayloadType.DICT,
            )
            return await self.teacher_client.issue_calc_hidden_states(
                payload, curr_ppo_step, sample_idx
            )

        return await asyncio.gather(
            *[issue_one(rollout_batch, s_idx + rbi) for rbi, rollout_batch in enumerate(rbs)]
        )

    async def get_teacher_hidden_states(
        self,
        rbs: List[Dict[str, List[Any]]],
        curr_ppo_step,
    ):
        """Fetch Teacher TQ slots and restore hidden states into ``rbs``."""
        s_idx = self.sample_idx

        async def get_one(rollout_batch, sample_idx):
            result = await self.teacher_client.get_calc_hidden_states_result(
                curr_ppo_step, sample_idx
            )
            restored = await async_restore_from_tq(result["teacher_hidden_states"])
            hidden_states = restored["teacher_hidden_states"]
            assert len(hidden_states) == len(next(iter(rollout_batch.values())))
            rollout_batch["teacher_hidden_states"] = hidden_states

        return await asyncio.gather(
            *[get_one(rollout_batch, s_idx + rbi) for rbi, rollout_batch in enumerate(rbs)]
        )

    async def calc_teacher_hidden_states(
        self,
        rbs: List[Dict[str, List[Any]]],
        curr_train_step: int,
    ) -> None:
        """Issue Teacher compute, then restore hidden states from TQ."""
        await self.teacher_client.mark_ppo_step_begin(0, curr_train_step)
        cpu_barrier()

        if self.is_mp_and_cp_head:
            await self.issue_teacher_hidden_states(rbs, curr_train_step)
        cpu_barrier()

        await self.get_teacher_hidden_states(rbs, curr_train_step)
        cpu_barrier()

        await self.teacher_client.mark_ppo_step_end(0, curr_train_step)
        cpu_barrier()

    @override
    async def __call__(self, data_iter, num_microbatches, curr_train_step):
        #TODO: support timer record times per stage
        dp_rank = mpu.get_data_parallel_rank()
        offload_process_group = self.config.training.offload_process_group
        if offload_process_group:
            destroy_process_groups()
            clear_memory()

        if self.config.training.enable_teacher_rollout:
            rbs = await self.rollout_samples(
                data_iter, num_microbatches, curr_train_step, dp_rank=dp_rank
            )
        else:
            rbs = self.gen_data_from_data_iter(data_iter, num_microbatches, curr_train_step)

        logging_memory_usage_details("memory tracking before bcast data", rank=0)
        rbs = BroadcastUtils.broadcast_rollout_batch(rbs)
        logging_memory_usage_details("memory tracking after bcast data", rank=0)

        if self.config.training.enable_teacher_kl_loss and self.config.training.setup_teacher_in_independent_topo:
            await self.calc_teacher_hidden_states(rbs, curr_train_step)

        if self.is_mp_and_cp_head:
            assert check_rollout_batches(rbs), f"rbs format error, may need pop('ready'): {rbs=}"
        cpu_barrier()
        if offload_process_group:
            reload_process_groups()

        clear_memory()

        self.sample_idx += num_microbatches
        return rbs
