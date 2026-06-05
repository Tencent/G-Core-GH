import asyncio
import os
import random
import time
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.client.base_client import BaseClientAbc, SamplerClientMixin
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils import log, logging_rank0, perf_time


class InferenceClient(BaseClientAbc, SamplerClientMixin):
    """Client for standalone inference (no weight updates).

    Parameters
    ----------
    config : RlConfig
    """
    def __init__(self, config: RlConfig):
        self.config = config

        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_size = mpu.get_data_parallel_world_size()
        self.rpc_type = "ray"

        rpc_client_lst = []
        svr_cluster_num_per_sampler = []
        sampler_config = self.config.sampler
        self.num_samplers = len(sampler_config.model_info)
        assert self.num_samplers == 1, f"{self.num_samplers=} only one type of sampler is supported"

        for sampler_idx in range(self.num_samplers):
            infer_engine_config = sampler_config.infer_engine_configs[sampler_idx]
            dist_config = infer_engine_config.dist_config

            num_clusters, num_engines, num_engine_per_cluster = self.get_engine_info(dist_config)

            rpc_client = self.build_rpc_client(
                self.rpc_type,
                None,
                None,
                num_clusters,
                num_engine_per_cluster,
                ray_actor_pname=f"sampler_{sampler_idx}"
            )
            svr_cluster_num_per_sampler.append(num_clusters)

            rpc_client_lst.append(rpc_client)

        assert len(rpc_client_lst) == self.num_samplers
        self.rpc_client_lst = rpc_client_lst
        self.svr_cluster_num_per_sampler = svr_cluster_num_per_sampler

        logging_rank0(f"SamplerClient init with {self.rpc_type} client")

    async def generate(self, sampler_idx, sidx, batched_data) -> Dict[str, List[Any]]:
        """Send a generation request to the sampler.

        Parameters
        ----------
        sampler_idx : int
        sidx : int
        batched_data : dict

        Returns
        -------
        dict[str, list]
            Generated outputs.
        """
        target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
            sample_idx=sidx, ep_idx=None
        )
        req_dict = {
            'actor_dp_rank': self.dp_rank,
            'actor_dp_size': self.dp_size,
            'sampler_dp_rank': sampler_idx,
            'ppo_step': 0,
            'sample_idx': sidx,
            'sampling_repeat': self.config.infer_result.sampling_repeat,
            'batched_data': batched_data,
        }
        fut = self.rpc_client_lst[sampler_idx].call(target_ep, 'generate', req_dict)
        resp = await fut
        log(f'generate sample_idx={sidx} completed')
        return resp

    async def infer_engine_flush_cache(self, sampler_idx):
        if torch.distributed.get_rank() == 0:
            await self._batch_rpc_call(sampler_idx, 'flush_cache', {})

    @override
    async def mark_ppo_step_begin(self, idx, ppo_step):
        pass

    @override
    async def mark_ppo_step_end(self, idx, ppo_step):
        pass

    @override
    async def wake_up(self, idx, tag_names=None):
        pass

    @override
    async def sleep(self, idx):
        pass
