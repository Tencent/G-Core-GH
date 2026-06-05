import asyncio
from abc import ABC, abstractmethod

import torch
import torch.distributed

from megatron.core import mpu

from gpatch_v4.configs.config import MappingProtocol
from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.orches.placement_group import compute_gen_rm_config_placement
from gpatch_v4.rpc_client import RpcClientFactory
from gpatch_v4.utils import logging_rank0, perf_time


class RmClientMixin:
    """Mixin providing RPC client construction and batch-call for reward models."""
    def build_rpc_client(
        self,
        rpc_type,
        ep_ips,
        ep_ports,
        num_clusters,
        num_engine_per_cluster,
        ray_actor_pname,
        dp_rank=None,
        dp_size=None,
    ):
        rpc_client = RpcClientFactory.create_client(
            self.config,
            rpc_type,
            ep_ips,
            ep_ports,
            num_clusters,
            num_engine_per_cluster,
            ray_actor_pname=ray_actor_pname,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
        return rpc_client

    async def _batch_rpc_call(self, rm_idx: int, action: str, req_data: dict, timeout: int = None):
        """Broadcast an RPC call to all cluster endpoints for a reward model.

        Parameters
        ----------
        rm_idx : int
        action : str
        req_data : dict
        timeout : int, optional

        Returns
        -------
        list
            Gathered responses from all endpoints.
        """
        rpc_cos = []
        for ep_i in range(self.svr_cluster_num_per_rm[rm_idx]):
            target_ep = self.rpc_client_lst[rm_idx].get_target_endpoint(
                sample_idx=None, ep_idx=ep_i
            )
            rpc_cos.append(self.rpc_client_lst[rm_idx].call(target_ep, action, req_data))

        return await asyncio.gather(*rpc_cos)

    @staticmethod
    def _compute_per_rm_gpus(gen_rm_config):
        """Return ``{rm_idx: allocated_gpus}`` from gen-rm config placement."""
        total_gpus = gen_rm_config.dist_config.nnodes * gen_rm_config.dist_config.num_gpus_per_node
        return compute_gen_rm_config_placement(gen_rm_config, total_gpus)


class SamplerClientMixin:
    """Mixin providing RPC client construction and batch-call for samplers."""
    def build_rpc_client(
        self,
        rpc_type,
        ep_ips,
        ep_ports,
        num_clusters,
        num_engine_per_cluster,
        ray_actor_pname,
        dp_rank=None,
        dp_size=None,
    ):
        rpc_client = RpcClientFactory.create_client(
            self.config,
            rpc_type,
            ep_ips,
            ep_ports,
            num_clusters,
            num_engine_per_cluster,
            ray_actor_pname=ray_actor_pname,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
        return rpc_client

    async def _batch_rpc_call(
        self, sampler_idx: int, action: str, req_data: dict, timeout: int = None
    ):
        """Broadcast an RPC call to all cluster endpoints for a sampler.

        Parameters
        ----------
        sampler_idx : int
        action : str
        req_data : dict
        timeout : int, optional

        Returns
        -------
        list
            Gathered responses from all endpoints.
        """
        rpc_cos = []
        for ep_i in range(self.svr_cluster_num_per_sampler[sampler_idx]):
            target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                sample_idx=None, ep_idx=ep_i
            )
            rpc_cos.append(self.rpc_client_lst[sampler_idx].call(target_ep, action, req_data))

        return await asyncio.gather(*rpc_cos)


class TeacherClientMixin:
    """Mixin providing RPC client construction and batch-call for teacher models."""
    def build_rpc_client(
        self,
        rpc_type,
        ep_ips,
        ep_ports,
        num_clusters,
        num_engine_per_cluster,
        ray_actor_pname,
        dp_rank=None,
        dp_size=None,
    ):
        rpc_client = RpcClientFactory.create_client(
            self.config,
            rpc_type,
            ep_ips,
            ep_ports,
            num_clusters,
            num_engine_per_cluster,
            ray_actor_pname=ray_actor_pname,
            multi_cast=True,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
        return rpc_client

    async def _batch_rpc_call(self, idx: int, action: str, req_data: dict, timeout: int = None):
        """Broadcast an RPC call to all teacher cluster endpoints.

        Parameters
        ----------
        idx : int
            Must be 0 (single teacher).
        action : str
        req_data : dict
        timeout : int, optional

        Returns
        -------
        list
            Gathered responses from all endpoints.
        """
        assert idx == 0, f"idx should be 0, but got {idx}"
        rpc_cos = []
        for ep_i in range(self.svr_cluster):
            target_ep = self.rpc_client.get_target_endpoint(sample_idx=None, ep_idx=ep_i)
            rpc_cos.append(self.rpc_client.call(target_ep, action, req_data))

        return await asyncio.gather(*rpc_cos)


class BaseClientAbc(ABC):
    """Abstract base for clients that communicate with inference engines."""
    def get_engine_info(self, dist_config, allocated_gpus=None):
        """Extract engine sizing info from a distributed config.

        Parameters
        ----------
        dist_config : object
        allocated_gpus : int, optional
            Actual GPU count allocated by placement logic. When provided,
            overrides ``dist_config.nnodes * num_gpus_per_node``.

        Returns
        -------
        tuple[int, int, int]
            ``(num_clusters, num_engines, num_engine_per_cluster)``.
        """
        svr_mp_size = dist_config.tensor_model_parallel_size * dist_config.pipeline_model_parallel_size
        num_gpus_per_engine = min(svr_mp_size, dist_config.num_gpus_per_node)

        if allocated_gpus is not None:
            svr_world_size = allocated_gpus
        else:
            svr_world_size = dist_config.nnodes * dist_config.num_gpus_per_node

        # CPU-only path: nnodes=0 but num_cpu_nodes>0 → single cluster,
        # one engine per CPU node.
        num_cpu_nodes = getattr(dist_config, 'num_cpu_nodes', 0)
        if svr_world_size == 0:
            assert num_cpu_nodes > 0
            return num_cpu_nodes, num_cpu_nodes, 1

        num_engines = svr_world_size // num_gpus_per_engine
        num_clusters = svr_world_size // svr_mp_size
        assert num_clusters > 0
        assert num_engines > 0
        assert num_engines % num_clusters == 0
        num_engine_per_cluster = num_engines // num_clusters
        return num_clusters, num_engines, num_engine_per_cluster

    @abstractmethod
    async def mark_ppo_step_begin(self, idx, ppo_step):
        ...

    @abstractmethod
    async def mark_ppo_step_end(self, idx, ppo_step):
        ...

    @abstractmethod
    async def wake_up(self, idx, tag_names=None):
        ...

    @abstractmethod
    async def sleep(self, idx):
        ...
