import asyncio
from typing import Any, Dict, List

import ray
import torch
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.client.base_client import BaseClientAbc, RmClientMixin
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils.common_utils import log, logging_rank0


class GenRmClient(BaseClientAbc, RmClientMixin):
    """Client for communicating with generative reward model engines.

    Parameters
    ----------
    config : RlConfig
    dp_rank : int, optional
        Data-parallel rank override.  When provided, the client does **not**
        query ``mpu`` and can run outside a Megatron parallel context (e.g.
        inside a ``RolloutController``).
    dp_size : int, optional
    """
    def __init__(self, config: RlConfig, dp_rank: int = None, dp_size: int = None):
        self.config = config

        policy_config = self.config.policy
        self.ep_ips = policy_config.gen_rm_client.endpoint_ips
        self.ep_ports = policy_config.gen_rm_client.endpoint_ports
        self.timeout = policy_config.gen_rm_client.rpc_timeout
        self.rpc_type = policy_config.gen_rm_client.rpc_type

        self.dp_rank = dp_rank if dp_rank is not None else mpu.get_data_parallel_rank()
        self.dp_size = dp_size if dp_size is not None else mpu.get_data_parallel_world_size()

        # When dp_rank/dp_size are explicitly provided, we are running in
        # standalone / controller mode without torch.distributed.
        self._standalone = dp_rank is not None

        gen_rm_config = self.config.gen_rm
        num_rms = len(gen_rm_config.reward_model_info)
        self.num_rms = num_rms

        rpc_client_lst = []
        svr_cluster_num_per_rm = []
        all_engine_actors_per_rm = []
        per_rm_gpus = None

        for rm_idx in range(num_rms):
            if self.rpc_type == "ray":
                if per_rm_gpus is None:
                    # HTTP endpoints are explicit; only Ray needs placement-derived sizing.
                    per_rm_gpus = self._compute_per_rm_gpus(gen_rm_config)
                dist_config = gen_rm_config.infer_engine_configs[rm_idx].dist_config
                num_clusters, num_engines, num_engine_per_cluster = self.get_engine_info(
                    dist_config, allocated_gpus=per_rm_gpus[rm_idx]
                )
                actor_pname = f"gen_rm_{rm_idx}"
                rpc_client = self.build_rpc_client(
                    self.rpc_type,
                    None,
                    None,
                    num_clusters,
                    num_engine_per_cluster,
                    ray_actor_pname=actor_pname,
                    dp_rank=self.dp_rank,
                    dp_size=self.dp_size,
                )
                svr_cluster_num_per_rm.append(num_clusters)
                all_engine_actors_per_rm.append(
                    [ray.get_actor(f"{actor_pname}_{ep_idx}") for ep_idx in range(num_engines)]
                )
            elif self.rpc_type == "http":
                # self.ep_ips 是一个 list of list of str
                # 第一个 list 维度是 num_rms, 第二个 list 维度是 num_of cluster
                rpc_client = self.build_rpc_client(
                    self.rpc_type, self.ep_ips[rm_idx], self.ep_ports[rm_idx], None, None, None
                )
                svr_cluster_num_per_rm.append(len(self.ep_ips[rm_idx]))
            else:
                raise ValueError(f"invalid rpc type: {self.rpc_type}")

            rpc_client_lst.append(rpc_client)
        assert len(rpc_client_lst) == num_rms
        self.rpc_client_lst = rpc_client_lst
        self.svr_cluster_num_per_rm = svr_cluster_num_per_rm
        self.all_engine_actors_per_rm = all_engine_actors_per_rm
        logging_rank0(
            f"GenRmClient init with {self.rpc_type} client (standalone={self._standalone})"
        )

    def _is_rpc_leader(self) -> bool:
        """Return True if this process should issue RPC broadcast calls."""
        if self._standalone:
            return True
        return torch.distributed.get_rank() == 0

    def get_input_token_keys(self, rm_idx: int):
        """Return the input_token_key list for a given RM.

        Parameters
        ----------
        rm_idx : int

        Returns
        -------
        list of str or None
            Routing keys, or None for broadcast mode.
        """
        return self.config.gen_rm.reward_model_info[rm_idx].input_token_key

    @override
    async def _batch_rpc_call(self, rm_idx: int, action: str, req_data: dict, timeout: int = None):
        """Broadcast Ray lifecycle RPCs to every engine actor for this gen-RM."""
        if self.rpc_type != "ray":
            return await RmClientMixin._batch_rpc_call(self, rm_idx, action, req_data, timeout)
        return await asyncio.gather(
            *[
                self.rpc_client_lst[rm_idx].call(actor, action, req_data)
                for actor in self.all_engine_actors_per_rm[rm_idx]
            ]
        )

    @override
    async def mark_ppo_step_begin(self, rm_idx, ppo_step):
        if self._is_rpc_leader():
            req_dict = {
                'ppo_step': ppo_step,
                'actor_dp_rank': self.dp_rank,
                'tag_names': ['weights', 'kv_cache'],
            }
            await self._batch_rpc_call(rm_idx, 'mark_ppo_step_begin', req_dict)

    @override
    async def mark_ppo_step_end(self, rm_idx, ppo_step):
        if self._is_rpc_leader():
            req_dict = {
                'ppo_step': ppo_step,
                'actor_dp_rank': self.dp_rank,
            }
            await self._batch_rpc_call(rm_idx, 'mark_ppo_step_end', req_dict)

    @override
    async def wake_up(self, rm_idx, tag_names=None):
        assert self.config.placement_type != "disaggregated"
        if self._is_rpc_leader():
            if tag_names is None:
                tag_names = ['weights', 'kv_cache']

            req_dict = {
                'tag_names': tag_names,
            }
            await self._batch_rpc_call(rm_idx, 'wake_up', req_dict)

    @override
    async def sleep(self, rm_idx):
        assert self.config.placement_type != "disaggregated"
        if self._is_rpc_leader():
            await self._batch_rpc_call(rm_idx, 'sleep', {})

    async def generate_rewards(self, rm_idx, ppo_step, sample_idx,
                               batched_data) -> Dict[str, List[Any]]:
        """Request reward scores from a generative reward model.

        Parameters
        ----------
        rm_idx : int
        ppo_step : int
        sample_idx : int
        batched_data : dict

        Returns
        -------
        dict[str, list]
            Reward outputs.
        """
        target_ep = self.rpc_client_lst[rm_idx].get_target_endpoint(
            sample_idx=sample_idx, ep_idx=None
        )
        req_dict = {
            'actor_dp_rank': self.dp_rank,
            'actor_dp_size': self.dp_size,
            'ppo_step': ppo_step,
            'sample_idx': sample_idx,
            'batched_data': batched_data,
        }
        fut = self.rpc_client_lst[rm_idx].call(target_ep, 'generate_rewards', req_dict)
        resp = await fut
        log(f'GenRmClient.generate {ppo_step=} {sample_idx=}')
        return resp

    async def infer_engine_flush_cache(self, rm_idx):
        """Flush the inference engine KV cache for a gen-RM.

        Parameters
        ----------
        rm_idx : int
        """
        if self._is_rpc_leader():
            await self._batch_rpc_call(rm_idx, 'flush_cache', {})
