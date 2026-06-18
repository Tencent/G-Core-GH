import asyncio
from typing import Any, Dict, List

import ray
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.configs.config import MappingProtocol
from gpatch_v4.rpc_client.base_rpc_client import RpcClient


class RayRpcClient(RpcClient):
    """ Ray client implementation """
    def __init__(
        self,
        config: MappingProtocol,
        rpc_type: str,
        ep_ips: List[str],
        ep_ports: List[int],
        num_clusters: int,
        num_engine_per_cluster: int,
        ray_actor_pname: str = None,
        dp_rank: int = None,
        dp_size: int = None,
    ):
        super().__init__(
            config,
            rpc_type,
            ep_ips,
            ep_ports,
            num_clusters,
            ray_actor_pname,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
        self.infer_engines = []
        self.num_clusters = num_clusters
        for i in range(self.num_clusters):
            actor = ray.get_actor(f"{ray_actor_pname}_{i * num_engine_per_cluster}")
            self.infer_engines.append(actor)

    @override
    async def call(
        self,
        target_ep,
        action: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if hasattr(target_ep, action):
            func = getattr(target_ep, action)
            ret = await func.remote(req_dict=data)
            return ret
        else:
            raise NotImplementedError(f"error func name {action}")

    @override
    def get_target_endpoint(self, sample_idx: int, ep_idx: int = None):
        if ep_idx is None:
            target_ep_idx = self._pick_endpoint_idx(sample_idx)
        else:
            target_ep_idx = ep_idx
        return self.infer_engines[target_ep_idx]

    def fire(
        self,
        target_ep,
        action: str,
        data: Dict[str, Any],
    ):
        """Submit a Ray remote call without awaiting.

        Returns
        -------
        ray.ObjectRef
        """
        func = getattr(target_ep, action)
        return func.remote(req_dict=data)
