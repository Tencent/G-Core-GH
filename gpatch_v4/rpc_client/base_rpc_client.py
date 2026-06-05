import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import zmq
import zmq.asyncio
from typing_extensions import override

from megatron.core import mpu

from gpatch.rpc import call_once_rpc
from gpatch_v4.configs.config import MappingProtocol


class RpcClient(ABC):
    """ RPC client abstract class """
    def __init__(
        self,
        config: MappingProtocol,
        rpc_type: str,
        ep_ips: List[str],
        ep_ports: List[int],
        num_clusters: int,
        ray_actor_pname: str = None,
        dp_rank: int = None,
        dp_size: int = None,
    ):
        self.config = config
        self.rpc_type = rpc_type
        self.ep_ips = ep_ips
        self.ep_ports = ep_ports
        self.num_clusters = num_clusters
        self.ray_actor_pname = ray_actor_pname
        self.dp_size = dp_size if dp_size is not None else mpu.get_data_parallel_world_size()
        self.dp_rank = dp_rank if dp_rank is not None else mpu.get_data_parallel_rank()

    @abstractmethod
    async def call(
        self,
        target_ep,
        action: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """ Execute RPC call """
        ...

    @abstractmethod
    def get_target_endpoint(self, sample_idx: int, ep_idx: int = None):
        """ Get target endpoint """
        ...

    def _pick_endpoint_idx(self, sample_idx: int):
        """ Pick endpoint index """
        assert self.num_clusters is not None, f"{self.num_clusters=} is None"
        dp_size = self.dp_size
        dp_rank = self.dp_rank
        server_dp_size = self.num_clusters
        if dp_size == server_dp_size:
            ep_idx = dp_rank
        elif dp_size < server_dp_size:
            ep_idx = dp_rank * (server_dp_size // dp_size) + sample_idx
        else:
            ep_idx = dp_rank // (dp_size // server_dp_size) + sample_idx
        ep_idx %= server_dp_size
        return ep_idx


class HttpRpcClient(RpcClient):
    """ HTTP RPC client implementation """
    def __init__(
        self,
        config: MappingProtocol,
        rpc_type: str,
        ep_ips: List[str],
        ep_ports: List[int],
        num_clusters: int,
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
        assert len(ep_ips) == len(ep_ports), f"{ep_ips=} and {ep_ports=} must have the same length"
        self.num_clusters = len(ep_ips)

    @override
    async def call(
        self,
        target_ep,
        action: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """ Execute HTTP RPC call """
        url = f"{target_ep}/{action}"
        return await call_once_rpc(url, data)

    @override
    def get_target_endpoint(self, sample_idx: int, ep_idx: int = None):
        """ Build HTTP URL """
        if ep_idx is None:
            target_ep_idx = self._pick_endpoint_idx(sample_idx=sample_idx)
        else:
            target_ep_idx = ep_idx
        target_ip = self.ep_ips[target_ep_idx]
        target_port = self.ep_ports[target_ep_idx]
        return f'http://{target_ip}:{target_port}'


class ZeroMqRpcClient(RpcClient):
    """ ZeroMQ RPC client implementation """
    def __init__(
        self,
        config: MappingProtocol,
        rpc_type: str,
        ep_ips: List[str],
        ep_ports: List[int],
        num_clusters: int,
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
        # ZeroMQ-specific initialization logic
        # This needs to be completed based on the actual ZeroMQ implementation
        #TODO: build socket
        raise NotImplementedError("ZeroMQ RPC client is not implemented")

    @override
    async def call(
        self,
        target_ep,
        action: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError("ZeroMQ RPC call is not implemented")

    @override
    def get_target_endpoint(self, sample_idx: int, ep_idx: int = None):
        """ build ZMQ URL """
        raise NotImplementedError("ZeroMQ RPC endpoint is not implemented")
