import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import zmq
import zmq.asyncio

from gpatch.rpc import call_once_rpc
from gpatch_v4.configs.config import MappingProtocol
from gpatch_v4.rpc_client.base_rpc_client import HttpRpcClient, RpcClient, ZeroMqRpcClient
from gpatch_v4.rpc_client.multi_cast_ray_rpc_client import MultiCastRayRpcClient
from gpatch_v4.rpc_client.ray_rpc_client import RayRpcClient


class RpcClientFactory:
    @staticmethod
    def create_client(
        config: MappingProtocol,
        rpc_type: str,
        ep_ips: List[str],
        ep_ports: List[int],
        num_clusters: int,
        num_engine_per_cluster: int,
        ray_actor_pname: str = None,
        multi_cast=False,
        dp_rank: int = None,
        dp_size: int = None,
    ) -> RpcClient:
        assert config is not None, "config must be passed to RpcClientFactory.create_client"
        if rpc_type == "http":
            assert not multi_cast, "http does not support multi cast"
            return HttpRpcClient(
                config,
                rpc_type=rpc_type,
                ep_ips=ep_ips,
                ep_ports=ep_ports,
                num_clusters=num_clusters,
                ray_actor_pname=ray_actor_pname,
                dp_rank=dp_rank,
                dp_size=dp_size,
            )
        elif rpc_type == "zeromq":
            assert not multi_cast, "http does not support multi cast"
            return ZeroMqRpcClient(
                config,
                rpc_type=rpc_type,
                ep_ips=ep_ips,
                ep_ports=ep_ports,
                num_clusters=num_clusters,
                ray_actor_pname=ray_actor_pname,
                dp_rank=dp_rank,
                dp_size=dp_size,
            )
        elif rpc_type == "ray":
            if multi_cast is False:
                return RayRpcClient(
                    config,
                    rpc_type=rpc_type,
                    ep_ips=ep_ips,
                    ep_ports=ep_ports,
                    num_clusters=num_clusters,
                    num_engine_per_cluster=num_engine_per_cluster,
                    ray_actor_pname=ray_actor_pname,
                    dp_rank=dp_rank,
                    dp_size=dp_size,
                )
            else:
                return MultiCastRayRpcClient(
                    config,
                    rpc_type=rpc_type,
                    ep_ips=ep_ips,
                    ep_ports=ep_ports,
                    num_clusters=num_clusters,
                    num_engine_per_cluster=num_engine_per_cluster,
                    ray_actor_pname=ray_actor_pname,
                    dp_rank=dp_rank,
                    dp_size=dp_size,
                )
        else:
            raise ValueError(f"Unsupported RPC type: {rpc_type}")
