import asyncio
from typing import Any, Dict, List

import ray
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.configs.config import MappingProtocol
from gpatch_v4.rpc_client.base_rpc_client import RpcClient
from gpatch_v4.rpc_client.ray_rpc_client import RayRpcClient


class MultiCastRayRpcClient(RpcClient):
    """ 支持并行处理多个actor的Ray RPC客户端， 比如向 mcore 作为 server 节点发送 ray rpc 请求，和  RayRpcClient 略有不同 """

    # 先这么写着，不要求很干净了，后面再美化

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
        self.infer_engines = [[] for _ in range(num_clusters)]
        self.num_clusters = num_clusters
        for dp_i in range(self.num_clusters):
            for r_j in range(num_engine_per_cluster):
                actor = ray.get_actor(f"{ray_actor_pname}_{dp_i}_{r_j}")
                self.infer_engines[dp_i].append(actor)

    @override
    async def call(
        self,
        target_ep,
        action: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        futs = []
        for ti, tep in enumerate(target_ep):
            if hasattr(tep, action):
                func = getattr(tep, action)
                _data = data
                # if ti != 0:
                #     # 只需要向 mp_and_cp_head 发送真实数据
                #     _data = {}
                co = func.remote(_data)
                futs.append(co)
            else:
                raise NotImplementedError(f"error func name {action} {tep}")
        assert len(futs
                  ) == len(target_ep), f"len(futs) {len(futs)} != len(target_ep) {len(target_ep)}"

        # TODO(@xiaotaoliu): ray 是 remote 就 create task 了吗？还是这里会出现 block ？
        ret_lst = []
        for fut in futs:
            ret = await fut
            ret_lst.append(ret)
        return ret_lst[0]

    @override
    def get_target_endpoint(self, sample_idx: int, ep_idx: int = None):
        if ep_idx is None:
            target_ep_idx = self._pick_endpoint_idx(sample_idx)
        else:
            target_ep_idx = ep_idx
        return self.infer_engines[target_ep_idx]
