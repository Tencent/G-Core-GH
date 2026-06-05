import asyncio
from typing import Any, Dict, List

import torch
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.client.base_client import BaseClientAbc, RmClientMixin
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils.common_utils import log, logging_rank0


class BaseBtRmClient(BaseClientAbc, RmClientMixin):
    """Base client for communicating with Bradley-Terry reward model engines.

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

        self.ep_ips = policy_config.bt_rm_client.endpoint_ips
        self.ep_ports = policy_config.bt_rm_client.endpoint_ports
        self.timeout = policy_config.bt_rm_client.rpc_timeout
        self.rpc_type = policy_config.bt_rm_client.rpc_type

        self.dp_rank = dp_rank if dp_rank is not None else mpu.get_data_parallel_rank()
        self.dp_size = dp_size if dp_size is not None else mpu.get_data_parallel_world_size()

        # When dp_rank/dp_size are explicitly provided, we are running in
        # standalone / controller mode without torch.distributed.
        self._standalone = dp_rank is not None

        rpc_client_lst = []
        svr_cluster_num_per_rm = []
        bt_rm_config = self.config.bt_rm
        num_rms = len(bt_rm_config.reward_model_info)
        self.num_rms = num_rms

        for rm_idx in range(num_rms):
            if self.rpc_type == "ray":
                infer_engine_config = bt_rm_config.infer_engine_configs[rm_idx]
                dist_config = infer_engine_config.dist_config
                num_clusters, num_engines, num_engine_per_cluster = self.get_engine_info(
                    dist_config
                )

                rpc_client = self.build_rpc_client(
                    self.rpc_type,
                    None,
                    None,
                    num_clusters,
                    num_engine_per_cluster,
                    ray_actor_pname=f"bt_rm_{rm_idx}",
                    dp_rank=self.dp_rank,
                    dp_size=self.dp_size,
                )
                svr_cluster_num_per_rm.append(num_clusters)
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
        logging_rank0(
            f"{self.__class__.__name__} init with {self.rpc_type} client (standalone={self._standalone})"
        )

    def _is_rpc_leader(self) -> bool:
        """Return True if this process should issue RPC broadcast calls."""
        if self._standalone:
            return True
        return torch.distributed.get_rank() == 0

    #TODO: impl BtRmClient func
    @override
    async def mark_ppo_step_begin(self, rm_idx, ppo_step):
        if self._is_rpc_leader():
            req_dict = {
                'ppo_step': ppo_step,
                'actor_dp_rank': self.dp_rank,
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
            await self._batch_rpc_call(rm_idx, 'wake_up')

    @override
    async def sleep(self, rm_idx):
        assert self.config.placement_type != "disaggregated"
        if self._is_rpc_leader():
            await self._batch_rpc_call(rm_idx, 'sleep')


class BtRmClient(BaseBtRmClient):
    """BT reward model client with issue/get result pattern.

    Parameters
    ----------
    config : RlConfig
    dp_rank : int, optional
    dp_size : int, optional
    """
    def __init__(self, config: RlConfig, dp_rank: int = None, dp_size: int = None):
        super().__init__(config, dp_rank=dp_rank, dp_size=dp_size)

    async def issue_bt_rm(self, batched_data: Dict[str, List[Any]], rm_idx, ppo_step, sample_idx):
        """Submit data to the BT reward model for scoring.

        Parameters
        ----------
        batched_data : dict
        rm_idx : int
        ppo_step : int
        sample_idx : int

        Returns
        -------
        dict
            Response with ``ret=True`` on success.
        """
        target_ep = self.rpc_client_lst[rm_idx].get_target_endpoint(
            sample_idx=sample_idx, ep_idx=None
        )
        req_dict = {
            "actor_dp_rank": self.dp_rank,
            "sample_idx": sample_idx,
            "ppo_step": ppo_step,
        }
        for k in req_dict.keys():
            assert k not in batched_data
        req_dict.update(batched_data)
        rpc_co = self.rpc_client_lst[rm_idx].call(target_ep, 'issue_bt_rm', req_dict)
        resp = await rpc_co
        assert resp["ret"] is True
        return resp

    async def get_bt_rm_result(self, rm_idx, ppo_step, sample_idx) -> Dict[str, Any]:
        """Retrieve BT reward model results.

        Parameters
        ----------
        rm_idx : int
        ppo_step : int
        sample_idx : int

        Returns
        -------
        dict[str, Any]
            Reward results with ``'rewards'``, ``'values'``, etc.
        """
        target_ep = self.rpc_client_lst[rm_idx].get_target_endpoint(
            sample_idx=sample_idx, ep_idx=None
        )
        req_dict = {
            "actor_dp_rank": self.dp_rank,
            "sample_idx": sample_idx,
            "ppo_step": ppo_step,
            "sampling_repeat_n": self.config.training.sampling_repeat_n
        }
        rpc_co = self.rpc_client_lst[rm_idx].call(target_ep, 'get_bt_rm_result', req_dict)
        resp = await rpc_co
        resp = self.post_process_resp(resp)
        return resp

    def post_process_resp(self, resp: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process BT-RM response (truncate values and per-token rewards).

        Parameters
        ----------
        resp : dict

        Returns
        -------
        dict
            Post-processed response.
        """
        if resp.get("values", None) is not None and resp["values"][0] is not None:
            values = resp["values"]
            assert values[0].ndim == 1, f"ndim {values[0].ndim}"
            if self.config.ppo.ppo_value_truncate_head:
                values = [v[1:].contiguous() for v in values]
            else:
                values = [v[:-1].contiguous() for v in values]
            resp["values"] = values
        if resp.get("per_token_rewards",
                    None) is not None and resp["per_token_rewards"][0] is not None:
            per_token_rewards = resp["per_token_rewards"]
            assert per_token_rewards[0].ndim == 1, f"ndim {per_token_rewards[0].ndim}"
            if self.config.ppo.ppo_value_truncate_head:
                per_token_rewards = [ptr[1:].contiguous() for ptr in per_token_rewards]
            else:
                per_token_rewards = [ptr[:-1].contiguous() for ptr in per_token_rewards]
            resp["per_token_rewards"] = per_token_rewards

        if resp.get('rewards', None) is not None and resp["rewards"][0] is not None:
            report_data = {"rm_infer_results": len(resp['rewards'])}
            #TODO: report_ppo_metrics(report_data)
        return resp
