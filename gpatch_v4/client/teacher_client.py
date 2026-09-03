import asyncio
from typing import Any, Dict, List

import torch
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.client.base_client import BaseClientAbc, TeacherClientMixin
from gpatch_v4.configs.config import OffPolicyDistillConfig, OnPolicyDistillConfig
from gpatch_v4.utils.common_utils import log, logging_rank0


def get_mp_cp_rank():
    """Compute the combined model-parallel + context-parallel rank.

    Returns
    -------
    int
    """
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    tp_cp_size = tp_size * cp_size

    mp_cp_rank = pp_rank * tp_cp_size + cp_rank * tp_size + tp_rank
    return mp_cp_rank


class TeacherClient(BaseClientAbc, TeacherClientMixin):
    """Client for communicating with a teacher model.

    Parameters
    ----------
    config : OnPolicyDistillConfig or OffPolicyDistillConfig
    teacher_name : str or None
        Named teachers use the ``teacher_{teacher_name}`` Ray actor prefix;
        the singular off-policy teacher uses ``teacher``.
    teacher_config : object
        The teacher's ``BasePolicyConfig`` (or dict equivalent).
    """
    def __init__(
        self,
        config: OnPolicyDistillConfig | OffPolicyDistillConfig,
        teacher_name: str | None,
        teacher_config,
    ):
        self.config = config
        self.teacher_name = teacher_name
        policy_config = self.config.policy

        self.ep_ips = policy_config.teacher_client.endpoint_ips
        self.ep_ports = policy_config.teacher_client.endpoint_ports
        self.timeout = policy_config.teacher_client.rpc_timeout
        self.rpc_type = policy_config.teacher_client.rpc_type

        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_size = mpu.get_data_parallel_world_size()

        ray_actor_pname = ("teacher" if teacher_name is None else f"teacher_{teacher_name}")

        rpc_client = None
        svr_cluster = None

        if self.rpc_type == "ray":
            if isinstance(teacher_config, dict):
                from gpatch_v4.configs.dist_config import DistConfig
                dc = teacher_config.get('dist_config', {})
                dist_config = DistConfig(**dc) if isinstance(dc, dict) else dc
            else:
                dist_config = teacher_config.dist_config
            dp_size, mp_cp_size = self.get_mcore_svr_infos(dist_config)
            log(f"Teacher client [{ray_actor_pname}] {dp_size=} {mp_cp_size=}", rank=0)

            rpc_client = self.build_rpc_client(
                self.rpc_type, None, None, dp_size, mp_cp_size, ray_actor_pname=ray_actor_pname
            )
            svr_cluster = dp_size
        else:
            raise ValueError(f"Invalid rpc type: {self.rpc_type}")

        self.rpc_client = rpc_client
        self.svr_cluster = svr_cluster

        self._check_rollout_gbs_covers_all_teacher_ep_ranks(dist_config, svr_cluster)

    def _check_rollout_gbs_covers_all_teacher_ep_ranks(self, dist_config, server_dp_size):
        """Assert that rollout_gbs is large enough so every teacher EP group is
        either fully active or fully idle.  A partially-active EP group will
        deadlock on the all_reduce inside get_max_seqlen_within_ep."""
        ep_size = getattr(dist_config, 'expert_model_parallel_size', 1)
        if ep_size <= 1:
            return

        client_dp_size = self.dp_size
        rollout_gbs = self.config.training.rollout_gbs
        samples_per_dp_rank = rollout_gbs // client_dp_size

        covered = set()
        for dp_rank in range(client_dp_size):
            for sample_idx in range(samples_per_dp_rank):
                if client_dp_size == server_dp_size:
                    ep_idx = dp_rank
                elif client_dp_size < server_dp_size:
                    ep_idx = dp_rank * (server_dp_size // client_dp_size) + sample_idx
                else:
                    ep_idx = dp_rank // (client_dp_size // server_dp_size) + sample_idx
                covered.add(ep_idx % server_dp_size)

        num_ep_groups = server_dp_size // ep_size
        for g in range(num_ep_groups):
            group_members = set(range(g * ep_size, (g + 1) * ep_size))
            hit = group_members & covered
            if hit and hit != group_members:
                missing = sorted(group_members - covered)
                assert False, (
                    f"rollout_gbs={rollout_gbs} causes partial activation of teacher "
                    f"EP group {g} (ep_size={ep_size}): only {len(hit)}/{ep_size} "
                    f"clusters receive data, missing clusters {missing}. "
                    f"This will deadlock on all_reduce in get_max_seqlen_within_ep. "
                    f"Increase rollout_gbs so that every teacher EP group is fully covered "
                    f"(need rollout_gbs >= server_dp_size={server_dp_size})."
                )

    def get_mcore_svr_infos(self, dist_config):
        """Compute DP size and MP-CP size from the teacher dist config.

        Parameters
        ----------
        dist_config : object

        Returns
        -------
        tuple[int, int]
            ``(num_clusters, mp_cp_size)``.
        """
        svr_mp_cp_size = dist_config.tensor_model_parallel_size * dist_config.pipeline_model_parallel_size * dist_config.context_parallel_size
        svr_world_size = dist_config.nnodes * dist_config.num_gpus_per_node
        num_clusters = svr_world_size // svr_mp_cp_size
        return num_clusters, svr_mp_cp_size

    @override
    async def mark_ppo_step_begin(self, teacher_idx, ppo_step):
        if torch.distributed.get_rank() == 0:
            req_dict = {
                'ppo_step': ppo_step,
                'actor_dp_rank': self.dp_rank,
            }
            await self._batch_rpc_call(teacher_idx, 'mark_ppo_step_begin', req_dict)

    @override
    async def mark_ppo_step_end(self, teacher_idx, ppo_step):
        if torch.distributed.get_rank() == 0:
            req_dict = {
                'ppo_step': ppo_step,
                'actor_dp_rank': self.dp_rank,
            }
            await self._batch_rpc_call(teacher_idx, 'mark_ppo_step_end', req_dict)

    @override
    async def wake_up(self, teacher_idx, tag_names=None):
        if torch.distributed.get_rank() == 0:
            await self._batch_rpc_call(teacher_idx, 'wake_up', {})

    @override
    async def sleep(self, teacher_idx):
        if torch.distributed.get_rank() == 0:
            await self._batch_rpc_call(teacher_idx, 'sleep', {})

    async def issue_calc_logps(self, batched_data: Dict[str, List[Any]], ppo_step, sample_idx):
        """Submit a log-probability computation request to the teacher.

        Parameters
        ----------
        batched_data : dict
        ppo_step : int
        sample_idx : int

        Returns
        -------
        dict
            Response with ``ret=True`` on success.
        """
        target_ep = self.rpc_client.get_target_endpoint(sample_idx=sample_idx, ep_idx=None)
        req_dict = {
            "actor_dp_rank": self.dp_rank,
            "sample_idx": sample_idx,
            "ppo_step": ppo_step,
        }
        for k in req_dict.keys():
            assert k not in batched_data
        req_dict.update(batched_data)
        resp = await self.rpc_client.call(target_ep, "issue_calc_logps", req_dict)
        assert resp["ret"] is True
        return resp

    async def get_calc_logps_result(self, ppo_step, sample_idx) -> Dict[str, Any]:
        """Retrieve teacher log-probability results.

        Parameters
        ----------
        ppo_step : int
        sample_idx : int

        Returns
        -------
        dict[str, Any]
            Teacher log-prob results.
        """
        target_ep = self.rpc_client.get_target_endpoint(sample_idx=sample_idx, ep_idx=None)
        req_dict = {
            "actor_dp_rank": self.dp_rank,
            "sample_idx": sample_idx,
            "ppo_step": ppo_step,
        }
        resp = await self.rpc_client.call(target_ep, "get_calc_logps_result", req_dict)
        return resp

    async def issue_calc_hidden_states(
        self, batched_data: Dict[str, List[Any]], ppo_step, sample_idx
    ):
        """Submit a hidden-state computation request to the teacher.

        Parameters
        ----------
        batched_data : dict
        ppo_step : int
        sample_idx : int

        Returns
        -------
        dict
            Response with ``ret=True`` on success.
        """
        target_ep = self.rpc_client.get_target_endpoint(sample_idx=sample_idx, ep_idx=None)
        req_dict = {
            "actor_dp_rank": self.dp_rank,
            "sample_idx": sample_idx,
            "ppo_step": ppo_step,
        }
        for k in req_dict.keys():
            assert k not in batched_data
        req_dict.update(batched_data)
        resp = await self.rpc_client.call(target_ep, "issue_calc_hidden_states", req_dict)
        assert resp["ret"] is True
        return resp

    async def get_calc_hidden_states_result(self, ppo_step, sample_idx) -> Dict[str, Any]:
        """Retrieve teacher hidden-state results.

        Parameters
        ----------
        ppo_step : int
        sample_idx : int

        Returns
        -------
        dict[str, Any]
            Teacher logit results.
        """
        target_ep = self.rpc_client.get_target_endpoint(sample_idx=sample_idx, ep_idx=None)
        req_dict = {
            "actor_dp_rank": self.dp_rank,
            "sample_idx": sample_idx,
            "ppo_step": ppo_step,
        }
        resp = await self.rpc_client.call(target_ep, "get_calc_hidden_states_result", req_dict)
        return resp
