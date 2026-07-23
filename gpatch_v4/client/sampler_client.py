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
from gpatch_v4.client.mixin import TestFuncMixin, UpdateWeightMixin
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.utils import log, logging_rank0, perf_time
from gpatch_v4.utils.placement import is_partial_colocated, use_ipc_weight_update


class SamplerClient(
    BaseClientAbc,
    SamplerClientMixin,
    UpdateWeightMixin,
    TestFuncMixin,
):
    """Client for communicating with sampler inference engines.

    Supports both Ray and HTTP RPC backends.

    Parameters
    ----------
    config : RlConfig
    dp_rank : int, optional
        Data-parallel rank override.  When provided, the client does **not**
        query ``mpu`` and can run outside a Megatron parallel context (e.g.
        inside a ``RolloutController``).
    dp_size : int, optional
    """
    def __init__(
        self,
        config: RlConfig,
        dp_rank: int = None,
        dp_size: int = None,
        skip_init_ipc_meta: bool = False
    ):
        #TODO: ep ips and ports
        self.config = config

        policy_config = self.config.policy
        self.ep_ips = policy_config.sampler_client.endpoint_ips
        self.ep_ports = policy_config.sampler_client.endpoint_ports
        self.timeout = policy_config.sampler_client.rpc_timeout
        self.rpc_type = policy_config.sampler_client.rpc_type

        self.update_weight_max_size_bytes = policy_config.sampler_client.update_weight_max_size_mb * (
            1024**2
        )
        # self.update_weight_max_size_bytes = 1
        self.update_weight_use_bucketed_ipc = policy_config.sampler_client.update_weight_use_bucketed_ipc

        self.dp_rank = dp_rank if dp_rank is not None else mpu.get_data_parallel_rank()
        self.dp_size = dp_size if dp_size is not None else mpu.get_data_parallel_world_size()

        # When dp_rank/dp_size are explicitly provided, we are running in
        # standalone / controller mode without torch.distributed.
        self._standalone = dp_rank is not None

        rpc_client_lst = []
        svr_cluster_num_per_sampler = []
        sampler_config = self.config.sampler
        self.num_samplers = len(sampler_config.model_info)

        for sampler_idx in range(self.num_samplers):
            if self.rpc_type == "ray":
                infer_engine_config = sampler_config.infer_engine_configs[sampler_idx]
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
                    ray_actor_pname=f"sampler_{sampler_idx}",
                    dp_rank=self.dp_rank,
                    dp_size=self.dp_size,
                )
                svr_cluster_num_per_sampler.append(num_clusters)
            elif self.rpc_type == "http":
                # self.ep_ips 是一个 list of list of str
                # 第一个 list 维度是 num_rms, 第二个 list 维度是 num_of cluster
                rpc_client = self.build_rpc_client(
                    self.rpc_type, self.ep_ips[sampler_idx], self.ep_ports[sampler_idx], None, None,
                    None
                )
                svr_cluster_num_per_sampler.append(len(self.ep_ips[sampler_idx]))
            else:
                raise ValueError(f"invalid rpc type: {self.rpc_type}")
            rpc_client_lst.append(rpc_client)

        assert len(rpc_client_lst) == self.num_samplers
        self.rpc_client_lst = rpc_client_lst
        self.svr_cluster_num_per_sampler = svr_cluster_num_per_sampler

        self.infer_backend = getattr(self.config.sampler, "backend", "sglang")
        self.skip_init_ipc_meta = skip_init_ipc_meta
        self._use_ipc_weight_update = use_ipc_weight_update(self.config)
        if self._use_ipc_weight_update and not self.skip_init_ipc_meta:
            self.build_update_from_tensor_meta()
        log(
            f"SamplerClient init with {self.rpc_type} client "
            f"{self.update_weight_max_size_bytes} bytes "
            f"(weight_update={'ipc' if self._use_ipc_weight_update else 'nccl'}, "
            f"backend={self.infer_backend}, standalone={self._standalone}, "
            f"bucketed_ipc={self.update_weight_use_bucketed_ipc})"
        )

    def _is_rpc_leader(self) -> bool:
        """Return True if this process should issue RPC broadcast calls.

        In standalone / controller mode, always True (single process).
        In distributed mode, only rank 0 issues the calls.
        """
        if self._standalone:
            return True
        return torch.distributed.get_rank() == 0

    async def maybe_init_distributed_weight_group_for_disagg(self):
        """Init NCCL weight group for distributed weight update (disaggregated or vllm)."""
        if not self._use_ipc_weight_update:
            await self.init_distributed_weight_group()

    def build_update_from_tensor_meta(self):
        """Build distributed groups for IPC-based weight transfer."""
        # 目前基于只有一个同构 sampler 做
        sampler_config = self.config.sampler
        first_dist_config = sampler_config.infer_engine_configs[0].dist_config
        first_sampler_mp_size = first_dist_config.tensor_model_parallel_size * first_dist_config.pipeline_model_parallel_size
        num_clusters, _, _ = self.get_engine_info(first_dist_config)
        self._ipc_gather_dst_rank = None
        self._ipc_gather_group = None
        self._ipc_target = None

        for idx in range(num_clusters):
            start_rank = idx * first_sampler_mp_size
            end_rank = (idx + 1) * first_sampler_mp_size
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(
                ranks=group_ranks,
                backend="gloo",
            )
            if dist.get_rank() in group_ranks:
                self._ipc_gather_dst_rank = start_rank
                self._ipc_gather_group = new_group
                self._ipc_target = idx

        assert is_partial_colocated(self.config) or self._ipc_gather_group is not None, (
            "IPC weight update rank did not map to any sampler gather group"
        )

    async def generate(
        self,
        sampler_idx: int,
        ppo_step: int,
        sidx: int,
        batched_data: Dict[str, Any],
        repeat_n: int,
        load_aware: bool = False,
        busy_sleep_s: float = 10.0,
    ) -> Dict[str, List[Any]]:
        """Send a generation request to the sampler.

        When ``load_aware`` is *True*, queries all cluster loads first
        and picks the cluster with the fewest waiting requests.

        Parameters
        ----------
        sampler_idx : int
        ppo_step : int
        sidx : int
            Sample index within the current batch.
        batched_data : dict
        repeat_n : int
        load_aware : bool
            If *True*, route to the least busy cluster.
        busy_sleep_s : float
            Seconds to sleep when all clusters are busy (only used when
            ``load_aware=True``).

        Returns
        -------
        dict[str, list]
            Generated outputs.
        """

        default_ep_idx = self.rpc_client_lst[sampler_idx]._pick_endpoint_idx(sidx)
        ep_idx = default_ep_idx

        if load_aware:
            delay_s = self._get_load_aware_dispatch_delay(sidx)
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            loads = await self.get_all_loads(sampler_idx)
            if loads[ep_idx]["num_waiting_reqs"] > 0:
                num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
                best = min(
                    range(num_clusters),
                    key=lambda i: (loads[i]["num_waiting_reqs"], loads[i]["num_reqs"]),
                )
                if loads[best]["num_waiting_reqs"] > 0:
                    log(
                        f"[SamplerClient] all {num_clusters} clusters busy "
                        f"(waiting_reqs={[ld['num_waiting_reqs'] for ld in loads]}), "
                        f"sleeping {busy_sleep_s}s"
                    )
                    await asyncio.sleep(busy_sleep_s)
                ep_idx = best

        target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
            sample_idx=None, ep_idx=ep_idx
        )

        req_dict = {
            'actor_dp_rank': self.dp_rank,
            'actor_dp_size': self.dp_size,
            'sampler_dp_rank': sampler_idx,
            'ppo_step': ppo_step,
            'sample_idx': sidx,
            'sampling_repeat': repeat_n,
            'batched_data': batched_data,
        }
        fut = self.rpc_client_lst[sampler_idx].call(target_ep, 'generate', req_dict)
        resp = await fut
        log(
            f'SamplerClient.generate {ppo_step=} sample_idx={sidx} default_ep_idx={default_ep_idx} select_ep_idx={ep_idx}'
        )
        return resp

    @override
    async def mark_ppo_step_begin(self, sampler_idx, ppo_step):
        if self._is_rpc_leader():
            req_dict = {
                'ppo_step': ppo_step,
                'actor_dp_rank': self.dp_rank,
                'tag_names': ['weights', 'kv_cache'],
            }
            await self._batch_rpc_call(sampler_idx, 'mark_ppo_step_begin', req_dict)

    @override
    async def mark_ppo_step_end(self, sampler_idx, ppo_step):
        if self._is_rpc_leader():
            req_dict = {
                'ppo_step': ppo_step,
                'actor_dp_rank': self.dp_rank,
            }
            await self._batch_rpc_call(sampler_idx, 'mark_ppo_step_end', req_dict)

    @override
    async def wake_up(self, sampler_idx, tag_names=None):
        assert self.config.placement_type != "disaggregated"
        if self._is_rpc_leader():
            if tag_names is None:
                tag_names = ['weights', 'kv_cache']
            req_dict = {
                'tag_names': tag_names,
            }
            await self._batch_rpc_call(sampler_idx, 'wake_up', req_dict)

    @override
    async def sleep(self, sampler_idx):
        assert self.config.placement_type != "disaggregated"
        if self._is_rpc_leader():
            await self._batch_rpc_call(sampler_idx, 'sleep', {})

    async def update_weights(self, model_engine, replace_zeros=False):
        """Wake up the sampler (if colocated) and push updated model weights.

        * colocated (``placement_type != "disaggregated"``) →
          :meth:`UpdateWeightMixin.update_weights_by_ipc_handle`
        * disaggregated →
          :meth:`UpdateWeightMixin.update_weights_by_distributed`

        Parameters
        ----------
        model_engine : object
            Training engine with ``export_weights()`` generator.
        replace_zeros : bool, optional

        Returns
        -------
        bool
        """
        # 目前只有一种同构 sampler, 所以只调 sampler_idx = 0，如果是异构的也不能update weights
        sampler_idx = 0
        if self.config.placement_type != "disaggregated":
            log("before wake up sampler weight")
            await self.wake_up(sampler_idx, tag_names=["weights"])
            cpu_barrier()
            log("after wake up sampler weight")

        if self._use_ipc_weight_update:
            resp = await self.update_weights_by_ipc_handle(sampler_idx, model_engine, replace_zeros)
        else:
            if self.infer_backend == "sglang":
                await self._release_kv_cache_for_distributed_update(sampler_idx)
                cpu_barrier()
            resp = self.update_weights_by_distributed(sampler_idx, model_engine, replace_zeros)
            if self.infer_backend == "sglang":
                await self._resume_kv_cache_after_distributed_update(sampler_idx)
                cpu_barrier()

        return resp

    async def _release_kv_cache_for_distributed_update(self, sampler_idx):
        if self._is_rpc_leader():
            await self._batch_rpc_call(sampler_idx, "release_kv_cache_for_weight_update", {})

    async def _resume_kv_cache_after_distributed_update(self, sampler_idx):
        if self._is_rpc_leader():
            await self._batch_rpc_call(sampler_idx, "resume_kv_cache_after_weight_update", {})

    async def infer_engine_flush_cache(self, sampler_idx):
        """Flush the inference engine KV cache.

        Parameters
        ----------
        sampler_idx : int
        """
        if self._is_rpc_leader():
            await self._batch_rpc_call(sampler_idx, 'flush_cache', {})

    async def get_all_loads(self, sampler_idx: int = 0) -> List[Dict[str, Any]]:
        """Query sglang load metrics from all clusters in parallel.

        Parameters
        ----------
        sampler_idx : int
            Sampler index, by default 0.

        Returns
        -------
        list[dict]
            One ``{"num_reqs", "num_tokens", "num_waiting_reqs"}`` per cluster.
        """
        return await self._batch_rpc_call(sampler_idx, 'get_load', {})

    def _get_load_aware_dispatch_delay(self, sidx: int) -> float:
        dispatch_stagger_s = self.config.training.load_aware_sampler_dispatch_stagger_s
        if dispatch_stagger_s <= 0:
            return 0.0
        stagger_slots = max(1, self.config.training.rollout_gbs)
        slot = (sidx * self.dp_size + self.dp_rank) % stagger_slots
        return slot * dispatch_stagger_s

    # ---- async rollout helpers (used by GrpoAsyncTrainActor) ----

    async def fire_generate(
        self,
        sampler_idx: int,
        ppo_step: int,
        sidx: int,
        batched_data: Dict[str, Any],
        repeat_n: int,
        load_aware: bool = False,
        busy_sleep_s: float = 10.0,
    ):
        """Send a generation request without awaiting the result.

        When ``load_aware`` is *True*, queries all cluster loads first and
        picks the cluster with the fewest waiting requests.  If every
        cluster has a non-empty wait queue, sleeps ``busy_sleep_s``
        seconds and retries.

        Parameters
        ----------
        sampler_idx : int
        ppo_step : int
        sidx : int
            Sample index within the current batch.
        batched_data : dict
        repeat_n : int
        load_aware : bool
            If *True*, route to the least busy cluster.
        busy_sleep_s : float
            Seconds to sleep when all clusters are busy (only used when
            ``load_aware=True``).

        Returns
        -------
        ray.ObjectRef
            Object reference that can be awaited later via :meth:`await_generate`.
        """
        default_ep_idx = self.rpc_client_lst[sampler_idx]._pick_endpoint_idx(sidx)
        ep_idx = default_ep_idx

        if load_aware:
            delay_s = self._get_load_aware_dispatch_delay(sidx)
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            loads = await self.get_all_loads(sampler_idx)
            if loads[ep_idx]["num_waiting_reqs"] > 0:
                num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
                best = min(
                    range(num_clusters),
                    key=lambda i: (loads[i]["num_waiting_reqs"], loads[i]["num_reqs"]),
                )
                if loads[best]["num_waiting_reqs"] > 0:
                    log(
                        f"[SamplerClient] all {num_clusters} clusters busy "
                        f"(waiting_reqs={[ld['num_waiting_reqs'] for ld in loads]}), "
                        f"sleeping {busy_sleep_s}s"
                    )
                    await asyncio.sleep(busy_sleep_s)
                ep_idx = best

        target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
            sample_idx=None, ep_idx=ep_idx
        )
        req_dict = {
            'actor_dp_rank': self.dp_rank,
            'actor_dp_size': self.dp_size,
            'sampler_dp_rank': sampler_idx,
            'ppo_step': ppo_step,
            'sample_idx': sidx,
            'sampling_repeat': repeat_n,
            'batched_data': batched_data,
        }
        return self.rpc_client_lst[sampler_idx].fire(target_ep, 'generate', req_dict)

    async def await_generate(self, object_ref):
        """Await a previously fired generation request.

        Parameters
        ----------
        object_ref : ray.ObjectRef

        Returns
        -------
        dict[str, list]
            Generated outputs.
        """
        return await object_ref
