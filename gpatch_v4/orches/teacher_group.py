import os
from typing import Optional

import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from gpatch_v4.actor import DistillTeacherActor
from gpatch_v4.configs.config import OffPolicyDistillConfig, OnPolicyDistillConfig
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.orches.utils import build_actor_env_vars
from gpatch_v4.utils import log, logging_rank0


def cal_dp_and_mp_rank(world_size, tp_size, pp_size, cp_size, rank):
    dp_size = world_size // (tp_size * pp_size * cp_size)
    tp_cp_size = tp_size * cp_size
    mp_cp_size = tp_size * pp_size * cp_size
    tp_dp_cp_size = tp_size * dp_size * cp_size

    curr_rank = rank
    pp_rank = 0
    if curr_rank >= tp_dp_cp_size:
        pp_rank = curr_rank // tp_dp_cp_size
        curr_rank = curr_rank % tp_dp_cp_size

    dp_rank = curr_rank // tp_cp_size
    tp_cp_rank = curr_rank % tp_cp_size
    mp_rank = tp_cp_rank + pp_rank * tp_cp_size
    return dp_rank, mp_rank


class RayTeacherGroup:
    def __init__(
        self,
        config,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int]],
        num_gpus_per_actor: float = 0.01,
        role: str = "role",
    ) -> None:
        self.config = config
        self.dist_config = config.teacher.dist_config
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        assert pg is not None
        pg, reordered_bundle_indices = pg

        env_vars = build_actor_env_vars({"NCCL_CUMEM_ENABLE": "0"})
        if isinstance(self.config, (OnPolicyDistillConfig, OffPolicyDistillConfig)):
            actor_impl = DistillTeacherActor
        else:
            raise NotImplementedError(f"unknown config {type(self.config)}")

        ActorClass = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)
        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None

        tp_size = self.dist_config.tensor_model_parallel_size
        pp_size = self.dist_config.pipeline_model_parallel_size
        cp_size = self.dist_config.context_parallel_size

        for rank in range(world_size):
            dp_rank, mp_cp_rank = cal_dp_and_mp_rank(world_size, tp_size, pp_size, cp_size, rank)
            actor = ActorClass.options(
                name=f'{self.role}_{dp_rank}_{mp_cp_rank}',
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)

            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
                log(
                    f"{self.__class__.__name__} world_size {world_size} master_addr {master_addr}, master_port {master_port}"
                )
            self._actor_handlers.append(actor)

    async def init(self):
        futs = [actor.init.remote(self.config) for actor in self._actor_handlers]
        for fut in futs:
            await fut

        # debeg 不需要加载 teacher 模型
        if not self.config.debug.debug_engine_update_weight:
            futs = [actor.setup_model.remote() for actor in self._actor_handlers]
            for fut in futs:
                await fut

            futs = [actor.sleep.remote() for actor in self._actor_handlers]
            for fut in futs:
                await fut

    async def wake_up(self):
        futs = [actor.wake_up.remote() for actor in self._actor_handlers]
        for fut in futs:
            await fut
