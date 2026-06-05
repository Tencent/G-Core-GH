import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from gpatch_v4.actor import GrpoSamplerActor, OffPolicyDistillSamplerActor
from gpatch_v4.configs.config import (
    EvaluateConfig,
    InferenceConfig,
    OffPolicyDistillConfig,
    OnPolicyDistillConfig,
    RlConfig,
)
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.orches.gen_rm_group import _allocate_rollout_engine_addr_and_ports_normal
from gpatch_v4.orches.utils import build_actor_env_vars
from gpatch_v4.utils import log, logging_rank0


class RaySamplerGroup:
    """Ray actor group managing sampler (inference engine) workers.

    Parameters
    ----------
    config : object
    num_nodes : int
    num_gpus_per_node : int
    pg : tuple[PlacementGroup, list[int]]
    num_gpus_per_actor : float, optional
    role : str, optional
    """
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
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def get_sampler_engine_info(self, sampler_idx):
        """Return engine sizing info for a given sampler index.

        Parameters
        ----------
        sampler_idx : int

        Returns
        -------
        tuple[int, int, int, int, int]
            ``(num_clusters, num_engines, num_gpus_per_engine,
            mp_size, rollout_num_gpus)``.
        """
        infer_engine_config = self.config.sampler.infer_engine_configs[sampler_idx]
        dist_config = infer_engine_config.dist_config
        assert dist_config.pipeline_model_parallel_size == 1
        mp_size = dist_config.tensor_model_parallel_size * dist_config.pipeline_model_parallel_size
        num_gpus_per_engine = min(mp_size, dist_config.num_gpus_per_node)
        rollout_num_gpus = dist_config.nnodes * dist_config.num_gpus_per_node
        num_engines = rollout_num_gpus // num_gpus_per_engine
        num_clusters = rollout_num_gpus // mp_size
        return num_clusters, num_engines, num_gpus_per_engine, mp_size, rollout_num_gpus

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        assert pg is not None
        pg, reordered_bundle_indices = pg
        self._reordered_bundle_indices = reordered_bundle_indices

        env_vars = build_actor_env_vars({"NCCL_CUMEM_ENABLE": "0"})

        if isinstance(self.config, (RlConfig, OnPolicyDistillConfig, InferenceConfig)):
            actor_impl = GrpoSamplerActor
        elif isinstance(self.config, (OffPolicyDistillConfig, EvaluateConfig)):
            actor_impl = OffPolicyDistillSamplerActor
        else:
            raise NotImplementedError(f"unknown config {type(self.config)}")
        ActorClass = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        # Create worker actors
        num_samplers = len(self.config.sampler.model_info)
        assert num_samplers == len(self.config.sampler.infer_engine_configs)
        self._actor_handlers = [None for _ in range(num_samplers)]
        for sampler_idx in range(num_samplers):
            num_clusters, num_engines, num_gpus_per_engine, _, rollout_num_gpus = self.get_sampler_engine_info(
                sampler_idx
            )
            assert 0 == num_engines % num_clusters

            master_addr, master_port = None, None
            sampler_i_hdls = []
            for rank in range(num_engines):
                actor = ActorClass.options(
                    name=f'sampler_{sampler_idx}_{rank}',
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=reordered_bundle_indices[rank *
                                                                              num_gpus_per_engine],
                    ),
                ).remote(num_engines, rank, master_addr, master_port)
                if rank == 0:
                    master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
                    log(
                        f"{self.__class__.__name__} master_addr {master_addr}, master_port {master_port}"
                    )
                sampler_i_hdls.append(actor)
            self._actor_handlers[sampler_idx] = sampler_i_hdls

    async def init(self):
        """Initialize all sampler actors, create inference engines, and sleep them."""
        num_samplers = len(self.config.sampler.model_info)
        for sampler_idx in range(num_samplers):
            infer_engine_config = self.config.sampler.infer_engine_configs[sampler_idx]
            dist_config = infer_engine_config.dist_config
            num_clusters, num_engines, num_gpus_per_engine, num_gpus_per_cluster, rollout_num_gpus = self.get_sampler_engine_info(
                sampler_idx
            )
            assert 0 == num_engines % num_clusters
            num_engines_per_cluster = num_engines // num_clusters

            futs = []
            for ep_idx in range(num_engines):
                actor = self._actor_handlers[sampler_idx][ep_idx]
                fut = actor.init.remote(self.config)
                futs.append(fut)
            for fut in futs:
                await fut

            addr_and_ports = _allocate_rollout_engine_addr_and_ports_normal(
                dist_config.num_gpus_per_node,
                rollout_num_gpus,
                num_gpus_per_cluster,
                infer_engine_config.dp_size,
                num_engines,
                self._actor_handlers[sampler_idx],
            )
            for ep_idx in range(num_engines):
                for key in ["port", "nccl_port", "dist_init_addr"]:
                    assert key in addr_and_ports[ep_idx], f"engine {ep_idx} {key} is not set."
                logging_rank0(f"ports for engine {ep_idx} {addr_and_ports[ep_idx]}")

            logging_rank0(f'create infer engines for sampler {sampler_idx}')
            futs = []
            for ep_idx in range(num_engines):
                actor = self._actor_handlers[sampler_idx][ep_idx]
                tp_rank = (ep_idx * num_gpus_per_engine) % num_gpus_per_cluster
                start = ep_idx * num_gpus_per_engine
                tp_size = dist_config.tensor_model_parallel_size
                engine_pg_bundles = list(self._reordered_bundle_indices[start:start + tp_size])
                fut = actor.init_infer_engine.remote(
                    self.config,
                    addr_and_ports[ep_idx]['dist_init_addr'],
                    sampler_idx,
                    ep_idx,
                    tp_rank,
                    ep_idx % num_engines_per_cluster == 0,
                    pg_bundle_indices=engine_pg_bundles,
                )
                futs.append(fut)
            for fut in futs:
                await fut

            if isinstance(self.config, InferenceConfig):
                return

            if hasattr(self.config, 'debug') and self.config.debug.debug_engine_update_weight:
                # 测试 update weight，就不把 sglang sleep 了
                return

            if self.config.placement_type == "disaggregated":
                continue

            logging_rank0(f'make infer engines sleep')
            futs = []
            for ep_idx in range(num_engines):
                actor = self._actor_handlers[sampler_idx][ep_idx]
                fut = actor.sleep.remote({'tag_names': ['weights', 'kv_cache']})
                futs.append(fut)
            for fut in futs:
                await fut

    async def write_engine_log_marker(self, ppo_step: int, phase: str = "begin"):
        """Broadcast a PPO step marker to all sampler engine log files via RPC."""
        futs = []
        for sampler_idx in range(len(self.config.sampler.model_info)):
            _, num_engines, _, _, _ = self.get_sampler_engine_info(sampler_idx)
            for ep_idx in range(num_engines):
                actor = self._actor_handlers[sampler_idx][ep_idx]
                futs.append(actor.write_engine_log_marker.remote(ppo_step, phase))
        for fut in futs:
            await fut

    async def wake_up(self, sampler_idx):
        """Wake up all actors for the given sampler index.

        Parameters
        ----------
        sampler_idx : int
        """
        assert self.config.placement_type != "disaggregated", (
            f"{self.__class__.__name__}.wake_up should not be called in disaggregated placement"
        )
        _, num_engines, _, _, _ = self.get_sampler_engine_info(sampler_idx)
        futs = []
        for ep_idx in range(num_engines):
            actor = self._actor_handlers[sampler_idx][ep_idx]
            fut = actor.wake_up.remote({'tag_names': ['weights', 'kv_cache']})
            futs.append(fut)
        for fut in futs:
            await fut
