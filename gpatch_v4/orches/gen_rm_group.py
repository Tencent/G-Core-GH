import os

import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from gpatch_v4.actor import GrpoGenRmActor, T2iGrpoGenRmActor
from gpatch_v4.configs.config import OnPolicyDistillConfig, RlConfig, T2iRlConfig
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.orches.custom_actor_registry import CustomActorRegistry
from gpatch_v4.orches.utils import build_actor_env_vars
from gpatch_v4.utils import log, logging_rank0


def _allocate_rollout_engine_addr_and_ports_normal(
    num_gpus_per_node,
    rollout_num_gpus,
    rollout_num_gpus_per_cluster,
    dp_size,
    num_engines,
    rollout_actors,
    base_port=11000,
):
    """Allocate server, NCCL, and dist-init ports for rollout engines.

    Parameters
    ----------
    num_gpus_per_node : int
    rollout_num_gpus : int
    rollout_num_gpus_per_cluster : int
        GPUs per engine cluster (TP size).
    dp_size : int
    num_engines : int
    rollout_actors : list

    Returns
    -------
    list of dict
        Per-engine dicts with ``'port'``, ``'nccl_port'``, ``'dist_init_addr'``.
    """
    addr_and_ports = [{} for _ in range(num_engines)]

    # Query actual node IP for each engine actor
    engine_ips = [
        ip for ip, _ in
        ray.get([actor._get_current_node_ip_and_free_port.remote() for actor in rollout_actors])
    ]

    # Group consecutive engines by node IP
    visited_ips = set()
    for rank in range(num_engines):
        node_ip = engine_ips[rank]
        if node_ip in visited_ips:
            continue
        visited_ips.add(node_ip)

        # Collect all engine ranks on this node (consecutive from this rank)
        node_ranks = []
        for r in range(rank, num_engines):
            if engine_ips[r] == node_ip:
                node_ranks.append(r)
            elif node_ranks:
                break
        num_engines_on_this_node = len(node_ranks)
        engine = rollout_actors[rank]

        def get_addr_and_ports():
            start_port = base_port

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports()

        for i in range(num_engines_on_this_node):
            addr_and_ports[node_ranks[i]]["port"] = get_port()
            addr_and_ports[node_ranks[i]]["nccl_port"] = get_port()

        if rollout_num_gpus_per_cluster > num_gpus_per_node:
            num_node_per_cluster = rollout_num_gpus_per_cluster // num_gpus_per_node
            if rank % num_node_per_cluster == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(6 + dp_size)}"
                for i in range(num_node_per_cluster):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[node_ranks[i]
                              ]["dist_init_addr"] = f"{get_addr()}:{get_port(6 + dp_size)}"

    return addr_and_ports


class RayGenRmGroup:
    """Ray actor group managing a single generative reward model's workers.

    Each ``RayGenRmGroup`` owns exactly **one** RM and its allocated
    GPU bundles. ``create_gen_rm_group`` creates one group per RM.

    Parameters
    ----------
    config : object
    pg : tuple[PlacementGroup, list[int]]
        Placement group and bundle indices allocated to **this** RM.
    rm_idx : int
    allocated_num_gpus : int
    num_gpus_per_actor : float, optional
    role : str, optional
    """
    def __init__(
        self,
        config,
        pg: tuple[PlacementGroup, list[int]],
        rm_idx: int,
        allocated_num_gpus: int,
        num_gpus_per_actor: float = 0.01,
        role: str = "role",
    ) -> None:
        self.config = config
        self.role = role
        self.rm_idx = rm_idx
        self.allocated_num_gpus = allocated_num_gpus
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def get_custom_actor_cls(self, actor_impl_name, env_vars, default):
        """Resolve a custom actor class or fall back to the default.

        Parameters
        ----------
        actor_impl_name : str or None
        env_vars : dict
        default : type

        Returns
        -------
        type
            Ray remote actor class.
        """
        if actor_impl_name is None or CustomActorRegistry.get(actor_impl_name) is None:
            return default
        actor_impl = CustomActorRegistry.get(actor_impl_name)
        return ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

    def get_rm_engine_info(self):
        """Return engine sizing info for this group's reward model.

        Returns
        -------
        tuple[int, int, int, int, int]
            ``(num_clusters, num_engines, num_gpus_per_engine,
            mp_size, rollout_num_gpus)``.
        """
        # 如果按照 megatron 的逻辑，显然是 tp * pp * cp * ep，然而推理没有 cp，而 ep 藏在 tp 内。
        rm_idx = self.rm_idx
        infer_engine_config = self.config.gen_rm.infer_engine_configs[rm_idx]
        dist_config = infer_engine_config.dist_config
        assert dist_config.pipeline_model_parallel_size == 1

        mp_size = dist_config.tensor_model_parallel_size * dist_config.pipeline_model_parallel_size
        num_gpus_per_engine = min(mp_size, dist_config.num_gpus_per_node)
        rollout_num_gpus = self.allocated_num_gpus

        assert rollout_num_gpus > 0, (
            f"rollout_num_gpus must be positive, got {rollout_num_gpus} for rm_idx={rm_idx}"
        )
        assert rollout_num_gpus >= mp_size, (
            f"rollout_num_gpus({rollout_num_gpus}) must be >= mp_size({mp_size}) for rm_idx={rm_idx}"
        )

        num_engines = rollout_num_gpus // num_gpus_per_engine
        num_clusters = rollout_num_gpus // mp_size
        assert num_clusters > 0, (
            f"num_clusters must be positive: rollout_num_gpus={rollout_num_gpus}, mp_size={mp_size}"
        )
        assert num_engines > 0, (
            f"num_engines must be positive: rollout_num_gpus={rollout_num_gpus}, "
            f"num_gpus_per_engine={num_gpus_per_engine}"
        )
        return num_clusters, num_engines, num_gpus_per_engine, mp_size, rollout_num_gpus

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        assert pg is not None
        pg, reordered_bundle_indices = pg
        self._reordered_bundle_indices = reordered_bundle_indices

        env_vars = build_actor_env_vars({"NCCL_CUMEM_ENABLE": "0"})

        if isinstance(self.config, T2iRlConfig):
            actor_impl = T2iGrpoGenRmActor
        elif isinstance(self.config, RlConfig):
            actor_impl = GrpoGenRmActor
        elif isinstance(self.config, OnPolicyDistillConfig):
            actor_impl = GrpoGenRmActor
        else:
            raise NotImplementedError(f"unknown config {type(self.config)}")

        DefaultActorClass = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        rm_idx = self.rm_idx
        ie_cfg = self.config.gen_rm.infer_engine_configs[rm_idx]
        ActorClass = self.get_custom_actor_cls(
            ie_cfg.custom_rm_actor_impl,
            env_vars,
            default=DefaultActorClass,
        )
        num_clusters, num_engines, num_gpus_per_engine, _, _ = self.get_rm_engine_info()
        assert 0 == num_engines % num_clusters

        master_addr, master_port = None, None
        self._actor_handlers = []
        for rank in range(num_engines):
            # NOTE(astrachang): 这里的rank其实是local rank，不是global rank
            bundle_idx = reordered_bundle_indices[rank * num_gpus_per_engine]
            actor = ActorClass.options(
                name=f'gen_rm_{rm_idx}_{rank}',
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                ),
            ).remote(num_engines, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
                log(
                    f"{self.__class__.__name__} rm_idx={rm_idx} master_addr {master_addr}, master_port {master_port}"
                )
            self._actor_handlers.append(actor)

    async def init_setup(self):
        """Allocate ports. Each RM group uses a distinct base_port to avoid conflicts."""
        rm_idx = self.rm_idx
        infer_engine_config = self.config.gen_rm.infer_engine_configs[rm_idx]
        dist_config = infer_engine_config.dist_config
        (num_clusters, num_engines, num_gpus_per_engine, num_gpus_per_cluster,
         rollout_num_gpus) = self.get_rm_engine_info()
        assert 0 == num_engines % num_clusters

        futs = []
        for ep_idx in range(num_engines):
            actor = self._actor_handlers[ep_idx]
            fut = actor.init.remote(self.config)
            futs.append(fut)
        for fut in futs:
            await fut

        self._addr_and_ports = _allocate_rollout_engine_addr_and_ports_normal(
            dist_config.num_gpus_per_node,
            rollout_num_gpus,
            num_gpus_per_cluster,
            infer_engine_config.dp_size,
            num_engines,
            self._actor_handlers,
            base_port=11000 + rm_idx * 100,
        )
        for ep_idx in range(num_engines):
            for key in ["port", "nccl_port", "dist_init_addr"]:
                assert key in self._addr_and_ports[ep_idx], f"engine {ep_idx} {key} is not set."
            logging_rank0(f"ports for engine {ep_idx} {self._addr_and_ports[ep_idx]}")

    async def init_load(self):
        """Load model checkpoints and sleep engines. Can run in parallel across groups."""
        rm_idx = self.rm_idx
        infer_engine_config = self.config.gen_rm.infer_engine_configs[rm_idx]
        dist_config = infer_engine_config.dist_config
        (num_clusters, num_engines, num_gpus_per_engine, num_gpus_per_cluster,
         rollout_num_gpus) = self.get_rm_engine_info()
        num_engines_per_cluster = num_engines // num_clusters

        logging_rank0(f'create infer engines for gen rm {rm_idx}')
        futs = []
        for ep_idx in range(num_engines):
            actor = self._actor_handlers[ep_idx]
            tp_rank = (ep_idx * num_gpus_per_engine) % num_gpus_per_cluster
            start = ep_idx * num_gpus_per_engine
            tp_size = dist_config.tensor_model_parallel_size
            engine_pg_bundles = list(self._reordered_bundle_indices[start:start + tp_size])
            fut = actor.init_infer_engine.remote(
                self.config,
                self._addr_and_ports[ep_idx]['dist_init_addr'],
                rm_idx,
                ep_idx,
                tp_rank,
                ep_idx % num_engines_per_cluster == 0,
                pg_bundle_indices=engine_pg_bundles,
            )
            futs.append(fut)
        for fut in futs:
            await fut

        if self.config.placement_type == "disaggregated":
            return

        logging_rank0(f'make infer engines sleep')
        futs = []
        for ep_idx in range(num_engines):
            actor = self._actor_handlers[ep_idx]
            fut = actor.sleep.remote({'tag_names': ['weights', 'kv_cache']})
            futs.append(fut)
        for fut in futs:
            await fut

    async def init(self):
        """Full init: setup then load. For backward compatibility."""
        await self.init_setup()
        await self.init_load()

    async def write_engine_log_marker(self, ppo_step: int, phase: str = "begin"):
        """Broadcast a PPO step marker to all gen-RM engine log files via RPC."""
        _, num_engines, _, _, _ = self.get_rm_engine_info()
        futs = []
        for ep_idx in range(num_engines):
            actor = self._actor_handlers[ep_idx]
            futs.append(actor.write_engine_log_marker.remote(ppo_step, phase))
        for fut in futs:
            await fut

    async def wake_up(self):
        """Wake up all engines in this group."""
        assert self.config.placement_type != "disaggregated", (
            f"{self.__class__.__name__}.wake_up should not be called in disaggregated placement"
        )
        _, num_engines, _, _, _ = self.get_rm_engine_info()
        futs = []
        for ep_idx in range(num_engines):
            actor = self._actor_handlers[ep_idx]
            fut = actor.wake_up.remote({'tag_names': ['weights', 'kv_cache']})
            futs.append(fut)
        for fut in futs:
            await fut
