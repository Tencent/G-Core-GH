import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from gpatch_v4.actor import KvStoreActor
from gpatch_v4.orches.utils import build_actor_env_vars
from gpatch_v4.utils import log, logging_rank0


async def _allocate_kv_http_ports(num_gpus_per_node, world_size, actors):
    """Allocate one available HTTP port per KV actor, grouped by node.

    Similar to ``_allocate_rollout_engine_addr_and_ports_normal`` in gen_rm_group
    but simpler: each actor gets exactly one port.
    """
    num_actors_per_node = num_gpus_per_node
    endpoints: list[str] = ["" for _ in range(world_size)]

    for rank, actor in enumerate(actors):
        if rank % num_actors_per_node != 0:
            continue

        # Use a starting port range that avoids ephemeral ports
        start_port = 12000

        node_ip, port = await actor._get_current_node_ip_and_free_port.remote(
            start_port=start_port, consecutive=num_actors_per_node
        )
        for i in range(num_actors_per_node):
            endpoints[rank + i] = f"{node_ip}:{port + i}"

    return endpoints


class RayKvStoreGroup:
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

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        assert pg is not None
        pg, reordered_bundle_indices = pg

        env_vars = build_actor_env_vars({"NCCL_CUMEM_ENABLE": "0"})

        actor_impl = KvStoreActor
        ActorClass = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = ActorClass.options(
                name=f'{self.role}_{rank}',
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
                    f"{self.__class__.__name__} master_addr {master_addr}, master_port {master_port}"
                )
            self._actor_handlers.append(actor)

    async def init(self):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Step 1: init actors (lmdb, etc.)
        futs = [actor.init.remote(self.config) for actor in self._actor_handlers]
        for fut in futs:
            await fut

        # Step 2: allocate available HTTP ports for each actor
        endpoints = await _allocate_kv_http_ports(
            self._num_gpus_per_node, world_size, self._actor_handlers
        )
        logging_rank0(f"KV store actor HTTP endpoint {endpoints}")

        # Step 3: start HTTP servers on each actor
        futs = []
        for rank, actor in enumerate(self._actor_handlers):
            host, port_str = endpoints[rank].split(":")
            futs.append(actor.start_http_server.remote(host, int(port_str)))
        for fut in futs:
            await fut

        return endpoints
