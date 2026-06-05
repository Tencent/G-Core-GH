import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from gpatch_v4.actor import TrainingPltActor
from gpatch_v4.orches.utils import build_actor_env_vars


class RayTrainingPltGroup:
    def __init__(
        self,
        config,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int]],
        num_gpus_per_actor: float = 0.01,
        role: str = "training_plt",
    ) -> None:
        self.config = config
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        assert world_size == 1, f"only need single process, got {world_size}"
        assert pg is not None
        pg, reordered_bundle_indices = pg
        assert len(
            reordered_bundle_indices
        ) == world_size, f"only need single process, got {world_size} {len(reordered_bundle_indices)}"

        env_vars = build_actor_env_vars({"NCCL_CUMEM_ENABLE": "0"})

        actor_impl = TrainingPltActor
        ActorClass = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        actor_unique_name = f"{self.role}"
        self.actor = ActorClass.options(
            name=actor_unique_name,
            num_cpus=num_gpus_per_actor,
            num_gpus=num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=reordered_bundle_indices[0],
            ),
        ).remote()

    async def run_loop(self):
        return await self.actor.run_loop.remote()

    def stop(self):
        self.actor.stop.remote()
