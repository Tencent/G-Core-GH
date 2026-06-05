import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from gpatch_v4.actor import GrpoBtRmActor, T2iGrpoBtRmActor
from gpatch_v4.configs.config import OnPolicyDistillConfig, RlConfig, T2iRlConfig
from gpatch_v4.orches.utils import build_actor_env_vars
from gpatch_v4.utils import log, logging_rank0


class RayBtRmGroup:
    """Ray actor group managing BT (Bradley-Terry) reward model workers.

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

    def get_rm_engine_info(self, rm_idx):
        """Return engine sizing info for a given reward model index.

        Parameters
        ----------
        rm_idx : int

        Returns
        -------
        tuple[int, int, int, int]
            ``(num_clusters, num_engines, num_gpus_per_engine,
            rollout_num_gpus)``.
        """
        infer_engine_config = self.config.bt_rm.infer_engine_configs[rm_idx]
        dist_config = infer_engine_config.dist_config
        assert dist_config.pipeline_model_parallel_size == 1
        mp_size = dist_config.tensor_model_parallel_size * dist_config.pipeline_model_parallel_size
        num_gpus_per_engine = min(mp_size, dist_config.num_gpus_per_node)
        rollout_num_gpus = dist_config.nnodes * dist_config.num_gpus_per_node
        num_engines = rollout_num_gpus // num_gpus_per_engine
        num_clusters = rollout_num_gpus // mp_size
        return num_clusters, num_engines, num_gpus_per_engine, rollout_num_gpus

    def _get_actor_impl(self):
        """Return the concrete actor class for the current config type."""
        if isinstance(self.config, T2iRlConfig):
            return T2iGrpoBtRmActor
        elif isinstance(self.config, (RlConfig, OnPolicyDistillConfig)):
            return GrpoBtRmActor
        else:
            raise NotImplementedError(f"unknown config {type(self.config)}")

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        assert pg is not None
        pg, reordered_bundle_indices = pg

        env_vars = build_actor_env_vars({"NCCL_CUMEM_ENABLE": "0"})

        actor_impl = self._get_actor_impl()
        GpuActorClass = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)
        CpuActorClass = ray.remote(num_gpus=0, runtime_env={"env_vars": env_vars})(actor_impl)

        # Create worker actors
        num_rms = len(self.config.bt_rm.reward_model_info)
        assert num_rms == len(self.config.bt_rm.infer_engine_configs)
        self._actor_handlers = [None for _ in range(num_rms)]
        for rm_idx in range(num_rms):
            infer_engine_config = self.config.bt_rm.infer_engine_configs[rm_idx]
            dist_config = infer_engine_config.dist_config
            num_cpu_nodes = getattr(dist_config, 'num_cpu_nodes', 0)

            num_clusters, num_engines, num_gpus_per_engine, rollout_num_gpus = (
                self.get_rm_engine_info(rm_idx)
            )

            # CPU-only path: nnodes=0, num_cpu_nodes>0 → create CPU actors
            if num_engines == 0:
                self._actor_handlers[rm_idx] = self._create_cpu_actors(
                    CpuActorClass,
                    rm_idx,
                    num_cpu_nodes,
                )
            else:
                # Normal GPU path
                master_addr, master_port = None, None
                rm_i_hdls = []
                for rank in range(num_engines):
                    actor = GpuActorClass.options(
                        name=f'bt_rm_{rm_idx}_{rank}',
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=reordered_bundle_indices[
                                rank * num_gpus_per_engine],
                        ),
                    ).remote(num_engines, rank, master_addr, master_port)
                    if rank == 0:
                        master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
                        log(
                            f"{self.__class__.__name__} master_addr {master_addr}, "
                            f"master_port {master_port}"
                        )
                    rm_i_hdls.append(actor)
                self._actor_handlers[rm_idx] = rm_i_hdls

    def _create_cpu_actors(self, CpuActorClass, rm_idx, num_cpu_nodes):
        """Create CPU-only actors for rule-only reward models.

        Parameters
        ----------
        CpuActorClass : ray.actor.ActorClass
        rm_idx : int
        num_cpu_nodes : int

        Returns
        -------
        list
            Actor handles.
        """
        world_size = num_cpu_nodes
        master_addr, master_port = None, None
        actors = []
        for rank in range(world_size):
            actor = CpuActorClass.options(
                name=f'bt_rm_{rm_idx}_{rank}',
                num_cpus=1,
                num_gpus=0,
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
                log(
                    f"{self.__class__.__name__} (CPU-only) "
                    f"master_addr {master_addr}, master_port {master_port}"
                )
            actors.append(actor)
        return actors

    async def init(self):
        """Initialize all BT-RM actors, reward models, and sleep them."""
        num_rms = len(self.config.bt_rm.reward_model_info)
        for rm_idx in range(num_rms):
            infer_engine_config = self.config.bt_rm.infer_engine_configs[rm_idx]
            dist_config = infer_engine_config.dist_config
            actors = self._actor_handlers[rm_idx]

            futs = []
            for actor in actors:
                fut = actor.init.remote(self.config, rm_idx)
                futs.append(fut)
            for fut in futs:
                await fut

            logging_rank0(f'create infer engines for bt rm {rm_idx}')
            futs = []
            for actor in actors:
                fut = actor.init_reward_model.remote(self.config, rm_idx)
                futs.append(fut)
            for fut in futs:
                await fut

            if self.config.placement_type == "disaggregated":
                continue

            logging_rank0(f'make bt reward engine sleep')
            futs = []
            for actor in actors:
                fut = actor.sleep.remote()
                futs.append(fut)
            for fut in futs:
                await fut

    async def wake_up(self, rm_idx):
        """Wake up all actors for the given BT reward model index.

        Parameters
        ----------
        rm_idx : int
        """
        assert self.config.placement_type != "disaggregated", (
            f"{self.__class__.__name__}.wake_up should not be called in disaggregated placement"
        )
        logging_rank0(f'make bt reward engine wake_up')
        actors = self._actor_handlers[rm_idx]
        futs = []
        for actor in actors:
            fut = actor.wake_up.remote()
            futs.append(fut)
        for fut in futs:
            await fut
