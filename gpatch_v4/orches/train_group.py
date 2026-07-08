import asyncio
import os
import time
from typing import Optional

import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from gpatch_v4.actor import (
    DistillStudentActor,
    DpoActor,
    EvaluateActor,
    FinetuneActor,
    GrpoAsyncTrainActor,
    GrpoTrainActor,
    OffPolicyDistillStudentActor,
    RewardActor,
    T2iEditSftActor,
    T2iGrpoTrainActor,
)
from gpatch_v4.configs.config import (
    DpoConfig,
    EvaluateConfig,
    FinetuneConfig,
    OffPolicyDistillConfig,
    OnPolicyDistillConfig,
    RewardConfig,
    RlConfig,
    T2iEditSftConfig,
    T2iRlConfig,
)
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.orches.failure import FailureEvent, FailureType
from gpatch_v4.orches.utils import build_actor_env_vars
from gpatch_v4.utils import log


class RayTrainGroup:
    """Group of distributed training actors.

    ``async`` methods return list of object refs.

    Args:
        config (Config): Config for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): GPUs per node.
        pg (PlacementGroup, optional): Placement group; ``None`` to auto-create.
        num_gpus_per_actor (float, optional): GPUs allocated per actor. If
            < 1.0 multiple models can share a GPU. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources per actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Custom resources per node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
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

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        assert pg is not None
        pg, reordered_bundle_indices = pg

        extra = {"NCCL_CUMEM_ENABLE": "0"}
        # Disaggregated (NCCL) vLLM trainer benefits from expandable_segments
        # to reduce fragmentation alongside the vLLM worker CUDA context.
        # Colocated vLLM uses IPCWeightTransferEngine via reduce_tensor,
        # which requires expandable_segments=False (the fd-based IPC path
        # depends on pidfd_getfd kernel syscall support).
        if (
            hasattr(self.config, 'sampler') and self.config.sampler.backend == "vllm" and
            self.config.placement_type == "disaggregated"
        ):
            extra["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env_vars = build_actor_env_vars(extra)

        if isinstance(self.config, T2iRlConfig):
            actor_impl = T2iGrpoTrainActor
        elif isinstance(self.config, RlConfig):
            if getattr(self.config.training, 'single_controller', False):
                actor_impl = GrpoAsyncTrainActor
            else:
                actor_impl = GrpoTrainActor
        elif isinstance(self.config, OnPolicyDistillConfig):
            actor_impl = DistillStudentActor
        elif isinstance(self.config, OffPolicyDistillConfig):
            actor_impl = OffPolicyDistillStudentActor
        elif isinstance(self.config, T2iEditSftConfig):
            actor_impl = T2iEditSftActor
        elif isinstance(self.config, FinetuneConfig):
            actor_impl = FinetuneActor
        elif isinstance(self.config, EvaluateConfig):
            actor_impl = EvaluateActor
        elif isinstance(self.config, DpoConfig):
            actor_impl = DpoActor
        elif isinstance(self.config, RewardConfig):
            actor_impl = RewardActor
        else:
            raise NotImplementedError(f"unknown config {type(self.config)}")
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
        """Initialize all actors in the training group."""
        futs = [actor.init.remote(self.config) for actor in self._actor_handlers]
        for fut in futs:
            await fut

    async def setup_client(self):
        """Set up RPC clients on all actors."""
        futs = [actor.setup_client.remote() for actor in self._actor_handlers]
        for fut in futs:
            await fut

    async def setup_rollout_generator(self):
        """Set up rollout generators on all actors.

        Only valid for ``RlConfig``, ``OnPolicyDistillConfig``, or
        ``OffPolicyDistillConfig``.
        """
        assert isinstance(
            self.config, (RlConfig, OnPolicyDistillConfig, OffPolicyDistillConfig)
        ), f"config {type(self.config)}"
        futs = [actor.setup_rollout_generator.remote() for actor in self._actor_handlers]
        for fut in futs:
            await fut

    async def setup_model_and_optimizer(self):
        """Build models and optimizers on all actors.

        Only valid for training-capable configurations.
        """
        assert isinstance(
            self.config, (
                FinetuneConfig,
                RlConfig,
                OnPolicyDistillConfig,
                OffPolicyDistillConfig,
                DpoConfig,
                RewardConfig,
            )
        ), f"config {type(self.config)}"
        futs = [actor.setup_model_and_optimizer.remote() for actor in self._actor_handlers]
        for fut in futs:
            await fut

    async def train_loop(self):
        """Execute the main training loop across all actors.

        Returns
        -------
        list
            Per-actor return values from ``train_loop``.
        """
        futs = [actor.train_loop.remote() for actor in self._actor_handlers]
        rets = []
        for fut in futs:
            rets.append(await fut)
        return rets

    async def log_memory(self, tag: str, rank: int = 0):
        """Log GPU memory on all policy actors (rank 0 only)."""
        futs = [actor.log_memory.remote(tag, rank) for actor in self._actor_handlers]
        for fut in futs:
            await fut

    async def update_weights(self, offload, flush_cache=False):
        """Broadcast updated policy weights to all actors."""
        futs = [
            actor.update_weights.remote(offload=offload, flush_cache=flush_cache)
            for actor in self._actor_handlers
        ]
        for fut in futs:
            await fut

    async def convert_to_hf_checkpoint(self):
        """Convert checkpoints to HuggingFace format on all actors."""
        futs = [actor.convert_to_hf_checkpoint.remote() for actor in self._actor_handlers]
        for fut in futs:
            await fut

    async def evaluate(self):
        """Run evaluation across all actors.

        Only valid for ``EvaluateConfig``.
        """
        assert isinstance(self.config, EvaluateConfig), f"config {type(self.config)}"
        futs = [actor.evaluate.remote() for actor in self._actor_handlers]
        for fut in futs:
            await fut

    async def debug_update_weight(self, stage):
        """Debug helper to test weight updates.

        Parameters
        ----------
        stage : int
        """
        futs = [actor.debug_update_weight.remote(stage) for actor in self._actor_handlers]
        for fut in futs:
            await fut

    async def check_liveness(self) -> Optional[FailureEvent]:
        """Monitor training actors for hang (loss of liveness).

        Returns
        -------
        FailureEvent or None
            *None* if training finished normally or no time limit is set.
        """
        while True:
            if self.config.training.max_train_step_waiting_time is None:
                if await self._actor_handlers[0].is_train_step_finished.remote():
                    return None
                await asyncio.sleep(30)
                continue

            if await self._actor_handlers[0].is_train_step_finished.remote():
                return None
            if await self._actor_handlers[0].check_liveness.remote():
                # Identify the node where actor[0] is running
                try:
                    node_ip = await self._actor_handlers[0].get_node_ip.remote()
                except Exception as e:
                    log(f"Failed to get node IP for {self.role}_0: {e}")
                    node_ip = "unknown"
                return FailureEvent(
                    failure_type=FailureType.HANG,
                    failed_node_ips=[node_ip],
                    failed_actor_names=[f"{self.role}_0"],
                    timestamp=time.time(),
                    details=(
                        f"Actor {self.role}_0 exceeded "
                        f"max_train_step_waiting_time="
                        f"{self.config.training.max_train_step_waiting_time}s"
                    ),
                )

            await asyncio.sleep(30)

    async def debug_scatter_and_gather(self):
        futs = [actor.debug_scatter_and_gather.remote() for actor in self._actor_handlers]
        for fut in futs:
            await fut

    async def test_ray_rpc(self):
        futs = [actor.test_ray_rpc.remote() for actor in self._actor_handlers]
        for fut in futs:
            await fut

    # ------------------------------------------------------------------ #
    #  Single-controller dispatch (used by GrpoSingleCtrlTrainer)
    # ------------------------------------------------------------------ #

    async def init_train_state(self):
        """Initialize async training state on all actors.

        Returns
        -------
        dict
            Training schedule info from actor 0.
        """
        futs = [actor.init_train_state.remote() for actor in self._actor_handlers]
        results = [await fut for fut in futs]
        return results[0]

    async def get_overlap_stats(self):
        """Collect overlap timestamps from all actors.

        Returns
        -------
        list[dict]
        """
        futs = [actor.get_overlap_stats.remote() for actor in self._actor_handlers]
        return [await fut for fut in futs]

    async def train_step(self, epoch, ppo_step, dp_refs, extra_metrics=None):
        """Process rollout data and run PPO training step on all actors.

        Parameters
        ----------
        epoch : int
        ppo_step : int
        dp_refs : list[ray.ObjectRef]
            One ObjectRef per DP rank from the ``RolloutController``.
        extra_metrics : dict or None
            Driver-side timing or other metrics merged into the actor's
            metrics for wandb logging.

        Returns
        -------
        list[dict]
        """
        futs = [
            actor.train_step.remote(epoch, ppo_step, dp_refs, extra_metrics=extra_metrics)
            for actor in self._actor_handlers
        ]
        return [await fut for fut in futs]

    async def save_checkpoint(self, step):
        """Save checkpoint on all actors."""
        futs = [actor.save_checkpoint.remote(step) for actor in self._actor_handlers]
        for fut in futs:
            await fut
