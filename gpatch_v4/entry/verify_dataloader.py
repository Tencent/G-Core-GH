import asyncio
import inspect

import hydra
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from megatron.core import mpu

import gpatch_v4.configs.config as train_cfg
from gpatch_v4 import orches
from gpatch_v4.actor.mixin import TokenizerMixin
from gpatch_v4.configs.utils import merge_hydra_config
from gpatch_v4.core.parallel_state import init_pg, initlize_parallel_state
from gpatch_v4.orches.placement_group import create_placement_groups
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.orches.utils import build_actor_env_vars
from gpatch_v4.trainer.helper import set_nnodes_default
from gpatch_v4.utils import log
from gpatch_v4.utils.common_utils import import_fn_from_path


class TestDLFinetuneActor(BaseActor, TokenizerMixin):
    async def init(self, config):
        super().init(config)

        # init parallel group
        initlize_parallel_state(config, config.policy.dist_config)
        init_pg(config.policy.dist_config)

        self.build_tokenizer()
        self.tokenizer = self.actor_tokenizer

        self.build_dataset_and_dataloader()

    def build_dataset_and_dataloader(self):
        fn = import_fn_from_path(self.config.data.py_path, self.config.data.fn_name)
        fn_kwargs = inspect.signature(fn).parameters

        cond1 = all(
            [
                len(fn_kwargs) == 4,
                'config' in fn_kwargs,
                'tokenizer' in fn_kwargs,
                'dp_rank' in fn_kwargs,
                'dp_size' in fn_kwargs,
            ]
        )
        if cond1:
            fn_ret = fn(
                config=self.config,
                tokenizer=self.tokenizer,
                dp_rank=mpu.get_data_parallel_rank(),
                dp_size=mpu.get_data_parallel_world_size(),
            )
        else:
            raise ValueError(f'unexpected signature {fn_kwargs}')

        self.train_dataset = fn_ret.get('train_dataset')
        self.train_dataloader = fn_ret.get('train_dataloader')
        self.train_sampler = fn_ret.get('train_sampler', None)

    def run_verify_func(self):
        fn = import_fn_from_path(
            self.config.data.py_path, self.config.data.dataloader_verify_fn_name
        )
        fn_kwargs = inspect.signature(fn).parameters
        cond1 = all(
            [
                len(fn_kwargs) == 3,
                'train_dataset' in fn_kwargs,
                'train_sampler' in fn_kwargs,
                'train_dataloader' in fn_kwargs,
            ]
        )
        if cond1:
            fn_ret = fn(
                train_dataset=self.train_dataset,
                train_sampler=self.train_sampler,
                train_dataloader=self.train_dataloader,
            )
        else:
            raise ValueError(f'unexpected signature {fn_kwargs}')


class RayTestGroup:
    """A group of dis train actors.

    Functions starting with ``async`` should return list of object refs.

    Args:
        config (Config): Actor group config.
        num_nodes (int): Nodes for this actor group.
        num_gpus_per_node (int): GPUs per node.
        pg (PlacementGroup, optional): Placement group; auto-created when None.
        num_gpus_per_actor (float, optional): GPUs per actor; ``<1.0`` enables
            sharing. Defaults to 1.
        resources (Dict[str, float], optional): Custom per-actor resources.
            https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Custom resources per node.
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

        env_vars = build_actor_env_vars({"NCCL_CUMEM_ENABLE": "0"})

        ActorClass = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(TestDLFinetuneActor)

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
        futs = [actor.init.remote(self.config) for actor in self._actor_handlers]
        for fut in futs:
            await fut

    async def run_verify_func(self):
        futs = [actor.run_verify_func.remote() for actor in self._actor_handlers]
        for fut in futs:
            await fut


@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg=None):
    config_cls = hydra.utils.get_class(cfg._target_)
    merged_obj = merge_hydra_config(config_cls, cfg)

    orches.init(merged_obj)
    set_nnodes_default(merged_obj)
    pgs = create_placement_groups(merged_obj)

    test_group = RayTestGroup(
        config=merged_obj,
        num_nodes=merged_obj.policy.dist_config.nnodes,
        num_gpus_per_node=merged_obj.policy.dist_config.num_gpus_per_node,
        pg=pgs['policy'],
        role="policy",
    )
    asyncio.run(test_group.init())
    asyncio.run(test_group.run_verify_func())


if __name__ == "__main__":
    main()
