import asyncio

from gpatch_v4 import orches
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.orches.placement_group import (
    create_bt_rm_group,
    create_gen_rm_group,
    create_placement_groups,
    create_sampler_group,
    create_train_group,
)
from gpatch_v4.trainer.helper import convert_mcore_to_hf, set_nnodes_default
from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin


class GrpoTrainer(TrainerRetryMixin):
    """Trainer for LLM GRPO (Group Relative Policy Optimization).

    Orchestrates sampler, reward model groups, and the training group
    lifecycle for language-model RL tasks.
    """
    _supports_async_rollout = False

    def __init__(self):
        self.sampler_group = None
        self.gen_rm_group = None
        self.bt_rm_group = None
        self.train_group = None

    async def conv_mcore_to_hf(self, config: RlConfig):
        """Convert a Megatron-Core checkpoint to HuggingFace format.

        Parameters
        ----------
        config : RlConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)
        await convert_mcore_to_hf(config, pgs)

    async def _init_gen_rm_groups(self, config, pgs):
        """Initialize gen-RM groups.

        ``create_gen_rm_group`` always returns a list of groups.
        """
        if config.gen_rm.destroy_engine_after_generation:
            assert config.gen_rm.backend == "sglang", (
                "gen_rm.destroy_engine_after_generation only supports sglang backend"
            )
        self.gen_rm_group = create_gen_rm_group(config, pgs)
        for grp in self.gen_rm_group:
            await grp.init_setup()
        await asyncio.gather(*(grp.init_load() for grp in self.gen_rm_group))

    async def launch(self, config: RlConfig):
        """Launch the full LLM GRPO training pipeline.

        Parameters
        ----------
        config : RlConfig
        """
        assert self._supports_async_rollout or not config.training.async_rollout, (
            f"{self.__class__.__name__} does not support async rollout; "
            "use GrpoSingleCtrlTrainer when training.single_controller=True"
        )
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        self.sampler_group = create_sampler_group(config, pgs)
        await self.sampler_group.init()

        if config.training.use_gen_rm_reward:
            await self._init_gen_rm_groups(config, pgs)

        if config.training.use_bt_rm_reward:
            self.bt_rm_group = create_bt_rm_group(config, pgs)
            await self.bt_rm_group.init()

        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()

        await self.train_group.setup_client()
        await self.train_group.setup_rollout_generator()
        await self.train_group.setup_model_and_optimizer()

    async def debug_update_weight(self, config: RlConfig):
        """Debug helper that exercises the weight-update path end-to-end.

        Parameters
        ----------
        config : RlConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        self.sampler_group = create_sampler_group(config, pgs)
        await self.sampler_group.init()

        if config.training.use_gen_rm_reward:
            await self._init_gen_rm_groups(config, pgs)

        if config.training.use_bt_rm_reward:
            self.bt_rm_group = create_bt_rm_group(config, pgs)
            await self.bt_rm_group.init()

        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()

        await self.train_group.setup_client()
        await self.train_group.setup_rollout_generator()

        await self.train_group.debug_update_weight(stage=1)
        await self.train_group.setup_model_and_optimizer()

        await self.train_group.debug_update_weight(stage=2)
