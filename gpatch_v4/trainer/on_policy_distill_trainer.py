from gpatch_v4 import orches
from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.orches.placement_group import (
    create_bt_rm_group,
    create_gen_rm_group,
    create_placement_groups,
    create_sampler_group,
    create_teacher_groups,
    create_train_group,
)
from gpatch_v4.trainer.helper import convert_mcore_to_hf, set_nnodes_default
from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin
from gpatch_v4.utils import log


class OnPolicyDistillTrainer(TrainerRetryMixin):
    """Trainer for on-policy distillation."""
    def __init__(self):
        self.sampler_group = None
        self.gen_rm_group = None
        self.bt_rm_group = None
        self.teacher_groups: dict = {}
        self.train_group = None

    async def conv_mcore_to_hf(self, config: OnPolicyDistillConfig):
        """Convert Megatron-Core checkpoint to HuggingFace format.

        Parameters
        ----------
        config : OnPolicyDistillConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)
        await convert_mcore_to_hf(config, pgs)

    async def launch(self, config: OnPolicyDistillConfig):
        """Launch on-policy distillation training.

        Parameters
        ----------
        config : OnPolicyDistillConfig
        """
        orches.init(config)
        set_nnodes_default(config)

        pgs = create_placement_groups(config)

        self.sampler_group = create_sampler_group(config, pgs)
        await self.sampler_group.init()

        if config.training.use_gen_rm_reward:
            self.gen_rm_group = create_gen_rm_group(config, pgs)
            for grp in self.gen_rm_group:
                await grp.init()

        if config.training.use_bt_rm_reward:
            self.bt_rm_group = create_bt_rm_group(config, pgs)
            await self.bt_rm_group.init()

        self.teacher_groups = create_teacher_groups(config, pgs)
        for t_name, t_group in self.teacher_groups.items():
            await t_group.init()

        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()
        await self.train_group.setup_client()
        await self.train_group.setup_rollout_generator()
        await self.train_group.setup_model_and_optimizer()

    async def debug_update_weight(self, config: OnPolicyDistillConfig):
        """Debug mode: test weight updates without full training.

        Parameters
        ----------
        config : OnPolicyDistillConfig
        """
        orches.init(config)
        set_nnodes_default(config)

        pgs = create_placement_groups(config)

        self.sampler_group = create_sampler_group(config, pgs)
        await self.sampler_group.init()

        if config.training.use_gen_rm_reward:
            self.gen_rm_group = create_gen_rm_group(config, pgs)
            for grp in self.gen_rm_group:
                await grp.init()

        if config.training.use_bt_rm_reward:
            self.bt_rm_group = create_bt_rm_group(config, pgs)
            await self.bt_rm_group.init()

        self.teacher_groups = create_teacher_groups(config, pgs)
        for t_name, t_group in self.teacher_groups.items():
            await t_group.init()

        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()
        await self.train_group.setup_client()
        await self.train_group.setup_rollout_generator()

        await self.train_group.debug_update_weight(stage=1)
        await self.train_group.setup_model_and_optimizer()

        await self.train_group.debug_update_weight(stage=2)
