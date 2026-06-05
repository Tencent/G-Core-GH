from gpatch_v4 import orches
from gpatch_v4.configs.config import OffPolicyDistillConfig
from gpatch_v4.orches.placement_group import (
    create_placement_groups,
    create_sampler_group,
    create_teacher_group,
    create_train_group,
)
from gpatch_v4.trainer.helper import convert_mcore_to_hf, set_nnodes_default
from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin


class OffPolicyDistillTrainer(TrainerRetryMixin):
    """Trainer for off-policy distillation."""
    def __init__(self):
        self.sampler_group = None
        self.teacher_group = None
        self.train_group = None

    async def conv_mcore_to_hf(self, config: OffPolicyDistillConfig):
        """Convert Megatron-Core checkpoint to HuggingFace format.

        Parameters
        ----------
        config : OffPolicyDistillConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)
        await convert_mcore_to_hf(config, pgs)

    async def launch(self, config: OffPolicyDistillConfig):
        """Launch off-policy distillation training.

        Parameters
        ----------
        config : OffPolicyDistillConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        if config.training.enable_teacher_rollout:
            self.sampler_group = create_sampler_group(config, pgs)
            await self.sampler_group.init()

        if config.training.enable_teacher_kl_loss and config.training.setup_teacher_in_independent_topo:
            self.teacher_group = create_teacher_group(config, pgs)
            await self.teacher_group.init()

        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()
        await self.train_group.setup_client()
        await self.train_group.setup_rollout_generator()
        await self.train_group.setup_model_and_optimizer()

    async def test_ray_rpc(self, config: OffPolicyDistillConfig):
        """Test Ray RPC connectivity (debug utility).

        Parameters
        ----------
        config : OffPolicyDistillConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)
        if config.training.enable_teacher_kl_loss and config.training.setup_teacher_in_independent_topo:
            self.teacher_group = create_teacher_group(config, pgs)
            await self.teacher_group.init()
        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()
        await self.train_group.test_ray_rpc()
