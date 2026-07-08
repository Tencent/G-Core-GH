from gpatch_v4 import orches
from gpatch_v4.configs.config import RewardConfig
from gpatch_v4.orches.placement_group import create_placement_groups, create_train_group
from gpatch_v4.trainer.helper import convert_mcore_to_hf, set_nnodes_default
from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin


class RewardTrainer(TrainerRetryMixin):
    """Trainer for Bradley-Terry reward-model training (output_scalar)."""
    def __init__(self):
        self.train_group = None

    async def conv_mcore_to_hf(self, config: RewardConfig):
        """Convert Megatron-Core checkpoint to HuggingFace format.

        Parameters
        ----------
        config : RewardConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)
        await convert_mcore_to_hf(config, pgs)

    async def launch(self, config: RewardConfig):
        """Launch reward-model training: create groups, init, and start training.

        Parameters
        ----------
        config : RewardConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()
        await self.train_group.setup_model_and_optimizer()
