from gpatch_v4 import orches
from gpatch_v4.configs.config import DpoConfig
from gpatch_v4.orches.placement_group import create_placement_groups, create_train_group
from gpatch_v4.trainer.helper import convert_mcore_to_hf, set_nnodes_default
from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin


class DpoTrainer(TrainerRetryMixin):
    """Trainer for Direct Preference Optimization (DPO)."""
    def __init__(self):
        self.train_group = None

    async def conv_mcore_to_hf(self, config: DpoConfig):
        """Convert Megatron-Core checkpoint to HuggingFace format.

        Parameters
        ----------
        config : DpoConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)
        await convert_mcore_to_hf(config, pgs)

    async def launch(self, config: DpoConfig):
        """Launch DPO training.

        Parameters
        ----------
        config : DpoConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()

        await self.train_group.setup_model_and_optimizer()
