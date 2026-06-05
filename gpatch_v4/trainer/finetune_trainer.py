from gpatch_v4 import orches
from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.orches.placement_group import (
    create_placement_groups,
    create_train_group,
    create_training_plt_group,
)
from gpatch_v4.trainer.helper import convert_mcore_to_hf, set_nnodes_default
from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin


class FinetuneTrainer(TrainerRetryMixin):
    """Trainer for supervised fine-tuning."""
    def __init__(self):
        self.train_group = None

    async def conv_mcore_to_hf(self, config: FinetuneConfig):
        """Convert Megatron-Core checkpoint to HuggingFace format.

        Parameters
        ----------
        config : FinetuneConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)
        await convert_mcore_to_hf(config, pgs)

    async def launch(self, config: FinetuneConfig):
        """Launch fine-tuning: create groups, init, and start training.

        Parameters
        ----------
        config : FinetuneConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()
        await self.train_group.setup_model_and_optimizer()

        #FIXME(guanyouhe): 这里 training_plt_group 后续处理会有问题，导致训练结束存完 ckpt 无法退出卡住
        # self.training_plt_group = create_training_plt_group(config, pgs)
