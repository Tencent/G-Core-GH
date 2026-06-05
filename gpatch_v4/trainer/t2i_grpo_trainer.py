from gpatch_v4 import orches
from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.orches.placement_group import (
    create_bt_rm_group,
    create_gen_rm_group,
    create_placement_groups,
    create_train_group,
)
from gpatch_v4.trainer.helper import set_nnodes_default
from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin


class T2iGrpoTrainer(TrainerRetryMixin):
    """Trainer for Text-to-Image GRPO (Group Relative Policy Optimization).

    Orchestrates placement group creation, reward model group
    initialization, and the training group lifecycle for T2I tasks.
    """
    def __init__(self):
        self.gen_rm_group = None
        self.bt_rm_group = None
        self.train_group = None

    async def launch(self, config: T2iRlConfig):
        """Launch the full T2I GRPO training pipeline.

        Parameters
        ----------
        config : T2iRlConfig
        """
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)
        self.train_group = create_train_group(config, pgs)

        if config.training.use_gen_rm_reward:
            self.gen_rm_group = create_gen_rm_group(config, pgs)
            for grp in self.gen_rm_group:
                await grp.init()

        if config.training.use_bt_rm_reward:
            self.bt_rm_group = create_bt_rm_group(config, pgs)
            await self.bt_rm_group.init()

        await self.train_group.init()
        await self.train_group.setup_client()
