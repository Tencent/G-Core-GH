from gpatch_v4 import orches
from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.orches.placement_group import (
    create_kv_store_group,
    create_placement_groups,
    create_train_group,
)
from gpatch_v4.trainer.helper import convert_mcore_to_hf, set_nnodes_default
from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin


class T2iEditSftTrainer(TrainerRetryMixin):
    def __init__(self):
        self.train_group = None

    async def launch(self, config: FinetuneConfig):
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        self.kv_store_group = create_kv_store_group(config, pgs)
        config.kv.http_endpoints = await self.kv_store_group.init()

        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()
