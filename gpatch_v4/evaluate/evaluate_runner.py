from gpatch_v4 import orches
from gpatch_v4.configs.config import OffPolicyDistillConfig
from gpatch_v4.orches.placement_group import (
    create_placement_groups,
    create_sampler_group,
    create_train_group,
)
from gpatch_v4.trainer.helper import set_nnodes_default


class EvaluateRunner:
    async def evaluate(self, config: OffPolicyDistillConfig):
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)
        sampler_group = create_sampler_group(config, pgs)
        await sampler_group.init()
        train_group = create_train_group(config, pgs)
        await train_group.init()
        await train_group.setup_client()

        await train_group.evaluate()
