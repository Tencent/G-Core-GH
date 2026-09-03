from typing_extensions import override

from gpatch_v4.rollout_generator.base_generator import BaseRolloutGenerator


class ReplayRolloutGenerator(BaseRolloutGenerator):
    """Rollout generator with experience replay (not yet implemented)."""
    @override
    async def __call__(self, data_iter, num_microbatches, curr_ppo_step):
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")
