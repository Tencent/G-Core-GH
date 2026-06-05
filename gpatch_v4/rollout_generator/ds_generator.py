from typing import Any, Dict, List

import torch
from typing_extensions import override

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.rollout_generator.generator_abc import RolloutGeneratorAbc


class DynamicSamplingRolloutGenerator(RolloutGeneratorAbc):
    """Rollout generator with dynamic sampling (not yet implemented)."""
    def __init__(
        self, config: RlConfig, sampler_client, gen_rm_client, bt_rm_client, run_eval=False
    ):
        super().__init__(config, sampler_client, gen_rm_client, bt_rm_client, run_eval)

        # 看看是否需要额外定义什么？
        pass

    @override
    async def rollout_samples(self, data_iter, num_microbatches, curr_ppo_step):
        raise NotImplementedError(f"{self.__class__.__name__}.rollout_samples is not implemented")

    @override
    async def generate_gen_rm_reward(
        self, rbs: List[Dict[str, List[Any]]], num_microbatches, curr_ppo_step
    ):
        raise NotImplementedError(
            f"{self.__class__.__name__}.generate_gen_rm_reward is not implemented"
        )

    @override
    async def calc_bt_rm_reward(
        self, rbs: List[Dict[str, List[Any]]], num_microbatches, curr_ppo_step
    ):
        raise NotImplementedError(f"{self.__class__.__name__}.calc_bt_rm_reward is not implemented")

    @override
    async def __call__(self, data_iter, num_microbatches, curr_ppo_step):
        raise NotImplementedError(f"{self.__class__.__name__}.rollout_samples is not implemented")
