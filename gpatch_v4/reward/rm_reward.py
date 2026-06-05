from abc import ABC
from typing import Any, Dict, List, Union

from transformers import AutoTokenizer
from typing_extensions import override

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.reward.base_reward import RewardAbc
from gpatch_v4.reward.mixin import RewardMixin


class RmReward(RewardAbc, RewardMixin):
    """Reward model-based reward (not yet fully implemented).

    Subclasses implement ``setup_reward_model`` and ``compute_rewards``.

    Parameters
    ----------
    config : RlConfig
    rm_idx : int
    tokenizer : AutoTokenizer
        Reward model tokenizer.
    actor_tokenizer : AutoTokenizer
        Policy actor tokenizer.
    """
    def __init__(
        self, config: RlConfig, rm_idx: int, tokenizer: AutoTokenizer,
        actor_tokenizer: AutoTokenizer
    ):
        super().__init__(config, rm_idx, tokenizer, actor_tokenizer)
        # construct model
        self.setup_reward_model()

    @override
    def setup_reward_model(self):

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement setup_reward_model"
        )

    def _get_rm_reward(self, data):
        #TODO：
        # 在 mp group 内广播这个 data，让其他rank 也拿到数据，否则 mcore_engine 没法跑 forward
        raise NotImplementedError(f"{self.__class__.__name__} does not implement _get_rm_reward")

    @override
    def compute_rewards(self, batched_data: List[Dict[str, Union[int, List[Any]]]]):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement compute_rewards")
