from typing import List

from transformers import AutoTokenizer

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.reward.base_reward_model_t2i import BaseT2iRewardModel
from gpatch_v4.reward.mix_rm_and_rule_reward import MixRmAndRuleReward
from gpatch_v4.reward.rm_reward import RmReward
from gpatch_v4.reward.rule_reward import RuleReward


class RewardFactory:
    """Factory that returns the appropriate reward engine."""
    @staticmethod
    def get_reward_engine(config: RlConfig, **kwargs):
        """Instantiate a reward engine based on ``config.bt_rm.reward_type``.

        Parameters
        ----------
        config : RlConfig
        **kwargs : dict
            Extra kwargs (``rm_idx``, ``tokenizer``, ``actor_tokenizer``).

        Returns
        -------
        RewardAbc
            Concrete reward engine.
        """
        if config.bt_rm.reward_type == "rule_only":
            return RuleReward(config, **kwargs)
        elif config.bt_rm.reward_type == "rm_only":
            return RmReward(config, **kwargs)
        elif config.bt_rm.reward_type == "rm_and_rule":
            return MixRmAndRuleReward(config, **kwargs)
