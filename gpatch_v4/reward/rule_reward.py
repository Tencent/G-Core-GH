import importlib
import inspect
import traceback
from typing import Any, Dict, List, Union

import torch
import torch.distributed
from transformers import AutoTokenizer
from typing_extensions import override

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.parallel_state import is_mp_head
from gpatch_v4.reward.base_reward import RewardAbc
from gpatch_v4.reward.mixin import RewardMixin
from gpatch_v4.utils import import_fn_from_path, log


class RuleReward(RewardAbc, RewardMixin):
    """Rule-based reward computed from user-provided Python functions.

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
        self,
        config: RlConfig,
        rm_idx: int,
        tokenizer: AutoTokenizer,
        actor_tokenizer: AutoTokenizer,
    ):
        super().__init__(config, rm_idx, tokenizer, actor_tokenizer)

        reward_model_info = self.config.bt_rm.reward_model_info[rm_idx]
        rule_reward_func = import_fn_from_path(
            reward_model_info.reward_py_path, reward_model_info.parse_reward_fn_name
        )
        fn_kwargs = inspect.signature(rule_reward_func).parameters
        cond1 = all(
            [
                len(fn_kwargs) >= 3,
                "batched_data" in fn_kwargs,
                "tokenizer" in fn_kwargs,
                "actor_tokenizer" in fn_kwargs,
            ]
        )
        self.pass_config = "config" in fn_kwargs
        assert cond1, f"unexpected signature {cond1}"
        self.rule_reward_func = rule_reward_func

    @override
    def setup_reward_model(self):
        # rule has no reward model
        return

    @override
    @torch.no_grad()
    def compute_rewards(
        self,
        batched_data: List[Dict[str, Union[int, List[Any]]]],
        sampling_repeat_n: int = None,
    ) -> List[Dict[str, List[Any]]]:

        reward_ret, per_token_reward, metrics = self._get_rule_rewards(
            batched_data, sampling_repeat_n
        )

        resp_dict = {
            "values": None,
            "rewards": reward_ret,
            "per_token_rewards": per_token_reward,  # none, or tensor
        }
        if metrics is not None:
            for k, v in metrics.items():
                resp_dict[k] = v

        reward_result = self.post_process_rewards(
            resp_dict,
            len(batched_data),
            sampling_repeat_n,
        )
        return reward_result

    def _get_rule_rewards(self, batched_data: List[Dict[str, Union[int, List[Any]]]], repeat_n):
        """Invoke the user-provided rule reward function.

        Parameters
        ----------
        batched_data : list of dict
        repeat_n : int

        Returns
        -------
        tuple
            ``(reward_ret, per_token_reward, metrics)`` or all ``None`` on
            non-head ranks.
        """
        # 看看是否需要额外做一些处理
        if is_mp_head():
            try:
                extra_kwargs = {"config": self.config} if self.pass_config else {}
                rule_reward, per_token_reward, metrics = self.rule_reward_func(
                    batched_data, self.tokenizer, self.actor_tokenizer, **extra_kwargs
                )
            except Exception as e:
                traceback.print_exc()
                log.error(f"rule_reward_func error: {e}")
                raise e

            reward_ret = rule_reward
            metrics.update({"rm_rewards": torch.zeros_like(reward_ret)})

            reward_ret = reward_ret.cpu()
            if per_token_reward is not None:
                per_token_reward = per_token_reward.cpu()

            for k, v in metrics.items():
                if torch.is_tensor(v):
                    metrics[k] = v.cpu()
            return reward_ret, per_token_reward, metrics

        return None, None, None
