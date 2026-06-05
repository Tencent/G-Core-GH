# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from typing_extensions import override
from typing import List, Dict, Union, Any

import torch

from megatron.core import mpu
from megatron.training import get_tokenizer
from megatron.training.global_vars import get_args
from gpatch.training import get_actor_tokenizer, get_rm_tokenizer

from gpatch.core.models.gpt import GptPpoCriticModel
from gpatch.core.utils import list_for_tensor_tolist
from tasks.math_rl_v3.math_rule_rm import (
    cal_accuracy_reward,
    cal_format_reward,
    validate_samples_useful,
)


class RuleGptPpoCriticModel(GptPpoCriticModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _extract_retool_rewards_from_gt_label(
        self,
        batches: List[Dict[str, Union[int, List[Any]]]],
    ) -> tuple:
        """
        Extract pre-computed ReTool rewards from gt_label (pass-through mode).
        
        When using ReTool sampler with --no-hook-webapi, rewards are computed
        during rollout generation and stored in gt_label['retool_reward'].
        This method extracts those rewards for use in training.
        
        Based on Verl's reward computation logic:
        - retool_reward: Final score (correctness + tool usage bonus)
        - retool_acc: Binary accuracy (1 if correct, 0 otherwise)
        - retool_tool_bonus: Tool usage bonus (only when wrong)
        
        Returns:
            Tuple of (acc_rewards, fmt_rewards) as lists
        """
        acc_rewards = []
        fmt_rewards = []

        for batch in batches:
            for gt in batch["gt_label"]:
                if isinstance(gt, dict) and "retool_reward" in gt:
                    # ReTool mode: extract pre-computed reward
                    # retool_reward already includes both correctness and tool bonus
                    # We split it for logging compatibility:
                    #   acc_reward = base correctness score
                    #   fmt_reward = tool usage bonus
                    base_score = gt.get("retool_base_score", gt["retool_reward"])
                    tool_bonus = gt.get("retool_tool_bonus", 0.0)

                    acc_rewards.append(float(base_score))
                    fmt_rewards.append(float(tool_bonus))
                else:
                    # Fallback: no retool reward found
                    acc_rewards.append(0.0)
                    fmt_rewards.append(0.0)

        return acc_rewards, fmt_rewards

    def infer_rule_based_rm(
        self,
        rewards,
        per_token_rewards=None,
        sequence_lengths: torch.Tensor = None,
        prompt_lengths: torch.Tensor = None,
        batches: List[Dict[str, Union[int, List[Any]]]] = None,
    ):  # -> tuple[Any, Any | None, dict[str, Any]] | tuple[Tensor, No...:

        tokenizer = get_actor_tokenizer()
        args = get_args()

        acc_reward_tensor = None
        fmt_reward_tensor = None
        # TODO 支持 cp 的话，这里要改
        is_mp_head = mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0
        if not is_mp_head:
            assert rewards is None
            assert per_token_rewards is None
        else:
            if self.config.ppo_grpo_reward_type == "rm_only":
                acc_reward_tensor = torch.zeros_like(rewards).to(rewards.dtype)
                fmt_reward_tensor = torch.zeros_like(rewards).to(rewards.dtype)
                return rewards, per_token_rewards, {
                    'rm_rewards': rewards,
                    'acc_rewards': acc_reward_tensor,
                    'fmt_rewards': fmt_reward_tensor,
                }

            # Check if ReTool rewards are pre-computed in gt_label (--no-hook-webapi mode)
            # This is the pass-through path for ReTool sampler
            use_retool_passthrough = False
            if batches and len(batches) > 0:
                first_gt = batches[0].get("gt_label", [None])[0]
                if isinstance(first_gt, dict) and "retool_reward" in first_gt:
                    use_retool_passthrough = True

            assert use_retool_passthrough, "retool pass-through mode is required"
            # ReTool pass-through: extract pre-computed rewards from gt_label
            acc_reward, fmt_reward = self._extract_retool_rewards_from_gt_label(batches)

            acc_reward_tensor = torch.tensor(
                acc_reward, dtype=torch.float32, device=torch.cuda.current_device()
            ).view(-1, 1)
            fmt_reward_tensor = torch.tensor(
                fmt_reward, dtype=torch.float32, device=torch.cuda.current_device()
            ).view(-1, 1)

            rule_reward = acc_reward_tensor + fmt_reward_tensor

            if self.config.ppo_grpo_reward_type == "rule_only":
                return rule_reward, None, {
                    'rm_rewards': torch.zeros_like(acc_reward_tensor),
                    'acc_rewards': acc_reward_tensor,
                    'fmt_rewards': fmt_reward_tensor,
                }
            elif self.config.ppo_grpo_reward_type == "rm_with_rule":
                # rewards = rewards + self.config.ppo_rule_reward_beta * rule_reward
                combined_rewards = torch.sigmoid(self.config.ppo_rm_reward_alpha * rewards) + \
                    (self.config.ppo_rule_reward_beta * rule_reward - 1)
                #TODO: per_token_rewards 要如何修改呢？
                return combined_rewards, per_token_rewards, {
                    'rm_rewards': rewards,
                    'acc_rewards': acc_reward_tensor,
                    'fmt_rewards': fmt_reward_tensor,
                }

        return rewards, per_token_rewards, {
            'rm_rewards': rewards,
            'acc_rewards': acc_reward_tensor,
            'fmt_rewards': fmt_reward_tensor,
        }

    @override
    def validate_samples(self, rewards, sampling_repeat=None):
        """"
        Checks the validity of the given samples and returns a dictionary with the results.

        Parameters:
        rewards: a tensor of rewards, shape [b, 1].

        Returns:
        dict: a dict regarding the usefulness of samples. example:
                {'sample_useful': tensor of usefulness ([b])}
        """
        check_result = None
        is_mp_head = mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0
        if not is_mp_head:
            assert rewards is None
        else:
            check_result = {"sample_useful": validate_samples_useful(rewards, sampling_repeat)}
        return check_result
