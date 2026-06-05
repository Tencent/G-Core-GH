# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from typing_extensions import override
from typing import List, Dict, Union, Any

import torch

from megatron.core import mpu

from gpatch.core.models.gpt import GptPpoCriticModel
from gpatch.training import get_actor_tokenizer
from gpatch.core.utils import list_for_tensor_tolist
from tasks.math_rl_v3.math_rule_rm import (
    cal_accuracy_reward,
    cal_format_reward,
    validate_samples_useful,
)


class RuleGptPpoCriticModel(GptPpoCriticModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer_rule_based_rm(
        self,
        rewards,
        per_token_rewards=None,
        sequence_lengths: torch.Tensor = None,
        prompt_lengths: torch.Tensor = None,
        batches: List[Dict[str, Union[int, List[Any]]]] = None,
    ):  # -> tuple[Any, Any | None, dict[str, Any]] | tuple[Tensor, No...:

        tokenizer = get_actor_tokenizer()

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

            inputs_list: List[torch.Tensor] = []
            # gt_label's type may be change to str
            gt_label_list: List[torch.Tensor] = []
            for batch in batches:
                # batch["tokens"] shape is [pad_seqlen]
                inputs_list.extend(batch["tokens"])
                # batch["gt_label"] shape is [1]
                gt_label_list.extend(batch["gt_label"])

            tokens_cpu: List[List[int]] = list_for_tensor_tolist(inputs_list, False)
            seq_len_cpu: List[int] = sequence_lengths.tolist()
            # type may be change to List[str]
            gt_label: List[int] = list_for_tensor_tolist(gt_label_list, True)
            assert len(tokens_cpu) == len(seq_len_cpu)
            assert len(tokens_cpu) == len(gt_label)

            for i in range(len(tokens_cpu)):
                tokens_cpu[i] = tokens_cpu[i][:seq_len_cpu[i]]
            resp_strs = tokenizer._tokenizer.batch_decode(tokens_cpu, skip_special_tokens=False)

            acc_reward, boxed_content_tmp, boxed_value_tmp = cal_accuracy_reward(
                resp_strs, gt_label
            )
            fmt_reward = cal_format_reward(resp_strs)
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
