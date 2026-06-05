# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, guanyouhe@tencent.com

from functools import partial

from tasks.glm4v.train_glm4vl import add_glm4vl_extra_args
from tasks.multimodal_comm.extra_args import ppo_mm_extra_args
from tasks.multimodal_grpo_critic_utils import MultiModalRuleOnlyCriticModel

from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config
from megatron.training import get_args, get_tokenizer
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import unwrap_model

from gpatch.core.transformer.transformer_config import GpatchTransformerConfig
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.v3.grpo_rm import GrpoRmTrainerV3, run_grpo_rm_v3


def critic_provider(reward_model):
    args = get_args()
    if args.use_grpo and args.ppo_grpo_reward_type == "rule_only":
        assert reward_model is None
        config = core_transformer_config_from_args(get_args(), GpatchTransformerConfig)
    else:
        assert reward_model is not None
        assert len(reward_model) == 1
        config = get_model_config(reward_model[0])

    critic_model = MultiModalRuleOnlyCriticModel(
        config=config,
        reward_model=reward_model,
        unwrap_model_func=unwrap_model,
        forward_micro_batch_size=args.ppo_logps_fwd_micro_batch_size,
        reward_running=args,
        ppo_reward_len_penalty_coef=args.ppo_reward_len_penalty_coef,
        ppo_reward_len_penalty_mean=args.ppo_reward_len_penalty_mean,
        ppo_reward_len_penalty_std=args.ppo_reward_len_penalty_std,
        rm_outputs_modifier=None,
        rm_output_sequence=args.rm_output_sequence,
        rm_output_scalar=args.rm_output_scalar,
        # SMART-PAD args
        enable_smart_pad=args.ppo_smart_pad_infer,
        pad_to_multi_of=args.ppo_rollout_pad_to_multiple_of,
        pad_token_id=get_tokenizer()._tokenizer.pad_token_id,
    )

    return critic_model


if __name__ == "__main__":
    init_gpatch_for_mcore()
    trainer = GrpoRmTrainerV3()
    # not support reward model now
    run_grpo_rm_v3(
        trainer,
        None,
        critic_provider,
        ModelType.encoder_and_decoder,
        extra_args_provider=partial(ppo_mm_extra_args, add_glm4vl_extra_args),
    )
