# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import traceback
from packaging.version import Version

import torch
from torch import Tensor

from gpatch.patch_mcore import init_gpatch_for_mcore
from megatron.core import package_info
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import unwrap_model
try:
    from megatron.training import inprocess_restart
except ImportError:
    inprocess_restart = None

from tasks.retool.args import get_tasks_args
# Use ReTool-specific critic model with pass-through reward support
from tasks.retool.rule_critic_model import RuleGptPpoCriticModel
from gpatch.core.device_type import is_wxacc1
from gpatch.training.v3.grpo_rm import run_grpo_rm_v3, GrpoRmTrainerV3
from gpatch.training import get_actor_tokenizer, get_rm_tokenizer
from gpatch.training.v3.default_model_provider import default_reward_model_provider
from gpatch.core.transformer.transformer_config import GpatchTransformerConfig

mcore_version = Version(package_info.__version__)
model_provider = default_reward_model_provider


def critic_provider(reward_model):
    args = get_args()
    if args.use_grpo and args.ppo_grpo_reward_type == "rule_only":
        assert reward_model is None
        config = core_transformer_config_from_args(get_args(), GpatchTransformerConfig)
    else:
        assert reward_model is not None
        assert len(reward_model) == 1
        config = get_model_config(reward_model[0])
    actor_tokenizer = get_actor_tokenizer()
    rm_tokenizers = get_rm_tokenizer()
    print_rank_0(f"{actor_tokenizer=} {rm_tokenizers=}")

    critic_model = RuleGptPpoCriticModel(
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
        pad_token_id=rm_tokenizers._tokenizer.pad_token_id
    )
    return critic_model


if __name__ == "__main__":
    init_gpatch_for_mcore()
    trainer = GrpoRmTrainerV3()

    print(f"{mcore_version=} {Version('0.13.0')} {mcore_version < Version('0.13.0')}")
    if mcore_version >= Version("0.13.0"):
        assert inprocess_restart is not None
        # Optionally enable inprocess restart on pretrain
        run_grpo_rm_v3, store = inprocess_restart.maybe_wrap_for_inprocess_restart(run_grpo_rm_v3)
        extra_args = {"store": store}
    else:
        extra_args = {}

    run_grpo_rm_v3(
        trainer,
        model_provider,
        critic_provider,
        ModelType.encoder_or_decoder,
        extra_args_provider=get_tasks_args,
        **extra_args
    )
