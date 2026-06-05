# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import torch
from torch import Tensor

from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GptPpoCriticModel
from megatron.core.aligner_helper import retrieve_model_state_dict_in_cpu
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.utils import get_ltor_masks_and_position_ids

from gpatch.core.device_type import is_wxacc1
from tasks.math_rl_v3.args import get_tasks_args
from tests.rm_test.reward_run import infer_rm


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(get_args())
    if hasattr(args, "use_legacy_models"):
        assert not args.use_legacy_models
    else:
        assert args.use_mcore_models
    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    elif is_wxacc1():
        pass
    else:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm
        )

    model = GptPpoCriticModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.px_rope_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        # RM args
        output_sequence=True,
        output_scalar=False,
        use_avg_pool=args.rm_use_avg_pool,
        forward_micro_batch_size=args.ppo_logps_fwd_micro_batch_size,
        mask_prompt=args.ppo_rm_mask_prompt,
        reward_running=args,
        ppo_reward_len_penalty_coef=args.ppo_reward_len_penalty_coef,
        ppo_reward_len_penalty_mean=args.ppo_reward_len_penalty_mean,
        ppo_reward_len_penalty_std=args.ppo_reward_len_penalty_std,
        rm_outputs_modifier=None,
        rm_output_sequence=args.rm_output_sequence,
        rm_output_scalar=args.rm_output_scalar,
    )
    print_rank_0(f"model arch {model}")

    return model


if __name__ == "__main__":
    infer_rm(model_provider, ModelType.encoder_or_decoder, extra_args_provider=get_tasks_args)
