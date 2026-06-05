# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com
"""Pretrain and SFT GPT."""
import copy
import os
import token
import torch

from functools import partial
from typing import List, Optional, Union
from packaging.version import Version

import torch
from torch import Tensor

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core import package_info
try:
    from megatron.training import inprocess_restart
except ImportError:
    inprocess_restart = None

from megatron_datasets.tasks.math_rl_v3.ppo_actor_dataset import build_train_valid_test_datasets, DataCollator
from megatron_datasets.utils import get_iterator
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core import tensor_parallel
from megatron.training.utils import get_ltor_masks_and_position_ids
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron_datasets.tasks.math_rl_v3.sft_dataset import (
    update_epoch_and_line,
)
import megatron.legacy.model

from gpatch.training.v3.ppo_actor import train_ppo_actor_v3
from gpatch.training.v3.default_model_provider import (
    default_actor_provider,
    default_sampler_client_provider,
    default_rm_critic_client_provider,
    default_gen_rm_client_provider,
)
from gpatch.core.aligner_helper import retrieve_model_state_dict_in_cpu
from gpatch.core.transformer.transformer_config import GpatchTransformerConfig
from gpatch.core.device_type import is_wxacc1
from gpatch.core.parallel_state import is_mp_and_cp_head, get_mp_and_cp_size
from gpatch.core.models.gpt import (
    GptPpoActorModel,
    GptPpoRmCriticClientV3,
    GptPpoSamplerClientV3,
    GptPpoGenRmClientV3,
)
from gpatch.patch_mcore import init_gpatch_for_mcore

from tasks.math_rl_v3.args import get_tasks_args
from tasks.math_rl_v3.sp import get_ppo_prompt_format
from tasks.math_rl_v3.ppo_sampling import filter_samplings
from tasks.math_rl_v3.math_rl_actor_trainer import MathRLActorTrainer
try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

mcore_version = Version(package_info.__version__)

actor_provider = default_actor_provider
sampler_client_provider = default_sampler_client_provider
rm_critic_client_provider = default_rm_critic_client_provider
gen_rm_client_provider = default_gen_rm_client_provider


def model_provider(
    pre_process=True,
    post_process=True,
    vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    config = core_transformer_config_from_args(args, GpatchTransformerConfig)
    # TODO(astrachang): 这东西怎么用yaml传tuple的啊
    if args.model_arch == "gpt_oss_moe":
        config.window_size = (128, 0)
        config.window_size = tuple(config.window_size)
        config.bias_dropout_fusion = False

        # NV的Config里面还没yarn的参数呢，要手动设置
        config.position_embedding_type = args.position_embedding_type
        config.yarn_rotary_scaling_factor = args.yarn_rotary_scaling_factor
        config.yarn_original_max_position_embeddings = args.yarn_original_max_position_embeddings
        config.yarn_beta_fast = args.yarn_beta_fast
        config.yarn_beta_slow = args.yarn_beta_slow
        config.yarn_mscale = args.yarn_mscale
        config.yarn_mscale_all_dim = args.yarn_mscale_all_dim
        config.yarn_correction_range_round_to_int = args.yarn_correction_range_round_to_int

    print_rank_0(f'{config.recompute_method=}')
    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config,
                    use_transformer_engine=use_te,
                    normalization=args.normalization,
                    qk_l2_norm=args.qk_l2_norm,
                    vp_stage=vp_stage
                )
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            if hasattr(transformer_layer_spec,
                       'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
                # Get the decoder layer spec explicitly if no decoder layer in the last stage,
                # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
                transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
            else:
                transformer_layer_spec_for_mtp = transformer_layer_spec
            mtp_block_spec = get_gpt_mtp_block_spec(
                config,
                transformer_layer_spec_for_mtp,
                use_transformer_engine=use_te,
                vp_stage=vp_stage
            )

        model = GPTModel(
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
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0('> building train, validation, and test datasets ...')
    prompt_format, eos_token = get_ppo_prompt_format(args, tokenizer)
    if isinstance(prompt_format, list):
        for sub_pt in prompt_format:
            print_rank_0(f"building dataset with sub_prompt_format {sub_pt} eos_token {eos_token}")
    else:
        print_rank_0(f"building dataset with prompt_format {prompt_format} eos_token {eos_token}")

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        args,
        tokenizer,
        dp_rank=mpu.get_data_parallel_rank(),
        dp_size=mpu.get_data_parallel_world_size(),
        prompt_format=prompt_format,
        eos_token=eos_token
    )
    print_rank_0(f"> finished creating datasets ...")

    collate_fn = DataCollator(
        tokenizer=tokenizer,
        seq_len=args.seq_length,
        resp_seq_len=args.ppo_resp_seq_len,
        gen_left_pad=args.gen_left_pad,
        random_pad=True
    )
    batch_size = args.ppo_rollout_micro_batch_size
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=args.px_dataloader_prefetch_factor,
    )

    eval_dataloader = None
    if valid_ds is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=args.ppo_eval_rollout_micro_batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=args.px_dataloader_prefetch_factor,
        )
    test_dataloader = None
    if test_ds is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=args.px_dataloader_prefetch_factor,
        )
    print_rank_0(f"> finished creating dataloader ...")

    return get_iterator(train_dataloader), get_iterator(eval_dataloader
                                                       ), get_iterator(test_dataloader)


def rollout_get_batch(data_iterator):
    args = get_args()
    assert is_mp_and_cp_head(), f'只有 mp_head 会走到这里'

    # Broadcast data.
    assert data_iterator is not None
    data = next(data_iterator)

    tokens = data['input_ids']
    lpad_lens = data['lpad_lens']
    gt_label = data['gt_label']

    lpad_lens_list = lpad_lens.tolist()
    prompt_token_ids = []
    for i in range(len(tokens)):
        # tokens_prompt 的格式是：
        # tokens_prompt = {
        #     'prompt_token_ids': [0, 114, 514, 19, 19, 810, 1, 2, 2],
        # }
        # 之所以额外套一层，是因为 vllm asyncLLM 的输入 `vllm.inputs.data.TokensPrompt` 本身就是用 dict 来表示。
        # 虽然看着觉得有点别扭，但是兼容性更好。
        prompt_token_ids.append({
            'prompt_token_ids': tokens[i][:lpad_lens_list[i]].tolist(),
        })

    batch_data = {
        "prompt_token_ids": prompt_token_ids,
        "lpad_lens": lpad_lens,
        "gt_label": gt_label,
    }

    return batch_data


# 初始化MathRLActorTrainer时拿不到args，无法判断是rm还是gen-rm
# 通过extra_metric_info_provider判断
def extra_metric_info_provider():
    args = get_args()
    if args.use_gen_rm:
        extra_metric_info = [
            {
                "key_name": "rm_rewards",
                "dtype": torch.float32
            },
        ]
    else:
        extra_metric_info = [
            {
                'key_name': 'rm_rewards',
                'dtype': torch.float32
            },
            {
                'key_name': 'acc_rewards',
                'dtype': torch.float32
            },
            {
                'key_name': 'fmt_rewards',
                'dtype': torch.float32
            },
            {
                'key_name': 'sample_useful',
                'dtype': torch.bool
            },
        ]
    return extra_metric_info


if __name__ == "__main__":
    init_gpatch_for_mcore()
    train_valid_test_datasets_provider.is_distributed = True

    trainer = MathRLActorTrainer(extra_metric_info=extra_metric_info_provider)

    print(f"{mcore_version=} {Version('0.13.0')} {mcore_version < Version('0.13.0')}")
    if mcore_version >= Version("0.13.0"):
        assert inprocess_restart is not None
        # Optionally enable inprocess restart on pretrain
        train_ppo_actor_v3, store = inprocess_restart.maybe_wrap_for_inprocess_restart(
            train_ppo_actor_v3
        )
        extra_args = {"store": store}
    else:
        extra_args = {}

    train_ppo_actor_v3(
        trainer,
        model_provider,
        actor_provider,
        sampler_client_provider,
        rm_critic_client_provider,
        gen_rm_client_provider,
        train_valid_test_datasets_provider,
        rollout_get_batch,
        filter_samplings,
        ModelType.encoder_or_decoder,
        extra_args_provider=get_tasks_args,
        **extra_args
    )
