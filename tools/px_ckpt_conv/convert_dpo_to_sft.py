from types import SimpleNamespace
import os
import gc
from typing import Dict
import importlib

import torch
import transformers

from megatron.core import mpu
from megatron.legacy import fused_kernels
from megatron.training import get_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args, set_global_variables
from megatron.training.initialize import _set_random_seed, _initialize_distributed
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

from gpatch.training.v3.default_model_provider import (
    default_sft_model_provider, default_dpo_model_provider, default_reward_model_provider
)
from gpatch.training.arguments import gpatch_extra_args
from gpatch.patch_mcore import init_gpatch_for_mcore


def load_dpo():
    args = get_args()

    if args.load_model_provider is None:
        model_provider = default_dpo_model_provider
    else:
        mod = importlib.import_module(args.load_model_provider)
        model_provider = mod.model_provider
    model = model_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    iteration, _ = load_checkpoint([model], None, None)
    sd = model.state_dict()
    cpu_sd = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith('policy_model.'):
            new_k = k.replace('policy_model.', '')
            cpu_sd[new_k] = v.cpu()

    del sd
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cpu_sd


def load_sft(state_dict):
    args = get_args()

    if args.save_model_provider is None:
        model_provider = default_sft_model_provider
    else:
        mod = importlib.import_module(args.save_model_provider)
        model_provider = mod.model_provider

    model = model_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    init_gpatch_for_mcore()
    args = parse_args(extra_args_provider=gpatch_extra_args)
    assert not args.perform_initialization  # 0 as poison
    assert not args.sequence_parallel

    args = validate_args(args)
    set_global_variables(args, build_tokenizer=True)
    args = get_args()
    _initialize_distributed(None, None)
    _set_random_seed(args.seed, args.data_parallel_random_init)

    fused_kernels.load(args)
    torch.distributed.barrier()

    sd = load_dpo()
    torch.distributed.barrier()
    sft_model = load_sft(sd)
    torch.distributed.barrier()
    save_checkpoint(1, [sft_model], None, None, num_floating_point_operations_so_far=0)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        os.rename(os.path.join(args.save, 'iter_0000001'), os.path.join(args.save, 'release'))
        with open(os.path.join(args.save, 'latest_checkpointed_iteration.txt'), 'w') as outf:
            outf.write('release')


if __name__ == "__main__":
    main()
