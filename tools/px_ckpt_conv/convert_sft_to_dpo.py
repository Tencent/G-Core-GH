import os
import gc
from typing import Dict
import importlib

import torch

from megatron.core import mpu
from megatron.legacy import fused_kernels
from megatron.training import get_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args, set_global_variables
from megatron.training.initialize import _set_random_seed, _initialize_distributed

from gpatch.training.v3.default_model_provider import (
    default_sft_model_provider, default_dpo_model_provider, default_reward_model_provider
)
from gpatch.training.arguments import gpatch_extra_args
from gpatch.patch_mcore import init_gpatch_for_mcore


def load_sft():
    args = get_args()

    if args.load_model_provider is None:
        model_provider = default_sft_model_provider
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
        cpu_sd[f'policy_model.{k}'] = v.cpu()
        cpu_sd[f'ref_model.{k}'] = v.cpu().clone().detach()

    return cpu_sd


def load_rm(cpu_sd: Dict):
    args = get_args()
    reward_model_path_list = args.dpo_reward_model_paths
    if len(reward_model_path_list) == 0:
        return cpu_sd

    org_load = args.load
    for i in range(len(reward_model_path_list)):
        args.load = reward_model_path_list[i]
        model_provider = default_reward_model_provider
        model = model_provider(
            pre_process=mpu.is_pipeline_first_stage(),
            post_process=mpu.is_pipeline_last_stage(),
        )
        iteration, _ = load_checkpoint([model], None, None)
        sd = model.state_dict()

        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            cpu_sd[f'reward_models.{i}.{k}'] = v.cpu()

        del sd
        del model
        gc.collect()
        torch.cuda.empty_cache()

    args.load = org_load
    return cpu_sd


def compare_state_dict_key(key1, key2):
    set1 = set(key1)
    set2 = set(key2)
    print("================")
    print(f"key1: len:{len(key1)} set-size:{len(set1)}")
    print(f"key2: len:{len(key2)} set-size:{len(set2)}")
    print(f"key1 - key2:{set1 - set2}")
    print(f"key2 - key1:{set2 - set1}")
    print("================")


def load_dpo(state_dict):
    # TODO(@nrwu):
    # 1. 不应该与 task 绑定
    # 2. gpu mem bound
    args = get_args()
    len_rm_models = len(args.dpo_reward_model_paths)
    args.dpo_reward_models_cnt = len_rm_models

    if args.save_model_provider is None:
        model_provider = default_dpo_model_provider
    else:
        mod = importlib.import_module(args.save_model_provider)
        model_provider = mod.model_provider

    dpo_model = model_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    os.system("free -h")
    # compare_state_dict_key(state_dict.keys(), dpo_model.state_dict().keys())
    dpo_model.load_state_dict(state_dict, strict=False)
    return dpo_model


# yet another possible way to handle
#
# save
# ```python
# k = 'myrandkeyname'
# t = torch.ones((3, 100), device='cpu') * args.rank
# st = make_tp_sharded_tensor_for_checkpoint(
#     tensor=t, key=k, allow_shape_mismatch=True,
# )
# ssd = {'model': {
#     k: st,
# }}
# dist_checkpointing.save(ssd, 'my-test-ckpt')
# ```
#
# load
# ```python
# k = 'myrandkeyname'
# t = torch.zeros((1000, 1000, 1000), device='cpu')
# st = make_tp_sharded_tensor_for_checkpoint(
#     tensor=t, key=k, allow_shape_mismatch=True,
# )
# ssd = {'model': {
#     k: st,
# }}
# sd = dist_checkpointing.load(ssd, 'my-test-ckpt')
# print('loaded', sd)
# ```
'''
usage:
--load: is the sft model
--dpo-reward-model-paths: is the remard model list, empty is for dpo
'''


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

    sd = load_sft()
    sd = load_rm(sd)
    torch.distributed.barrier()
    dpo_model = load_dpo(sd)
    torch.distributed.barrier()
    save_checkpoint(1, [dpo_model], None, None, num_floating_point_operations_so_far=0)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        os.rename(os.path.join(args.save, 'iter_0000001'), os.path.join(args.save, 'release'))
        with open(os.path.join(args.save, 'latest_checkpointed_iteration.txt'), 'w') as outf:
            outf.write('release')


if __name__ == "__main__":
    main()
