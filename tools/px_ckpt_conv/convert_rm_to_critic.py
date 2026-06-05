from types import SimpleNamespace
import argparse
import json
import os
import pprint
import re
import shutil
import sys
import gc
import shutil
import importlib
from typing import Dict

import multi_device.platform

from tqdm import tqdm
import torch
import transformers

from gpatch.core.device_type import is_wxacc1
from megatron.core import dist_checkpointing
from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.core.enums import ModelType
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from megatron.legacy import fused_kernels
from megatron.training import get_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args, set_global_variables
from megatron.training.initialize import _set_random_seed, _initialize_distributed
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.training.utils import unwrap_model
from megatron.training import get_tokenizer


def load_rm():
    args = get_args()

    mod = importlib.import_module(args.load_model_provider)
    rm_model = mod.model_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    iteration, _ = load_checkpoint([rm_model], None, None)
    sd = rm_model.state_dict()
    cpu_sd = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        cpu_sd[k] = v.cpu()

    return cpu_sd


def load_critic(state_dict):
    args = get_args()

    mod = importlib.import_module(args.save_model_provider)
    critic_model = mod.model_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )

    if "rm_head.weight" in state_dict.keys():
        del state_dict["rm_head.weight"]
    critic_model.load_state_dict(state_dict, strict=False)
    return critic_model


def main():
    args = parse_args()
    assert args.perform_initialization  # 0 as poison, but rm head should be inited

    args = validate_args(args)
    set_global_variables(args, build_tokenizer=True)
    args = get_args()
    _initialize_distributed()
    _set_random_seed(args.seed, args.data_parallel_random_init)
    if not is_wxacc1():
        fused_kernels.load(args)
    torch.distributed.barrier()

    sd = load_rm()
    torch.distributed.barrier()
    critic_model = load_critic(sd)
    torch.distributed.barrier()
    save_checkpoint(1, [critic_model], None, None, num_floating_point_operations_so_far=0)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        os.rename(os.path.join(args.save, 'iter_0000001'), os.path.join(args.save, 'release'))
        with open(os.path.join(args.save, 'latest_checkpointed_iteration.txt'), 'w') as outf:
            outf.write('release')


if __name__ == "__main__":
    main()
