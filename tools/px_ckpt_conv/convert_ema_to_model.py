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
from typing import Dict
import importlib

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


def load_ema_and_assign_to_model():
    args = get_args()
    model_func = importlib.import_module(args.load_model_provider)
    model = model_func.model_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    ema_model = model_func.ema_model_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    iteration, _ = load_checkpoint([model, ema_model], None, None)
    sd = ema_model.ema_model.state_dict()
    cpu_sd = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        cpu_sd[k] = v.cpu()

    model.load_state_dict(cpu_sd, strict=False)
    return model


def main():
    args = parse_args()
    assert not args.sequence_parallel

    args = validate_args(args)
    set_global_variables(args, build_tokenizer=True)
    args = get_args()
    _initialize_distributed()
    _set_random_seed(args.seed, args.data_parallel_random_init)
    if not is_wxacc1():
        fused_kernels.load(args)
    torch.distributed.barrier()

    model = load_ema_and_assign_to_model()
    torch.distributed.barrier()
    args.apply_ema = False
    save_checkpoint(1, [model], None, None, num_floating_point_operations_so_far=0)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        os.rename(os.path.join(args.save, 'iter_0000001'), os.path.join(args.save, 'release'))
        with open(os.path.join(args.save, 'latest_checkpointed_iteration.txt'), 'w') as outf:
            outf.write('release')


if __name__ == "__main__":
    main()
