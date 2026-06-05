# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com, xiaotaoliu@tencent.com, nrwu@tencent.com

from functools import partial

import torch

from gpatch.training.v3.grpo_sampler import run_grpo_sampler_v3, GrpoSamplerV3
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.v3.default_model_provider import default_sampler_model_provider

from megatron_datasets.qwen2vl_dataset import get_processor
from tasks.qwen2vl.train_qwen2vl import add_qwen2vl_extra_args
from tasks.multimodal_grpo_sampler_utils import (
    SamplerGetBatch,
    gen_rollouts,
)

if __name__ == "__main__":
    init_gpatch_for_mcore()
    grpo_sampler = GrpoSamplerV3()
    get_batch_obj = SamplerGetBatch(get_processor, False)
    run_grpo_sampler_v3(
        grpo_sampler,
        default_sampler_model_provider,
        gen_func=partial(gen_rollouts, get_batch_obj=get_batch_obj),
        extra_args_provider=add_qwen2vl_extra_args,
    )
