# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com, xiaotaoliu@tencent.com, nrwu@tencent.com

from functools import partial

from gpatch.training.v3.grpo_gen_rm import GrpoGenRmTrainerV3, run_grpo_gen_rm_v3
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.v3.default_model_provider import default_gen_rm_model_provider

from megatron_datasets.qwen2vl_dataset import get_processor
from tasks.qwen3vl.train_qwen3vl import add_qwen3vl_extra_args
from tasks.multimodal_comm.extra_args import ppo_mm_extra_args

from tasks.multimodal_grpo_gen_rm_utils import gen_rm_func

if __name__ == "__main__":
    init_gpatch_for_mcore()

    trainer = GrpoGenRmTrainerV3()
    run_grpo_gen_rm_v3(
        trainer,
        default_gen_rm_model_provider,
        partial(gen_rm_func, get_processor_func=get_processor),
        extra_args_provider=partial(ppo_mm_extra_args, add_qwen3vl_extra_args),
    )
