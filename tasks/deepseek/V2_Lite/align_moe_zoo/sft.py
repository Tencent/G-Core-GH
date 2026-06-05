# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import os
import torch
from functools import partial
from packaging.version import Version

from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector

from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.arguments import gpatch_extra_args

stimer = StragglerDetector()

import pretrain_gpt

model_provider = pretrain_gpt.model_provider
forward_step = pretrain_gpt.forward_step
train_valid_test_datasets_provider = pretrain_gpt.train_valid_test_datasets_provider

if __name__ == "__main__":
    init_gpatch_for_mcore()
    # Temporary for transition to core datasetss
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=gpatch_extra_args,
    )
