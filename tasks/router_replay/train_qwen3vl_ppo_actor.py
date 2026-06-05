from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import torch
from packaging.version import Version
from typing_extensions import override

from megatron.core import mpu, package_info, parallel_state
from megatron.core.enums import ModelType
from megatron.core.models import vision
from megatron.training import get_args, get_tokenizer
from megatron.training.utils import unwrap_model

try:
    from megatron.training import inprocess_restart
except ImportError:
    inprocess_restart = None

from tasks.qwen3vl.train_qwen3vl import (
    add_qwen3vl_extra_args,
    model_provider,
    train_valid_test_data_iter_provider,
)

from tasks.qwen3vl.train_qwen3vl_ppo_actor import (
    actor_provider,
    extra_metric_info_provider,
    rollout_get_batch,
)

from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.v3.default_model_provider import (
    default_gen_rm_client_provider,
    default_rm_critic_client_provider,
    default_sampler_client_provider,
)
from gpatch.training.v3.ppo_actor import MultiModalPpoActorTrainer, train_ppo_actor_v3

mcore_version = Version(package_info.__version__)
sampler_client_provider = default_sampler_client_provider
rm_critic_client_provider = default_rm_critic_client_provider
gen_rm_client_provider = default_gen_rm_client_provider

if __name__ == "__main__":
    init_gpatch_for_mcore()

    print(f"{mcore_version=} {Version('0.13.0')} {mcore_version < Version('0.13.0')}")
    extra_args = {}
    if mcore_version >= Version("0.13.0"):
        assert inprocess_restart is not None
        # Optionally enable inprocess restart on pretrain
        train_ppo_actor_v3, store = inprocess_restart.maybe_wrap_for_inprocess_restart(
            train_ppo_actor_v3
        )
        extra_args = {"store": store}

    actor_trainer = MultiModalPpoActorTrainer(extra_metric_info=extra_metric_info_provider)
    train_ppo_actor_v3(
        actor_trainer,
        model_provider,
        actor_provider,
        sampler_client_provider,
        rm_critic_client_provider,
        gen_rm_client_provider,
        train_valid_test_data_iter_provider,
        rollout_get_batch,
        None,
        ModelType.encoder_and_decoder,
        extra_args_provider=add_qwen3vl_extra_args,
        **extra_args
    )
