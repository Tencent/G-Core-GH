# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from packaging.version import Version

import torch

from megatron.core import mpu, package_info
from megatron.core.enums import ModelType

from megatron.training import get_args, get_tokenizer, print_rank_0

try:
    from megatron.training import inprocess_restart
except ImportError:
    inprocess_restart = None

from megatron_datasets.tasks.math_rl_v3.ppo_actor_dataset import build_train_valid_test_datasets, DataCollator
from megatron_datasets.utils import get_iterator

from gpatch.training.v3.ppo_actor import train_ppo_actor_v3
from gpatch.training.v3.default_model_provider import (
    default_actor_provider,
    default_megatron_bridge_actor_model_provider,
    default_replay_buffer_provider,
    default_sampler_client_provider,
    default_rm_critic_client_provider,
    default_gen_rm_client_provider,
)
from gpatch.training.v3.replay_buffer import ReplayBufferWithDataSource
from gpatch.core.parallel_state import is_mp_and_cp_head

from gpatch.patch_mcore import init_gpatch_for_mcore

from tasks.math_rl_v3.args import get_tasks_args
from tasks.math_rl_v3.sp import get_ppo_prompt_format
from tasks.math_rl_v3.ppo_sampling import filter_samplings
from tasks.math_rl_v3.math_rl_actor_trainer import MathRLActorTrainer, MathRLHeteroGenRMActorTrainer

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
        gen_left_pad=args.gen_left_pad
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
    messages = data['messages']

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
        "messages": messages,
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
    for k, v in args.gdpo_reward_weights.items():
        extra_metric_info.append({'key_name': k, 'dtype': torch.float32})
    extra_metric_info = list({frozenset(d.items()): d for d in extra_metric_info}.values())
    return extra_metric_info


def replay_buffer_provider(repeat, data_iter):
    return ReplayBufferWithDataSource(repeat, data_iter)


if __name__ == "__main__":
    init_gpatch_for_mcore()
    train_valid_test_datasets_provider.is_distributed = True

    trainer = MathRLHeteroGenRMActorTrainer(extra_metric_info=extra_metric_info_provider)
    # trainer = MathRLActorTrainer(extra_metric_info=extra_metric_info_provider)

    print(f"{mcore_version=} = {Version('0.16.0rc0')} is {mcore_version == Version('0.16.0rc0')}")
    assert mcore_version >= Version('0.16.0rc0')

    assert inprocess_restart is not None
    # Optionally enable inprocess restart on pretrain
    train_ppo_actor_v3, store = inprocess_restart.maybe_wrap_for_inprocess_restart(
        train_ppo_actor_v3
    )
    extra_args = {"store": store}

    train_ppo_actor_v3(
        trainer,
        default_megatron_bridge_actor_model_provider,
        actor_provider,
        sampler_client_provider,
        rm_critic_client_provider,
        gen_rm_client_provider,
        train_valid_test_datasets_provider,
        rollout_get_batch,
        filter_samplings,
        ModelType.encoder_or_decoder,
        extra_args_provider=get_tasks_args,
        replay_buffer_provider=replay_buffer_provider,
        **extra_args
    )
