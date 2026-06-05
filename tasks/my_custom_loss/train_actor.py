# coding=utf-8
"""
自定义 Loss 的训练入口

基于 math_rl_v3 的训练脚本，替换为自定义的 Actor 模型。
"""
from packaging.version import Version

import torch

from megatron.core import mpu, package_info
from megatron.core.enums import ModelType
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.utils import unwrap_model

try:
    from megatron.training import inprocess_restart
except ImportError:
    inprocess_restart = None

from gpatch.training.v3.ppo_actor import train_ppo_actor_v3
from gpatch.training.v3.default_model_provider import (
    default_sampler_client_provider,
    default_rm_critic_client_provider,
    default_gen_rm_client_provider,
)
from gpatch.training.v3.replay_buffer import ReplayBufferWithDataSource
from gpatch.core.parallel_state import is_mp_and_cp_head
from gpatch.patch_mcore import init_gpatch_for_mcore

# ################################################################
# ################### 【开始】导入自定义 Actor 模型 ###################
# ################################################################

from tasks.my_custom_loss.custom_actor_model import GptPpoCustomLossActorModel

# ################################################################
# ################### 【结束】导入自定义 Actor 模型 ###################
# ################################################################

# 复用 math_rl_v3 的组件
from tasks.math_rl_v3.args import get_tasks_args
from tasks.math_rl_v3.sp import get_ppo_prompt_format
from tasks.math_rl_v3.ppo_sampling import filter_samplings
from tasks.math_rl_v3.math_rl_actor_trainer import MathRLActorTrainer
from megatron_datasets.tasks.math_rl_v3.ppo_actor_dataset import (
    build_train_valid_test_datasets, DataCollator
)
from megatron_datasets.utils import get_iterator

mcore_version = Version(package_info.__version__)

# ################################################################
# ################### 【开始】自定义 Actor Provider ###################
# ################################################################


def custom_actor_provider(model, ref_model_state):
    """
    自定义 Actor Provider
    创建自定义 Actor 模型实例
    """
    args = get_args()

    actor_model = GptPpoCustomLossActorModel(
        model=model,
        ref_model_state=ref_model_state,
        unwrap_model_func=unwrap_model,
        # PPO args
        forward_micro_batch_size=args.ppo_logps_fwd_micro_batch_size,
        ppo_rollout_temperature=args.ppo_rollout_temperature,
        # SMART-PAD args
        pad_to_multi_of=args.ppo_rollout_pad_to_multiple_of,
        pad_token_id=get_tokenizer()._tokenizer.pad_token_id,
        dynamic_mbs_target_seqlen=args.ppo_train_dynamic_mbs_target_seq,
        dynamic_mbs_limit=args.ppo_train_dynamic_mbs_limit,
        ppo_pack_seq=args.ppo_pack_seq,
    )

    return actor_model


# ################################################################
# ################### 【结束】自定义 Actor Provider ###################
# ################################################################

# 使用自定义的 actor_provider，其他保持默认
actor_provider = custom_actor_provider
sampler_client_provider = default_sampler_client_provider
rm_critic_client_provider = default_rm_critic_client_provider
gen_rm_client_provider = default_gen_rm_client_provider


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """复用 math_rl_v3 的数据集构建逻辑"""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0('> building train, validation, and test datasets ...')
    prompt_format, eos_token = get_ppo_prompt_format(args, tokenizer)
    print_rank_0(f"building dataset with prompt_format {prompt_format} eos_token {eos_token}")

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        args,
        tokenizer,
        dp_rank=mpu.get_data_parallel_rank(),
        dp_size=mpu.get_data_parallel_world_size(),
        prompt_format=prompt_format,
        eos_token=eos_token
    )

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
    )

    return get_iterator(train_dataloader), None, None


def rollout_get_batch(data_iterator):
    """复用 math_rl_v3 的 rollout_get_batch"""
    args = get_args()
    assert is_mp_and_cp_head()

    data = next(data_iterator)
    tokens = data['input_ids']
    lpad_lens = data['lpad_lens']
    gt_label = data['gt_label']
    messages = data['messages']

    lpad_lens_list = lpad_lens.tolist()
    prompt_token_ids = []
    for i in range(len(tokens)):
        prompt_token_ids.append({
            'prompt_token_ids': tokens[i][:lpad_lens_list[i]].tolist(),
        })

    return {
        "prompt_token_ids": prompt_token_ids,
        "lpad_lens": lpad_lens,
        "gt_label": gt_label,
        "messages": messages,
    }


def extra_metric_info_provider():
    """定义额外的 metric 信息"""
    return [
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


def replay_buffer_provider(repeat, data_iter):
    return ReplayBufferWithDataSource(repeat, data_iter)


if __name__ == "__main__":
    init_gpatch_for_mcore()
    train_valid_test_datasets_provider.is_distributed = True

    trainer = MathRLActorTrainer(extra_metric_info=extra_metric_info_provider)

    if mcore_version >= Version("0.13.0"):
        from gpatch.training.v3.default_model_provider_0_13 import default_actor_model_provider
        model_provider = default_actor_model_provider
        train_ppo_actor_v3, store = inprocess_restart.maybe_wrap_for_inprocess_restart(
            train_ppo_actor_v3
        )
        extra_args = {"store": store}
    else:
        from gpatch.training.v3.default_model_provider import default_actor_model_provider
        model_provider = default_actor_model_provider
        extra_args = {}

    train_ppo_actor_v3(
        trainer,
        model_provider,
        actor_provider,  # 使用自定义的 actor_provider
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
