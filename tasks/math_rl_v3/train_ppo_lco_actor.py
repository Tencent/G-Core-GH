# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, guanyouhe@tencent.com

import time

from typing import Any, Dict, List, Tuple
from typing_extensions import override
from packaging.version import Version
from functools import partial

import torch

from gpatch.core.utils import clear_memory, print_with_rank_and_datetime
from megatron.core import package_info
from megatron.core import mpu, parallel_state
from megatron.core.enums import ModelType
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.utils import unwrap_model

try:
    from megatron.training import inprocess_restart
except ImportError:
    inprocess_restart = None

from gpatch.training.v3.ppo_actor import train_ppo_actor_v3
from gpatch.training.v3.default_model_provider import (
    default_actor_provider,
    default_sampler_client_provider,
    default_rm_critic_client_provider,
    default_gen_rm_client_provider,
)
from gpatch.core.device_type import is_wxacc1
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.core.models.gpt import GptPpoActorModel
from gpatch.core.models.gpt.gpt_ppo_actor_model import GptPpoActorModel
from gpatch.core.smart_pad_helper import (
    preprocess_packed_seqs,
    postprocess_packed_seqs,
)
from gpatch.core.aligner_helper import (
    from_parallel_logits_to_logprobs,
    masked_mean,
    average_losses_across_data_parallel_group,
)
from gpatch.core.ppo_helper import vocab_parallel_entropy, calculate_kl_loss

from gpatch.training.v3.replay_buffer import ReplayBufferWithDataSource
from gpatch.core.parallel_state import is_mp_and_cp_head
from tasks.math_rl_v3.args import get_tasks_args
from tasks.math_rl_v3.sp import get_ppo_prompt_format
from tasks.math_rl_v3.ppo_sampling import filter_samplings
from tasks.math_rl_v3.math_rl_actor_trainer import MathRLActorTrainer

from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy, VocabUtility

from megatron_datasets.tasks.math_rl_v3.ppo_actor_dataset import build_train_valid_test_datasets, DataCollator
from megatron_datasets.utils import get_iterator

mcore_version = Version(package_info.__version__)

sampler_client_provider = default_sampler_client_provider
rm_critic_client_provider = default_rm_critic_client_provider
gen_rm_client_provider = default_gen_rm_client_provider


def vocab_parallel_lse(logits, dim=-1, keepdim=True):
    logits_max = logits.max(dim=-1, keepdim=True).values.detach()
    torch.distributed.all_reduce(
        logits_max, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group()
    )
    normalized_vocab_parallel_logits = logits - logits_max
    sum_exp_logits = torch.sum(
        torch.exp(normalized_vocab_parallel_logits), dim=dim, keepdim=keepdim
    )
    normalized_vocab_parallel_logits = None

    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=mpu.get_tensor_model_parallel_group(),
    )
    out = torch.log(sum_exp_logits)

    out = out + logits_max
    return out


def vocab_parallel_gather(vocab_parallel_logits, target):
    """
    在vocab parallel场景下gather target位置的logits
    
    Args:
        vocab_parallel_logits: [batch, seq_len, partition_vocab_size]
        target: [batch, seq_len] - 全局vocab索引
    
    Returns:
        gathered_logits: [batch, seq_len] - target位置的logits值
    """

    # 获取分区信息
    partition_vocab_size = vocab_parallel_logits.size(-1)
    rank = mpu.get_tensor_model_parallel_rank()
    world_size = mpu.get_tensor_model_parallel_world_size()
    vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
        partition_vocab_size, rank, world_size
    )

    # 创建mask并调整索引
    target_mask = (target >= vocab_start_index) & (target < vocab_end_index)
    masked_target = (target - vocab_start_index) * target_mask

    # Gather操作
    gathered_logits = torch.gather(vocab_parallel_logits, dim=-1, index=masked_target)

    # 置零不在当前分区的位置
    gathered_logits[~target_mask] = 0.0

    # All-reduce获取完整结果
    torch.distributed.all_reduce(
        gathered_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=mpu.get_tensor_model_parallel_group(),
    )

    return gathered_logits


def vocab_parallel_log_softmax(logits, dim=-1, keepdim=True):
    return logits - vocab_parallel_lse(logits, dim=dim, keepdim=keepdim)


class GptPpoLcoActorModel(GptPpoActorModel):
    def __init__(self, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lco_rho_pos = 1.8
        self.lco_rho_neg = 0.9
        self.lco_min_rho_prob = 0.9
        self.vocab_size = vocab_size
        self.kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)

    def create_target_distribution(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        actions_log_probs: torch.Tensor,
        rho: float,
        min_rho_prob: float,
    ):
        assert rho > 0
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        actions = actions.unsqueeze(-1)
        actions_log_probs = actions_log_probs.unsqueeze(-1)

        vocab_start = tp_rank * self.vocab_size // tp_size
        vocab_end = (tp_rank + 1) * self.vocab_size // tp_size

        # use prev_logprobs
        # action_logprobs = torch.gather(torch.log_softmax(logits, dim=-1), dim=-1, index=actions).squeeze(-1)

        # a = torch.logsumexp(
        #       torch.scatter_add(
        #       logits,
        #       dim=-1,
        #       index=actions,
        #       src=torch.full_like(actions, fill_value=float("-inf"), dtype=logits.dtype)
        #   ),
        #   dim=-1,
        #   keepdim=True
        # )
        # mask/index construction
        action_mask = (actions >= vocab_start) & (actions < vocab_end)
        actions_local_shift = torch.where(
            action_mask, actions - vocab_start + 1, torch.zeros_like(actions)
        )

        # src for first scatter
        src = torch.zeros_like(actions_local_shift, dtype=logits.dtype)
        src = src.masked_fill(action_mask.bool(), float("-inf"))

        logits_pad = torch.nn.functional.pad(logits, (1, 0), value=0)  # pad 1 col on the left

        # a = lse of updated logits with -inf on target pos (exclude pad column)
        updated = torch.scatter_add(logits_pad, dim=-1, index=actions_local_shift, src=src)[:, 1:,
                                                                                            ...]

        a = vocab_parallel_lse(updated, dim=-1, keepdim=True)
        # b = gathered target logits (masked and all-reduced)
        # b = torch.gather(logits, dim=-1, index=actions)
        b = vocab_parallel_gather(logits, actions)

        # rho_probs = rho * action_logprobs.float().exp()
        rho_probs = rho * actions_log_probs.float().exp()

        if min_rho_prob is not None and rho < 1:  # only consider the case when rho < 1
            assert 0 < min_rho_prob < 1
            rho_probs = torch.where(
                rho_probs < min_rho_prob,
                actions_log_probs.float().exp(), rho_probs
            )

        c = torch.where(
            1 - rho_probs > 0, torch.log(rho_probs / (1 - rho_probs)),
            torch.tensor(1000., device=logits.device)
        )
        src_update = a - b + c
        logits2 = torch.scatter_add(logits_pad, dim=-1, index=actions_local_shift,
                                    src=src_update)[:, 1:, ...]

        lse = vocab_parallel_lse(logits2, dim=-1, keepdim=True)
        out = logits2 - lse
        return out

    def calculate_loss_with_mask(
        self, parallel_logits, lco_mask, mask, advantages, actions, action_logprobs, rho
    ):

        if lco_mask.sum().item() == 0:
            return torch.tensor(0.0, device=parallel_logits.device), torch.tensor(
                0.0, device=parallel_logits.device
            )

        lco_logits = parallel_logits[lco_mask]
        lco_actions = actions[lco_mask]
        lco_action_logprobs = action_logprobs[lco_mask]

        target_logits = self.create_target_distribution(
            lco_logits, lco_actions, lco_action_logprobs, rho, self.lco_min_rho_prob
        )
        target_logits = target_logits.detach()

        original_logits = vocab_parallel_log_softmax(lco_logits, dim=-1, keepdim=True)
        ratios = self.kl_loss(original_logits, target_logits)
        ratios = ratios.sum(-1)

        torch.distributed.all_reduce(
            ratios, op=torch.distributed.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group()
        )

        masked_adv = advantages[lco_mask]

        loss1 = masked_adv * ratios
        clip_max_loss = loss1

        actor_loss = clip_max_loss
        actor_loss = torch.mean(actor_loss)
        return actor_loss, ratios

    @override
    def get_actor_grpo_forward_output_and_loss_func(self, seqlen: int):
        def fwd_output_and_loss_func(seqlen, data_iterator, model):
            batches: List[Dict[str, Any]] = next(data_iterator)
            unwrapped_model = self.unwrap_model_func(model)

            batch, fwd_kwargs = self.prepare_data_for_grpo_loss(batches, seqlen)
            for key in ["mask", "advantages", "prev_log_probs", "ref_log_probs", "target"]:
                assert key in batch

            if not self.ppo_pack_seq:
                # 我们发现多数情况 H20 这样就很足够了，pack seq 小部分情况有收益。
                # 1. 如果你有多个 global batch，smart padding 足够好。
                # 2. 如果你只有 1 个 global batch，
                #   2.1 在长度都很长的情况，pack seq 无作用；
                #   2.2 长度都很短，pack seq 和 dynamic mbs 相同；
                #   2.3 长度有长有短，由于 megatron 的 MBS
                #       不可变，只能按照最长来处理，MBS=1，导致实际上无法 pack。
                parallel_logits = model(**fwd_kwargs)
            else:
                # 如果打开 --ppo-pack-seq 应该要将 --ppo-rollout-pad-to-multiple-of 调到很小
                # 这个 cond 会 call MultimodalRotaryEmbedding，导致 position id 不对。
                # 其他情况，m-core gpt model 的 position id 其实没有作用。
                assert not (
                    unwrapped_model.position_embedding_type == 'mrope' and
                    not unwrapped_model.config.multi_latent_attention
                )
                # [b, s] tensor indicating pad (0) or not (1)
                cur_mbs, cur_max_seqlen = fwd_kwargs['input_ids'].shape[:2]
                cur_actual_seqlen = batch['sequence_lengths'].unsqueeze(1).expand(
                    -1, cur_max_seqlen
                )
                tmpa = torch.arange(cur_max_seqlen, device='cuda',
                                    dtype=torch.int32).unsqueeze(0).expand(cur_mbs, -1)
                pad_mask = torch.ones(cur_mbs, cur_max_seqlen, device='cuda', dtype=torch.bool)
                pad_mask[tmpa >= cur_actual_seqlen] = False
                input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
                    fwd_kwargs['input_ids'],
                    pad_mask,
                    pre_process=unwrapped_model.pre_process,
                )
                input_ids_rmpad = input_ids_rmpad.contiguous()
                output_rmpad = model(
                    input_ids=input_ids_rmpad,
                    position_ids=fwd_kwargs['position_ids'],
                    attention_mask=None,
                    labels=None,
                    packed_seq_params=packed_seq_params,
                )
                parallel_logits = postprocess_packed_seqs(
                    output_rmpad,
                    packed_seq_params,
                    pad_mask,
                    cur_mbs,
                    cur_max_seqlen,
                    post_process=unwrapped_model.post_process,
                )

            if isinstance(parallel_logits, tuple):
                parallel_logits = parallel_logits[0]
            assert isinstance(parallel_logits, torch.Tensor)

            def loss_func(parallel_logits):
                parallel_logits = parallel_logits.float()
                mask = batch["mask"]
                advantages = batch["advantages"]
                prev_log_probs = batch["prev_log_probs"]
                ref_log_probs = batch["ref_log_probs"]
                actions = fwd_kwargs['input_ids']
                assert advantages.dtype == torch.float32
                assert prev_log_probs.dtype == torch.float32
                target = batch["target"]
                parallel_logits_clone = parallel_logits.clone()

                scaled_entropy = vocab_parallel_entropy(
                    parallel_logits_clone, mask, ignore_cp=self.ppo_pack_seq
                )
                scaled_entropy = scaled_entropy.detach()

                advantages_mask_pos = (advantages > 0) & mask.bool()
                advantages_mask_neg = (~advantages_mask_pos) & mask.bool()

                # trunc head
                actions = actions[:, 1:]
                parallel_logits_clone = parallel_logits_clone[:, 1:, :]
                parallel_logits_clone = parallel_logits_clone.roll(1, dims=1)

                actor_loss_pos, ratios_pos = \
                    self.calculate_loss_with_mask(parallel_logits_clone, advantages_mask_pos, mask,
                                                  advantages, actions, prev_log_probs, self.lco_rho_pos)

                actor_loss_neg, ratios_neg = \
                    self.calculate_loss_with_mask(parallel_logits_clone, advantages_mask_neg, mask,
                                                  advantages, actions, prev_log_probs, self.lco_rho_neg)

                actor_loss = (actor_loss_pos - actor_loss_neg)

                loss = actor_loss
                reduced_actor_loss = average_losses_across_data_parallel_group([loss])

                if Version(package_info.__version__) < Version("0.12.1"):
                    bwd_loss = loss * self.config.context_parallel_size
                else:
                    bwd_loss = loss.clone()

                return (
                    bwd_loss,
                    {
                        "loss": reduced_actor_loss[0],
                        "ppo_ratio": torch.tensor(0.0, device=loss.device),
                        "ppo_ratio_clamped": torch.tensor(0.0, device=loss.device),
                        "scaled_entropy": scaled_entropy,
                        "grpo_kl_loss": torch.tensor(0.0, device=loss.device),
                    },
                )

            return parallel_logits, loss_func

        return partial(fwd_output_and_loss_func, seqlen)


def lco_actor_provider(model, ref_model_state):
    args = get_args()

    actor_model = GptPpoLcoActorModel(
        vocab_size=args.padded_vocab_size,
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


actor_provider = lco_actor_provider


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
    return extra_metric_info


def replay_buffer_provider(repeat, data_iter):
    return ReplayBufferWithDataSource(repeat, data_iter)


if __name__ == "__main__":
    init_gpatch_for_mcore()
    train_valid_test_datasets_provider.is_distributed = True

    trainer = MathRLActorTrainer(extra_metric_info=extra_metric_info_provider)

    print(f"{mcore_version=} {Version('0.13.0')} {mcore_version < Version('0.13.0')}")
    if mcore_version >= Version("0.13.0"):
        from gpatch.training.v3.default_model_provider_0_13 import default_actor_model_provider
        model_provider = default_actor_model_provider

        assert inprocess_restart is not None
        # Optionally enable inprocess restart on pretrain
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
