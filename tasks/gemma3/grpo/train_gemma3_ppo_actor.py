# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, guanyouhe@tencent.com

from typing import Any, Dict, List, Tuple
from typing_extensions import override
from packaging.version import Version

import torch

from megatron.core import package_info
from megatron.core import mpu, parallel_state
from megatron.core.enums import ModelType
from megatron.training import get_args, get_tokenizer
from megatron.training.utils import unwrap_model
from megatron.training.utils import get_ltor_masks_and_position_ids
try:
    from megatron.training import inprocess_restart
except ImportError:
    inprocess_restart = None

from gpatch.training.v3.ppo_actor import MultiModalPpoActorTrainer, train_ppo_actor_v3
from gpatch.core.device_type import is_wxacc1
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.core.models.gpt import GptPpoActorModel
from gpatch.core.utils import gen_unique_id, split_data_cp_rank
from gpatch.core.aligner_helper import pad_or_truncate_last_dim
from gpatch.training.v3.default_model_provider import (
    default_sampler_client_provider,
    default_rm_critic_client_provider,
    default_gen_rm_client_provider,
)

from megatron_datasets.mega_indexed_jsonl_dataset_v3 import update_consumed

from tasks.gemma3.train_gemma3 import (
    sft_model_provider,
    train_valid_test_data_iter_provider,
    add_extra_args,
)

mcore_version = Version(package_info.__version__)
sampler_client_provider = default_sampler_client_provider
rm_critic_client_provider = default_rm_critic_client_provider
gen_rm_client_provider = default_gen_rm_client_provider


class Gemma3PpoActorModel(GptPpoActorModel):
    @override
    def prepare_data_for_model_forward_only(self, batches: List[Dict[str, Any]],
                                            seqlen: int) -> Dict[str, Any]:
        tokens_l = []
        position_ids_l = []

        attention_mask_l = []
        sliding_window_attention_mask_l = []
        pixel_values_l = []
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seqlen
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, self.pad_token_id))
            assert batch["position_ids"].shape[-1] >= seqlen, "小于 seqlen 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seqlen, 0))

            assert batch["attention_mask"].shape[-1] >= seqlen
            assert batch["attention_mask"].shape[-2] >= seqlen
            attention_mask_l.append(batch["attention_mask"][..., :seqlen, :seqlen])
            sliding_window_attention_mask_l.append(
                batch["sliding_window_attention_mask"][..., :seqlen, :seqlen]
            )
            if batch["pixel_values"] is not None:
                pixel_values_l.append(batch["pixel_values"])

        non_blocking = False if is_wxacc1() else True
        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)

        attention_mask = torch.cat(attention_mask_l).cuda(non_blocking=non_blocking)
        sliding_window_attention_mask = torch.cat(sliding_window_attention_mask_l).cuda(
            non_blocking=non_blocking
        )
        if len(pixel_values_l) == 0:
            imgs = torch.tensor([], dtype=torch.bfloat16, device=tokens.device)
        else:
            imgs = torch.cat(pixel_values_l).cuda(non_blocking=non_blocking)
        if self.config.context_parallel_size > 1:
            if imgs.dim() > 1:
                imgs = split_data_cp_rank(imgs, self.config.context_parallel_size, 2)
                attention_mask = split_data_cp_rank(
                    attention_mask, self.config.context_parallel_size, 2
                )
                sliding_window_attention_mask = split_data_cp_rank(
                    sliding_window_attention_mask, self.config.context_parallel_size, 2
                )

        return dict(
            images=imgs,
            input_ids=tokens,
            target=tokens.detach().clone(),
            position_ids=position_ids,
            attention_mask=(attention_mask, sliding_window_attention_mask),
            image_token_index=get_tokenizer()._tokenizer.image_token_id,
        )

    @override
    def prepare_data_for_grpo_loss(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        tokens_l = []
        position_ids_l = []
        advantages_l = []
        mask_l = []
        logprobs_l = []
        ref_logprobs_l = []

        attention_mask_l = []
        sliding_window_attention_mask_l = []
        pixel_values_l = []
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seqlen
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, self.pad_token_id))
            assert batch["position_ids"].shape[-1] >= seqlen, "小于 seqlen 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seqlen, 0))

            advantages_l.append(pad_or_truncate_last_dim(batch["advantages"], seqlen - 1, 0))
            mask_l.append(pad_or_truncate_last_dim(batch["mask"], seqlen - 1, 0))
            logprobs_l.append(pad_or_truncate_last_dim(batch["logprobs"], seqlen - 1, 0))
            ref_logprobs_l.append(pad_or_truncate_last_dim(batch["ref_logprobs"], seqlen - 1, 0))

            assert batch["attention_mask"].shape[-1] >= seqlen
            assert batch["attention_mask"].shape[-2] >= seqlen
            attention_mask_l.append(batch["attention_mask"][..., :seqlen, :seqlen])
            sliding_window_attention_mask_l.append(
                batch["sliding_window_attention_mask"][..., :seqlen, :seqlen]
            )
            if batch["pixel_values"] is not None:
                pixel_values_l.append(batch["pixel_values"])

        non_blocking = False if is_wxacc1() else True
        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)

        advantages = torch.stack(advantages_l)
        mask = torch.stack(mask_l)
        logprobs = torch.stack(logprobs_l)
        ref_logprobs = torch.stack(ref_logprobs_l)

        attention_mask = torch.cat(attention_mask_l).cuda(non_blocking=non_blocking)
        sliding_window_attention_mask = torch.cat(sliding_window_attention_mask_l).cuda(
            non_blocking=non_blocking
        )
        if len(pixel_values_l) == 0:
            imgs = torch.tensor([], dtype=torch.bfloat16, device=tokens.device)
        else:
            imgs = torch.cat(pixel_values_l).cuda(non_blocking=non_blocking)

        if self.config.context_parallel_size > 1:
            # 放到GPU中切分可能更快
            if imgs.dim() > 1:
                imgs = split_data_cp_rank(imgs, self.config.context_parallel_size, 2)
                attention_mask = split_data_cp_rank(
                    attention_mask, self.config.context_parallel_size, 2
                )
                sliding_window_attention_mask = split_data_cp_rank(
                    sliding_window_attention_mask, self.config.context_parallel_size, 2
                )

        batch = {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "sliding_window_attention_mask": sliding_window_attention_mask,
            "imgs": imgs,
            "position_ids": position_ids,
            "advantages": advantages,
            "prev_log_probs": logprobs,
            "mask": mask,
            "ref_log_probs": ref_logprobs,
            'target': tokens.detach().clone(),
        }
        required_keys = set()
        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            required_keys.update(batch.keys())
        else:
            required_keys.update(("attention_mask", "sliding_window_attention_mask"))
            if parallel_state.is_pipeline_first_stage():
                required_keys.update(("tokens", "position_ids", "imgs"))
            if parallel_state.is_pipeline_last_stage():
                required_keys.update(
                    ("tokens", "advantages", "mask", "prev_log_probs", "ref_log_probs", 'target')
                )

        batch = {
            key: val.cuda(non_blocking=non_blocking) if key in required_keys else None
            for key, val in batch.items()
        }

        fwd_kwargs = dict(
            images=batch["imgs"],
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=(batch["attention_mask"], batch["sliding_window_attention_mask"]),
            image_token_index=get_tokenizer()._tokenizer.image_token_id,
        )

        return batch, fwd_kwargs


def actor_provider(model, ref_model_state):
    args = get_args()

    actor_model = Gemma3PpoActorModel(
        model=model,
        ref_model_state=ref_model_state,
        unwrap_model_func=unwrap_model,
        # PPO args
        forward_micro_batch_size=args.ppo_logps_fwd_micro_batch_size,
        ppo_rollout_temperature=args.ppo_rollout_temperature,
        # SMART-PAD args
        pad_to_multi_of=args.ppo_rollout_pad_to_multiple_of,
        pad_token_id=get_tokenizer()._tokenizer.pad_token_id,
    )

    return actor_model


# 这里不能做cp分割，要在model.forward前做
def rollout_get_batch(data_iterator):
    # 按照设计，只有 mp_head 会走到这里
    args = get_args()

    # Broadcast data.
    assert data_iterator is not None
    data = next(data_iterator)

    update_consumed(args.train_data_consuming_progresses, torch.distributed.get_rank(), data)

    json_data_list = data['json_data_list']

    tokens = data["input_ids"]
    attention_mask = data["attention_mask"]
    sliding_window_attention_mask = data["sliding_window_attention_mask"]
    prompt_len = data["prompt_len"]
    pixel_values = None
    if "pixel_values" in data:
        pixel_values = data["pixel_values"]

    # Position ids.
    _, _, position_ids = get_ltor_masks_and_position_ids(
        data["labels"],
        -100,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    batch_data = dict(
        # type is list
        unique_id=[gen_unique_id()],
        json_data_list=json_data_list,
        tokens=[tokens.squeeze(0)],
        prompt_len=[prompt_len],
        imgs_np_array_list=data["imgs_np_array_list"],
        # save at mm_data_cache, type is tensor
        position_ids=position_ids,
        attention_mask=attention_mask,
        sliding_window_attention_mask=sliding_window_attention_mask,
        ds_prompt_len=prompt_len,
        pixel_values=pixel_values,
        cache_keys=[
            "position_ids",
            "attention_mask",
            "sliding_window_attention_mask",
            "pixel_values",
        ],
    )
    return batch_data


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

    extra_metric_info = [
        {
            'key_name': 'acc_rewards',
            'dtype': torch.float32
        },
        {
            'key_name': 'fmt_rewards',
            'dtype': torch.float32
        },
    ]

    actor_trainer = MultiModalPpoActorTrainer(extra_metric_info=extra_metric_info)
    train_ppo_actor_v3(
        actor_trainer,
        sft_model_provider,
        actor_provider,
        sampler_client_provider,
        rm_critic_client_provider,
        gen_rm_client_provider,
        train_valid_test_data_iter_provider,
        rollout_get_batch,
        None,
        ModelType.encoder_and_decoder,
        extra_args_provider=add_extra_args,
        **extra_args
    )
