# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, guanyouhe@tencent.com

from typing import Any, Dict, List, Tuple

import torch
from typing_extensions import override

from megatron_datasets.mega_indexed_jsonl_dataset_v3 import update_consumed
from tasks.glm4v.train_glm4vl import (
    add_glm4vl_extra_args,
    model_provider,
    train_valid_test_data_iter_provider,
)

from megatron.core import mpu, parallel_state
from megatron.core.enums import ModelType
from megatron.core.models import vision
from megatron.training import get_args, get_tokenizer
from megatron.training.utils import unwrap_model

from gpatch.core.aligner_helper import pad_or_truncate_last_dim
from gpatch.core.device_type import is_wxacc1
from gpatch.core.models.gpt import (
    GptPpoActorModel,
    GptPpoGenRmClientV3,
    GptPpoRmCriticClientV3,
    GptPpoSamplerClientV3,
)
from gpatch.core.utils import gen_unique_id, split_data_ulysses_cp_rank
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.v3.default_model_provider import (
    default_gen_rm_client_provider,
    default_rm_critic_client_provider,
    default_sampler_client_provider,
)
from gpatch.training.v3.ppo_actor import MultiModalPpoActorTrainer, train_ppo_actor_v3


class GlmVLPpoActorModel(GptPpoActorModel):
    @override
    def prepare_data_for_model_forward_only(self, batches: List[Dict[str, Any]],
                                            seqlen: int) -> Dict[str, Any]:
        tokens_l = []
        position_ids_l = []
        image_grid_thw_l = []
        pixel_values_l = []
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seqlen
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, self.pad_token_id))
            assert batch["position_ids"].shape[-1] >= seqlen, "小于 seqlen 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seqlen, 0))
            image_grid_thw_l.append(batch["image_grid_thw"])
            pixel_values_l.append(batch["pixel_values"])

        non_blocking = False if is_wxacc1() else True
        tokens = (torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking))
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)
        image_grid_thw = torch.cat(image_grid_thw_l, dim=0).cuda(non_blocking=non_blocking)
        pixel_values = torch.cat(pixel_values_l, dim=0).cuda(non_blocking=non_blocking)

        return dict(
            input_ids=tokens,
            target=tokens.detach().clone(),
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            pixel_values=pixel_values,
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

        image_grid_thw_l = []
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

            image_grid_thw_l.append(batch["image_grid_thw"])
            pixel_values_l.append(batch["pixel_values"])

        non_blocking = False if is_wxacc1() else True
        tokens = (torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking))
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)

        advantages = torch.stack(advantages_l)
        mask = torch.stack(mask_l)
        logprobs = torch.stack(logprobs_l)
        ref_logprobs = torch.stack(ref_logprobs_l)

        image_grid_thw = torch.cat(image_grid_thw_l, dim=0).cuda(non_blocking=non_blocking)
        pixel_values = torch.cat(pixel_values_l, dim=0).cuda(non_blocking=non_blocking)

        batch = {
            "input_ids": tokens,
            "position_ids": position_ids.cuda(non_blocking=non_blocking),
            "image_grid_thw": image_grid_thw,
            "pixel_values": pixel_values,
            "advantages": advantages,
            "prev_log_probs": logprobs,
            "mask": mask,
            "ref_log_probs": ref_logprobs,
            "target": tokens.detach().clone(),
        }
        if parallel_state.is_pipeline_last_stage():
            for k in ["mask", "prev_log_probs", "ref_log_probs", "advantages"]:
                batch[k] = batch[k].cuda(non_blocking=non_blocking)

        fwd_kwargs = dict(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
        )

        return batch, fwd_kwargs


def actor_provider(model, ref_model_state):
    args = get_args()

    actor_model = GlmVLPpoActorModel(
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


def rollout_get_batch(data_iterator):
    # 按照设计，只有 mp_head 会走到这里
    args = get_args()

    # Broadcast data.
    assert data_iterator is not None
    data = next(data_iterator)

    if args.px_data_config_path is not None:
        update_consumed(args.train_data_consuming_progresses, torch.distributed.get_rank(), data)

    json_data_list = data["json_data_list"]

    tokens = data["input_ids"]
    assert (
        tokens.shape[0] == 1 and len(json_data_list) == 1
    ), "--ppo-rollout-micro-batch-size must be 1"
    image_grid_thw = data["image_grid_thw"]
    position_ids = data["position_ids"]
    prompt_len = data["prompt_len"]

    image_input_mask = data["image_input_mask"]
    image_padded = data["image_padded"]
    image_padded = image_padded.bool()[0].item()
    assert not image_padded, f"image padded 必须为 False，因为会被重新组合"

    pixel_values = None
    if "pixel_values" in data:
        pixel_values = data["pixel_values"].type(torch.bfloat16)

    tokens_for_gen = data.get("input_ids_for_gen", [])

    batch_data = dict(
        # type is list
        unique_id=[gen_unique_id()],
        json_data_list=json_data_list,
        tokens=[tokens.squeeze(0)],
        tokens_for_gen=tokens_for_gen,
        prompt_len=[prompt_len],
        imgs_np_array_list=data["imgs_np_array_list"],
        # save at mm_data_cache, type is tensor
        position_ids=position_ids,
        image_grid_thw=image_grid_thw,
        image_input_mask=image_input_mask,
        pixel_values=pixel_values,
        cache_keys=[
            "position_ids",
            "image_grid_thw",
            "image_input_mask",
            "pixel_values",
        ],
    )
    return batch_data


# 初始化 MultiModalPpoActorTrainer 时拿不到args，无法判断是rm还是gen-rm
# 通过extra_metric_info_provider判断
def extra_metric_info_provider():
    extra_metric_info = [
        {
            "key_name": "acc_rewards",
            "dtype": torch.float32
        },
        {
            "key_name": "fmt_rewards",
            "dtype": torch.float32
        },
    ]
    args = get_args()
    if args.use_gen_rm:
        extra_metric_info.append({"key_name": "rm_rewards", "dtype": torch.float32})
    return extra_metric_info


if __name__ == "__main__":
    init_gpatch_for_mcore()
    actor_trainer = MultiModalPpoActorTrainer(extra_metric_info=extra_metric_info_provider)
    train_ppo_actor_v3(
        actor_trainer,
        model_provider,
        actor_provider,
        default_sampler_client_provider,
        default_rm_critic_client_provider,
        default_gen_rm_client_provider,
        train_valid_test_data_iter_provider,
        rollout_get_batch,
        None,
        ModelType.encoder_or_decoder,
        extra_args_provider=add_glm4vl_extra_args,
    )
