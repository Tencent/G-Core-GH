from typing import Any, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.extended_model.base import (
    ApplySamplingRolloutAttrBase,
    PrepareDataForward,
    SamplerGenerateFunc,
)
from gpatch_v4.utils import (
    get_ltor_masks_and_position_ids,
    get_tensor_on_this_cp_rank,
    pad_or_truncate_last_dim,
)


class PrepareDataForwardAgentic(PrepareDataForward):
    @override
    def model_forward_only(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        model_fwd_args = {}

        tokens_l = []
        pixel_values_l = []
        image_grid_thw_l = []

        for batch in batches:
            tokens_l.append(
                pad_or_truncate_last_dim(
                    batch["tokens"],
                    seqlen,
                    pad_token_id,
                    pad_with_random_token=pad_with_random_token
                )
            )

            pixel_values = batch.get("pixel_values", None)
            image_grid_thw = batch.get("image_grid_thw", None)
            if pixel_values is not None:
                pixel_values_l.append(pixel_values)
                image_grid_thw_l.append(image_grid_thw)

        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=True)
        model_fwd_args["target"] = tokens.detach().clone()

        if pixel_values_l:
            pixel_values = torch.cat(pixel_values_l, dim=0).cuda(non_blocking=True)
            image_grid_thw = torch.cat(image_grid_thw_l, dim=0).cuda(non_blocking=True)
            model_fwd_args["pixel_values"] = pixel_values
            model_fwd_args["image_grid_thw"] = image_grid_thw

        if dist.get_world_size(mpu.get_context_parallel_group()) > 1:
            assert False

        model_fwd_args["input_ids"] = tokens
        model_fwd_args["position_ids"] = None
        model_fwd_args["attention_mask"] = None
        return model_fwd_args

    @override
    def grpo_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert not kwargs.get("distill_training", False)
        non_blocking = True
        tokens_l = []
        pixel_values_l = []
        image_grid_thw_l = []
        advantages_l = []
        mask_l = []
        logprobs_l = []
        ref_logprobs_l = []
        rollout_logprobs_l = []
        sequence_lengths_l = []
        for batch in batches:
            tokens_l.append(
                pad_or_truncate_last_dim(
                    batch['tokens'],
                    seqlen,
                    pad_token_id,
                    pad_with_random_token=pad_with_random_token
                )
            )
            advantages_l.append(pad_or_truncate_last_dim(batch['advantages'], seqlen - 1, 0))
            mask_l.append(pad_or_truncate_last_dim(batch['mask'], seqlen - 1, 0))
            logprobs_l.append(pad_or_truncate_last_dim(batch['logprobs'], seqlen - 1, 0))
            ref_logprobs_l.append(pad_or_truncate_last_dim(batch['ref_logprobs'], seqlen - 1, 0))
            rollout_logprobs_l.append(
                pad_or_truncate_last_dim(batch['rollout_log_probs'], seqlen - 1, 0.0)
            )
            sequence_lengths_l.append(batch['sequence_lengths'])
            pixel_values = batch.get("pixel_values", None)
            image_grid_thw = batch.get("image_grid_thw", None)
            if pixel_values is not None:
                pixel_values_l.append(pixel_values)
                image_grid_thw_l.append(image_grid_thw)

        tokens = torch.stack(tokens_l).cuda(non_blocking=non_blocking)
        advantages = torch.stack(advantages_l)
        mask = torch.stack(mask_l)
        logprobs = torch.stack(logprobs_l)
        ref_logprobs = torch.stack(ref_logprobs_l)
        rollout_log_probs = torch.stack(rollout_logprobs_l)
        sequence_lengths = torch.stack(sequence_lengths_l)

        target = tokens.detach().clone()

        # 数据先被 pad to multiple of 过了
        if dist.get_world_size(mpu.get_context_parallel_group()) > 1:
            assert False

        batch = {
            "tokens": tokens,
            "advantages": advantages,
            "prev_log_probs": logprobs,
            "mask": mask,
            "ref_log_probs": ref_logprobs,
            'target': target,
            'sequence_lengths': sequence_lengths,
            'rollout_log_probs': rollout_log_probs,
        }
        required_keys = set()
        if mpu.get_pipeline_model_parallel_world_size() == 1:
            required_keys.update(batch.keys())
        else:
            required_keys.add("attention_mask")
            required_keys.add("sequence_lengths")
            if mpu.is_pipeline_first_stage():
                required_keys.update(("tokens", "position_ids"))
            if mpu.is_pipeline_last_stage():
                required_keys.update(
                    (
                        "tokens", "advantages", "mask", "prev_log_probs", "ref_log_probs",
                        "rollout_log_probs", 'target'
                    )
                )

        batch = {
            key:
                (
                    val.cuda(non_blocking=non_blocking)
                    if key in required_keys and val is not None else None
                )
            for key, val in batch.items()
        }

        fwd_kwargs = dict(
            input_ids=batch.pop("tokens"),
            labels=None,
        )
        fwd_kwargs["position_ids"] = None
        fwd_kwargs["attention_mask"] = None

        if pixel_values_l:
            pixel_values = torch.cat(pixel_values_l, dim=0).cuda(non_blocking=True)
            image_grid_thw = torch.cat(image_grid_thw_l, dim=0).cuda(non_blocking=True)
            fwd_kwargs["pixel_values"] = pixel_values
            fwd_kwargs["image_grid_thw"] = image_grid_thw

        return batch, fwd_kwargs

    def ppo_value_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        non_blocking = True
        tokens_l = []
        values_l = []
        returns_l = []
        mask_l = []
        sequence_lengths_l = []
        pixel_values_l = []
        image_grid_thw_l = []
        for batch in batches:
            tokens_l.append(
                pad_or_truncate_last_dim(
                    batch['tokens'],
                    seqlen,
                    pad_token_id,
                    pad_with_random_token=pad_with_random_token
                )
            )
            values_l.append(pad_or_truncate_last_dim(
                batch['values'],
                seqlen - 1,
                0.0,
            ))
            returns_l.append(pad_or_truncate_last_dim(
                batch['returns'],
                seqlen - 1,
                0.0,
            ))
            mask_l.append(pad_or_truncate_last_dim(batch['mask'], seqlen - 1, 0))
            sequence_lengths_l.append(batch['sequence_lengths'])
            pixel_values = batch.get("pixel_values", None)
            image_grid_thw = batch.get("image_grid_thw", None)
            if pixel_values is not None:
                pixel_values_l.append(pixel_values)
                image_grid_thw_l.append(image_grid_thw)

        tokens = torch.stack(tokens_l).cuda(non_blocking=non_blocking)
        mask = torch.stack(mask_l)
        values = torch.stack(values_l).cuda(non_blocking=non_blocking)
        returns = torch.stack(returns_l).cuda(non_blocking=non_blocking)
        sequence_lengths = torch.stack(sequence_lengths_l).cuda(non_blocking=non_blocking)
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=0,  # unused
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            compute_attention_mask=False,
        )

        # 数据先被 pad to multiple of 过了
        if dist.get_world_size(mpu.get_context_parallel_group()) > 1 and not ppo_pack_seq:
            tokens = get_tensor_on_this_cp_rank(tokens, 1, key_name="tokens")
            attention_mask = get_tensor_on_this_cp_rank(
                attention_mask, 2, key_name="attention_mask"
            )
            position_ids = get_tensor_on_this_cp_rank(position_ids, 1, key_name="position_ids")

        batch = {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "mask": mask,
            "values": values,
            'sequence_lengths': sequence_lengths,
            'returns': returns,
        }
        required_keys = set()
        if mpu.get_pipeline_model_parallel_world_size() == 1:
            required_keys.update(batch.keys())
        else:
            required_keys.add("attention_mask")
            required_keys.add("sequence_lengths")
            required_keys.add("position_ids")
            if mpu.is_pipeline_first_stage():
                required_keys.update(("tokens", ))
            if mpu.is_pipeline_last_stage():
                required_keys.update(("tokens", "mask", "values", "returns"))

        batch = {
            key:
                (
                    val.cuda(non_blocking=non_blocking)
                    if key in required_keys and val is not None else None
                )
            for key, val in batch.items()
        }

        fwd_kwargs = dict(
            input_ids=batch.pop("tokens"),
            position_ids=batch.pop("position_ids"),
            attention_mask=batch.pop("attention_mask"),
            labels=None,
        )
        if pixel_values_l:
            pixel_values = torch.cat(pixel_values_l, dim=0).cuda(non_blocking=True)
            image_grid_thw = torch.cat(image_grid_thw_l, dim=0).cuda(non_blocking=True)
            fwd_kwargs["pixel_values"] = pixel_values
            fwd_kwargs["image_grid_thw"] = image_grid_thw
        return batch, fwd_kwargs

    @override
    def sft_train(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert False
