import functools
from typing import Any, Dict, List, Tuple

import torch
from typing_extensions import override

from gpatch_v4.extended_model.base import PrepareDataForward
from gpatch_v4.utils import get_ltor_masks_and_position_ids, pad_or_truncate_last_dim


def _patch_gemma4_kv_sharing_for_fsdp2():
    """Fix Gemma4 KV sharing under FSDP2.

    FSDP2's mixed-precision pre-hook uses tree_map on forward kwargs,
    which copies dicts/lists.  Gemma4 relies on in-place mutation of a
    ``shared_kv_states`` dict across decoder layers, so each layer ends
    up with its own empty copy and the KV-sharing lookup fails.

    We keep one canonical dict per forward pass and inject it into every
    decoder-layer call, bypassing the clone.
    """
    try:
        import transformers.models.gemma4.modeling_gemma4 as gem
    except ImportError:
        return

    if getattr(gem, "_kv_sharing_patched_for_fsdp2", False):
        return

    _kv_container = [{}]

    _orig_text_fwd = gem.Gemma4TextModel.forward

    @functools.wraps(_orig_text_fwd)
    def _text_model_forward(self, *args, **kwargs):
        _kv_container[0] = {}
        return _orig_text_fwd(self, *args, **kwargs)

    _orig_decoder_fwd = gem.Gemma4TextDecoderLayer.forward

    @functools.wraps(_orig_decoder_fwd)
    def _decoder_layer_forward(
        self,
        hidden_states,
        per_layer_input=None,
        shared_kv_states=None,
        **kwargs,
    ):
        return _orig_decoder_fwd(
            self,
            hidden_states,
            per_layer_input,
            shared_kv_states=_kv_container[0],
            **kwargs,
        )

    gem.Gemma4TextModel.forward = _text_model_forward
    gem.Gemma4TextDecoderLayer.forward = _decoder_layer_forward
    gem._kv_sharing_patched_for_fsdp2 = True


_patch_gemma4_kv_sharing_for_fsdp2()


class Gemma4PrepareDataForward(PrepareDataForward):
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
        vocab_size = kwargs.get("vocab_size", 0)
        token_list = []
        label_list = []
        pixel_values_list = []
        image_position_ids_list = []
        mm_token_type_ids_list = []
        attention_mask_list = []

        for batch in batches:
            token = batch['tokens']
            label = batch['labels']
            token_len = token.shape[-1]

            if token_len <= seq_len:
                token = pad_or_truncate_last_dim(
                    token,
                    seq_len + 1,
                    pad_token_id,
                    pad_with_random_token=pad_with_random_token,
                    vocab_size=vocab_size,
                )
                label = pad_or_truncate_last_dim(label, seq_len + 1, -100)
                token = token[:-1]
                label = label[1:]
            else:
                token = token[:-1]
                label = label[1:]
                token = token[-seq_len:]
                label = label[-seq_len:]

            token_list.append(token)
            label_list.append(label)

            if "pixel_values" in batch and batch["pixel_values"] is not None:
                pixel_values_list.append(batch["pixel_values"])
            if "image_position_ids" in batch and batch["image_position_ids"] is not None:
                image_position_ids_list.append(batch["image_position_ids"])
            if "mm_token_type_ids" in batch and batch["mm_token_type_ids"] is not None:
                mm_ids = pad_or_truncate_last_dim(batch["mm_token_type_ids"], seq_len, 0)
                mm_token_type_ids_list.append(mm_ids)
            if "attention_mask" in batch and batch["attention_mask"] is not None:
                amask = pad_or_truncate_last_dim(batch["attention_mask"], seq_len, 0)
                attention_mask_list.append(amask)

        tokens = torch.stack(token_list).cuda(non_blocking=True)
        labels = torch.stack(label_list).cuda(non_blocking=True)

        loss_mask = torch.ones(labels.size(), dtype=torch.float, device=labels.device)
        loss_mask[labels == pad_token_id] = 0.0
        loss_mask[labels == -100] = 0.0

        attention_mask = None
        if attention_mask_list:
            attention_mask = torch.stack(attention_mask_list).cuda(non_blocking=True)

        pixel_values = None
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0).cuda(non_blocking=True)

        image_position_ids = None
        if image_position_ids_list:
            image_position_ids = torch.cat(image_position_ids_list, dim=0).cuda(non_blocking=True)

        mm_token_type_ids = None
        if mm_token_type_ids_list:
            mm_token_type_ids = torch.stack(mm_token_type_ids_list).cuda(non_blocking=True)

        batch_out = {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

        fwd_kwargs = dict(
            input_ids=tokens,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            mm_token_type_ids=mm_token_type_ids,
            labels=None,
        )
        return batch_out, fwd_kwargs

    @override
    def model_forward_only(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        vocab_size = kwargs.get("vocab_size", 0)
        token_list = []
        pixel_values_list = []
        image_position_ids_list = []
        mm_token_type_ids_list = []
        attention_mask_list = []

        for batch in batches:
            token_list.append(
                pad_or_truncate_last_dim(
                    batch["tokens"],
                    seqlen,
                    pad_token_id,
                    pad_with_random_token=pad_with_random_token,
                    vocab_size=vocab_size,
                )
            )
            if "pixel_values" in batch and batch["pixel_values"] is not None:
                pixel_values_list.append(batch["pixel_values"])
            if "image_position_ids" in batch and batch["image_position_ids"] is not None:
                image_position_ids_list.append(batch["image_position_ids"])
            if "mm_token_type_ids" in batch and batch["mm_token_type_ids"] is not None:
                mm_ids = pad_or_truncate_last_dim(batch["mm_token_type_ids"], seqlen, 0)
                mm_token_type_ids_list.append(mm_ids)
            if "attention_mask" in batch and batch["attention_mask"] is not None:
                amask = pad_or_truncate_last_dim(batch["attention_mask"], seqlen, 0)
                attention_mask_list.append(amask)

        tokens = torch.stack(token_list).cuda(non_blocking=True)
        attention_mask = None
        if attention_mask_list:
            attention_mask = torch.stack(attention_mask_list).cuda(non_blocking=True)

        pixel_values = None
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0).cuda(non_blocking=True)

        image_position_ids = None
        if image_position_ids_list:
            image_position_ids = torch.cat(image_position_ids_list, dim=0).cuda(non_blocking=True)

        mm_token_type_ids = None
        if mm_token_type_ids_list:
            mm_token_type_ids = torch.stack(mm_token_type_ids_list).cuda(non_blocking=True)

        return dict(
            input_ids=tokens,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            mm_token_type_ids=mm_token_type_ids,
            labels=None,
            target=tokens.detach().clone(),
        )

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
        assert not ppo_pack_seq

        non_blocking = True
        tokens_l = []
        advantages_l = []
        mask_l = []
        logprobs_l = []
        ref_logprobs_l = []
        rollout_logprobs_l = []
        pixel_values_list = []
        image_position_ids_list = []
        mm_token_type_ids_list = []
        attention_mask_list = []

        has_rollout_logprobs = "rollout_log_probs" in batches[0]

        for batch in batches:
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, pad_token_id))
            advantages_l.append(pad_or_truncate_last_dim(batch["advantages"], seqlen - 1, 0))
            mask_l.append(pad_or_truncate_last_dim(batch["mask"], seqlen - 1, 0))
            logprobs_l.append(pad_or_truncate_last_dim(batch["logprobs"], seqlen - 1, 0))
            ref_logprobs_l.append(pad_or_truncate_last_dim(batch["ref_logprobs"], seqlen - 1, 0))
            if has_rollout_logprobs:
                rollout_logprobs_l.append(
                    pad_or_truncate_last_dim(batch["rollout_log_probs"], seqlen - 1, 0)
                )

            if "pixel_values" in batch and batch["pixel_values"] is not None:
                pixel_values_list.append(batch["pixel_values"])
            if "image_position_ids" in batch and batch["image_position_ids"] is not None:
                image_position_ids_list.append(batch["image_position_ids"])
            if "mm_token_type_ids" in batch and batch["mm_token_type_ids"] is not None:
                mm_ids = pad_or_truncate_last_dim(batch["mm_token_type_ids"], seqlen, 0)
                mm_token_type_ids_list.append(mm_ids)
            if "attention_mask" in batch and batch["attention_mask"] is not None:
                amask = pad_or_truncate_last_dim(batch["attention_mask"], seqlen, 0)
                attention_mask_list.append(amask)

        tokens = torch.stack(tokens_l).cuda(non_blocking=non_blocking)
        target = tokens.detach().clone()
        advantages = torch.stack(advantages_l)
        mask = torch.stack(mask_l)
        logprobs = torch.stack(logprobs_l)
        ref_logprobs = torch.stack(ref_logprobs_l)
        rollout_log_probs = (torch.stack(rollout_logprobs_l) if has_rollout_logprobs else None)

        attention_mask = None
        if attention_mask_list:
            attention_mask = torch.stack(attention_mask_list).cuda(non_blocking=non_blocking)

        pixel_values = None
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0).cuda(non_blocking=non_blocking)

        image_position_ids = None
        if image_position_ids_list:
            image_position_ids = torch.cat(image_position_ids_list,
                                           dim=0).cuda(non_blocking=non_blocking)

        mm_token_type_ids = None
        if mm_token_type_ids_list:
            mm_token_type_ids = torch.stack(mm_token_type_ids_list).cuda(non_blocking=non_blocking)

        batch_out = {
            "advantages": advantages.cuda(non_blocking=non_blocking),
            "prev_log_probs": logprobs.cuda(non_blocking=non_blocking),
            "mask": mask.cuda(non_blocking=non_blocking),
            "ref_log_probs": ref_logprobs.cuda(non_blocking=non_blocking),
            "target": target,
        }
        if rollout_log_probs is not None:
            batch_out["rollout_log_probs"] = rollout_log_probs.cuda(non_blocking=non_blocking)

        if "sample_mask" in batches[0]:
            sample_mask_l = [_batch["sample_mask"] for _batch in batches]
            batch_out["sample_mask"] = torch.stack(sample_mask_l).cuda(non_blocking=non_blocking)

        if "entropy_aux_figures" in batches[0]:
            batch_out["entropy_aux_figures"] = batches[0]["entropy_aux_figures"].cuda(
                non_blocking=non_blocking
            )

        fwd_kwargs = dict(
            input_ids=tokens,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            mm_token_type_ids=mm_token_type_ids,
            labels=None,
        )
        return batch_out, fwd_kwargs
