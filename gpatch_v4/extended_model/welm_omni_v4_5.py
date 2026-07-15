import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from typing_extensions import override

logger = logging.getLogger(__name__)

from megatron.core import mpu

from gpatch_v4.extended_model.base import PrepareDataForward
from gpatch_v4.utils import get_ltor_masks_and_position_ids, pad_or_truncate_last_dim


class WelmOmniV45PrepareDataForward(PrepareDataForward):
    def model_forward_only(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        raise NotImplementedError("model_forward_only is not implemented")

    def grpo_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError("grpo_train is not implemented")

    def prepare_loss_weights(
        self,
        loss_weights: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        raise NotImplementedError("prepare_loss_weights is not implemented")

    @property
    def audio_token_id(self) -> int:
        return self.config.policy.hf_config.audio_token_id

    def _prepare_tokens_and_labels(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        seq_len: int,
        pad_token_id: int,
        vocab_size: int,
        pad_with_random_token: bool = False,
    ):
        # Resolve the audio token id up front; forbid random padding from
        # producing it (handled inside pad_or_truncate_last_dim).
        audio_token_id = self.audio_token_id
        # 先判断 labels 是否有被 shift 过
        assert tokens.shape == labels.shape, f"{tokens.shape=}, {labels.shape=}"
        assert torch.equal(
            tokens == labels, labels >= 0
        ), f"labels should not be shifted:{tokens.tolist()=} {labels.tolist()=}"
        if tokens.shape[-1] <= seq_len:
            actual_len = tokens.shape[-1]
            # 多加一位是为了 shift
            tokens = pad_or_truncate_last_dim(
                tokens,
                seq_len + 1,
                pad_token_id,
                pad_with_random_token=pad_with_random_token,
                vocab_size=vocab_size,
                truncate_left=False,
                forbidden_token_ids=[audio_token_id],
            )
            labels = pad_or_truncate_last_dim(labels, seq_len + 1, -100, truncate_left=False)
            tokens = tokens[:-1]
            labels = labels[1:]
        else:
            tokens = tokens[:-1]
            labels = labels[1:]
            tokens = tokens[-seq_len:]
            labels = labels[-seq_len:]
            actual_len = tokens.shape[-1]
        return tokens, labels, actual_len

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
        input_features_list = []
        audio_feature_lengths_list = []
        for i, batch in enumerate(batches):
            token, label, _ = self._prepare_tokens_and_labels(
                batch["tokens"],
                batch["labels"],
                seq_len,
                pad_token_id,
                vocab_size,
                pad_with_random_token,
            )

            token_list.append(token)
            label_list.append(label)
            input_features = batch.get("input_features", None)
            audio_feature_lengths = batch.get("audio_feature_lengths", None)
            if input_features is None or audio_feature_lengths is None:
                continue
            input_features_list.append(input_features)
            audio_feature_lengths_list.append(audio_feature_lengths)

        tokens = torch.stack(token_list).view(len(token_list), -1).cuda(non_blocking=True)
        labels = torch.stack(label_list).view(len(token_list), -1).cuda(non_blocking=True)
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            labels, -100, False, False, True, compute_attention_mask=comput_attn_mask
        )

        input_features = None
        audio_feature_lengths = None
        if len(input_features_list) > 0:
            # (mel_bins, total_frames_in_microbatch), (num_audios_in_microbatch,)
            input_features = torch.cat(input_features_list, dim=1).cuda(non_blocking=True)
            audio_feature_lengths = torch.cat(audio_feature_lengths_list,
                                              dim=0).cuda(non_blocking=True)

        batch = {
            "tokens": tokens,
            "labels": labels,
            "input_features": input_features,
            "audio_feature_lengths": audio_feature_lengths,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
        fwd_kwargs = {
            "input_ids": tokens,
            "labels": None,
            "input_features": input_features,
            "audio_feature_lengths": audio_feature_lengths,
            "position_ids": position_ids,
        }
        return batch, fwd_kwargs


__all__ = ["WelmOmniV45PrepareDataForward"]
