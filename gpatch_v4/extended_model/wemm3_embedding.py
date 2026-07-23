from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from typing_extensions import override

try:
    from megatron.bridge.utils.context_parallel_utils import qwen3vl_parallel_split
except ImportError:
    qwen3vl_parallel_split = None
from megatron.core import mpu

from gpatch_v4.extended_model.base import PrepareDataForward
from gpatch_v4.utils import (
    get_tensor_on_this_cp_rank,
    pad_or_truncate_last_dim,
    qwen2vl_pad_and_split,
)


class Wemm3EmbeddingPrepareDataForward(PrepareDataForward):
    def _padding_images(
        self,
        vision_data: List[torch.Tensor],
        vision_grid_thw: List[torch.Tensor],
    ) -> Tuple[list[int], list[bool], torch.Tensor, torch.Tensor]:
        # megatron-bridge 要支持 TP/CP as DP
        if not self.config.training.build_from_mbridge:
            parallel_size = mpu.get_tensor_and_context_parallel_world_size()
            vision_data = torch.cat(vision_data, dim=0)
            vision_grid_thw = torch.cat(vision_grid_thw, dim=0)
            cp_img_num = None
            images_padded = None
            if parallel_size > 1:
                vision_data, vision_grid_thw, cp_img_num = qwen3vl_parallel_split(
                    parallel_size,
                    vision_data,
                    vision_grid_thw,
                )
            return cp_img_num, images_padded, vision_data, vision_grid_thw

        # not support padding image now
        hw_factor = 4
        cp_size = mpu.get_context_parallel_world_size()
        vision_data, vision_grid_thw, cp_img_num, images_padded = qwen2vl_pad_and_split(
            cp_size,
            hw_factor,
            vision_data,
            vision_grid_thw,
        )

        vision_data = torch.cat(vision_data, dim=0)
        vision_grid_thw = torch.cat(vision_grid_thw, dim=0)
        for i in range(len(images_padded)):
            images_padded[i] = bool(images_padded[i])
            assert not images_padded[i], "not support padding image now"
        return cp_img_num, images_padded, vision_data, vision_grid_thw

    @override
    def model_forward_only(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        raise NotImplementedError("model_forward_only is not implemented")

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
        raise NotImplementedError("model_forward_only is not implemented")

    def _shift_label(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        seq_len: int,
        pad_token_id: int,
        pad_with_random_token: bool = False,
    ):
        # 先判断 labels 是否有被 shift 过
        assert tokens.shape == labels.shape, f"{tokens.shape=}, {labels.shape=}"
        assert torch.equal(
            tokens == labels, labels >= 0
        ), f"labels should not be shifted:{tokens.tolist()=} {labels.tolist()=}"
        if tokens.shape[-1] <= seq_len:
            # 多加一位是为了 shift
            tokens = pad_or_truncate_last_dim(
                tokens, seq_len + 1, pad_token_id, pad_with_random_token=pad_with_random_token
            )
            labels = pad_or_truncate_last_dim(labels, seq_len + 1, -100)
            tokens = tokens[..., :-1]
            labels = labels[..., 1:]
        else:
            tokens = tokens[..., :-1]
            labels = labels[..., 1:]
            tokens = tokens[..., -seq_len:]
            labels = labels[..., -seq_len:]
        return tokens, labels

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
        tokens_l = []
        labels_l = []
        loss_mask_l = []
        position_ids_l = []
        attention_mask_l = []
        image_input_mask_l = []
        vision_grid_thw_l = []
        vision_data_l = []
        input_features_l = []
        feature_attention_mask_l = []
        video_second_per_grid_l = []
        audio_feature_l = []
        meta_info_l = []
        for batch in batches:
            attention_mask = pad_or_truncate_last_dim(batch["attention_mask"], seq_len, 0)
            tokens, labels = self._shift_label(
                batch["tokens"],
                batch["labels"],
                seq_len,
                pad_token_id,
                pad_with_random_token,
            )

            tokens_l.append(tokens)
            labels_l.append(labels)
            loss_mask = torch.ones(labels.size(), dtype=torch.float)
            loss_mask[labels == -100] = 0.0
            loss_mask_l.append(pad_or_truncate_last_dim(loss_mask, seq_len, 0))
            assert batch["position_ids"].shape[-1] >= seq_len, "小于 seq_len 时, 不能 pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seq_len, 0))
            attention_mask_l.append(attention_mask)

            if "vision_data" in batch and batch["vision_data"] is not None:
                vision_grid_thw_l.append(batch["vision_grid_thw"])
                vision_data_l.append(batch["vision_data"])
            if "image_input_mask" in batch and batch["image_input_mask"] is not None:
                image_input_mask = pad_or_truncate_last_dim(batch["image_input_mask"], seq_len, 0)
                image_input_mask_l.append(image_input_mask)

            if "input_features" in batch and batch["input_features"] is not None:
                input_features = pad_or_truncate_last_dim(
                    batch["input_features"], seq_len, pad_token_id
                )
                input_features_l.append(input_features)
                feature_attention_mask = pad_or_truncate_last_dim(
                    batch["feature_attention_mask"], seq_len, 0
                )
                feature_attention_mask_l.append(feature_attention_mask)
                video_second_per_grid_l.append(batch["video_second_per_grid"])

            if "audio_feature" in batch and batch["audio_feature"] is not None:
                # audio_feature = pad_or_truncate_last_dim(batch["audio_feature"], seq_len, pad_token_id)
                audio_feature = batch["audio_feature"]
                audio_feature_l.append(audio_feature)

            if "meta_info" in batch:
                meta_info_l.append(batch["meta_info"])

        non_blocking = True
        tokens = torch.cat(tokens_l, dim=0).cuda(non_blocking=non_blocking)
        labels = torch.cat(labels_l, dim=0).cuda(non_blocking=non_blocking)
        loss_mask = torch.cat(loss_mask_l, dim=0).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)
        attention_mask = torch.cat(attention_mask_l, dim=0).cuda(non_blocking=non_blocking)

        image_input_mask = None
        cp_img_num, images_padded, vision_data, vision_grid_thw = None, None, None, None
        if len(vision_data_l) > 0:
            cp_img_num, images_padded, vision_data, vision_grid_thw = self._padding_images(
                vision_data_l, vision_grid_thw_l
            )
            image_input_mask = torch.cat(image_input_mask_l, dim=0).cuda(non_blocking=non_blocking)
            vision_data = vision_data.cuda(non_blocking=non_blocking)
            vision_grid_thw = vision_grid_thw.cuda(non_blocking=non_blocking)

        full_loss_mask = loss_mask
        if mpu.get_context_parallel_world_size() > 1:
            labels = get_tensor_on_this_cp_rank(labels, 1, key_name="labels")
            loss_mask = get_tensor_on_this_cp_rank(loss_mask, 1, key_name="loss_mask")

        input_features = None
        feature_attention_mask = None
        video_second_per_grid = None
        if len(input_features_l) > 0:
            input_features = torch.cat(input_features_l, dim=0).cuda(non_blocking=non_blocking)
            feature_attention_mask = torch.cat(feature_attention_mask_l,
                                               dim=0).cuda(non_blocking=non_blocking)
            video_second_per_grid = torch.cat(video_second_per_grid_l,
                                              dim=0).cuda(non_blocking=non_blocking)

        audio_feature = None
        if len(audio_feature_l) > 0:
            audio_feature = torch.cat(audio_feature_l, dim=0).cuda(non_blocking=non_blocking)

        batch = {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "full_loss_mask": full_loss_mask,
            "image_input_mask": image_input_mask,
            "image_grid_thw": vision_grid_thw,
            "pixel_values": vision_data,
            "cp_img_num": cp_img_num,
            "images_padded": images_padded,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "video_second_per_grid": video_second_per_grid,
            "audio_feature": audio_feature,
            "meta_info": meta_info_l if len(meta_info_l) > 0 else None,
        }

        only_return_last_hidden_state = batches[0]['only_return_last_hidden_state'
                                                  ] if 'only_return_last_hidden_state' in batches[
                                                      0] else False
        fwd_kwargs = dict(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=None,
            labels=None,
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            image_input_mask=batch["image_input_mask"],
            images_padded=batch["images_padded"],
            cp_img_num=batch["cp_img_num"],
            only_return_last_hidden_state=only_return_last_hidden_state,
        )
        if batch["input_features"] is not None:
            fwd_kwargs["input_features"] = batch["input_features"]
            fwd_kwargs["feature_attention_mask"] = batch["feature_attention_mask"]
            fwd_kwargs["video_second_per_grid"] = batch["video_second_per_grid"]

        if audio_feature is not None:
            fwd_kwargs["audio_feature"] = audio_feature

        return batch, fwd_kwargs
