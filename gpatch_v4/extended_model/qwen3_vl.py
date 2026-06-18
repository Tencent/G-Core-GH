from typing import Any, Dict, List, Tuple

import torch
from typing_extensions import override

from megatron.core import mpu, parallel_state
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.data_schedule_utils import (
    _get_global_seqlens_and_ids,
    get_thd_partitioned_indices,
)
from megatron.core.packed_seq_params import PackedSeqParams

try:
    from megatron.bridge.utils.context_parallel_utils import qwen3vl_parallel_split
except ImportError:
    qwen3vl_parallel_split = None

from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.extended_model.base import PrepareDataForward
from gpatch_v4.extended_model.mtp_mixin import OnlineMtpSftMixin
from gpatch_v4.utils import (
    get_tensor_on_this_cp_rank,
    pad_or_truncate_last_dim,
    qwen2vl_pad_and_split,
)
from gpatch_v4.utils.dynamic_cp_utils import (
    _round_up,
    sft_dyn_cp_schedule_default,
    sft_dyn_cp_schedule_smart_padding,
)


class Qwen3VLPrepareDataForward(OnlineMtpSftMixin, PrepareDataForward):
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
        if self.config.policy.model_arch in [
            MODEL_ARCH.QWEN3_5, MODEL_ARCH.QWEN3_5_MOE, MODEL_ARCH.QWEN3_5_WEMM,
            MODEL_ARCH.QWEN3_5_MOE_WEMM
        ]:
            cp_size = mpu.get_tensor_and_context_parallel_world_size()
        else:
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
        tokens_l = []
        position_ids_l = []
        image_input_mask_l = []
        vision_grid_thw_l = []
        vision_data_l = []
        input_features_l = []
        feature_attention_mask_l = []
        audio_feature_l = []
        non_blocking = True
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seqlen
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, pad_token_id))
            assert batch["position_ids"].shape[-1] >= seqlen, "小于 seqlen 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seqlen, 0))
            image_input_mask_l.append(
                pad_or_truncate_last_dim(batch["image_input_mask"], seqlen, 0)
            )

            if "vision_data" in batch and batch["vision_data"] is not None:
                vision_grid_thw_l.append(batch["vision_grid_thw"])
                vision_data_l.append(batch["vision_data"])

            if "input_features" in batch and batch["input_features"] is not None:
                input_features_l.append(batch["input_features"].cuda(non_blocking=non_blocking))
                feature_attention_mask_l.append(
                    batch["feature_attention_mask"].cuda(non_blocking=non_blocking)
                )

            if "audio_feature" in batch and batch["audio_feature"] is not None:
                audio_feature = batch["audio_feature"]
                audio_feature_l.append(audio_feature)

        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)

        # 这里有一个 padding iamge，对齐 DataCollatorForQwen2Vl
        image_input_mask = None
        cp_img_num, images_padded, vision_data, vision_grid_thw = None, None, None, None
        if len(vision_data_l) > 0:
            cp_img_num, images_padded, vision_data, vision_grid_thw = self._padding_images(
                vision_data_l, vision_grid_thw_l
            )
            image_input_mask = torch.cat(image_input_mask_l, dim=0).cuda(non_blocking=non_blocking)
            vision_data = vision_data.cuda(non_blocking=non_blocking)
            vision_grid_thw = vision_grid_thw.cuda(non_blocking=non_blocking)

        input_features = None
        feature_attention_mask = None
        # audio 的数据不能放到一起处理，不同音频之后可能使用了
        # 同一个 attn。输出的 shape 也不对
        # shape 可以参考 megatron_datasets/qwenvl_dataset_map.py get_audio_token_cnt
        if len(input_features_l) > 0:
            input_features = input_features_l
            feature_attention_mask = feature_attention_mask_l

        audio_feature = None
        if len(audio_feature_l) > 0:
            audio_feature = torch.cat(audio_feature_l, dim=0).cuda(non_blocking=non_blocking)

        fwd_kwargs = dict(
            input_ids=tokens,
            target=tokens.detach().clone(),
            position_ids=position_ids,
            pixel_values=vision_data,
            image_grid_thw=vision_grid_thw,
            image_input_mask=image_input_mask,
            images_padded=images_padded,
            cp_img_num=cp_img_num,
        )
        if input_features is not None:
            fwd_kwargs["input_features"] = input_features
            fwd_kwargs["feature_attention_mask"] = feature_attention_mask

        if audio_feature is not None:
            fwd_kwargs["audio_feature"] = audio_feature

        return fwd_kwargs

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
        assert not ppo_pack_seq, f"not support ppo packed seq now"

        tokens_l = []
        position_ids_l = []
        image_input_mask_l = []
        advantages_l = []
        mask_l = []
        logprobs_l = []
        ref_logprobs_l = []
        rollout_logprobs_l = []

        vision_grid_thw_l = []
        vision_data_l = []
        input_features_l = []
        feature_attention_mask_l = []
        audio_feature_l = []
        non_blocking = True
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seqlen
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, pad_token_id))
            assert batch["position_ids"].shape[-1] >= seqlen, "小于 seqlen 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seqlen, 0))
            image_input_mask_l.append(
                pad_or_truncate_last_dim(batch["image_input_mask"], seqlen, 0)
            )

            advantages_l.append(pad_or_truncate_last_dim(batch["advantages"], seqlen - 1, 0))
            mask_l.append(pad_or_truncate_last_dim(batch["mask"], seqlen - 1, 0))
            logprobs_l.append(pad_or_truncate_last_dim(batch["logprobs"], seqlen - 1, 0))
            ref_logprobs_l.append(pad_or_truncate_last_dim(batch["ref_logprobs"], seqlen - 1, 0))

            rollout_logprobs_l.append(
                pad_or_truncate_last_dim(batch["rollout_log_probs"], seqlen - 1, 0)
            )

            if "vision_data" in batch and batch["vision_data"] is not None:
                vision_grid_thw_l.append(batch["vision_grid_thw"])
                vision_data_l.append(batch["vision_data"])

            if "input_features" in batch and batch["input_features"] is not None:
                input_features_l.append(batch["input_features"].cuda(non_blocking=non_blocking))
                feature_attention_mask_l.append(
                    batch["feature_attention_mask"].cuda(non_blocking=non_blocking)
                )

            if "audio_feature" in batch and batch["audio_feature"] is not None:
                audio_feature = batch["audio_feature"]
                audio_feature_l.append(audio_feature)

        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)

        advantages = torch.stack(advantages_l)
        mask = torch.stack(mask_l)
        logprobs = torch.stack(logprobs_l)
        ref_logprobs = torch.stack(ref_logprobs_l)
        rollout_log_probs = torch.stack(rollout_logprobs_l)

        mtp_labels, mtp_loss_mask = self._build_online_mtp_labels(
            tokens, mask, do_cp_split=not ppo_pack_seq
        )

        sample_mask = None
        if "sample_mask" in batches[0]:
            sample_mask_l = [_b["sample_mask"] for _b in batches]
            sample_mask = torch.stack(sample_mask_l).cuda()

        global_retention_ratio = None
        if "global_retention_ratio" in batches[0]:
            global_retention_ratio = batches[0]["global_retention_ratio"].cuda()

        entropy_aux_figures = None
        if "entropy_aux_figures" in batches[0]:
            entropy_aux_figures = batches[0]["entropy_aux_figures"].cuda()

        # 这里有一个 padding iamge，对齐 DataCollatorForQwen2Vl
        image_input_mask = None
        cp_img_num, images_padded, vision_data, vision_grid_thw = None, None, None, None
        if len(vision_data_l) > 0:
            cp_img_num, images_padded, vision_data, vision_grid_thw = self._padding_images(
                vision_data_l, vision_grid_thw_l
            )
            image_input_mask = torch.cat(image_input_mask_l, dim=0).cuda(non_blocking=non_blocking)
            vision_data = vision_data.cuda(non_blocking=non_blocking)
            vision_grid_thw = vision_grid_thw.cuda(non_blocking=non_blocking)

        input_features = None
        feature_attention_mask = None
        # audio 的数据不能放到一起处理，不同音频之后可能使用了
        # 同一个 attn。输出的 shape 也不对
        # shape 可以参考 megatron_datasets/qwenvl_dataset_map.py get_audio_token_cnt
        if len(input_features_l) > 0:
            input_features = input_features_l
            feature_attention_mask = feature_attention_mask_l

        audio_feature = None
        if len(audio_feature_l) > 0:
            audio_feature = torch.cat(audio_feature_l, dim=0).cuda(non_blocking=non_blocking)

        batch = {
            "input_ids": tokens,
            "position_ids": position_ids,
            "pixel_values": vision_data,
            "image_grid_thw": vision_grid_thw,
            "image_input_mask": image_input_mask,
            "images_padded": images_padded,
            "cp_img_num": cp_img_num,
            "advantages": advantages,
            "prev_log_probs": logprobs,
            "mask": mask,
            "ref_log_probs": ref_logprobs,
            "rollout_log_probs": rollout_log_probs,
            'target': tokens.detach().clone(),
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "mtp_labels": mtp_labels,
            "mtp_loss_mask": mtp_loss_mask,
            "sample_mask": sample_mask,
            "global_retention_ratio": global_retention_ratio,
            "entropy_aux_figures": entropy_aux_figures,
            "audio_feature": audio_feature,
        }
        if mpu.is_pipeline_last_stage():
            keys_to_cuda = [
                "mask", "prev_log_probs", "ref_log_probs", "advantages", "rollout_log_probs"
            ]
            if batch["sample_mask"] is not None:
                keys_to_cuda.append("sample_mask")
            if batch["global_retention_ratio"] is not None:
                keys_to_cuda.append("global_retention_ratio")
            if batch["entropy_aux_figures"] is not None:
                keys_to_cuda.append("entropy_aux_figures")
            for k in keys_to_cuda:
                batch[k] = batch[k].cuda(non_blocking=non_blocking)

        fwd_kwargs = dict(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            image_input_mask=batch["image_input_mask"],
            images_padded=batch["images_padded"],
            cp_img_num=batch["cp_img_num"],
        )
        if batch["input_features"] is not None:
            fwd_kwargs["input_features"] = batch["input_features"]
            fwd_kwargs["feature_attention_mask"] = batch["feature_attention_mask"]
        self._maybe_set_mtp_sft_fwd_kwargs(fwd_kwargs, batch["mtp_labels"], batch["mtp_loss_mask"])

        if audio_feature is not None:
            fwd_kwargs["audio_feature"] = audio_feature

        return batch, fwd_kwargs

    def _prepare_tokens_and_labels(
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
            tokens = tokens[:-1]
            labels = labels[1:]
        else:
            tokens = tokens[:-1]
            labels = labels[1:]
            tokens = tokens[-seq_len:]
            labels = labels[-seq_len:]
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
        square_averaging_weight_list = []
        input_features_l = []
        feature_attention_mask_l = []
        video_second_per_grid_l = []
        audio_feature_l = []
        meta_info_l = []
        non_blocking = True
        for batch in batches:
            attention_mask = pad_or_truncate_last_dim(batch["attention_mask"], seq_len, 0)
            tokens, labels = self._prepare_tokens_and_labels(
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
            image_input_mask = pad_or_truncate_last_dim(batch["image_input_mask"], seq_len, 0)
            image_input_mask_l.append(image_input_mask)

            if "input_features" in batch and batch["input_features"] is not None:
                input_features_l.append(batch["input_features"].cuda(non_blocking=non_blocking))
                feature_attention_mask_l.append(
                    batch["feature_attention_mask"].cuda(non_blocking=non_blocking)
                )
            if "video_second_per_grid" in batch and batch["video_second_per_grid"] is not None:
                video_second_per_grid_l.append(batch["video_second_per_grid"])

            if "audio_feature" in batch and batch["audio_feature"] is not None:
                audio_feature = batch["audio_feature"]
                audio_feature_l.append(audio_feature)

            if "square_averaging_weight" in batch:
                square_averaging_weight_list.append(batch["square_averaging_weight"])

            if "meta_info" in batch:
                meta_info_l.append(batch["meta_info"])

        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        # tokens = torch.cat(tokens_l, dim=0).cuda(non_blocking=non_blocking)
        labels = torch.stack(labels_l).view(len(labels_l), -1).cuda(non_blocking=non_blocking)
        loss_mask = torch.stack(loss_mask_l).view(len(loss_mask_l),
                                                  -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)
        attention_mask = torch.stack(attention_mask_l).cuda(non_blocking=non_blocking)
        square_averaging_weights = None
        if len(square_averaging_weight_list) > 0:
            assert len(square_averaging_weight_list) == len(tokens_l)
            square_averaging_weights = torch.stack(square_averaging_weight_list).view(
                len(square_averaging_weight_list), -1
            ).cuda(non_blocking=True)

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
        # audio 的数据不能放到一起处理，不同音频之后可能使用了
        # 同一个 attn。输出的 shape 也不对
        # shape 可以参考 megatron_datasets/qwenvl_dataset_map.py get_audio_token_cnt
        if len(input_features_l) > 0:
            input_features = input_features_l
            feature_attention_mask = feature_attention_mask_l
        if len(video_second_per_grid_l) > 0:
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
            "square_averaging_weights": square_averaging_weights,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "video_second_per_grid": video_second_per_grid,
            "audio_feature": audio_feature,
            "meta_info": meta_info_l if len(meta_info_l) > 0 else None,
        }

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
        )
        if 'only_return_last_hidden_state' in batches[0] and \
            batches[0]['only_return_last_hidden_state']:
            fwd_kwargs["only_return_last_hidden_state"] = True

        if batch["input_features"] is not None:
            fwd_kwargs["input_features"] = batch["input_features"]
            fwd_kwargs["feature_attention_mask"] = batch["feature_attention_mask"]
        if batch["video_second_per_grid"] is not None:
            fwd_kwargs["video_second_per_grid"] = batch["video_second_per_grid"]

        if audio_feature is not None:
            fwd_kwargs["audio_feature"] = audio_feature

        # labels/loss_mask are already CP-split above; let the model compute the
        # MTP loss from them when online_mtp_sft is enabled.
        self._maybe_set_mtp_sft_fwd_kwargs(fwd_kwargs, batch["labels"], batch["loss_mask"])

        return batch, fwd_kwargs

    @override
    def sft_reroute_data_for_dynamic_cp(
        self,
        gbs_batches: List[Dict[str, Any]],
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float]:
        dp_group = mpu.get_data_parallel_group()
        tp_group = mpu.get_tensor_model_parallel_group()
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)

        dp_cp_size = dp_cp_group.size()
        tp_size = tp_group.size()

        dp_size = dp_group.size()
        cp_size = dp_cp_group.size() // dp_size
        assert cp_size == mpu.get_context_parallel_world_size()

        dist_config = self.config.policy.dist_config
        scheduler_type = dist_config.dynamic_cp_scheduler_type
        if scheduler_type == "smart_padding":
            dp_cp_pad = 2 * cp_size
        else:
            dp_cp_pad = 2 * dp_cp_size if dp_cp_size > 1 else 1
        tp_pad = tp_size if tp_size > 1 else 1
        pad_div = dp_cp_pad * tp_pad

        # 1. 将所有的张量都 reshape 成一维，方便后续的动态 cp 调度
        dtype_map = {
            "vision_data": torch.float32,
            "vision_grid_thw": torch.int64,
        }
        vision_data_last_dim = None
        for i, batch in enumerate(gbs_batches):
            if batch.get("vision_data") is not None:
                assert batch.get("vision_grid_thw") is not None
                assert dtype_map["vision_data"] == batch["vision_data"].dtype
                assert dtype_map["vision_grid_thw"] == batch["vision_grid_thw"].dtype
                if batch["vision_data"].numel() > 0:
                    if vision_data_last_dim is None:
                        vision_data_last_dim = batch["vision_data"].shape[-1]
                    else:
                        assert vision_data_last_dim == batch["vision_data"].shape[-1], (
                            f"inconsistent vision_data last dim across samples: "
                            f"{vision_data_last_dim=} != {batch['vision_data'].shape[-1]=}"
                        )

            tokens = batch["tokens"]
            labels = batch["labels"]

            raw_len = tokens.shape[-1]
            pad_len = _round_up(raw_len, pad_div)
            tokens, labels = self._prepare_tokens_and_labels(
                tokens,
                batch["labels"],
                pad_len,
                pad_token_id,
                pad_with_random_token,
            )

            gbs_batches[i] = dict(
                tokens=tokens,
                labels=labels,
                loss_mask=(labels != -100).to(torch.float32),
                original_seq_len=torch.tensor([raw_len], dtype=torch.int32),
                padded_seq_len=torch.tensor([tokens.shape[-1]], dtype=torch.int32),
                position_ids=pad_or_truncate_last_dim(batch["position_ids"], pad_len,
                                                      0).permute(1, 2, 0).reshape(-1).contiguous(),
                image_input_mask=pad_or_truncate_last_dim(batch["image_input_mask"], pad_len,
                                                          0).reshape(-1),
                vision_data=batch["vision_data"].reshape(-1) if "vision_data" in batch else None,
                vision_grid_thw=batch["vision_grid_thw"].reshape(-1)
                if "vision_grid_thw" in batch else None,
            )

        # 2. 根据调度器类型执行不同的调度和 packing 策略
        dev = torch.cuda.current_device()
        packed_keys = ["tokens", "labels", "loss_mask", "image_input_mask", "position_ids"]
        cat_keys = ["vision_data", "vision_grid_thw"]

        if scheduler_type == "smart_padding":
            assert len(gbs_batches) % cp_size == 0, (
                f"gbs/dp_size ({len(gbs_batches)}) must be divisible by config_cp_size ({cp_size}). "
                f"Adjust gbs or context_parallel_size."
            )
            new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum = (
                sft_dyn_cp_schedule_smart_padding(
                    gbs_batches,
                    dp_group,
                    cp_size,
                    dist_config,
                    dev,
                    packed_keys,
                    cat_keys,
                )
            )
        else:
            assert scheduler_type == "default"
            global_id_seqlens_keys = [
                "tokens",
                "labels",
                "loss_mask",
                "image_input_mask",
                "original_seq_len",
                "padded_seq_len",
                "vision_data",
                "vision_grid_thw",
                "position_ids",
            ]
            new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum = (
                sft_dyn_cp_schedule_default(
                    gbs_batches,
                    dp_group,
                    tp_group,
                    dp_cp_group,
                    cp_size,
                    dp_size,
                    dist_config,
                    dev,
                    packed_keys,
                    cat_keys,
                    global_id_seqlens_keys,
                    dtype_map,
                )
            )

        # 3. 恢复 vision 张量的原始形状
        for sample in new_samples:
            for k in sample.keys():
                if sample[k] is None:
                    continue
                if k in ["vision_data"]:
                    if sample[k].numel() == 0:
                        continue
                    # dynamic-cp 交换数据后，不是所有 rank 的 vision_data_last_dim 都有值
                    if vision_data_last_dim is None:
                        vision_grid_thw = sample["vision_grid_thw"].reshape(-1, 3)
                        vision_rows = vision_grid_thw.prod(dim=1).sum().item()
                        assert vision_rows > 0, f"{vision_grid_thw=}"
                        assert sample[k].numel() % vision_rows == 0, (
                            f"invalid vision_data/grid_thw shape: {sample[k].shape=}, "
                            f"{vision_grid_thw=}"
                        )
                        vision_data_last_dim = sample[k].numel() // vision_rows
                    sample[k] = sample[k].reshape(-1, vision_data_last_dim)
                elif k in ["vision_grid_thw"]:
                    sample[k] = sample[k].reshape(-1, 3)

        return new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum

    @override
    def sft_train_with_dynamic_cp(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert len(batches) == 1, "sft_train_with_dynamic_cp only supports one batch"
        batch = batches[0]
        assert "local_cp_size" in batch

        # 0. Lazy transfer: move tensors to GPU on demand (they may reside on
        #    CPU to reduce peak memory when GBS is large).
        dev = torch.cuda.current_device()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and not v.is_cuda:
                batch[k] = v.to(dev, non_blocking=True)

        # 1. get cp_group
        lcp = batch.get("local_cp_size")
        if lcp is not None:
            lcp_val = lcp.item() if isinstance(lcp, torch.Tensor) else int(lcp)
            cp_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=lcp_val)
        else:
            cp_group = parallel_state.get_context_parallel_group()

        # 2. do cp slicing (THD load balancing)
        cp_size = cp_group.size()
        total_tokens = batch["tokens"].size(0)
        if cp_size > 1:
            cp_rank = cp_group.rank()
            # Pass cu_seqlens_padded as cu_seqlens to work around a TE bug
            # in thd_get_partitioned_indices.
            cu_seqlens_for_partition = batch["cu_seqlens_padded"]
            index = get_thd_partitioned_indices(
                cu_seqlens_for_partition, total_tokens, cp_size, cp_rank
            )
            # tokens 需要全量的输入, sft_train 函数也是这样
            for key in ["labels", "loss_mask"]:
                assert key in batch
                batch[key] = batch[key].index_select(0, index)
            # position_ids 在 sft_reroute_data_for_dynamic_cp 里被 reshape 成扁平的 [3*pad_len]
            # (mrope 三个维度按维度连续排列, 见 sft_reroute_data_for_dynamic_cp), 多 sample pack
            # 之后是 [3*total_tokens]. index 的值域只在 [0, total_tokens),
            # 直接 index_select(0, index) 只会从第一个 mrope 维度取数据, 必须
            # 先 reshape 成 [3, total_tokens] 再按 token 轴切分.
            pos_ids = batch["position_ids"].view(1, total_tokens, 3)
            batch["position_ids"] = pos_ids.index_select(1, index).view(-1).contiguous()

        # 3. align tp
        tp_size = parallel_state.get_tensor_model_parallel_group().size()
        assert batch["tokens"].size(0) % tp_size == 0, (
            f"post-CP tokens ({batch['tokens'].size(0)}) not aligned to tp_size={tp_size}"
        )

        # 4. change view
        cp_tokens_val = torch.tensor(batch["labels"].size(0), dtype=torch.int32)
        batch["tokens"] = batch["tokens"].view(1, total_tokens).contiguous()
        batch["image_input_mask"] = batch["image_input_mask"].view(1, total_tokens).contiguous()
        batch["labels"] = batch["labels"].view(1, cp_tokens_val).contiguous()
        batch["loss_mask"] = batch["loss_mask"].view(1, cp_tokens_val).contiguous()
        batch["position_ids"] = batch["position_ids"].view(1, cp_tokens_val,
                                                           3).permute(2, 0, 1).contiguous()
        if "square_averaging_weight" in batch:
            batch["square_averaging_weights"] = batch.pop("square_averaging_weight").view(1, -1).to(
                torch.cuda.current_device()
            )

        cu_seqlens_padded = batch["cu_seqlens_padded"]
        max_seqlen = batch["max_seqlen"].item()
        local_cp_size = batch["local_cp_size"].item()
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            local_cp_size=local_cp_size,
            cp_group=cp_group,
        )

        fwd_kwargs = dict(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=None,
            labels=None,
            pixel_values=batch["vision_data"],
            image_grid_thw=batch["vision_grid_thw"],
            image_input_mask=batch["image_input_mask"],
            images_padded=None,
            cp_img_num=None,
            packed_seq_params=packed_seq_params,
        )

        # Store cp_group in batch so the loss function can use it for CP reduction.
        batch["cp_group"] = cp_group

        return batch, fwd_kwargs

    @override
    def opd_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert not ppo_pack_seq, f"not support ppo packed seq now"

        tokens_l = []
        position_ids_l = []
        image_input_mask_l = []
        advantages_l = []
        mask_l = []
        logprobs_l = []
        ref_logprobs_l = []
        teacher_logprobs_l = []
        rollout_logprobs_l = []
        has_ref_logprobs = "ref_logprobs" in batches[0]

        vision_grid_thw_l = []
        vision_data_l = []
        audio_feature_l = []
        teacher_names = list(self.config.teachers.keys())
        is_single_teacher = len(teacher_names) == 1
        routing_field = getattr(self.config.ppo, "g_opd_teacher_routing_field", "teacher_type")
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seqlen
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, pad_token_id))
            assert batch["position_ids"].shape[-1] >= seqlen, "小于 seqlen 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seqlen, 0))
            image_input_mask_l.append(
                pad_or_truncate_last_dim(batch["image_input_mask"], seqlen, 0)
            )

            advantages_l.append(pad_or_truncate_last_dim(batch["advantages"], seqlen - 1, 0))
            mask_l.append(pad_or_truncate_last_dim(batch["mask"], seqlen - 1, 0))
            logprobs_l.append(pad_or_truncate_last_dim(batch["logprobs"], seqlen - 1, 0))

            if is_single_teacher:
                teacher_name = teacher_names[0]
            else:
                assert routing_field in batch, (
                    f"multi-teacher opd requires per-sample routing field "
                    f"'{routing_field}' in batch, got keys={list(batch.keys())}"
                )
                teacher_name = batch[routing_field]
                assert teacher_name in teacher_names, (
                    f"sample routed to teacher '{teacher_name}' which is not in "
                    f"configured teachers {teacher_names}"
                )
            teacher_logprobs_l.append(
                pad_or_truncate_last_dim(batch[f'teacher_logprobs_{teacher_name}'], seqlen - 1, 0)
            )
            if has_ref_logprobs:
                ref_logprobs_l.append(
                    pad_or_truncate_last_dim(batch['ref_logprobs'], seqlen - 1, 0)
                )
            rollout_logprobs_l.append(
                pad_or_truncate_last_dim(batch["rollout_log_probs"], seqlen - 1, 0)
            )

            if "vision_data" in batch and batch["vision_data"] is not None:
                vision_grid_thw_l.append(batch["vision_grid_thw"])
                vision_data_l.append(batch["vision_data"])

            if "audio_feature" in batch and batch["audio_feature"] is not None:
                audio_feature = batch["audio_feature"]
                audio_feature_l.append(audio_feature)

        non_blocking = True
        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)

        advantages = torch.stack(advantages_l)
        mask = torch.stack(mask_l)
        logprobs = torch.stack(logprobs_l)
        teacher_logprobs = torch.stack(teacher_logprobs_l)
        ref_logprobs = torch.stack(ref_logprobs_l) if has_ref_logprobs else teacher_logprobs
        rollout_log_probs = torch.stack(rollout_logprobs_l)

        mtp_labels, mtp_loss_mask = self._build_online_mtp_labels(
            tokens, mask, do_cp_split=not ppo_pack_seq
        )

        # 这里有一个 padding iamge，对齐 DataCollatorForQwen2Vl
        image_input_mask = None
        cp_img_num, images_padded, vision_data, vision_grid_thw = None, None, None, None
        if len(vision_data_l) > 0:
            cp_img_num, images_padded, vision_data, vision_grid_thw = self._padding_images(
                vision_data_l, vision_grid_thw_l
            )
            image_input_mask = torch.cat(image_input_mask_l, dim=0).cuda(non_blocking=non_blocking)
            vision_data = vision_data.cuda(non_blocking=non_blocking)
            vision_grid_thw = vision_grid_thw.cuda(non_blocking=non_blocking)

        audio_feature = None
        if len(audio_feature_l) > 0:
            audio_feature = torch.cat(audio_feature_l, dim=0).cuda(non_blocking=non_blocking)

        batch = {
            "input_ids": tokens,
            "position_ids": position_ids,
            "pixel_values": vision_data,
            "image_grid_thw": vision_grid_thw,
            "image_input_mask": image_input_mask,
            "images_padded": images_padded,
            "cp_img_num": cp_img_num,
            "advantages": advantages,
            "prev_log_probs": logprobs,
            "mask": mask,
            "ref_log_probs": ref_logprobs,
            "teacher_log_probs": teacher_logprobs,
            "rollout_log_probs": rollout_log_probs,
            'target': tokens.detach().clone(),
            "mtp_labels": mtp_labels,
            "mtp_loss_mask": mtp_loss_mask,
            "audio_feature": audio_feature,
        }
        if mpu.is_pipeline_last_stage():
            for k in [
                "mask", "prev_log_probs", "ref_log_probs", "teacher_log_probs", "advantages",
                "rollout_log_probs"
            ]:
                batch[k] = batch[k].cuda(non_blocking=non_blocking)

        fwd_kwargs = dict(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            image_input_mask=batch["image_input_mask"],
            images_padded=batch["images_padded"],
            cp_img_num=batch["cp_img_num"],
        )
        self._maybe_set_mtp_sft_fwd_kwargs(fwd_kwargs, batch["mtp_labels"], batch["mtp_loss_mask"])

        if audio_feature is not None:
            fwd_kwargs["audio_feature"] = audio_feature

        return batch, fwd_kwargs


class Qwen3VLOffPoilicyDistillPrepareDataForward(Qwen3VLPrepareDataForward):
    @override
    def sft_train(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ):
        assert "input_teacher_logits" in kwargs, f"{kwargs=}"
        input_teacher_logits = kwargs["input_teacher_logits"]
        seq_len_shard_by_cp = seq_len // mpu.get_context_parallel_world_size()

        tokens_l = []
        labels_l = []
        loss_mask_l = []
        position_ids_l = []
        image_input_mask_l = []
        vision_grid_thw_l = []
        vision_data_l = []
        teacher_logits_list = []
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seq_len
            tokens, labels = self._prepare_tokens_and_labels(
                batch["tokens"],
                batch["labels"],
                seq_len,
                pad_token_id,
                pad_with_random_token,
            )

            tokens_l.append(tokens)
            labels_l.append(labels)
            loss_mask_l.append(pad_or_truncate_last_dim(batch["loss_mask"], seq_len, 0.0))

            assert batch["position_ids"].shape[-1] >= seq_len, "小于 seq_len 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seq_len, 0))
            image_input_mask_l.append(
                pad_or_truncate_last_dim(batch["image_input_mask"], seq_len, 0)
            )
            if "vision_data" in batch and batch["vision_data"] is not None:
                vision_grid_thw_l.append(batch["vision_grid_thw"])
                vision_data_l.append(batch["vision_data"])

            teacher_logits = None
            if input_teacher_logits and mpu.is_pipeline_last_stage():
                teacher_logits = batch["teacher_logits"]
                assert seq_len % mpu.get_context_parallel_world_size(
                ) == 0, f"{seq_len=} {mpu.get_context_parallel_world_size()=}"
                assert teacher_logits.ndim == 2, f"teacher_logits.ndim={teacher_logits.ndim}"
                assert seq_len_shard_by_cp == teacher_logits.shape[
                    0], f"{seq_len_shard_by_cp=} != {teacher_logits.shape[0]}"
                # teacher 计算 logits 的时候就已经 pad 过了，而且是按照相同 tp 和 cp 拆分，所以不需要额外做 pad 或者cp 拆分这些
                # teacher_logits 本身是 pin_memory 的，所以直接 non_blocking = True 转 gpu 上速度最快
                teacher_logits_list.append(teacher_logits.cuda(non_blocking=True))

        non_blocking = True
        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        labels = torch.stack(labels_l).view(len(labels_l), -1).cuda(non_blocking=non_blocking)
        loss_mask = torch.stack(loss_mask_l).view(len(loss_mask_l),
                                                  -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)
        batch_size = tokens.shape[0]
        teacher_logits = None
        if input_teacher_logits and mpu.is_pipeline_last_stage():
            teacher_logits = torch.stack(teacher_logits_list)
            teacher_logits = teacher_logits.view(batch_size, seq_len_shard_by_cp, -1)
            assert teacher_logits.ndim == 3, f"{teacher_logits.ndim=}"

        # 这里有一个 padding iamge，对齐 DataCollatorForQwen2Vl
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

        batch = dict(
            tokens=tokens,
            labels=labels,
            position_ids=position_ids,
            pixel_values=vision_data,
            image_grid_thw=vision_grid_thw,
            image_input_mask=image_input_mask,
            images_padded=images_padded,
            cp_img_num=cp_img_num,
            loss_mask=loss_mask,
            full_loss_mask=full_loss_mask,
        )

        fwd_kwargs = dict(
            input_ids=tokens,
            position_ids=position_ids,
            pixel_values=vision_data,
            image_grid_thw=vision_grid_thw,
            image_input_mask=image_input_mask,
            images_padded=images_padded,
            cp_img_num=cp_img_num,
            labels=None,
        )
        # labels/loss_mask are already CP-split above; let the model compute the
        # MTP loss from them when online_mtp_sft is enabled.
        self._maybe_set_mtp_sft_fwd_kwargs(fwd_kwargs, batch["labels"], batch["loss_mask"])

        if input_teacher_logits:
            if mpu.is_pipeline_last_stage():
                assert teacher_logits.ndim == 3, f"teacher_logits.ndim={teacher_logits.ndim}"
                batch["teacher_logits"] = teacher_logits
            else:
                batch["teacher_logits"] = None
        return batch, fwd_kwargs


class Qwen3VLDpoPrepareDataForward(Qwen3VLPrepareDataForward):
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
        ref_logps_l = []
        for batch in batches:
            assert "ref_logprobs" in batch, f"{batch=}"
            ref_logps = pad_or_truncate_last_dim(batch["ref_logprobs"], seq_len - 1, 0)
            ref_logps_l.append(ref_logps)

        non_blocking = True
        ref_logprobs = torch.stack(ref_logps_l).view(len(ref_logps_l),
                                                     -1).cuda(non_blocking=non_blocking)
        batch, fwd_kwargs = super(Qwen3VLDpoPrepareDataForward, self).sft_train(
            batches, seq_len, pad_token_id, comput_attn_mask, pad_with_random_token, **kwargs
        )

        assert "ref_logprobs" not in batch, f"{batch=}"
        batch["ref_logprobs"] = ref_logprobs
        return batch, fwd_kwargs
