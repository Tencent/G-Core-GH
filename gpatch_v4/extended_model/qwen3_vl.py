import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from typing_extensions import override

from megatron.core import mpu, parallel_state
from megatron.core.datasets.data_schedule_utils import get_thd_partitioned_indices
from megatron.core.packed_seq_params import PackedSeqParams

try:
    from megatron.bridge.utils.context_parallel_utils import qwen3vl_parallel_split
except ImportError:
    qwen3vl_parallel_split = None

from megatron_datasets.utils import build_forbidden_token_ids
try:
    from megatron.lite.model.qwen3_5.lite.vision import Qwen35VisionInputs
    from megatron.lite.runtime.contracts import LossContext, PackedBatch

    from gpatch_v4.utils.mlite_batch_bridge import build_mlite_finetune_source_batch
except ImportError:
    Qwen35VisionInputs = None
    LossContext = None
    PackedBatch = None
    build_mlite_finetune_source_batch = None

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.extended_model.base import PrepareDataForward
from gpatch_v4.extended_model.mtp_mixin import OnlineMtpSftMixin
from gpatch_v4.utils import (
    get_tensor_on_this_cp_rank,
    metadata_scalar,
    pad_3d_seq_dim,
    pad_or_truncate_last_dim,
    qwen2vl_pad_and_split,
)
from gpatch_v4.utils.dynamic_cp_utils import (
    _flatten_routed_experts_for_dynamic_cp,
    _restore_packed_routed_experts,
    _round_up,
    compute_dyn_cp_response_span,
    dyn_cp_schedule_default,
    dyn_cp_schedule_smart_padding,
)


class Qwen3VLPrepareDataForward(OnlineMtpSftMixin, PrepareDataForward):
    # Dyn-CP cat_keys for Qwen3-Omni audio. Same WeLM layout as packed SFT:
    # ``(mel, T)`` + ``audio_feature_lengths``. Flatten to 1-D for all-to-all.
    _OMNI_AUDIO_CAT_KEYS = (
        "input_features",
        "audio_feature_lengths",
    )

    @property
    def forbidden_token_ids(self):
        # Multimodal special token ids that must not be produced by random
        # padding. Walks nested Omni thinker/talker configs; None if empty.
        if not hasattr(self, "_forbidden_token_ids"):
            hf_config = getattr(self.config.policy, "hf_config", None)
            self._forbidden_token_ids = build_forbidden_token_ids(hf_config) or None
        return self._forbidden_token_ids

    def rl_train_cp_chunk_single_data(
        self,
        data: torch.Tensor,
    ):
        local_data = get_tensor_on_this_cp_rank(data, 1, key_name="target")
        return local_data

    def _is_qwen3_omni_moe(self) -> bool:
        return self.config.policy.model_arch == MODEL_ARCH.QWEN3_OMNI_MOE

    def _omni_audio_cat_keys(self) -> Tuple[str, ...]:
        if not self._is_qwen3_omni_moe():
            return ()
        return self._OMNI_AUDIO_CAT_KEYS

    def _omni_audio_dyn_cp_dtype_map(self) -> Dict[str, torch.dtype]:
        # Always register so text-only local ranks can allocate empty send
        # buffers during dyn-cp all-to-all (same pattern as WelmOmni).
        if not self._is_qwen3_omni_moe():
            return {}
        return {
            "input_features": torch.bfloat16,
            "audio_feature_lengths": torch.int64,
        }

    def _flatten_omni_audio_for_dyn_cp(
        self,
        batch: Dict[str, Any],
        dtype_map: Dict[str, torch.dtype],
        mel_bins_box: List[Optional[int]],
    ) -> Dict[str, Any]:
        """Flatten one sample's WeLM-packed Omni audio for dyn-cp ``cat_keys``."""
        if not self._is_qwen3_omni_moe():
            return {}
        has_feat = "input_features" in batch and batch["input_features"] is not None
        has_lens = ("audio_feature_lengths" in batch and batch["audio_feature_lengths"] is not None)
        if not has_feat:
            assert not has_lens, ("audio_feature_lengths set without input_features")
            return {k: None for k in self._OMNI_AUDIO_CAT_KEYS}

        assert has_lens, "input_features set without audio_feature_lengths"
        feats = batch["input_features"]
        lengths = batch["audio_feature_lengths"]
        assert feats.ndim == 2, f"expected input_features (mel, T), got {feats.shape=}"
        assert feats.dtype == dtype_map["input_features"], (
            f"input_features dtype must be {dtype_map['input_features']}, got {feats.dtype}"
        )
        assert lengths.dtype == dtype_map["audio_feature_lengths"], (
            f"audio_feature_lengths dtype must be {dtype_map['audio_feature_lengths']}, "
            f"got {lengths.dtype}"
        )
        mel = int(feats.shape[0])
        if mel_bins_box[0] is None:
            mel_bins_box[0] = mel
        else:
            assert mel_bins_box[0] == mel, (f"inconsistent mel_bins: {mel_bins_box[0]=} != {mel=}")
        return {
            "input_features": feats.transpose(0, 1).contiguous().reshape(-1),
            "audio_feature_lengths": lengths,
        }

    def _restore_omni_audio_after_dyn_cp(
        self,
        samples: List[Dict[str, Any]],
        dp_cp_group,
        mel_bins: Optional[int],
    ) -> None:
        """Restore packed Omni audio 1-D flats to ``(mel, T)`` after dyn-cp."""
        if not self._is_qwen3_omni_moe():
            return
        dev = torch.cuda.current_device()
        mel_bins_t = torch.tensor(
            [mel_bins if mel_bins is not None else 0],
            dtype=torch.int32,
            device=dev,
        )
        dist.all_reduce(mel_bins_t, op=dist.ReduceOp.MAX, group=dp_cp_group)
        mel_bins_global = int(mel_bins_t.item())
        for sample in samples:
            feats = sample["input_features"]
            if feats is None or (isinstance(feats, torch.Tensor) and feats.numel() == 0):
                sample["input_features"] = None
                sample["audio_feature_lengths"] = None
                continue
            assert mel_bins_global > 0, (
                "input_features present after dyn-cp pack but mel_bins is 0 on all ranks"
            )
            assert feats.numel() % mel_bins_global == 0, (
                f"invalid input_features shape after pack: {feats.shape=}, {mel_bins_global=}"
            )
            sample["input_features"] = (
                feats.reshape(-1, mel_bins_global).transpose(0, 1).contiguous()
            )

    def _cat_packed_omni_audio(
        self,
        batches: List[Dict[str, Any]],
        *,
        non_blocking: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Cat per-sample WeLM audio into one encoder input."""
        if not self._is_qwen3_omni_moe():
            return None, None
        feat_parts = []
        len_parts = []
        for batch in batches:
            if "input_features" not in batch or batch["input_features"] is None:
                assert (
                    "audio_feature_lengths" not in batch or batch["audio_feature_lengths"] is None
                ), "audio_feature_lengths set without input_features"
                continue
            feat = batch["input_features"]
            lengths = batch["audio_feature_lengths"]
            assert lengths is not None, "input_features set without audio_feature_lengths"
            assert torch.is_tensor(feat) and feat.ndim == 2, (
                f"expected (mel, T), got {type(feat)} {getattr(feat, 'shape', None)}"
            )
            feat_parts.append(feat.cuda(non_blocking=non_blocking))
            len_parts.append(lengths.cuda(non_blocking=non_blocking))
        if not feat_parts:
            return None, None
        return torch.cat(feat_parts, dim=1), torch.cat(len_parts, dim=0)

    def _set_omni_audio_fwd_kwargs(
        self,
        fwd_kwargs: Dict[str, Any],
        input_features: Optional[torch.Tensor],
        audio_feature_lengths: Optional[torch.Tensor],
    ) -> None:
        if input_features is None:
            assert audio_feature_lengths is None, (
                "audio_feature_lengths set without input_features"
            )
            return
        assert audio_feature_lengths is not None, ("input_features requires audio_feature_lengths")
        fwd_kwargs["input_features"] = input_features
        fwd_kwargs["audio_feature_lengths"] = audio_feature_lengths

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
        video_second_per_grid_l = []
        non_blocking = True
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seqlen
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, pad_token_id))
            assert batch["position_ids"].shape[-1] >= seqlen, "小于 seqlen 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seqlen, 0))
            if "image_input_mask" in batch and batch["image_input_mask"] is not None:
                image_input_mask_l.append(
                    pad_or_truncate_last_dim(batch["image_input_mask"], seqlen, 0)
                )

            if "vision_data" in batch and batch["vision_data"] is not None:
                vision_grid_thw_l.append(batch["vision_grid_thw"])
                vision_data_l.append(batch["vision_data"])

            if "video_second_per_grid" in batch and batch["video_second_per_grid"] is not None:
                video_second_per_grid_l.append(batch["video_second_per_grid"])

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

        input_features, audio_feature_lengths = self._cat_packed_omni_audio(batches)
        video_second_per_grid = None
        if len(video_second_per_grid_l) > 0:
            video_second_per_grid = torch.cat(video_second_per_grid_l,
                                              dim=0).cuda(non_blocking=non_blocking)

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
        self._set_omni_audio_fwd_kwargs(fwd_kwargs, input_features, audio_feature_lengths)
        if video_second_per_grid is not None:
            fwd_kwargs["video_second_per_grid"] = video_second_per_grid

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
        prev_per_token_entropies_l = []
        ref_logprobs_l = []
        rollout_logprobs_l = []

        vision_grid_thw_l = []
        vision_data_l = []
        video_second_per_grid_l = []
        non_blocking = True
        has_ref_logprobs = "ref_logprobs" in batches[0]
        has_rollout_logprobs = "rollout_log_probs" in batches[0]
        has_prev_per_token_entropies = "prev_per_token_entropies" in batches[0]
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seqlen
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, pad_token_id))
            assert batch["position_ids"].shape[-1] >= seqlen, "小于 seqlen 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seqlen, 0))
            if "image_input_mask" in batch and batch["image_input_mask"] is not None:
                image_input_mask_l.append(
                    pad_or_truncate_last_dim(batch["image_input_mask"], seqlen, 0)
                )

            advantages_l.append(pad_or_truncate_last_dim(batch["advantages"], seqlen - 1, 0))
            mask_l.append(pad_or_truncate_last_dim(batch["mask"], seqlen - 1, 0))
            logprobs_l.append(pad_or_truncate_last_dim(batch["logprobs"], seqlen - 1, 0))
            if has_prev_per_token_entropies:
                prev_per_token_entropies_l.append(
                    pad_or_truncate_last_dim(batch["prev_per_token_entropies"], seqlen - 1, 0)
                )
            if has_ref_logprobs:
                ref_logprobs_l.append(
                    pad_or_truncate_last_dim(batch["ref_logprobs"], seqlen - 1, 0)
                )

            if has_rollout_logprobs:
                rollout_logprobs_l.append(
                    pad_or_truncate_last_dim(batch["rollout_log_probs"], seqlen - 1, 0)
                )

            if "vision_data" in batch and batch["vision_data"] is not None:
                vision_grid_thw_l.append(batch["vision_grid_thw"])
                vision_data_l.append(batch["vision_data"])

            if "video_second_per_grid" in batch and batch["video_second_per_grid"] is not None:
                video_second_per_grid_l.append(batch["video_second_per_grid"])

        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)

        advantages = torch.stack(advantages_l)
        mask = torch.stack(mask_l)
        logprobs = torch.stack(logprobs_l)
        if has_prev_per_token_entropies:
            prev_per_token_entropies = torch.stack(prev_per_token_entropies_l)
        ref_logprobs = torch.stack(ref_logprobs_l) if has_ref_logprobs else None
        rollout_log_probs = torch.stack(rollout_logprobs_l) if has_rollout_logprobs else None

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

        input_features, audio_feature_lengths = self._cat_packed_omni_audio(batches)
        video_second_per_grid = None
        if len(video_second_per_grid_l) > 0:
            video_second_per_grid = torch.cat(video_second_per_grid_l,
                                              dim=0).cuda(non_blocking=non_blocking)

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
            "audio_feature_lengths": audio_feature_lengths,
            "video_second_per_grid": video_second_per_grid,
            "mtp_labels": mtp_labels,
            "mtp_loss_mask": mtp_loss_mask,
            "sample_mask": sample_mask,
            "global_retention_ratio": global_retention_ratio,
            "entropy_aux_figures": entropy_aux_figures,
        }
        if has_prev_per_token_entropies:
            batch["prev_per_token_entropy"] = prev_per_token_entropies
        if mpu.is_pipeline_last_stage():
            keys_to_cuda = ["mask", "prev_log_probs", "advantages"]
            if has_prev_per_token_entropies:
                keys_to_cuda.append("prev_per_token_entropy")
            if batch["ref_log_probs"] is not None:
                keys_to_cuda.append("ref_log_probs")
            if batch["rollout_log_probs"] is not None:
                keys_to_cuda.append("rollout_log_probs")
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
        self._set_omni_audio_fwd_kwargs(
            fwd_kwargs, batch["input_features"], batch["audio_feature_lengths"]
        )
        if batch["video_second_per_grid"] is not None:
            fwd_kwargs["video_second_per_grid"] = batch["video_second_per_grid"]
        self._maybe_set_mtp_sft_fwd_kwargs(fwd_kwargs, batch["mtp_labels"], batch["mtp_loss_mask"])

        return batch, fwd_kwargs

    @override
    def ppo_value_train(
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
        values_l = []
        returns_l = []
        mask_l = []

        vision_grid_thw_l = []
        vision_data_l = []
        non_blocking = True
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seqlen
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, pad_token_id))
            assert batch["position_ids"].shape[-1] >= seqlen, "小于 seqlen 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seqlen, 0))
            if "image_input_mask" in batch and batch["image_input_mask"] is not None:
                image_input_mask_l.append(
                    pad_or_truncate_last_dim(batch["image_input_mask"], seqlen, 0)
                )

            values_l.append(pad_or_truncate_last_dim(batch["values"], seqlen - 1, 0.0))
            returns_l.append(pad_or_truncate_last_dim(batch["returns"], seqlen - 1, 0.0))
            mask_l.append(pad_or_truncate_last_dim(batch["mask"], seqlen - 1, 0))

            if "vision_data" in batch and batch["vision_data"] is not None:
                vision_grid_thw_l.append(batch["vision_grid_thw"])
                vision_data_l.append(batch["vision_data"])

        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)

        values = torch.stack(values_l)
        returns = torch.stack(returns_l)
        mask = torch.stack(mask_l)

        image_input_mask = None
        cp_img_num, images_padded, vision_data, vision_grid_thw = None, None, None, None
        if len(vision_data_l) > 0:
            cp_img_num, images_padded, vision_data, vision_grid_thw = self._padding_images(
                vision_data_l, vision_grid_thw_l
            )
            image_input_mask = torch.cat(image_input_mask_l, dim=0).cuda(non_blocking=non_blocking)
            vision_data = vision_data.cuda(non_blocking=non_blocking)
            vision_grid_thw = vision_grid_thw.cuda(non_blocking=non_blocking)

        input_features, audio_feature_lengths = self._cat_packed_omni_audio(batches)

        batch = {
            "input_ids": tokens,
            "position_ids": position_ids,
            "pixel_values": vision_data,
            "image_grid_thw": vision_grid_thw,
            "image_input_mask": image_input_mask,
            "images_padded": images_padded,
            "cp_img_num": cp_img_num,
            "values": values,
            "returns": returns,
            "mask": mask,
            "input_features": input_features,
            "audio_feature_lengths": audio_feature_lengths,
        }
        if mpu.is_pipeline_last_stage():
            for k in ["mask", "values", "returns"]:
                batch[k] = batch[k].cuda(non_blocking=non_blocking)

        fwd_kwargs = dict(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            image_input_mask=batch["image_input_mask"],
            images_padded=batch["images_padded"],
            cp_img_num=batch["cp_img_num"],
            labels=None,
        )
        self._set_omni_audio_fwd_kwargs(
            fwd_kwargs, batch["input_features"], batch["audio_feature_lengths"]
        )

        return batch, fwd_kwargs

    def _get_default_vision_type(self, config):
        if isinstance(config, FinetuneConfig):
            return torch.float32
        return torch.bfloat16

    @override
    def prepare_loss_weights(
        self,
        loss_weights: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        if loss_weights.shape[-1] <= seq_len:
            loss_weights = pad_or_truncate_last_dim(loss_weights, seq_len + 1, 0.0)
            loss_weights = loss_weights[1:]
        else:
            loss_weights = loss_weights[1:]
            loss_weights = loss_weights[-seq_len:]
        return loss_weights

    def _prepare_tokens_and_labels(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        seq_len: int,
        pad_token_id: int,
        pad_with_random_token: bool = False,
        vocab_size: int = 0,
        forbidden_token_ids=None,
    ):
        # 先判断 labels 是否有被 shift 过
        assert tokens.shape == labels.shape, f"{tokens.shape=}, {labels.shape=}"
        assert torch.equal(
            tokens == labels, labels >= 0
        ), f"labels should not be shifted:{tokens.tolist()=} {labels.tolist()=}"
        if tokens.shape[-1] <= seq_len:
            # 多加一位是为了 shift
            tokens = pad_or_truncate_last_dim(
                tokens,
                seq_len + 1,
                pad_token_id,
                pad_with_random_token=pad_with_random_token,
                vocab_size=vocab_size,
                forbidden_token_ids=forbidden_token_ids,
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
        video_second_per_grid_l = []
        meta_info_l = []
        non_blocking = True
        loss_weights_list = []
        loss_weights = None
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
            loss_mask = torch.ones(labels.size(), dtype=torch.float, device=labels.device)
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

            if "video_second_per_grid" in batch and batch["video_second_per_grid"] is not None:
                video_second_per_grid_l.append(batch["video_second_per_grid"])

            if "square_averaging_weight" in batch:
                square_averaging_weight_list.append(batch["square_averaging_weight"])

            if "loss_weights" in batch:
                lw = batch["loss_weights"]
                lw = self.prepare_loss_weights(lw, seq_len)
                loss_weights_list.append(lw)

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

        if len(loss_weights_list) > 0:
            assert len(loss_weights_list) == len(tokens_l)
            loss_weights = torch.stack(loss_weights_list).view(len(loss_weights_list),
                                                               -1).cuda(non_blocking=non_blocking)

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
            if loss_weights is not None:
                loss_weights = get_tensor_on_this_cp_rank(loss_weights, 1, key_name="loss_weights")

        input_features, audio_feature_lengths = self._cat_packed_omni_audio(batches)
        video_second_per_grid = None
        if len(video_second_per_grid_l) > 0:
            video_second_per_grid = torch.cat(video_second_per_grid_l,
                                              dim=0).cuda(non_blocking=non_blocking)

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
            "loss_weights": loss_weights,
            "input_features": input_features,
            "audio_feature_lengths": audio_feature_lengths,
            "video_second_per_grid": video_second_per_grid,
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
        self._set_omni_audio_fwd_kwargs(
            fwd_kwargs, batch["input_features"], batch["audio_feature_lengths"]
        )
        if batch["video_second_per_grid"] is not None:
            fwd_kwargs["video_second_per_grid"] = batch["video_second_per_grid"]

        # labels/loss_mask are already CP-split above; let the model compute the
        # MTP loss from them when online_mtp_sft is enabled.
        self._maybe_set_mtp_sft_fwd_kwargs(fwd_kwargs, batch["labels"], batch["loss_mask"])

        return batch, fwd_kwargs

    @override
    def sft_to_mlite_packed(
        self,
        batch: List[Dict[str, Any]],
        *,
        num_microbatches: int,
        seq_length: int,
        device: torch.device,
        dp_size: int,
        dp_group,
    ):
        """Pack step samples into mlite PackedBatch (no pad; protocol owns label roll).

        Assumes QwenVL map/collator already bounded length: tokens length ``T`` satisfies
        ``2 <= T <= seq_length + 1``. Does not invent a second truncation policy.
        ``position_ids`` / ``image_input_mask`` may still be longer (map pad to
        ``seq_length + 1``); align them with ``tokens`` by taking the prefix ``[:T]``.
        """
        if (
            Qwen35VisionInputs is None or LossContext is None or PackedBatch is None or
            build_mlite_finetune_source_batch is None
        ):
            raise ImportError(
                "mlite packing requires megatron.lite; put "
                "mlite/experimental/lite before Megatron-LM in PYTHONPATH"
            )

        if not batch:
            raise ValueError("mlite finetune batch must not be empty")
        if num_microbatches <= 0:
            raise ValueError(f"num_microbatches must be positive, got {num_microbatches}")
        if len(batch) % num_microbatches != 0:
            raise ValueError(f"batch size {len(batch)} must be divisible by {num_microbatches=}")

        microbatch_size = len(batch) // num_microbatches
        prepared = []
        max_prediction_length = 0
        for start in range(0, len(batch), microbatch_size):
            microbatch = batch[start:start + microbatch_size]
            vision_flags = [self._mlite_sample_has_vision(sample) for sample in microbatch]
            has_any_vision = any(vision_flags)
            # Like sft_train: text rows still carry MRoPE ids when dataset provides them.
            require_mrope = has_any_vision or any(
                "position_ids" in sample and sample["position_ids"] is not None
                for sample in microbatch
            )

            token_rows = []
            packing_masks = []
            aligned_masks = []
            position_rows = []
            vision_data_rows = []
            vision_grid_rows = []
            image_mask_rows = []
            for sample, has_vision in zip(microbatch, vision_flags, strict=True):
                tokens, valid_labels = self._prepare_mlite_sequence(sample, seq_length, device)
                token_count = int(tokens.numel())
                # Align mask to next-token targets; do not shift tokens here.
                aligned_mask = torch.cat(
                    [valid_labels[1:], valid_labels.new_zeros(1)],
                    dim=0,
                )
                token_rows.append(tokens)
                packing_masks.append(valid_labels)
                aligned_masks.append(aligned_mask)
                max_prediction_length = max(max_prediction_length, token_count - 1)
                if require_mrope:
                    position_rows.append(
                        self._prepare_mlite_position_ids(sample, token_count, device)
                    )
                if has_any_vision:
                    image_mask_rows.append(
                        self._prepare_mlite_image_input_mask(
                            sample,
                            token_count=token_count,
                            require_vision_tokens=has_vision,
                            device=device,
                        )
                    )
                if has_vision:
                    vision_data, vision_grid_thw = self._prepare_mlite_vision_pixels(
                        sample,
                        device=device,
                    )
                    vision_data_rows.append(vision_data)
                    vision_grid_rows.append(vision_grid_thw)

            sequence_lengths = torch.tensor(
                [row.numel() for row in token_rows],
                dtype=torch.long,
                device=device,
            )
            flat_tokens = torch.cat(token_rows, dim=0).contiguous()
            position_ids = (torch.cat(position_rows, dim=2).contiguous() if position_rows else None)
            extras = {}
            if has_any_vision:
                spatial_merge_sizes = set()
                vision_index = 0
                for has_vision, image_input_mask in zip(
                    vision_flags,
                    image_mask_rows,
                    strict=True,
                ):
                    if not has_vision:
                        continue
                    patch_count = int(vision_grid_rows[vision_index].prod().item())
                    image_token_count = int(image_input_mask.sum().item())
                    merge_area, remainder = divmod(patch_count, image_token_count)
                    spatial_merge_size = math.isqrt(merge_area)
                    if remainder or spatial_merge_size**2 != merge_area:
                        raise ValueError(
                            "vision patch count must equal image token count times "
                            "spatial_merge_size squared"
                        )
                    spatial_merge_sizes.add(spatial_merge_size)
                    vision_index += 1
                if len(spatial_merge_sizes) != 1:
                    raise ValueError("all images in an mlite batch must use one spatial_merge_size")
                extras["vision"] = Qwen35VisionInputs(
                    pixel_values=torch.cat(vision_data_rows, dim=0).contiguous(),
                    image_grid_thw=torch.cat(vision_grid_rows, dim=0).contiguous(),
                    image_input_mask=torch.cat(image_mask_rows, dim=0).contiguous(),
                    num_images_per_sequence=torch.tensor(
                        [int(flag) for flag in vision_flags],
                        dtype=torch.long,
                        device=device,
                    ),
                    spatial_merge_size=spatial_merge_sizes.pop(),
                )
            packed_batch = PackedBatch(
                input_ids=flat_tokens,
                labels=flat_tokens,
                loss_mask=torch.cat(packing_masks, dim=0).float().contiguous(),
                seq_lens=sequence_lengths,
                position_ids=position_ids,
                extras=extras,
            )
            source_batch = build_mlite_finetune_source_batch(aligned_masks)
            prepared.append((packed_batch, source_batch))

        global_valid_tokens = sum(
            source_batch["aligned_loss_mask"].values().sum() for _packed, source_batch in prepared
        ).to(dtype=torch.float32)
        if dist.is_initialized() and dp_size > 1:
            if dp_group is None:
                raise RuntimeError("mlite DP group is required when dp_size > 1")
            dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.SUM, group=dp_group)
        if global_valid_tokens.item() <= 0:
            raise ValueError("mlite finetune batch has no valid prediction tokens")

        loss_scale = dp_size * num_microbatches / float(global_valid_tokens.item())
        runtime_batches = [
            (
                packed_batch,
                LossContext(
                    return_log_probs=True,
                    loss_scale=loss_scale,
                    source_batch=source_batch,
                ),
            ) for packed_batch, source_batch in prepared
        ]
        return runtime_batches, global_valid_tokens, max_prediction_length

    @staticmethod
    def _mlite_sample_has_vision(sample: Dict[str, Any]) -> bool:
        vision_data = sample["vision_data"] if "vision_data" in sample else None
        vision_grid_thw = sample["vision_grid_thw"] if "vision_grid_thw" in sample else None
        if (vision_data is None) != (vision_grid_thw is None):
            raise ValueError(
                "vision_data and vision_grid_thw must both be provided or both be None"
            )
        return vision_data is not None

    @staticmethod
    def _prepare_mlite_position_ids(
        sample: Dict[str, Any],
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Align collator MRoPE ``[3, 1, S]`` to token length ``T`` via prefix ``[:T]``."""
        if "position_ids" not in sample or sample["position_ids"] is None:
            raise ValueError("sample requires non-null position_ids")
        position_ids = torch.as_tensor(sample["position_ids"])
        if (
            position_ids.dim() != 3 or position_ids.shape[:2] != (3, 1) or
            position_ids.shape[-1] < token_count
        ):
            raise ValueError(
                "position_ids must have shape [3, 1, S] with "
                f"S >= {token_count}, got {tuple(position_ids.shape)}"
            )
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("position_ids must be an integer tensor")
        return position_ids[..., :token_count].to(
            device=device, dtype=torch.long, non_blocking=True
        )

    @staticmethod
    def _prepare_mlite_image_input_mask(
        sample: Dict[str, Any],
        *,
        token_count: int,
        require_vision_tokens: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Align ``image_input_mask`` to token length ``T`` via prefix ``[:T]``."""
        if "image_input_mask" in sample and sample["image_input_mask"] is not None:
            image_input_mask = torch.as_tensor(sample["image_input_mask"])
            if (
                image_input_mask.dim() != 2 or image_input_mask.shape[0] != 1 or
                image_input_mask.shape[-1] < token_count
            ):
                raise ValueError(
                    "image_input_mask must have shape [1, S] with "
                    f"S >= {token_count}, got {tuple(image_input_mask.shape)}"
                )
            if image_input_mask.dtype != torch.bool:
                raise TypeError("image_input_mask must be a bool tensor")
            image_input_mask = image_input_mask[..., :token_count]
        else:
            if require_vision_tokens:
                raise ValueError("vision sample requires non-null image_input_mask")
            image_input_mask = torch.zeros((1, token_count), dtype=torch.bool)

        if require_vision_tokens:
            if not image_input_mask.any():
                raise ValueError("vision image_input_mask must select at least one token")
        elif image_input_mask.any():
            raise ValueError("text-only sample cannot set image_input_mask without vision_data")
        return image_input_mask[0].to(device=device, non_blocking=True)

    @staticmethod
    def _prepare_mlite_vision_pixels(
        sample: Dict[str, Any],
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_data = torch.as_tensor(sample["vision_data"])
        if vision_data.dim() < 2 or vision_data.size(0) == 0:
            raise ValueError("vision_data must have shape [num_patches, ...]")

        vision_grid_thw = torch.as_tensor(sample["vision_grid_thw"])
        if vision_grid_thw.shape != (1, 3):
            raise ValueError(
                "single-image mlite samples require vision_grid_thw shape [1, 3], "
                f"got {tuple(vision_grid_thw.shape)}"
            )
        if vision_grid_thw.dtype not in (torch.int32, torch.int64):
            raise TypeError("vision_grid_thw must be an integer tensor")
        if (vision_grid_thw <= 0).any():
            raise ValueError("vision_grid_thw entries must be positive")
        patch_count = int(vision_grid_thw.prod().item())
        if vision_data.size(0) != patch_count:
            raise ValueError(
                f"vision_data has {vision_data.size(0)} patches, "
                f"but vision_grid_thw describes {patch_count}"
            )
        return (
            vision_data.to(device=device, non_blocking=True),
            vision_grid_thw.to(device=device, dtype=torch.long, non_blocking=True),
        )

    @staticmethod
    def _prepare_mlite_sequence(
        sample: Dict[str, Any],
        seq_length: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate one unshifted sample; refuse lengths above ``seq_length + 1``."""
        tokens = torch.as_tensor(sample["tokens"]).reshape(-1)
        labels = torch.as_tensor(sample["labels"]).reshape(-1)
        if tokens.shape != labels.shape:
            raise ValueError(
                f"tokens and labels must have identical shapes, "
                f"got {tokens.shape} and {labels.shape}"
            )
        if tokens.numel() < 2:
            raise ValueError("mlite finetune samples must contain at least two tokens")
        max_tokens = seq_length + 1
        if tokens.numel() > max_tokens:
            raise ValueError(
                "mlite expects QwenVL map/collator to bound tokens to "
                f"seq_length+1={max_tokens}, got {tokens.numel()}"
            )
        valid_labels = labels != -100
        if not torch.equal(labels[valid_labels], tokens[valid_labels].to(labels.device)):
            raise ValueError("every non-ignored label must equal its corresponding token")
        return (
            tokens.to(device=device, dtype=torch.long, non_blocking=True),
            valid_labels.to(device=device, non_blocking=True),
        )

    @override
    def pretrain_packed(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare one dataset-packed Qwen VLM THD microbatch.

        Packing happens in Energon, so this path only performs lazy H2D,
        tensor reshaping and static-CP forward construction.
        """
        assert not self.config.policy.dist_config.dynamic_context_parallel, (
            "pretrain_packed is incompatible with dynamic_context_parallel"
        )
        assert "cu_seqlens_padded" in batch, "packed batch missing cu_seqlens_padded"
        assert batch["tokens"].ndim == 1, (
            f"expected 1-D packed tokens, got {batch['tokens'].shape}"
        )

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        for key, value in list(batch.items()):
            if torch.is_tensor(value) and value.device != device:
                batch[key] = value.to(device, non_blocking=True)

        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        assert cp_size == 1, (
            "Qwen pretrain_packed currently supports context_parallel_size=1 "
            f"(got {cp_size})"
        )

        tokens = batch["tokens"]
        total_tokens = int(tokens.shape[0])
        tp_size = parallel_state.get_tensor_model_parallel_group().size()
        assert total_tokens % tp_size == 0, (
            f"packed tokens ({total_tokens}) not aligned to tp_size={tp_size}"
        )

        position_ids = batch["position_ids"]
        assert position_ids.ndim == 2 and position_ids.shape == (3, total_tokens), (
            f"expected packed mRoPE position_ids [3, {total_tokens}], "
            f"got {tuple(position_ids.shape)}"
        )
        assert batch["image_input_mask"].shape == (1, total_tokens), (
            f"expected packed vision mask [1, {total_tokens}], "
            f"got {tuple(batch['image_input_mask'].shape)}"
        )

        batch["tokens"] = tokens.view(1, total_tokens).contiguous()
        batch["labels"] = batch["labels"].view(1, total_tokens).contiguous()
        batch["loss_mask"] = batch["loss_mask"].view(1, total_tokens).contiguous()
        batch["position_ids"] = position_ids.view(3, 1, total_tokens).contiguous()
        if "square_averaging_weight" in batch:
            batch["square_averaging_weights"] = batch.pop("square_averaging_weight").view(1, -1)
        max_seqlen = int(batch["max_seqlen"])
        padded_seq_len = batch["padded_seq_len"]
        if torch.is_tensor(padded_seq_len):
            padded_seq_len = [int(x) for x in padded_seq_len.detach().cpu().tolist()]
        else:
            padded_seq_len = [int(x) for x in padded_seq_len]
        cu_seqlens_padded = batch["cu_seqlens_padded"]
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
        packed_seq_params._myfa_padded_lens_cache = padded_seq_len

        fwd_kwargs = dict(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=None,
            labels=None,
            pixel_values=batch.get("vision_data"),
            image_grid_thw=batch.get("vision_grid_thw"),
            image_input_mask=batch["image_input_mask"],
            images_padded=None,
            cp_img_num=None,
            packed_seq_params=packed_seq_params,
        )
        if "input_features" in batch and batch["input_features"] is not None:
            fwd_kwargs["input_features"] = batch["input_features"]
            assert batch["audio_feature_lengths"] is not None, (
                "packed input_features requires audio_feature_lengths"
            )
            fwd_kwargs["audio_feature_lengths"] = batch["audio_feature_lengths"]
        if batch.get("video_second_per_grid") is not None:
            fwd_kwargs["video_second_per_grid"] = batch["video_second_per_grid"]

        batch["cp_group"] = cp_group
        batch["max_seqlen"] = max_seqlen
        batch["padded_seq_len"] = padded_seq_len
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
        scheduler_max_seqlen = kwargs.get(
            "max_seqlen_per_dp_cp_rank", dist_config.max_seqlen_per_dp_cp_rank
        )
        if scheduler_type == "smart_padding":
            dp_cp_pad = 2 * cp_size
        else:
            dp_cp_pad = 2 * dp_cp_size if dp_cp_size > 1 else 1
        tp_pad = tp_size if tp_size > 1 else 1
        pad_div = dp_cp_pad * tp_pad

        # 1. 将所有的张量都 reshape 成一维，方便后续的动态 cp 调度
        dtype_map = {
            "vision_grid_thw": torch.int64,
            "vision_data": self._get_default_vision_type(self.config),
            **self._omni_audio_dyn_cp_dtype_map(),
        }
        vision_data_last_dim = None
        mel_bins_box: List[Optional[int]] = [None]
        for i, batch in enumerate(gbs_batches):
            if batch.get("vision_data") is not None:
                assert batch.get("vision_grid_thw") is not None
                assert batch["vision_data"].dtype == dtype_map["vision_data"], (
                    f"inconsistent vision_data dtype: {batch['vision_data'].dtype} vs {dtype_map['vision_data']}"
                )
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
            _vocab_size = kwargs.get("vocab_size", 0)
            tokens, labels = self._prepare_tokens_and_labels(
                tokens,
                batch["labels"],
                pad_len,
                pad_token_id,
                pad_with_random_token,
                vocab_size=_vocab_size,
                forbidden_token_ids=self.forbidden_token_ids,
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
                vision_data=batch["vision_data"].reshape(-1)
                if batch.get("vision_data") is not None else None,
                vision_grid_thw=batch["vision_grid_thw"].reshape(-1)
                if batch.get("vision_grid_thw") is not None else None,
                **self._flatten_omni_audio_for_dyn_cp(batch, dtype_map, mel_bins_box),
            )

        # 2. 根据调度器类型执行不同的调度和 packing 策略
        dev = torch.cuda.current_device()
        packed_keys = ["tokens", "labels", "loss_mask", "image_input_mask", "position_ids"]
        cat_keys = ["vision_data", "vision_grid_thw", *self._omni_audio_cat_keys()]

        if scheduler_type == "smart_padding":
            assert len(gbs_batches) % cp_size == 0, (
                f"gbs/dp_size ({len(gbs_batches)}) must be divisible by config_cp_size ({cp_size}). "
                f"Adjust gbs or context_parallel_size."
            )
            new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum = (
                dyn_cp_schedule_smart_padding(
                    gbs_batches,
                    dp_group,
                    cp_size,
                    dist_config,
                    dev,
                    packed_keys,
                    cat_keys,
                    max_seqlen_per_dp_cp_rank=scheduler_max_seqlen,
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
                *self._omni_audio_cat_keys(),
            ]
            new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum, _ = (
                dyn_cp_schedule_default(
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
                    max_seqlen_per_dp_cp_rank=scheduler_max_seqlen,
                    need_routing_info=False,
                )
            )

        # 3. 恢复 vision 张量的原始形状
        for sample in new_samples:
            for k in list(sample.keys()):
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
        self._restore_omni_audio_after_dyn_cp(new_samples, dp_cp_group, mel_bins_box[0])

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
        if self._is_qwen3_omni_moe():
            assert "input_features" in batch
            assert "audio_feature_lengths" in batch
            self._set_omni_audio_fwd_kwargs(
                fwd_kwargs, batch["input_features"], batch["audio_feature_lengths"]
            )

        # Store cp_group in batch so the loss function can use it for CP reduction.
        batch["cp_group"] = cp_group

        return batch, fwd_kwargs

    @override
    def rl_reroute_data_for_dynamic_cp(
        self,
        gbs_batches: List[Dict[str, Any]],
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float, Dict[str, Any]]:
        dp_group = mpu.get_data_parallel_group()
        tp_group = mpu.get_tensor_model_parallel_group()
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)

        dp_cp_size = dp_cp_group.size()
        tp_size = tp_group.size()
        dp_cp_pad = 2 * dp_cp_size if dp_cp_size > 1 else 1
        tp_pad = tp_size if tp_size > 1 else 1
        pad_div = dp_cp_pad * tp_pad

        dp_size = dp_group.size()
        cp_size = dp_cp_group.size() // dp_size

        dist_config = self.config.policy.dist_config
        scheduler_max_seqlen = kwargs.get(
            "max_seqlen_per_dp_cp_rank", dist_config.max_seqlen_per_dp_cp_rank
        )

        first = gbs_batches[0]
        global_retention_ratio = first.get("global_retention_ratio")
        entropy_aux_figures = first.get("entropy_aux_figures")

        _optional_rl_keys = [
            ("mask", "loss_mask"),
            ("advantages", "advantages"),
            ("logprobs", "prev_log_probs"),
            ("ref_logprobs", "ref_log_probs"),
            ("rollout_log_probs", "rollout_log_probs"),
            ("prev_per_token_entropy", "prev_per_token_entropy"),
            ("token_weights", "token_weights"),
        ]
        # OPD teacher logprobs: detect teacher_logprobs_* keys and unify to
        # teacher_log_probs. For multi-teacher, select per-sample based on routing.
        teacher_logprobs_keys = [k for k in first if k.startswith("teacher_logprobs_")]
        if teacher_logprobs_keys:
            _opd_teacher_names = [k[len("teacher_logprobs_"):] for k in teacher_logprobs_keys]
            _opd_single_teacher = len(_opd_teacher_names) == 1
            _opd_routing_field = "teacher_type"
        else:
            _opd_teacher_names = None

        rl_key_map = [(src, dst) for src, dst in _optional_rl_keys if src in first]
        has_sample_mask = first.get("sample_mask") is not None
        has_routed_experts = self.config.training.moe_router_replay
        routed_experts_shape = None
        if has_routed_experts:
            assert "routed_experts" in first, (
                "moe_router_replay is enabled but routed_experts is missing"
            )
            first_routed_experts = first["routed_experts"]
            assert first_routed_experts is not None, (
                "moe_router_replay is enabled but routed_experts is missing"
            )
            assert first_routed_experts.ndim == 3, (
                f"routed_experts must be [seq, layer, topk], got "
                f"{first_routed_experts.shape=}"
            )
            routed_experts_shape = tuple(first_routed_experts.shape[1:])
            dp_rank = mpu.get_data_parallel_rank()

        dtype_map = {
            "vision_grid_thw": torch.int64,
            "vision_data": self._get_default_vision_type(self.config),
            **self._omni_audio_dyn_cp_dtype_map(),
        }
        vision_data_last_dim = None
        mel_bins_box: List[Optional[int]] = [None]
        for i, batch in enumerate(gbs_batches):
            if batch.get("vision_data") is not None:
                assert batch.get("vision_grid_thw") is not None
                assert batch["vision_data"].dtype == dtype_map["vision_data"], (
                    f"inconsistent vision_data dtype: {batch['vision_data'].dtype} vs {dtype_map['vision_data']}"
                )
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
            shifted_tokens = tokens[:-1]
            shifted_labels = tokens[1:]
            actual_len = shifted_tokens.shape[-1]
            pad_len = _round_up(actual_len, pad_div)
            prompt_len = metadata_scalar(batch, "prompt_lengths")
            sequence_len = metadata_scalar(batch, "sequence_lengths")
            response_start, response_len = compute_dyn_cp_response_span(
                prompt_len=prompt_len,
                sequence_len=sequence_len,
                actual_len=actual_len,
                # Optional: logprob-only reroute runs before create_response_mask.
                response_mask=batch.get("mask"),
            )

            # tokens / image_input_mask / position_ids index absolute positions, so the
            # pre-shift (dropping the last token) needs no extra slicing here.
            sample = dict(
                tokens=pad_or_truncate_last_dim(shifted_tokens, pad_len, pad_token_id),
                labels=pad_or_truncate_last_dim(shifted_labels, pad_len, 0),
                position_ids=pad_or_truncate_last_dim(batch["position_ids"], pad_len,
                                                      0).permute(1, 2, 0).reshape(-1).contiguous(),
                image_input_mask=pad_or_truncate_last_dim(batch["image_input_mask"], pad_len,
                                                          0).reshape(-1),
                original_seq_len=torch.tensor([actual_len], dtype=torch.int32),
                padded_seq_len=torch.tensor([pad_len], dtype=torch.int32),
                dyn_cp_response_start=torch.tensor([response_start], dtype=torch.int32),
                dyn_cp_response_length=torch.tensor([response_len], dtype=torch.int32),
                vision_data=batch["vision_data"].reshape(-1)
                if batch.get("vision_data") is not None else None,
                vision_grid_thw=batch["vision_grid_thw"].reshape(-1)
                if batch.get("vision_grid_thw") is not None else None,
                **self._flatten_omni_audio_for_dyn_cp(batch, dtype_map, mel_bins_box),
            )
            for src_key, dst_key in rl_key_map:
                value = batch[src_key]
                if src_key == "token_weights" and value.numel() == 1:
                    value = value.reshape(1).expand(actual_len)
                sample[dst_key] = pad_or_truncate_last_dim(value, pad_len, 0).to(torch.float32)
            if _opd_teacher_names is not None:
                if _opd_single_teacher:
                    tname = _opd_teacher_names[0]
                else:
                    tname = batch[_opd_routing_field]
                sample["teacher_log_probs"] = pad_or_truncate_last_dim(
                    batch[f"teacher_logprobs_{tname}"], pad_len, 0
                ).to(torch.float32)
                if "ref_log_probs" not in sample:
                    sample["ref_log_probs"] = sample["teacher_log_probs"].clone()
            if has_sample_mask:
                sm = batch["sample_mask"]
                if sm.dim() == 0:
                    sm = sm.unsqueeze(0)
                sm = sm.expand(actual_len).to(torch.float32).contiguous()
                sample["sample_mask"] = pad_or_truncate_last_dim(sm, pad_len, 0)
            if has_routed_experts:
                routed_experts = batch["routed_experts"]
                assert tuple(routed_experts.shape[1:]) == routed_experts_shape, (
                    "all routed_experts tensors must share [layer, topk]: "
                    f"{tuple(routed_experts.shape[1:])=} != {routed_experts_shape}"
                )
                sample["routed_experts"] = _flatten_routed_experts_for_dynamic_cp(
                    routed_experts,
                    actual_len=actual_len,
                    padded_len=pad_len,
                    dp_rank=dp_rank,
                )
            gbs_batches[i] = sample

        # Schedule and pack with default dynamic CP scheduler.
        dev = torch.cuda.current_device()
        packed_keys = [
            "tokens",
            "labels",
            "image_input_mask",
            "position_ids",
            "dyn_cp_response_start",
            "dyn_cp_response_length",
        ]
        packed_keys.extend(dst for _, dst in rl_key_map)
        if _opd_teacher_names is not None:
            packed_keys.append("teacher_log_probs")
            if "ref_log_probs" not in packed_keys:
                packed_keys.append("ref_log_probs")
        if has_sample_mask:
            packed_keys.append("sample_mask")
        if has_routed_experts:
            packed_keys.append("routed_experts")

        cat_keys = ["vision_data", "vision_grid_thw", *self._omni_audio_cat_keys()]
        global_id_seqlens_keys = (packed_keys + cat_keys + ["original_seq_len", "padded_seq_len"])
        need_routing_info = kwargs.get("need_routing_info", True)
        new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum, routing_info = (
            dyn_cp_schedule_default(
                gbs_batches,
                dp_group,
                tp_group,
                dp_cp_group,
                cp_size,
                dp_size,
                dist_config,
                dev,
                packed_keys=packed_keys,
                cat_keys=cat_keys,
                global_id_seqlens_keys=global_id_seqlens_keys,
                dtype_map=dtype_map,
                max_seqlen_per_dp_cp_rank=scheduler_max_seqlen,
                need_routing_info=need_routing_info,
            )
        )

        # Restore vision tensors to their original shapes.
        for sample in new_samples:
            for k in list(sample.keys()):
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
            if global_retention_ratio is not None:
                sample["global_retention_ratio"] = global_retention_ratio
            if entropy_aux_figures is not None:
                sample["entropy_aux_figures"] = entropy_aux_figures
        self._restore_omni_audio_after_dyn_cp(new_samples, dp_cp_group, mel_bins_box[0])
        if has_routed_experts:
            assert routed_experts_shape is not None
            _restore_packed_routed_experts(
                new_samples,
                num_layers=routed_experts_shape[0],
                topk=routed_experts_shape[1],
            )

        return new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum, routing_info

    @override
    def grpo_train_with_dynamic_cp(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert len(batches) == 1, "grpo_train_with_dynamic_cp only supports one batch"
        batch = batches[0]
        # restore to gpu before forward.
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and not v.is_cuda:
                batch[k] = v.cuda(non_blocking=True)
        assert "local_cp_size" in batch

        lcp = batch.get("local_cp_size")
        if lcp is not None:
            lcp_val = lcp.item() if isinstance(lcp, torch.Tensor) else int(lcp)
            cp_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=lcp_val)
        else:
            cp_group = parallel_state.get_context_parallel_group()

        rollout_token_keys = [
            key for key in (
                "advantages",
                "prev_log_probs",
                "ref_log_probs",
                "rollout_log_probs",
                "teacher_log_probs",
                "prev_per_token_entropy",
                "sample_mask",
                "loss_mask",
                "token_weights",
            ) if key in batch
        ]

        total_tokens = batch["tokens"].size(0)
        cp_size = cp_group.size()
        if cp_size > 1:
            cp_rank = cp_group.rank()
            # Pass cu_seqlens_padded as cu_seqlens to work around a TE bug in
            # thd_get_partitioned_indices.
            index = get_thd_partitioned_indices(
                batch["cu_seqlens_padded"], total_tokens, cp_size, cp_rank
            )
            # Qwen3-VL consumes full tokens/image_input_mask, while only the
            # model label stream is CP-sharded. Keep rollout tensors replicated
            # in the DCP subgroup for the verl-style loss reconstruction path.
            cp_split_keys = ["labels"]
            for key in cp_split_keys:
                batch[key] = batch[key].index_select(0, index)
            # position_ids is flattened mrope [3 * total_tokens]; reshape to the
            # token axis before slicing.
            pos_ids = batch["position_ids"].view(1, total_tokens, 3)
            batch["position_ids"] = pos_ids.index_select(1, index).view(-1).contiguous()

        tp_size = parallel_state.get_tensor_model_parallel_group().size()
        assert batch["tokens"].size(0) % tp_size == 0, (
            f"post-CP tokens ({batch['tokens'].size(0)}) not aligned to tp_size={tp_size}"
        )

        cp_tokens = batch["labels"].size(0)
        batch["tokens"] = batch["tokens"].view(1, total_tokens).contiguous()
        batch["image_input_mask"] = batch["image_input_mask"].view(1, total_tokens).contiguous()
        batch["labels"] = batch["labels"].view(1, cp_tokens).contiguous()
        batch["position_ids"] = batch["position_ids"].view(1, cp_tokens, 3).permute(2, 0,
                                                                                    1).contiguous()
        for key in rollout_token_keys:
            if batch[key].numel() != total_tokens:
                raise ValueError(
                    f"dynamic-CP rollout tensor {key!r} has {batch[key].numel()} values, "
                    f"expected {total_tokens}"
                )
            batch[key] = batch[key].view(1, total_tokens).contiguous()
        if "routed_experts" in batch:
            assert batch["routed_experts"].shape[0] == total_tokens
            assert batch["routed_experts"].ndim == 3

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

        if "loss_mask" in batch:
            batch["mask"] = batch.pop("loss_mask")
        batch["target"] = batch.pop("labels")
        if batch.get("global_retention_ratio") is not None:
            batch["global_retention_ratio"] = batch["global_retention_ratio"].cuda()

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
        if self._is_qwen3_omni_moe():
            assert "input_features" in batch
            assert "audio_feature_lengths" in batch
            self._set_omni_audio_fwd_kwargs(
                fwd_kwargs, batch["input_features"], batch["audio_feature_lengths"]
            )
        return batch, fwd_kwargs

    @override
    def opd_train_with_dynamic_cp(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert "teacher_log_probs" in batches[0], (
            "OPD dynamic CP requires teacher_log_probs in packed batch"
        )
        return self.grpo_train_with_dynamic_cp(
            batches, seqlen, pad_token_id, ppo_pack_seq, pad_with_random_token, **kwargs
        )

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
        prev_topk_logprobs_l = []
        opd_topk_ids_l = []
        has_ref_logprobs = "ref_logprobs" in batches[0]
        has_topk = "prev_topk_logprobs" in batches[0] and "opd_topk_ids" in batches[0]

        vision_grid_thw_l = []
        vision_data_l = []
        video_second_per_grid_l = []
        teacher_names = list(self.config.teachers.keys())
        is_single_teacher = len(teacher_names) == 1
        routing_field = getattr(self.config.ppo, "g_opd_teacher_routing_field", "teacher_type")
        for batch in batches:
            assert batch["tokens"].shape[-1] <= seqlen
            tokens_l.append(pad_or_truncate_last_dim(batch["tokens"], seqlen, pad_token_id))
            assert batch["position_ids"].shape[-1] >= seqlen, "小于 seqlen 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seqlen, 0))
            if "image_input_mask" in batch and batch["image_input_mask"] is not None:
                image_input_mask_l.append(
                    pad_or_truncate_last_dim(batch["image_input_mask"], seqlen, 0)
                )

            adv = batch["advantages"]
            if adv.dim() == 2:
                advantages_l.append(pad_3d_seq_dim(adv, seqlen - 1, 0))
            else:
                advantages_l.append(pad_or_truncate_last_dim(adv, seqlen - 1, 0))
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
            if has_topk:
                prev_topk_logprobs_l.append(
                    pad_3d_seq_dim(batch['prev_topk_logprobs'], seqlen - 1, 0)
                )
                opd_topk_ids_l.append(pad_3d_seq_dim(batch['opd_topk_ids'], seqlen - 1, 0))

            if "vision_data" in batch and batch["vision_data"] is not None:
                vision_grid_thw_l.append(batch["vision_grid_thw"])
                vision_data_l.append(batch["vision_data"])

            if "video_second_per_grid" in batch and batch["video_second_per_grid"] is not None:
                video_second_per_grid_l.append(batch["video_second_per_grid"])

        non_blocking = True
        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)

        advantages = torch.stack(advantages_l)
        mask = torch.stack(mask_l)
        logprobs = torch.stack(logprobs_l)
        teacher_logprobs = torch.stack(teacher_logprobs_l)
        ref_logprobs = torch.stack(ref_logprobs_l) if has_ref_logprobs else teacher_logprobs
        rollout_log_probs = torch.stack(rollout_logprobs_l)
        prev_topk_logprobs = torch.stack(prev_topk_logprobs_l) if has_topk else None
        opd_topk_ids = torch.stack(opd_topk_ids_l) if has_topk else None

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

        input_features, audio_feature_lengths = self._cat_packed_omni_audio(batches)
        video_second_per_grid = None
        if len(video_second_per_grid_l) > 0:
            video_second_per_grid = torch.cat(video_second_per_grid_l,
                                              dim=0).cuda(non_blocking=non_blocking)

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
            "input_features": input_features,
            "audio_feature_lengths": audio_feature_lengths,
            "video_second_per_grid": video_second_per_grid,
        }
        if has_topk:
            batch["prev_topk_logprobs"] = prev_topk_logprobs
            batch["opd_topk_ids"] = opd_topk_ids
        if mpu.is_pipeline_last_stage():
            keys_to_cuda = [
                "mask",
                "prev_log_probs",
                "ref_log_probs",
                "teacher_log_probs",
                "advantages",
                "rollout_log_probs",
            ]
            if has_topk:
                keys_to_cuda.extend(["prev_topk_logprobs", "opd_topk_ids"])
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
        self._maybe_set_mtp_sft_fwd_kwargs(fwd_kwargs, batch["mtp_labels"], batch["mtp_loss_mask"])

        self._set_omni_audio_fwd_kwargs(
            fwd_kwargs, batch["input_features"], batch["audio_feature_lengths"]
        )
        if batch["video_second_per_grid"] is not None:
            fwd_kwargs["video_second_per_grid"] = batch["video_second_per_grid"]

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
        assert "input_teacher_hidden_states" in kwargs, f"{kwargs=}"
        input_teacher_hidden_states = kwargs["input_teacher_hidden_states"]
        seq_len_shard_by_cp = seq_len // mpu.get_context_parallel_world_size()

        tokens_l = []
        labels_l = []
        loss_mask_l = []
        position_ids_l = []
        image_input_mask_l = []
        vision_grid_thw_l = []
        vision_data_l = []
        teacher_outputs_list = []
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
            loss_mask = torch.ones(labels.size(), dtype=torch.float, device=labels.device)
            loss_mask[labels == -100] = 0.0
            loss_mask_l.append(pad_or_truncate_last_dim(loss_mask, seq_len, 0))

            assert batch["position_ids"].shape[-1] >= seq_len, "小于 seq_len 时，不能 Pad 0"
            position_ids_l.append(pad_or_truncate_last_dim(batch["position_ids"], seq_len, 0))
            if "image_input_mask" in batch and batch["image_input_mask"] is not None:
                image_input_mask_l.append(
                    pad_or_truncate_last_dim(batch["image_input_mask"], seq_len, 0)
                )
            if "vision_data" in batch and batch["vision_data"] is not None:
                vision_grid_thw_l.append(batch["vision_grid_thw"])
                vision_data_l.append(batch["vision_data"])

            teacher_output = None
            if input_teacher_hidden_states and mpu.is_pipeline_last_stage():
                teacher_output = batch["teacher_hidden_states"]
                assert seq_len % mpu.get_context_parallel_world_size(
                ) == 0, f"{seq_len=} {mpu.get_context_parallel_world_size()=}"
                assert teacher_output.ndim == 2
                if teacher_output.shape[0] < seq_len:
                    teacher_output = torch.cat(
                        (
                            teacher_output,
                            teacher_output.new_zeros(
                                seq_len - teacher_output.shape[0],
                                teacher_output.shape[1],
                            ),
                        ),
                        dim=0,
                    )
                else:
                    teacher_output = teacher_output[:seq_len]
                teacher_output = get_tensor_on_this_cp_rank(teacher_output, seq_dim=0)
                assert seq_len_shard_by_cp == teacher_output.shape[0], (
                    f"{seq_len_shard_by_cp=} != {teacher_output.shape[0]}"
                )
                teacher_outputs_list.append(teacher_output.cuda(non_blocking=True))

        non_blocking = True
        tokens = torch.stack(tokens_l).view(len(tokens_l), -1).cuda(non_blocking=non_blocking)
        labels = torch.stack(labels_l).view(len(labels_l), -1).cuda(non_blocking=non_blocking)
        loss_mask = torch.stack(loss_mask_l).view(len(loss_mask_l),
                                                  -1).cuda(non_blocking=non_blocking)
        position_ids = torch.cat(position_ids_l, dim=1).cuda(non_blocking=non_blocking)
        batch_size = tokens.shape[0]
        teacher_output = None
        if input_teacher_hidden_states and mpu.is_pipeline_last_stage():
            teacher_output = torch.stack(teacher_outputs_list)
            teacher_output = teacher_output.view(batch_size, seq_len_shard_by_cp, -1)
            assert teacher_output.ndim == 3

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

        if input_teacher_hidden_states:
            if mpu.is_pipeline_last_stage():
                batch["teacher_hidden_states"] = teacher_output
            else:
                batch["teacher_hidden_states"] = None
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
        batch["full_tokens"] = batch["tokens"]
        return batch, fwd_kwargs
