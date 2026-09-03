from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from typing_extensions import override

try:
    from megatron.bridge.utils.context_parallel_utils import qwen3vl_parallel_split
except ImportError:
    qwen3vl_parallel_split = None
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from gpatch_v4.core.constants import MODEL_ARCH
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
        if self.config.policy.model_arch in (
            MODEL_ARCH.WEMM3_5_EMBEDDING,
            MODEL_ARCH.WEMM3_5_MOE_EMBEDDING,
        ):
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
    def prepare_loss_weights(
        self,
        loss_weights: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        raise NotImplementedError("prepare_loss_weights is not implemented")

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
        """Consume collator-prepacked THD batches (one packed sample per list item).

        Collator already packed ``tokens``/``labels`` (labels already
        next-token shifted)/``cu_seqlens_padded``/``eos_positions``.
        Here we only H2D + build ``PackedSeqParams``. Do **not** call
        ``_shift_label``, and do **not** pad to a uniform ``seq_len`` —
        flash THD uses the packed tensor length as-is.
        """
        assert mpu.get_context_parallel_world_size(
        ) == 1, ("embedding pack_seq requires context_parallel_size == 1")
        assert len(batches) == 1, (
            f"pack_seq expects one packed sample per microbatch, got {len(batches)}"
        )
        sample = batches[0]
        assert "cu_seqlens_padded" in sample and "eos_positions" in sample, (
            "pack_seq batch missing cu_seqlens_padded/eos_positions from collator"
        )

        non_blocking = True
        tokens = sample["tokens"].cuda(non_blocking=non_blocking)
        labels = sample["labels"].cuda(non_blocking=non_blocking)
        attention_mask = sample["attention_mask"].cuda(non_blocking=non_blocking)
        position_ids = sample["position_ids"].cuda(non_blocking=non_blocking)
        cu_seqlens_padded = sample["cu_seqlens_padded"].cuda(non_blocking=non_blocking)
        eos_positions = sample["eos_positions"].cuda(non_blocking=non_blocking)
        max_seqlen = int(sample["max_seqlen"].item()) if "max_seqlen" in sample else int(
            (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]).max().item()
        )

        cur_len = tokens.shape[-1]
        # if cur_len > seq_len:
        #     raise RuntimeError(
        #         f"packed tokens length {cur_len} exceeds seq_len={seq_len}; "
        #         "raise training.seq_length or reduce train_mbs"
        #     )
        assert int(cu_seqlens_padded[-1].item()) == cur_len, (
            f"cu_seqlens_padded[-1]={int(cu_seqlens_padded[-1].item())} != tokens length {cur_len}"
        )

        loss_mask = (labels != -100).to(torch.float)
        full_loss_mask = loss_mask

        image_input_mask = None
        cp_img_num, images_padded, vision_data, vision_grid_thw = None, None, None, None
        if sample.get("vision_data") is not None:
            cp_img_num, images_padded, vision_data, vision_grid_thw = self._padding_images(
                [sample["vision_data"]], [sample["vision_grid_thw"]]
            )
            image_input_mask = sample["image_input_mask"].cuda(non_blocking=non_blocking)
            vision_data = vision_data.cuda(non_blocking=non_blocking)
            vision_grid_thw = vision_grid_thw.cuda(non_blocking=non_blocking)

        is_source = sample.get("is_source").cuda(non_blocking=non_blocking)

        # GradCache 阶段2 (use_gbs_embedding_in_loss) 写入 sample 的回放数据:
        # embedding 梯度切片 / logit_scale 梯度 / 阶段2 真实指标。
        embeddings_grad = sample.get("embeddings_grad")
        logit_scale_grad = sample.get("logit_scale_grad")
        cache_metrics = sample.get("cache_metrics")

        # HF PairQwen35EmbCollator 对齐：batch 级监督信号（可选）
        relevant_score = sample.get("relevant_score")
        if relevant_score is not None:
            relevant_score = relevant_score.cuda(non_blocking=non_blocking)
        teacher_scores = sample.get("teacher_scores")
        if teacher_scores is not None:
            teacher_scores = teacher_scores.cuda(non_blocking=non_blocking)

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

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
            "input_features": None,
            "feature_attention_mask": None,
            "video_second_per_grid": None,
            "audio_feature": None,
            "is_source": is_source,
            "embeddings_grad": embeddings_grad,
            "logit_scale_grad": logit_scale_grad,
            "cache_metrics": cache_metrics,
            "cu_seqlens_padded": cu_seqlens_padded,
            "eos_positions": eos_positions,
            "max_seqlen": max_seqlen,
            "packed_seq_params": packed_seq_params,
            "relevant_score": relevant_score,
            "teacher_scores": teacher_scores,
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
            packed_seq_params=packed_seq_params,
        )
        return batch, fwd_kwargs
