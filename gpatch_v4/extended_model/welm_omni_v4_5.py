import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from typing_extensions import override

from megatron.core import mpu, parallel_state
from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices
from megatron.core.packed_seq_params import PackedSeqParams

from gpatch_v4.extended_model.base import PrepareDataForward
from gpatch_v4.utils import (
    get_ltor_masks_and_position_ids,
    get_tensor_on_this_cp_rank,
    pad_or_truncate_last_dim,
)
from gpatch_v4.utils.dynamic_cp_utils import _round_up, dyn_cp_schedule_default

logger = logging.getLogger(__name__)


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
                forbidden_token_ids=[audio_token_id],
            )
            labels = pad_or_truncate_last_dim(labels, seq_len + 1, -100)
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
            has_feats = input_features is not None
            has_lens = audio_feature_lengths is not None
            assert has_feats == has_lens, (
                "input_features and audio_feature_lengths must both be set or both be None, "
                f"got features={'set' if has_feats else 'None'} "
                f"lengths={'set' if has_lens else 'None'}"
            )
            if not has_feats:
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

        full_loss_mask = loss_mask
        if mpu.get_context_parallel_world_size() > 1:
            assert seq_len % mpu.get_context_parallel_world_size() == 0, (
                f"{seq_len=} not divisible by "
                f"context_parallel_size={mpu.get_context_parallel_world_size()}"
            )
            labels = get_tensor_on_this_cp_rank(labels, 1, key_name="labels")
            loss_mask = get_tensor_on_this_cp_rank(loss_mask, 1, key_name="loss_mask")
            position_ids = get_tensor_on_this_cp_rank(position_ids, 1, key_name="position_ids")

        batch = {
            "tokens": tokens,
            "labels": labels,
            "input_features": input_features,
            "audio_feature_lengths": audio_feature_lengths,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "full_loss_mask": full_loss_mask,
        }
        fwd_kwargs = {
            "input_ids": tokens,
            "labels": None,
            "input_features": input_features,
            "audio_feature_lengths": audio_feature_lengths,
            "position_ids": position_ids,
        }
        return batch, fwd_kwargs

    @override
    def sft_reroute_data_for_dynamic_cp(
        self,
        gbs_batches: List[Dict[str, Any]],
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float]:
        """Schedule + pack GBS samples for Dynamic CP SFT.

        Token fields follow the LLM dyn-cp path; audio features use
        ``cat_keys`` like Qwen3VL vision. N-grams are **not** packed here --
        ``WelmOmniV45Model`` computes them per ``cu_seqlens`` segment.
        """
        dp_group = mpu.get_data_parallel_group()
        tp_group = mpu.get_tensor_model_parallel_group()
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)

        dp_cp_size = dp_cp_group.size()
        tp_size = tp_group.size()

        dp_size = dp_group.size()
        cp_size = dp_cp_group.size() // dp_size

        dist_config = self.config.policy.dist_config
        scheduler_type = dist_config.dynamic_cp_scheduler_type
        assert scheduler_type == "default", (
            f"WelmOmniV45 only supports dyn_cp_schedule_default, got {scheduler_type!r}"
        )
        scheduler_max_seqlen = kwargs.get(
            "max_seqlen_per_dp_cp_rank", dist_config.max_seqlen_per_dp_cp_rank
        )
        dp_cp_pad = 2 * dp_cp_size if dp_cp_size > 1 else 1
        tp_pad = tp_size if tp_size > 1 else 1
        pad_div = dp_cp_pad * tp_pad

        # 1. Shift labels; flatten audio features for cat_keys transfer.
        vocab_size = kwargs.get("vocab_size", 0)
        mel_bins = None
        # Always register audio dtypes so text-only local ranks can still
        # allocate empty send buffers during dyn-cp all-to-all. Keep a fixed
        # dtype across ranks (do not overwrite from local tensors).
        dtype_map = {
            "audio_feature_lengths": torch.int64,
            "input_features": torch.bfloat16,
        }
        for i, batch in enumerate(gbs_batches):
            raw_len = batch["tokens"].shape[-1]
            pad_len = _round_up(raw_len, pad_div)
            tokens, labels, actual_len = self._prepare_tokens_and_labels(
                batch["tokens"],
                batch["labels"],
                pad_len,
                pad_token_id,
                vocab_size,
                pad_with_random_token,
            )

            input_features = batch.get("input_features", None)
            audio_feature_lengths = batch.get("audio_feature_lengths", None)
            has_feats = input_features is not None
            has_lens = audio_feature_lengths is not None
            assert has_feats == has_lens, (
                "input_features and audio_feature_lengths must both be set or both be None, "
                f"got features={'set' if has_feats else 'None'} "
                f"lengths={'set' if has_lens else 'None'}"
            )
            if input_features is not None:
                # (mel_bins, frames) -> 1-D (frames * mel_bins) for dyn-cp all-to-all
                assert input_features.ndim == 2, f"{input_features.shape=}"
                if mel_bins is None:
                    mel_bins = int(input_features.shape[0])
                else:
                    assert mel_bins == int(
                        input_features.shape[0]
                    ), (f"inconsistent mel_bins: {mel_bins=} != {input_features.shape[0]=}")
                assert input_features.dtype == dtype_map["input_features"], (
                    f"input_features dtype must be {dtype_map['input_features']}, "
                    f"got {input_features.dtype} (dyn-cp all-to-all requires uniform dtype)"
                )
                assert audio_feature_lengths is not None
                assert audio_feature_lengths.dtype == dtype_map["audio_feature_lengths"]
                input_features = input_features.transpose(0, 1).contiguous().reshape(-1)

            gbs_batches[i] = dict(
                tokens=tokens,
                labels=labels,
                loss_mask=(labels != -100).to(torch.float32),
                position_ids=torch.arange(
                    tokens.shape[-1], dtype=torch.int64, device=tokens.device
                ),
                original_seq_len=torch.tensor([actual_len], dtype=torch.int32),
                padded_seq_len=torch.tensor([tokens.shape[-1]], dtype=torch.int32),
                input_features=input_features,
                audio_feature_lengths=audio_feature_lengths,
            )

        # 2. Schedule + pack (default only; smart_padding not supported)
        dev = torch.cuda.current_device()
        packed_keys = ["tokens", "labels", "loss_mask", "position_ids"]
        cat_keys = ["input_features", "audio_feature_lengths"]
        global_id_seqlens_keys = [
            "tokens",
            "labels",
            "loss_mask",
            "position_ids",
            "original_seq_len",
            "padded_seq_len",
            "input_features",
            "audio_feature_lengths",
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

        # 3. Restore input_features to (mel_bins, total_frames).
        # mel_bins may be unknown on ranks that only received audio via reroute.
        mel_bins_t = torch.tensor(
            [mel_bins if mel_bins is not None else 0],
            dtype=torch.int32,
            device=dev,
        )
        dist.all_reduce(mel_bins_t, op=dist.ReduceOp.MAX, group=dp_cp_group)
        mel_bins_global = int(mel_bins_t.item())
        for sample in new_samples:
            feats = sample.get("input_features", None)
            if feats is None or feats.numel() == 0:
                sample["input_features"] = None
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
        """Prepare one packed Dynamic-CP microbatch for WeLM-Omni SFT.

        Keep full ``tokens`` for the model's full-sequence audio scatter /
        OE fusion; CP-slice ``labels`` / ``loss_mask`` / ``position_ids``.
        N-grams are computed in ``WelmOmniV45Model`` from ``cu_seqlens``.
        """
        assert len(batches) == 1, "sft_train_with_dynamic_cp only supports one batch"
        batch = batches[0]
        assert "local_cp_size" in batch

        # 0. Lazy H2D
        dev = torch.cuda.current_device()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and not v.is_cuda:
                batch[k] = v.to(dev, non_blocking=True)

        # 1. Dynamic CP group
        lcp = batch.get("local_cp_size")
        if lcp is not None:
            lcp_val = lcp.item() if isinstance(lcp, torch.Tensor) else int(lcp)
            cp_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=lcp_val)
        else:
            cp_group = parallel_state.get_context_parallel_group()

        # 2. CP-slice labels/loss_mask/position_ids (tokens stay full for audio)
        cp_size = cp_group.size()
        total_tokens = batch["tokens"].size(0)
        if cp_size > 1:
            cp_rank = cp_group.rank()
            cu_seqlens_for_partition = batch["cu_seqlens_padded"]
            index = get_thd_partitioned_indices(
                cu_seqlens_for_partition, total_tokens, cp_size, cp_rank
            )
            for key in ["labels", "loss_mask", "position_ids"]:
                assert key in batch, f"{key} missing from dyn-cp packed batch"
                batch[key] = batch[key].index_select(0, index)
            if "loss_weights" in batch:
                batch["loss_weights"] = batch["loss_weights"].index_select(0, index)

        # 3. TP align on full packed tokens (model shards embeddings later)
        tp_size = parallel_state.get_tensor_model_parallel_group().size()
        assert batch["tokens"].size(0) % tp_size == 0, (
            f"post-CP tokens ({batch['tokens'].size(0)}) not aligned to tp_size={tp_size}"
        )

        # 4. Views
        cp_tokens_val = torch.tensor(batch["labels"].size(0), dtype=torch.int32)
        batch["tokens"] = batch["tokens"].view(1, total_tokens).contiguous()
        batch["position_ids"] = batch["position_ids"].view(1, cp_tokens_val).contiguous()
        batch["labels"] = batch["labels"].view(1, cp_tokens_val).contiguous()
        batch["loss_mask"] = batch["loss_mask"].view(1, cp_tokens_val).contiguous()
        if "loss_weights" in batch:
            batch["loss_weights"] = batch["loss_weights"].view(1, cp_tokens_val).contiguous()

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
        packed_seq_params._myfa_padded_lens_cache = (
            (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]).to(torch.long).tolist()
        )

        input_features = batch.get("input_features", None)
        audio_feature_lengths = batch.get("audio_feature_lengths", None)

        fwd_kwargs = dict(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=None,
            labels=None,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            packed_seq_params=packed_seq_params,
        )

        batch["cp_group"] = cp_group
        return batch, fwd_kwargs


__all__ = ["WelmOmniV45PrepareDataForward"]
