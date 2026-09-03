# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen3.8-Flash-Next (``qwen4_exp``) engine hooks.

Two overrides are needed on top of the generic LLM path:

* :class:`Qwen4ExpPrepareDataForwardLLM` — hand the model a **2-D padding mask** in
  HuggingFace convention instead of gcore's Megatron-convention 4-D mask.
* :class:`Qwen4ExpPostInitModel` — keep the Engram-owning layer out of activation
  checkpointing, which can only be done after the engine enables it.
"""
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.extended_model.base import PostInitModel
from gpatch_v4.extended_model.llm import PrepareDataForwardLLM
from gpatch_v4.models.qwen4_exp.cp import (
    Qwen4ExpCPContext,
    qwen4_exp_cp_chunk_data,
    qwen4_exp_pack_sequences,
)
from gpatch_v4.models.qwen4_exp import Qwen4ExpConfig, set_activation_checkpointing

__all__ = [
    "Qwen4ExpPostInitModel",
    "Qwen4ExpPrepareDataForwardLLM",
]


class Qwen4ExpPrepareDataForwardLLM(PrepareDataForwardLLM):
    """Prepare SFT data with the 2-D padding mask expected by HuggingFace models."""

    _cp_global_padding_mask: torch.Tensor
    _cp_context: Qwen4ExpCPContext | None

    def __init__(self, config) -> None:
        super().__init__(config)
        model_config = Qwen4ExpConfig.from_pretrained(config.policy.hf_model_path)
        self._pad_each_doc_to_multi_of = (
            model_config.get_text_config().indexer_compress_ratio
        )

    @override
    def _sft_train_cp_chunk_data(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        if attention_mask is not None:
            raise RuntimeError("qwen4_exp CP data prep unexpectedly received a 4-D mask")
        cp_group = mpu.get_context_parallel_group()
        cp_size = dist.get_world_size(cp_group)
        cp_rank = mpu.get_context_parallel_rank()
        tokens, labels, loss_mask, position_ids, self._cp_context = (
            qwen4_exp_cp_chunk_data(
                cp_rank,
                cp_size,
                cp_group,
                tokens=tokens,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                global_padding_mask=self._cp_global_padding_mask,
            )
        )
        return tokens, labels, loss_mask, position_ids, None

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
        if self.config.policy.ppo_pack_seq:
            return self._sft_train_thd(batches, seq_len, pad_token_id)

        cp_size = mpu.get_context_parallel_world_size()
        global_loss_weights = None
        if cp_size > 1 and any("loss_weights" in sample for sample in batches):
            if not all("loss_weights" in sample for sample in batches):
                raise ValueError("loss_weights must be present on every sample in a batch")
            global_loss_weights = torch.stack(
                [
                    self.prepare_loss_weights(sample["loss_weights"], seq_len)
                    for sample in batches
                ]
            ).view(len(batches), -1).cuda(non_blocking=True)

        if cp_size > 1:
            device = torch.device("cuda", torch.cuda.current_device())
            global_attention_mask = self._hf_padding_mask(batches, seq_len, device).bool()
            self._cp_global_padding_mask = global_attention_mask.logical_not()
            self._cp_context = None
        try:
            batch, fwd_kwargs = super().sft_train(
                batches,
                seq_len,
                pad_token_id,
                # The generic 4-D Megatron mask has opposite polarity and is replaced below.
                comput_attn_mask=False,
                pad_with_random_token=pad_with_random_token,
                **kwargs,
            )
            if cp_size > 1:
                cp_context = self._cp_context
                if cp_context is None:
                    raise RuntimeError("qwen4_exp CP data prep did not build its batch context")
        finally:
            if cp_size > 1:
                del self._cp_global_padding_mask
                del self._cp_context

        if cp_size == 1:
            fwd_kwargs["attention_mask"] = self._hf_padding_mask(
                batches, seq_len, fwd_kwargs["input_ids"].device
            )
        else:
            fwd_kwargs["attention_mask"] = cp_context.local_attention_mask.to(torch.long)
            fwd_kwargs["_qwen4_exp_cp_context"] = cp_context
            if global_loss_weights is not None:
                batch["loss_weights"] = global_loss_weights[
                    :, cp_context.local_sequence_start : cp_context.local_sequence_end
                ].contiguous()
        return batch, fwd_kwargs

    def _sft_train_thd(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Pack shifted SFT documents into one boundary-aware THD row."""
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        cp_group = mpu.get_context_parallel_group()

        document_ids = []
        document_labels = []
        document_loss_weights = []
        has_loss_weights = any("loss_weights" in sample for sample in batches)
        if has_loss_weights and not all("loss_weights" in sample for sample in batches):
            raise ValueError("loss_weights must be present on every packed sample")

        for index, sample in enumerate(batches):
            tokens = torch.as_tensor(sample["tokens"], dtype=torch.long).reshape(-1)
            labels = torch.as_tensor(sample["labels"], dtype=torch.long).reshape(-1)
            if tokens.shape != labels.shape:
                raise ValueError(
                    f"sample {index} token/label shapes differ: {tokens.shape} and {labels.shape}"
                )
            if not torch.equal(tokens == labels, labels >= 0):
                raise ValueError(f"sample {index} labels must be unshifted token IDs or -100")
            if tokens.numel() > seq_len + 1:
                tokens = tokens[-(seq_len + 1):]
                labels = labels[-(seq_len + 1):]
            if tokens.numel() < 2:
                raise ValueError(f"sample {index} needs at least two tokens for SFT shifting")

            document_ids.append(tokens[:-1].cuda(non_blocking=True))
            document_labels.append(labels[1:].cuda(non_blocking=True))
            if has_loss_weights:
                loss_weights = torch.as_tensor(
                    sample["loss_weights"], dtype=torch.float32
                ).reshape(-1)
                if loss_weights.numel() != torch.as_tensor(sample["tokens"]).numel():
                    raise ValueError(
                        f"sample {index} loss_weights must match its unshifted tokens"
                    )
                if loss_weights.numel() > seq_len + 1:
                    loss_weights = loss_weights[-(seq_len + 1):]
                document_loss_weights.append(loss_weights[1:].cuda(non_blocking=True))

        packed_ids, packed_positions, packed_labels, packed_seq_params = (
            qwen4_exp_pack_sequences(
                document_ids,
                document_labels,
                cp_size=cp_size,
                pad_multiple=self._pad_each_doc_to_multi_of,
                pad_token_id=pad_token_id,
            )
        )
        if packed_seq_params.total_seqlen > seq_len:
            raise ValueError(
                f"packed row length {packed_seq_params.total_seqlen} exceeds seq_len={seq_len}"
            )

        global_padding_mask = torch.arange(
            packed_seq_params.total_seqlen, device=packed_ids.device
        ).unsqueeze(0) >= int(packed_seq_params.cu_seqlens_q[-1])
        loss_mask = (packed_labels != -100).float()
        local_ids, local_labels, local_loss_mask, local_positions, cp_context = (
            qwen4_exp_cp_chunk_data(
                cp_rank,
                cp_size,
                cp_group,
                tokens=packed_ids,
                labels=packed_labels,
                loss_mask=loss_mask,
                position_ids=packed_positions,
                global_padding_mask=global_padding_mask,
                global_cu_seqlens=packed_seq_params.cu_seqlens_q,
            )
        )

        local_loss_weights = None
        if document_loss_weights:
            packed_loss_weights = torch.zeros(
                (1, packed_seq_params.total_seqlen),
                dtype=torch.float32,
                device=packed_ids.device,
            )
            for start, weights in zip(
                packed_seq_params.cu_seqlens_q[:-1].tolist(),
                document_loss_weights,
            ):
                packed_loss_weights[0, start:start + weights.numel()] = weights
            local_loss_weights = packed_loss_weights[
                :, cp_context.local_sequence_start : cp_context.local_sequence_end
            ].contiguous()

        batch = {
            "labels": local_labels,
            "loss_mask": local_loss_mask,
            "loss_weights": local_loss_weights,
            "full_labels": packed_labels,
            "full_loss_mask": loss_mask,
            "full_packed_seq_params": packed_seq_params,
        }
        fwd_kwargs = {
            "input_ids": local_ids,
            "position_ids": local_positions,
            "attention_mask": cp_context.local_attention_mask.to(torch.long),
            "labels": None,
            "_qwen4_exp_cp_context": cp_context,
        }
        return batch, fwd_kwargs

    @staticmethod
    def _hf_padding_mask(
        batches: List[Dict[str, Any]],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """``[batch, seq_len]`` int mask, ``1`` for real tokens and ``0`` for right padding.

        Lengths come from the unpadded ``tokens`` entries, so this stays correct however
        the base class chose to pad.
        """
        mask = torch.zeros(len(batches), seq_len, dtype=torch.long, device=device)
        for index, sample in enumerate(batches):
            length = min(int(sample["tokens"].shape[0]), seq_len)
            mask[index, :length] = 1
        return mask


class Qwen4ExpPostInitModel(PostInitModel):
    """Keep Engram out of recompute, which would replay its owner-sharded collectives."""
    @override
    def __call__(self, model) -> None:
        set_activation_checkpointing(model, enabled=bool(self.config.training.recompute))
