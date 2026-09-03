from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from transformers import DeepseekV4Config
from typing_extensions import override

from megatron.core import mpu, parallel_state
from megatron.core.packed_seq_params import PackedSeqParams as McorePackedSeqParams

from gpatch_v4.configs.config import FinetuneConfig, RlConfig
from gpatch_v4.extended_model.base import (
    CheckpointContextFn,
    PostInitModel,
    ResetRouterCorrectionBiasAccum,
    UpdateRouterCorrectionBias,
)
from gpatch_v4.extended_model.llm import DpoPrepareDataForwardLLM, PrepareDataForwardLLM
from gpatch_v4.models.deepseek_v4.cp import cp_chunk_data
from gpatch_v4.models.deepseek_v4.freeze_csa_indexer import freeze_csa_indexer_params
from gpatch_v4.models.deepseek_v4.freeze_update_router import (
    checkpoint_context_fn,
    freeze_router_weights,
    init_router_correction_bias_accumulators,
    register_router_correction_bias_accum_tracking_hook,
    reset_router_correction_bias_accum,
    update_router_correction_bias,
)
from gpatch_v4.models.deepseek_v4.thd import pack_sequences
from gpatch_v4.utils import pad_or_truncate_last_dim


class DeepseekV4PrepareDataForwardLLM(PrepareDataForwardLLM):
    """DeepSeek-V4 SFT data preparation (HpModule / mcore, EP + CP).

    Three modes, selected by training backend and ``ppo_pack_seq``:

    **mcore CP THD** (mcore backend + CP > 1, ``ppo_pack_seq=False``):
    packs samples into a single ``[1, T]`` sequence and builds a mcore
    :class:`megatron.core.packed_seq_params.PackedSeqParams` with
    ``qkv_format='thd'`` and ``cp_partition_mode='contiguous'``.
    Required by :class:`DSv4HybridSelfAttention` which raises when
    ``cp_size > 1`` but no THD PSP is provided.

    **BSHD** (``ppo_pack_seq=False``, non-mcore-CP): pads + stacks into
    ``[B, S]`` via the parent class, then overrides CP slicing to use
    contiguous ``cp_chunk_data`` instead of Megatron-style zigzag.

    **THD** (``ppo_pack_seq=True``): packs variable-length samples into a
    single ``[1, T]`` sequence via :func:`pack_sequences`, builds the
    HpModule :class:`PackedSeqParams`, and CP-slices the layout.
    """
    def __init__(self, config):
        super().__init__(config)
        # Dynamic CP and mcore contiguous THD also need sliding-window and
        # compression metadata when ppo_pack_seq=False.
        self._model_config = DeepseekV4Config.from_pretrained(config.policy.hf_model_path)
        if config.debug.debug_truncate_num_hidden_layers is not None:
            nl = config.debug.debug_truncate_num_hidden_layers
            self._model_config.num_hidden_layers = nl
            self._model_config.layer_types = self._model_config.layer_types[:nl]
            self._model_config.mlp_layer_types = self._model_config.mlp_layer_types[:nl]

        # TODO 设计在这里不太合理，如果后续有需求再调整。
        self._pad_each_doc_to_multi_of = max(self._model_config.compress_rates.values())

    def _grpo_fsdp2_train_thd(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        *,
        include_rl_fields: bool,
        pad_with_random_token: bool = False,
        vocab_size: int = 0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Pack pre-shifted GRPO tensors into one DSV4 THD sequence.

        GRPO token-level tensors already live on the next-token ``S-1`` axis.
        Shift every sample before packing so targets never roll across segment
        boundaries, then place every RL field at the same padded segment offset.
        """
        cp_size = dist.get_world_size(mpu.get_context_parallel_group())
        cp_rank = mpu.get_context_parallel_rank() if cp_size > 1 else 0

        ids_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        actual_lens: list[int] = []
        for batch in batches:
            tokens = batch["tokens"]
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.reshape(-1)
            current_seq_len = int(batch["sequence_lengths"].item())
            assert current_seq_len == tokens.numel(
            ), (f"{current_seq_len=} != tokens.numel()={tokens.numel()}")
            assert current_seq_len <= seq_len, (
                f"sample sequence length {current_seq_len} exceeds configured "
                f"sequence length {seq_len}"
            )
            assert current_seq_len >= 2, "GRPO samples need at least two tokens"

            shifted_input = tokens[:-1].cuda(non_blocking=True)
            shifted_target = tokens[1:].cuda(non_blocking=True)
            ids_list.append(shifted_input)
            targets_list.append(shifted_target)
            actual_lens.append(shifted_input.numel())

        packed_ids, packed_pos, _, full_psp = pack_sequences(
            ids_list,
            None,
            config=self._model_config,
            pad_to_multiple_of=self._pad_each_doc_to_multi_of,
            cp_size=cp_size,
            pad_token_id=pad_token_id,
        )
        # 暂时先不做 random pad
        # if pad_with_random_token:
        #     assert vocab_size > 0, "vocab_size must be positive for random THD padding"
        #     pad_mask = full_psp.layout.pad_token_mask
        #     packed_ids.view(-1)[pad_mask] = torch.randint(
        #         0,
        #         vocab_size,
        #         (int(pad_mask.sum().item()),),
        #         dtype=packed_ids.dtype,
        #         device=packed_ids.device,
        #     )

        total_tokens = full_psp.total_seqlen
        packed_target = torch.zeros((1, total_tokens), dtype=torch.long, device=packed_ids.device)
        rl_key_map = (
            ("advantages", "advantages"),
            ("mask", "mask"),
            ("logprobs", "prev_log_probs"),
            ("ref_logprobs", "ref_log_probs"),
            ("rollout_log_probs", "rollout_log_probs"),
        )
        present_rl_keys = [(src, dst) for src, dst in rl_key_map
                           if src in batches[0]] if include_rl_fields else []
        if include_rl_fields:
            for required in ("advantages", "mask", "logprobs"):
                assert required in batches[0], f"THD GRPO batch is missing {required}"

        packed_rl = {
            dst: torch.zeros((1, total_tokens), dtype=torch.float32, device=packed_ids.device)
            for _, dst in present_rl_keys
        }
        cu_padded = full_psp.cu_seqlens_q_padded
        for sample_idx, (batch, target, actual_len) in enumerate(
            zip(batches, targets_list, actual_lens, strict=True)
        ):
            offset = int(cu_padded[sample_idx].item())
            packed_target[0, offset:offset + actual_len] = target
            for src, dst in present_rl_keys:
                assert src in batch, f"sample {sample_idx} is missing {src}"
                value = batch[src]
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)
                assert value.dim() <= 1, (
                    f"sample {sample_idx} {src} has shape {tuple(value.shape)}; "
                    "FSDP2 DSV4 THD GRPO only supports 1D token-level fields "
                    "(e.g. 3D topk advantages are not supported yet)"
                )
                value = value.reshape(-1)
                # Rollout stores one trailing dummy slot on the original
                # S-token axis; policy loss lives on the shifted S-1 axis.
                if src == "rollout_log_probs" and value.numel() == actual_len + 1:
                    value = value[:actual_len]
                assert value.numel() == actual_len, (
                    f"sample {sample_idx} {src} length {value.numel()} "
                    f"!= shifted token length {actual_len}"
                )
                packed_rl[dst][0, offset:offset + actual_len] = value.to(
                    device=packed_ids.device, dtype=torch.float32
                )
        assert int(cu_padded[-1].item()) == total_tokens

        if cp_size > 1:
            local_ids, _, _, local_pos, local_psp = cp_chunk_data(
                cp_rank,
                cp_size,
                tokens=packed_ids,
                position_ids=packed_pos,
                packed_seq_params=full_psp,
            )
        else:
            local_ids = packed_ids
            local_pos = packed_pos
            local_psp = full_psp

        batch_out: Dict[str, Any] = {
            "packed_token_ids": packed_ids,
            "target": packed_target,
            "full_packed_seq_params": full_psp,
            "cu_seqlens": full_psp.cu_seqlens_q,
            "cu_seqlens_padded": full_psp.cu_seqlens_q_padded,
            **packed_rl,
        }
        for key in [
            "entropy_aux_figures",
        ]:
            if include_rl_fields and key in batches[0]:
                batch_out[key] = torch.as_tensor(batches[0][key]).to(device=packed_ids.device)

        fwd_kwargs: Dict[str, Any] = {
            "input_ids": local_ids,
            "position_ids": local_pos,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": local_psp,
        }
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
        cp_size = mpu.get_context_parallel_world_size()
        is_mcore = getattr(self.config.training, "training_backend", "") == "mcore"

        if is_mcore and cp_size > 1:
            return self._model_forward_only_mcore_thd(batches, seqlen, pad_token_id)

        if not self.config.policy.ppo_pack_seq:
            return super().model_forward_only(
                batches,
                seqlen,
                pad_token_id,
                pad_with_random_token=pad_with_random_token,
                **kwargs,
            )
        batch, fwd_kwargs = self._grpo_fsdp2_train_thd(
            batches,
            seqlen,
            pad_token_id,
            include_rl_fields=False,
            pad_with_random_token=pad_with_random_token,
            vocab_size=kwargs.get("vocab_size", 0),
        )
        fwd_kwargs["target"] = batch["target"]
        fwd_kwargs["full_packed_seq_params"] = batch["full_packed_seq_params"]
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

        cp_size = mpu.get_context_parallel_world_size()
        is_mcore = getattr(self.config.training, "training_backend", "") == "mcore"
        if is_mcore and cp_size > 1:
            assert ppo_pack_seq, "dsv4 mcore cp require thd format for rl (grpo)"
            # Seq-mean per-sample GRPO loss cannot be normalized correctly under
            # contiguous static CP: _grpo_train_mcore_thd packs N samples into a
            # single [1, T] row and each CP rank owns a contiguous [T/cp] slice.
            # The loss factory's per-sample path only supports either a plain 2D
            # [B, S] batch (collapses the packed row into one "sample") or the
            # dynamic-CP TE layout (cu // cp_size) — neither matches contiguous
            # static CP, and no cross-CP per-sample reduction is done. Only
            # per-token normalization (calculate_per_token_loss=True) aggregates
            # correctly over DP×CP. Fail fast instead of silently mis-normalizing.
            if getattr(self.config, "ppo", None) is not None \
                    and self.config.ppo.loss_func == "grpo":
                otc = self.config.policy.override_transformer_config or {}
                assert otc.get("calculate_per_token_loss", False), (
                    "DSv4 mcore CP>1 RL with loss_func='grpo' (seq-mean per-sample) "
                    "requires calculate_per_token_loss=True; per-sample loss "
                    "normalization is not supported under contiguous static CP. "
                    "Set policy.override_transformer_config.calculate_per_token_loss"
                    "=True (or switch to a per-token loss)."
                )
            return self._grpo_train_mcore_thd(batches, seqlen, pad_token_id)

        if not ppo_pack_seq:
            return super().grpo_train(
                batches,
                seqlen,
                pad_token_id,
                ppo_pack_seq=False,
                pad_with_random_token=pad_with_random_token,
                **kwargs,
            )
        assert self.config.policy.ppo_pack_seq
        return self._grpo_fsdp2_train_thd(
            batches,
            seqlen,
            pad_token_id,
            include_rl_fields=True,
            pad_with_random_token=pad_with_random_token,
            vocab_size=kwargs.get("vocab_size", 0),
        )

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
        # contiguous CP requires THD format; choose PSP type by training backend.
        cp_size = mpu.get_context_parallel_world_size()
        is_mcore = getattr(self.config.training, "training_backend", "") == "mcore"

        if is_mcore and cp_size > 1:
            assert self.config.policy.ppo_pack_seq, ("dsv4 mcore cp require thd format")

        if not self.config.policy.ppo_pack_seq:
            return super().sft_train(
                batches,
                seq_len,
                pad_token_id,
                comput_attn_mask=comput_attn_mask,
                pad_with_random_token=pad_with_random_token,
                **kwargs,
            )
        if is_mcore:
            return self._sft_train_mcore_thd(batches, seq_len, pad_token_id)
        return self._sft_train_thd(batches, seq_len, pad_token_id)

    def _sft_train_mcore_thd(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Pack sequences into THD + mcore PackedSeqParams for DSv4 mcore CP.

        Creates a ``megatron.core.packed_seq_params.PackedSeqParams`` with
        ``qkv_format='thd'`` and ``cp_partition_mode='contiguous'``.  Each
        sample is padded to the next multiple of ``cp_size`` so that the
        total packed length ``T`` is exactly divisible by ``cp_size``; each
        CP rank then receives a contiguous ``T // cp_size`` token slice.

        The caller (``_sft_train_func`` in mixin.py) detects the non-None
        ``packed_seq_params`` in ``fwd_kwargs`` and calls the model directly
        instead of routing through ``gptmodel_pack_foward``.
        """
        from megatron.core.packed_seq_params import (
            PackedSeqParams as McorePackedSeqParams,
        )

        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()

        ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        ref_logprobs_list: List[torch.Tensor] = []
        has_ref_logprobs = "ref_logprobs" in batches[0]
        for b in batches:
            tok = b["tokens"]
            # labels has not be shifted yet
            lab = b["labels"]
            if not isinstance(tok, torch.Tensor):
                tok = torch.tensor(tok, dtype=torch.long)
            if not isinstance(lab, torch.Tensor):
                lab = torch.tensor(lab, dtype=torch.long)
            assert tok.shape == lab.shape
            if tok.shape[-1] > seq_len + 1:
                tok = tok[-(seq_len + 1):]
                lab = lab[-(seq_len + 1):]
            # Next-token shift
            tok = tok[:-1]
            lab = lab[1:]
            ids_list.append(tok.cuda(non_blocking=True))
            labels_list.append(lab.cuda(non_blocking=True))
            if has_ref_logprobs:
                ref_logprobs = torch.as_tensor(b["ref_logprobs"], dtype=torch.float32).reshape(-1)
                ref_logprobs_list.append(
                    pad_or_truncate_last_dim(ref_logprobs, tok.numel(),
                                             0.0).cuda(non_blocking=True)
                )

        device = ids_list[0].device
        seqlens = [t.shape[0] for t in ids_list]
        # Pad each sample to the next cp_size multiple so that T % cp_size == 0.
        padded_seqlens = [((s + cp_size - 1) // cp_size) * cp_size for s in seqlens]
        T = sum(padded_seqlens)

        # at least sliding window * cp
        sliding_window = self._model_config.sliding_window
        align_length = math.lcm(sliding_window, self.config.training.pad_to_mulitiple_of)
        T = max(T, sliding_window * cp_size)
        # pad to multiple of align_length, avoid endless cuteSDL compile
        if cp_size > 1:
            T = (T + align_length - 1) // align_length * align_length

        packed_ids = torch.full((1, T), pad_token_id, dtype=torch.long, device=device)
        packed_labels = torch.full((1, T), -100, dtype=torch.long, device=device)
        packed_ref_logprobs = (
            torch.zeros((1, T), dtype=torch.float32, device=device) if has_ref_logprobs else None
        )

        # Keep logical and physical boundaries separate: the former excludes
        # inter-sequence CP padding while the latter indexes packed storage.
        cu_seqlens_cpu = torch.zeros(len(ids_list) + 1, dtype=torch.int32)
        cu_seqlens_padded_cpu = torch.zeros(len(ids_list) + 1, dtype=torch.int32)
        offset = 0
        logical_offset = 0
        for i, (s, s_pad) in enumerate(zip(seqlens, padded_seqlens)):
            packed_ids[0, offset:offset + s] = ids_list[i]
            packed_labels[0, offset:offset + s] = labels_list[i]
            if packed_ref_logprobs is not None:
                packed_ref_logprobs[0, offset:offset + s] = ref_logprobs_list[i]
            offset += s_pad
            logical_offset += s
            cu_seqlens_cpu[i + 1] = logical_offset
            cu_seqlens_padded_cpu[i + 1] = offset

        max_seqlen = max(padded_seqlens)
        # Sliding-window inflate may make T > sum(padded_seqlens). Fold the
        # physical tail into the last segment so RoPE/CSA (which fall back to
        # cu_seqlens when *_padded is None) see a consistent length.
        if T > offset:
            cu_seqlens_padded_cpu[-1] = T
            max_seqlen = max(max_seqlen, int(padded_seqlens[-1]) + (T - offset))

        cu_seqlens = cu_seqlens_cpu.to(device)
        cu_seqlens_padded = cu_seqlens_padded_cpu.to(device)

        psp = McorePackedSeqParams(
            qkv_format="thd",
            cp_partition_mode="contiguous",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

        # Contiguous CP slice: each rank owns T // cp_size tokens.
        chunk = T // cp_size
        start, end = cp_rank * chunk, (cp_rank + 1) * chunk
        local_ids = packed_ids[:, start:end].contiguous()
        local_labels = packed_labels[:, start:end].contiguous()
        local_loss_mask = (local_labels != -100).float()
        full_loss_mask = (packed_labels != -100).float()

        batch_out: Dict[str, Any] = {
            "labels": local_labels,
            "loss_mask": local_loss_mask,
            "full_labels": packed_labels,
            "full_loss_mask": full_loss_mask,
            "full_packed_seq_params": psp,
        }
        if self.config.training.online_train_dspark:
            # TODO: 此路径暂时还没支持
            assert False, "online_train_dspark is not supported for DSv4 mcore CP"
            batch_out["full_input_ids"] = packed_ids
        if packed_ref_logprobs is not None:
            batch_out["ref_logprobs"] = packed_ref_logprobs[:, start:end].contiguous()
        fwd_kwargs: Dict[str, Any] = {
            "input_ids": local_ids,
            # position_ids: THD CP mode computes RoPE positions internally from
            # cu_seqlens via _thd_cp_position_ids; no external position_ids needed.
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            # Non-None PSP signals mixin.py to skip gptmodel_pack_foward.
            "packed_seq_params": psp,
        }
        return batch_out, fwd_kwargs

    def _sft_train_thd(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        cp_size = dist.get_world_size(mpu.get_context_parallel_group())
        cp_rank = mpu.get_context_parallel_rank() if cp_size > 1 else 0

        ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        for b in batches:
            tok = b["tokens"]
            lab = b["labels"]
            if not isinstance(tok, torch.Tensor):
                tok = torch.tensor(tok, dtype=torch.long)
            if not isinstance(lab, torch.Tensor):
                lab = torch.tensor(lab, dtype=torch.long)

            assert tok.shape == lab.shape
            # Truncate if longer than seq_len + 1 (need +1 for shift)
            if tok.shape[-1] > seq_len + 1:
                tok = tok[-(seq_len + 1):]
                lab = lab[-(seq_len + 1):]
            # Next-token shift
            tok = tok[:-1]
            lab = lab[1:]

            ids_list.append(tok.cuda(non_blocking=True))
            labels_list.append(lab.cuda(non_blocking=True))

        # NOTE：由于目前 pack seq 的算法是 heuristic 的，所以其实有可能超过最大长度限制，但一般程度比较轻微。
        packed_ids, packed_pos, packed_labels, psp = pack_sequences(
            ids_list,
            labels_list,
            config=self._model_config,
            pad_to_multiple_of=self._pad_each_doc_to_multi_of,
            cp_size=cp_size,
            pad_token_id=pad_token_id,
            label_ignore_index=-100,
        )

        if cp_size > 1:
            local_ids, local_labels, _, local_pos, local_psp = cp_chunk_data(
                cp_rank,
                cp_size,
                tokens=packed_ids,
                labels=packed_labels,
                position_ids=packed_pos,
                packed_seq_params=psp,
            )
        else:
            local_ids = packed_ids
            local_labels = packed_labels
            local_pos = packed_pos
            local_psp = psp
        local_loss_mask = (local_labels != -100).float()
        full_loss_mask = (packed_labels != -100).float()

        batch_out: Dict[str, Any] = {
            "labels": local_labels,
            "loss_mask": local_loss_mask,
            "full_labels": packed_labels,
            "full_loss_mask": full_loss_mask,
            "full_packed_seq_params": psp,
        }
        if self.config.training.online_train_dspark:
            batch_out["full_input_ids"] = packed_ids
        fwd_kwargs: Dict[str, Any] = {
            "input_ids": local_ids,
            "position_ids": local_pos,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": local_psp,
        }
        return batch_out, fwd_kwargs

    def _model_forward_only_mcore_thd(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
    ) -> Dict[str, Any]:
        """Pack samples into training-compatible THD format for CP logprob computation.

        Returns a dict consumed by ``get_logprob_output_only_func``'s
        ``log_prob_output_only_func``.  The standard keys (``input_ids``,
        ``position_ids``, ``attention_mask``, ``packed_seq_params``, ``target``)
        drive the model forward.  Three protocol flags are added for the
        ``id_func`` closure to handle logprob computation correctly:

        ``_logprob_ignore_cp=True``
            Skip ``from_parallel_logits_to_logprobs``'s internal CP reorder /
            slice / all_gather — the CP slice was already applied here.

        ``_logprob_pre_shifted=True``
            ``target`` is pre-built as ``rolled_tokens[:, start:end]``
            (next-token shift already applied), so ``from_parallel_logits_to_logprobs``
            must not roll again.

        ``_logprob_contiguous_gather=True``
            After computing local logprobs ``[1, T/cp]``, ``id_func`` must
            explicitly all-gather across CP ranks in contiguous order
            (``dist.all_gather`` + ``torch.cat``), then split the physical
            packed buffer back into one fixed-width result per input sample.
            This differs from the zigzag all-gather used by standard CP.

        The packing deliberately reuses ``_sft_train_mcore_thd`` so reference /
        rollout logprob forwards have the identical sequence boundaries, CP
        alignment padding, and final sliding-window inflation as training.
        """
        packed_batches = [{**batch, "labels": batch["tokens"]} for batch in batches]
        packed_batch, fwd_kwargs = self._sft_train_mcore_thd(
            packed_batches, seqlen - 1, pad_token_id
        )
        fwd_kwargs["target"] = packed_batch["labels"]
        fwd_kwargs.update(
            {
                # Protocol flags for get_logprob_output_only_func's id_func.
                "_logprob_ignore_cp": True,
                "_logprob_pre_shifted": True,
                "_logprob_contiguous_gather": True,
            }
        )
        return fwd_kwargs

    def _grpo_train_mcore_thd(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Pack RL samples into THD + mcore PackedSeqParams for DSv4 mcore CP.

        Analogous to ``_sft_train_mcore_thd`` but additionally packs per-token
        RL fields (advantages, mask, prev_log_probs, ref_log_probs,
        rollout_log_probs) alongside input_ids/target.

        Each sample is padded to the next multiple of ``cp_size`` so that the
        total packed length ``T`` is divisible by ``cp_size``; each CP rank
        then receives a contiguous ``T // cp_size`` token slice.

        Batch-level scalars (sequence_lengths, sample_mask,
        global_retention_ratio) are stacked unchanged and returned in
        ``batch_out``.

        The returned ``fwd_kwargs["packed_seq_params"]`` is detected by
        ``gptmodel_pack_foward`` in ``model_forward.py``, which then bypasses
        the HpModule rmpad path and calls ``model(...)`` directly.
        """
        from megatron.core.packed_seq_params import (
            PackedSeqParams as McorePackedSeqParams,
        )

        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()

        has_ref_logprobs = "ref_logprobs" in batches[0]
        has_rollout_logprobs = "rollout_log_probs" in batches[0]
        has_sample_mask = "sample_mask" in batches[0]
        has_global_retention = "global_retention_ratio" in batches[0]

        ids_list: List[torch.Tensor] = []
        target_list: List[torch.Tensor] = []
        adv_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        logprobs_list: List[torch.Tensor] = []
        ref_lp_list: List[torch.Tensor] = []
        rollout_lp_list: List[torch.Tensor] = []
        sequence_lengths_l: List[torch.Tensor] = []
        sample_mask_l: List[torch.Tensor] = []

        for b in batches:
            tok = b["tokens"]
            if not isinstance(tok, torch.Tensor):
                tok = torch.tensor(tok, dtype=torch.long)
            if tok.shape[-1] > seqlen + 1:
                tok = tok[-(seqlen + 1):]
            # Input/target shift; RL per-token fields are already seqlen-1 long.
            inp = tok[:-1]
            tgt = tok[1:]
            actual_len = inp.shape[0]

            def _f32(v):
                if not isinstance(v, torch.Tensor):
                    return torch.tensor(v, dtype=torch.float32)
                return v.float()

            ids_list.append(inp.cuda(non_blocking=True))
            target_list.append(tgt.cuda(non_blocking=True))
            adv_list.append(
                pad_or_truncate_last_dim(_f32(b["advantages"]), actual_len,
                                         0.0).cuda(non_blocking=True)
            )
            mask_list.append(
                pad_or_truncate_last_dim(_f32(b["mask"]), actual_len, 0.0).cuda(non_blocking=True)
            )
            logprobs_list.append(
                pad_or_truncate_last_dim(_f32(b["logprobs"]), actual_len,
                                         0.0).cuda(non_blocking=True)
            )
            sequence_lengths_l.append(b["sequence_lengths"])
            if has_ref_logprobs:
                ref_lp_list.append(
                    pad_or_truncate_last_dim(_f32(b["ref_logprobs"]), actual_len,
                                             0.0).cuda(non_blocking=True)
                )
            if has_rollout_logprobs:
                rollout_lp_list.append(
                    pad_or_truncate_last_dim(_f32(b["rollout_log_probs"]), actual_len,
                                             0.0).cuda(non_blocking=True)
                )
            if has_sample_mask:
                sample_mask_l.append(b["sample_mask"])

        device = ids_list[0].device
        seqlens = [t.shape[0] for t in ids_list]
        padded_seqlens = [((s + cp_size - 1) // cp_size) * cp_size for s in seqlens]
        T = sum(padded_seqlens)

        # at least sliding_window * cp, aligned to sliding_window
        sliding_window = self._model_config.sliding_window
        T = max(T, sliding_window * cp_size)
        if cp_size > 1:
            T = (T + sliding_window - 1) // sliding_window * sliding_window

        packed_ids = torch.full((1, T), pad_token_id, dtype=torch.long, device=device)
        packed_tgt = torch.zeros((1, T), dtype=torch.long, device=device)
        packed_adv = torch.zeros((1, T), dtype=torch.float32, device=device)
        packed_mask = torch.zeros((1, T), dtype=torch.float32, device=device)
        packed_lp = torch.zeros((1, T), dtype=torch.float32, device=device)
        packed_ref_lp = torch.zeros((1, T), dtype=torch.float32, device=device) \
            if has_ref_logprobs else None
        packed_rollout_lp = torch.zeros((1, T), dtype=torch.float32, device=device) \
            if has_rollout_logprobs else None

        cu_seqlens_cpu = torch.zeros(len(ids_list) + 1, dtype=torch.int32)
        cu_seqlens_padded_cpu = torch.zeros(len(ids_list) + 1, dtype=torch.int32)
        offset = 0
        logical_offset = 0
        for i, (s, s_pad) in enumerate(zip(seqlens, padded_seqlens)):
            sl = slice(offset, offset + s)
            packed_ids[0, sl] = ids_list[i]
            packed_tgt[0, sl] = target_list[i]
            packed_adv[0, sl] = adv_list[i]
            packed_mask[0, sl] = mask_list[i]
            packed_lp[0, sl] = logprobs_list[i]
            if has_ref_logprobs:
                packed_ref_lp[0, sl] = ref_lp_list[i]
            if has_rollout_logprobs:
                packed_rollout_lp[0, sl] = rollout_lp_list[i]
            offset += s_pad
            logical_offset += s
            cu_seqlens_cpu[i + 1] = logical_offset
            cu_seqlens_padded_cpu[i + 1] = offset
        max_seqlen = max(padded_seqlens)
        if T > offset:
            cu_seqlens_padded_cpu[-1] = T
            max_seqlen = max(max_seqlen, int(padded_seqlens[-1]) + (T - offset))
        cu_seqlens = cu_seqlens_cpu.to(device)
        cu_seqlens_padded = cu_seqlens_padded_cpu.to(device)

        psp = McorePackedSeqParams(
            qkv_format="thd",
            cp_partition_mode="contiguous",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

        # Contiguous CP slice: each rank owns T // cp_size tokens.
        chunk = T // cp_size
        start, end = cp_rank * chunk, (cp_rank + 1) * chunk

        def _cp(t: torch.Tensor) -> torch.Tensor:
            return t[:, start:end].contiguous()

        batch_out: Dict[str, Any] = {
            "advantages": _cp(packed_adv),
            "prev_log_probs": _cp(packed_lp),
            "mask": _cp(packed_mask),
            "target": _cp(packed_tgt),
            "sequence_lengths": torch.stack(sequence_lengths_l).cuda(),
            "ref_log_probs": _cp(packed_ref_lp) if has_ref_logprobs else None,
            "rollout_log_probs": _cp(packed_rollout_lp) if has_rollout_logprobs else None,
            "full_packed_seq_params": psp,
        }
        if has_sample_mask:
            batch_out["sample_mask"] = torch.stack(sample_mask_l).cuda()
        if has_global_retention:
            batch_out["global_retention_ratio"] = batches[0]["global_retention_ratio"].cuda()

        fwd_kwargs: Dict[str, Any] = {
            "input_ids": _cp(packed_ids),
            # THD CP mode computes RoPE from cu_seqlens internally; no external position_ids.
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            # Non-None PSP: gptmodel_pack_foward's fast path calls model(...) directly.
            "packed_seq_params": psp,
        }
        return batch_out, fwd_kwargs

    def _train_with_dynamic_cp_contiguous(
        self,
        batches: List[Dict[str, Any]],
        *,
        rl: bool,
        pad_token_id: int = 0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Build a DSv4 THD microbatch using the selected dynamic CP group."""
        assert len(batches) == 1, "DSv4 dynamic CP only supports one packed microbatch"
        batch = batches[0]
        assert "local_cp_size" in batch

        dev = torch.cuda.current_device()
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor) and not value.is_cuda:
                batch[key] = value.to(dev, non_blocking=True)

        local_cp_size = int(batch["local_cp_size"].item())
        cp_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()

        if rl:
            token_keys = [
                key for key in (
                    "tokens",
                    "labels",
                    "advantages",
                    "prev_log_probs",
                    "ref_log_probs",
                    "rollout_log_probs",
                    "teacher_log_probs",
                    "sample_mask",
                ) if key in batch
            ]
            if "loss_mask" in batch:
                token_keys.append("loss_mask")
        else:
            token_keys = ["tokens", "labels", "loss_mask"]
            if "loss_weights" in batch:
                token_keys.append("loss_weights")

        total_tokens = int(batch["cu_seqlens_padded"][-1].item())
        if not rl:
            tp_size = parallel_state.get_tensor_model_parallel_group().size()
            configured_pad = int(self.config.training.pad_to_mulitiple_of)
            if configured_pad <= 0:
                raise ValueError(
                    "training.pad_to_mulitiple_of must be positive for DSV4 "
                    f"dynamic CP, got {configured_pad}"
                )
            sliding_window = int(self._model_config.sliding_window)
            # Match _sft_train_mcore_thd: keep at least one sliding window per
            # CP rank and round the packed tail before contiguous CP slicing.
            # The configured and TP factors are additional dynamic-CP constraints.
            alignment = math.lcm(configured_pad, sliding_window, cp_size * tp_size)
            padded_total = max(total_tokens, sliding_window * cp_size)
            padded_total = ((padded_total + alignment - 1) // alignment * alignment)
            padding = padded_total - total_tokens
            if padding:
                # Pad only the end of the packed microbatch instead of every
                # sample. -100/zero keep these synthetic rows out of SFT loss.
                pad_values = {
                    "tokens": pad_token_id,
                    "labels": -100,
                    "loss_mask": 0,
                    "loss_weights": 0,
                }
                for key in token_keys:
                    value = batch[key].reshape(-1)
                    batch[key] = torch.cat(
                        (
                            value,
                            torch.full(
                                (padding, ),
                                pad_values[key],
                                dtype=value.dtype,
                                device=value.device,
                            ),
                        )
                    )
                # Extend only the padded boundary of the final segment. Keep
                # cu_seqlens unchanged so attention can identify real tokens
                # and mask the synthetic tail.
                cu_seqlens_padded = batch["cu_seqlens_padded"].clone()
                last_start = int(cu_seqlens_padded[-2].item())
                cu_seqlens_padded[-1] = padded_total
                batch["cu_seqlens_padded"] = cu_seqlens_padded
                batch["max_seqlen"] = torch.maximum(
                    batch["max_seqlen"],
                    batch["max_seqlen"].new_tensor(padded_total - last_start),
                )
                total_tokens = padded_total
            # THD contiguous CP derives RoPE positions from packed sequence
            # metadata, as in _sft_train_mcore_thd.
            batch.pop("position_ids", None)

        assert batch["tokens"].numel() == total_tokens
        assert total_tokens % cp_size == 0, (
            f"DSv4 contiguous dynamic CP requires total_tokens={total_tokens} "
            f"divisible by local_cp_size={cp_size}"
        )
        local_tokens = total_tokens // cp_size
        # Dynamic CP uses equal contiguous token ranges; a range may cross
        # packed sample boundaries, which are described by cu_seqlens below.
        row_slice = slice(cp_rank * local_tokens, (cp_rank + 1) * local_tokens)

        if not rl:
            # Preserve the packed tensors before CP slicing for metrics/debug
            # paths, matching the output contract of _sft_train_mcore_thd.
            if self.config.training.online_train_dspark:
                # TODO: 此路径暂时还没支持
                assert False, "online_train_dspark is not supported for DSv4 mcore CP"
                full_input_ids = batch["tokens"].reshape(1, total_tokens)
            full_labels = batch["labels"].reshape(1, total_tokens)
            full_loss_mask = (full_labels != -100).float()
        for key in token_keys:
            assert batch[key].numel() == total_tokens, (
                f"DSv4 dynamic CP field {key} has {batch[key].numel()} rows, "
                f"expected {total_tokens}"
            )
            batch[key] = batch[key][row_slice].view(1, local_tokens).contiguous()
        if not rl:
            # Keep labels as the source of truth for ignored/padded tokens,
            # matching _sft_train_mcore_thd.
            batch["loss_mask"] = (batch["labels"] != -100).float()

        tp_size = parallel_state.get_tensor_model_parallel_group().size()
        assert local_tokens % tp_size == 0, (
            f"post-CP tokens ({local_tokens}) not aligned to tp_size={tp_size}"
        )

        cu_seqlens = batch["cu_seqlens"]
        cu_seqlens_padded = batch["cu_seqlens_padded"]
        # Megatron consumes real and padded boundaries separately: real
        # boundaries define valid attention rows, while padded boundaries
        # describe physical THD offsets after post-pack padding.
        packed_seq_params = McorePackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=int(batch["max_seqlen"].item()),
            max_seqlen_kv=int(batch["max_seqlen"].item()),
            local_cp_size=local_cp_size,
            cp_group=cp_group,
            cp_partition_mode="contiguous",
        )

        if rl:
            if "loss_mask" in batch:
                batch["mask"] = batch.pop("loss_mask")
            batch["target"] = batch.pop("labels")
            if batch.get("global_retention_ratio") is not None:
                batch["global_retention_ratio"] = batch["global_retention_ratio"].to(dev)
        else:
            if self.config.training.online_train_dspark:
                # TODO: 此路径暂时还没支持
                assert False, "online_train_dspark is not supported for DSv4 mcore CP"
                batch["full_input_ids"] = full_input_ids
            batch["full_labels"] = full_labels
            batch["full_loss_mask"] = full_loss_mask
            batch["full_packed_seq_params"] = packed_seq_params
        batch["cp_group"] = cp_group

        fwd_kwargs = {
            "input_ids": batch["tokens"],
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": packed_seq_params,
        }
        return batch, fwd_kwargs

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
        """Use contiguous THD slicing required by DSv4 dynamic CP."""
        return self._train_with_dynamic_cp_contiguous(
            batches,
            rl=False,
            pad_token_id=pad_token_id,
        )

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
        """Use contiguous THD slicing required by DSv4 dynamic CP."""
        return self._train_with_dynamic_cp_contiguous(batches, rl=True)

    @override
    def _sft_train_cp_chunk_data(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: None | torch.Tensor,
        attention_mask: None | torch.Tensor,
    ):
        cp_size = dist.get_world_size(mpu.get_context_parallel_group())
        if cp_size <= 1:
            return tokens, labels, loss_mask, position_ids, attention_mask

        cp_rank = mpu.get_context_parallel_rank()
        # SFT 路径无 packed_seq_params（BSHD），cp_chunk_data 第 5 个返回值 None 丢弃。
        tokens, labels, loss_mask, position_ids, _ = cp_chunk_data(
            cp_rank,
            cp_size,
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
        )

        # DSV4 HpModule path: attention_mask must be None.
        assert attention_mask is None

        return tokens, labels, loss_mask, position_ids, attention_mask

    @override
    def _rl_train_cp_chunk_data(
        self, tokens: torch.Tensor, position_ids: None | torch.Tensor,
        attention_mask: None | torch.Tensor
    ):
        cp_size = dist.get_world_size(mpu.get_context_parallel_group())
        if cp_size <= 1:
            return tokens, position_ids, attention_mask

        cp_rank = mpu.get_context_parallel_rank()
        # SFT 路径无 packed_seq_params（BSHD），cp_chunk_data 第 5 个返回值 None 丢弃。
        tokens, _, _, position_ids, _ = cp_chunk_data(
            cp_rank,
            cp_size,
            tokens=tokens,
            labels=None,
            loss_mask=None,
        )

        # DSV4 HpModule path: attention_mask must be None.
        # assert attention_mask is None
        attention_mask = None
        return tokens, position_ids, attention_mask

    @override
    def rl_train_cp_chunk_single_data(
        self,
        data: torch.Tensor,
    ):
        cp_size = dist.get_world_size(mpu.get_context_parallel_group())
        cp_rank = mpu.get_context_parallel_rank()
        if cp_size <= 1:
            return data

        local_data = cp_chunk_data(cp_rank, cp_size, tokens=data)[0]
        return local_data


class DeepseekV4DpoPrepareDataForwardLLM(DpoPrepareDataForwardLLM, DeepseekV4PrepareDataForwardLLM):
    """DSV4 DPO data preparation: ref_logprobs injection + contiguous CP slicing.

    MRO resolves ``sft_train`` to ``DpoPrepareDataForwardLLM`` (injects
    ``ref_logprobs``), which calls ``super().sft_train`` →
    ``DeepseekV4PrepareDataForwardLLM`` (BSHD/THD dispatch + contiguous CP).
    CP chunk overrides come from ``DeepseekV4PrepareDataForwardLLM``.
    """
    def __init__(self, config):
        super().__init__(config)
        is_mcore = config.training.training_backend == "mcore"
        if not is_mcore:
            assert not config.policy.ppo_pack_seq, "DPO + THD pack_seq not supported for DSV4"
        elif config.policy.dist_config.context_parallel_size > 1:
            assert getattr(config.policy, "forward_only_mbs", 1) == 1, (
                "DSv4 mcore THD DPO requires policy.forward_only_mbs=1 for "
                "reference log-prob computation"
            )

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
        """Use the DSv4 THD path directly for mcore contiguous CP DPO."""
        is_mcore = self.config.training.training_backend == "mcore"
        if is_mcore and mpu.get_context_parallel_world_size() > 1:
            return DeepseekV4PrepareDataForwardLLM.sft_train(
                self,
                batches,
                seq_len,
                pad_token_id,
                comput_attn_mask,
                pad_with_random_token,
                **kwargs,
            )
        return super().sft_train(
            batches,
            seq_len,
            pad_token_id,
            comput_attn_mask,
            pad_with_random_token,
            **kwargs,
        )


class DeepseekV4PostInitModel(PostInitModel):
    """Post-init model handler for DeepSeek-V4."""
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model):
        """Post-init the model.

        Parameters
        ----------
        model : torch.nn.Module
        """
        # freeze MoE TopKRouter weight
        if self.config.training.freeze_router_weight:
            freeze_router_weights(model)
        # freeze CSA Lightning Indexer (default): no KL / top-k grad, avoid WD shrink
        if self.config.training.freeze_csa_indexer:
            freeze_csa_indexer_params(model)
        # Load-only DSpark: keep mtp.* in ckpt but out of the optimizer.
        if (self.config.training.enable_dspark and not self.config.training.online_train_dspark):
            # 避免 wegiht decay 对 mtp 的影响。
            assert model.mtp is not None
            model.mtp.requires_grad_(False)
        # init buffers for per-step router correction bias updates
        # register forward hook to track router correction bias accum
        if not self.config.training.freeze_router_correction_bias:
            init_router_correction_bias_accumulators(model)
            register_router_correction_bias_accum_tracking_hook(model)


class DeepseekV4CheckpointContextFn(CheckpointContextFn):
    """torch.utils.checkpoint.checkpoint context_fn for DeepSeek-V4."""
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, *args, **kwargs):
        """torch.utils.checkpoint.checkpoint context_fn.

        Parameters
        ----------
        *args : any
        **kwargs : any

        Returns
        -------
        union[tuple[nullcontext, nullcontext], tuple[contextmanager, contextmanager]]
        """
        if self.config.training.freeze_router_correction_bias or not isinstance(
            self.config, (FinetuneConfig, RlConfig)
        ):
            return nullcontext(), nullcontext()
        return checkpoint_context_fn()


class DeepseekV4ResetRouterCorrectionBiasAccum(ResetRouterCorrectionBiasAccum):
    """Reset router correction bias accum for DeepSeek-V4."""
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model):
        """Reset router correction bias accum.

        Parameters
        ----------
        model : torch.nn.Module
        """
        reset_router_correction_bias_accum(model)


class DeepseekV4UpdateRouterCorrectionBias(UpdateRouterCorrectionBias):
    """Update router correction bias for DeepSeek-V4."""
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model, update_speed, use_abs_update):
        """Update router correction bias.

        Parameters
        ----------
        model : torch.nn.Module
        update_speed : float
        use_abs_update : bool

        Returns
        -------
        tuple[Optional[float], Optional[float]]
        """
        return update_router_correction_bias(
            model,
            update_speed,
            use_abs_update,
            dump_and_exit=self.config.debug.debug_dump_expert_token_counts,
            dump_path=self.config.debug.debug_dump_expert_token_counts_path,
        )
