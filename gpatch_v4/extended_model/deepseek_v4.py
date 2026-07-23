from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from transformers import DeepseekV4Config
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.configs.config import FinetuneConfig, RlConfig
from gpatch_v4.extended_model.base import (
    CheckpointContextFn,
    PostInitModel,
    ResetRouterCorrectionBiasAccum,
    UpdateRouterCorrectionBias,
)
from gpatch_v4.extended_model.llm import DpoPrepareDataForwardLLM, PrepareDataForwardLLM
from gpatch_v4.models.deepseek_v4.cp import cp_chunk_data
from gpatch_v4.models.deepseek_v4.freeze_update_router import (
    checkpoint_context_fn,
    freeze_router_weights,
    init_router_correction_bias_accumulators,
    register_router_correction_bias_accum_tracking_hook,
    reset_router_correction_bias_accum,
    update_router_correction_bias,
)
from gpatch_v4.models.deepseek_v4.thd import pack_sequences


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
        if self.config.policy.ppo_pack_seq:
            self._model_config = DeepseekV4Config.from_pretrained(config.policy.hf_model_path, )
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
            assert self.config.policy.ppo_pack_seq, "dsv4 mcore cp require thd format"

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

        device = ids_list[0].device
        seqlens = [t.shape[0] for t in ids_list]
        # Pad each sample to the next cp_size multiple so that T % cp_size == 0.
        padded_seqlens = [((s + cp_size - 1) // cp_size) * cp_size for s in seqlens]
        T = sum(padded_seqlens)

        # at least sliding window * cp
        sliding_window = self._model_config.sliding_window
        T = max(T, sliding_window * cp_size)

        packed_ids = torch.full((1, T), pad_token_id, dtype=torch.long, device=device)
        packed_labels = torch.full((1, T), -100, dtype=torch.long, device=device)

        # Build cu_seqlens on CPU first to avoid per-element CPU→GPU sync, then transfer once.
        cu_seqlens_cpu = torch.zeros(len(ids_list) + 1, dtype=torch.int32)
        offset = 0
        for i, (s, s_pad) in enumerate(zip(seqlens, padded_seqlens)):
            packed_ids[0, offset:offset + s] = ids_list[i]
            packed_labels[0, offset:offset + s] = labels_list[i]
            offset += s_pad
            cu_seqlens_cpu[i + 1] = offset
        cu_seqlens = cu_seqlens_cpu.to(device)

        psp = McorePackedSeqParams(
            qkv_format="thd",
            cp_partition_mode="contiguous",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(padded_seqlens),
            max_seqlen_kv=max(padded_seqlens),
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
        fwd_kwargs: Dict[str, Any] = {
            "input_ids": local_ids,
            "position_ids": local_pos,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": local_psp,
        }
        return batch_out, fwd_kwargs

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
        assert not config.policy.ppo_pack_seq, ("DPO + THD pack_seq not supported for DSV4")


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
        return update_router_correction_bias(model, update_speed, use_abs_update)
