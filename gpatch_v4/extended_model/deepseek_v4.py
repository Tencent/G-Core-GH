from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from transformers import DeepseekV4Config
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.extended_model.llm import PrepareDataForwardLLM
from gpatch_v4.models.deepseek_v4.cp import cp_chunk_data
from gpatch_v4.models.deepseek_v4.thd import pack_sequences


class DeepseekV4PrepareDataForwardLLM(PrepareDataForwardLLM):
    """DeepSeek-V4 SFT data preparation (HpModule, EP + CP).

    Two modes selected by ``config.training.pack_seq``:

    **BSHD** (``pack_seq=False``, default): pads + stacks into ``[B, S]``
    via the parent class, then overrides CP slicing to use contiguous
    ``cp_chunk_data`` instead of Megatron-style zigzag.

    **THD** (``pack_seq=True``): packs variable-length samples into a
    single ``[1, T]`` sequence via :func:`pack_sequences`, builds
    :class:`PackedSeqParams`, and CP-slices the layout.
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
        if not self.config.policy.ppo_pack_seq:
            return super().sft_train(
                batches,
                seq_len,
                pad_token_id,
                comput_attn_mask=comput_attn_mask,
                pad_with_random_token=pad_with_random_token,
                **kwargs,
            )
        return self._sft_train_thd(batches, seq_len, pad_token_id)

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
    def _rl_train_cp_chunk_single_data(
        self,
        data: torch.Tensor,
    ):
        cp_size = dist.get_world_size(mpu.get_context_parallel_group())
        cp_rank = mpu.get_context_parallel_rank()
        if cp_size <= 1:
            return data

        local_data = cp_chunk_data(cp_rank, cp_size, tokens=data)[0]
        return local_data
