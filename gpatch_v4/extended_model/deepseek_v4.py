import torch
import torch.distributed as dist
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.extended_model.llm import PrepareDataForwardLLM
from gpatch_v4.models.deepseek_v4.cp import cp_chunk_data


class DeepseekV4PrepareDataForwardLLM(PrepareDataForwardLLM):
    """DeepSeek-V4 SFT data preparation (HpModule, EP + CP).

    Overrides :meth:`_sft_train_cp_chunk_data` to follow the HpModule CP
    contract (different from Megatron-style zigzag), under ``cp_size > 1``:

    - ``tokens`` / ``labels`` / ``loss_mask`` / ``position_ids``: all sliced
      contiguously to ``[B, s_local]`` in one call to
      :func:`gpatch_v4.models.deepseek_v4.cp.cp_chunk_data`; ``position_ids``
      is rebuilt as ``arange(s_local) + cp_rank * s_local``. The model
      forward consumes them directly without re-slicing; the invariant
      ``loss_mask.shape == labels.shape`` is preserved by construction.
    - ``attention_mask``: must be ``None`` for DSV4 (asserted below).
    """
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
