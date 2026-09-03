# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Contiguous context parallelism for Qwen3.8-Flash-Next."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

__all__ = [
    "Qwen4ExpCPContext",
    "Qwen4ExpPackedSeqParams",
    "qwen4_exp_cp_all_gather",
    "qwen4_exp_cp_chunk_data",
    "qwen4_exp_cp_left_halo",
    "qwen4_exp_pack_sequences",
]


@dataclass(frozen=True)
class Qwen4ExpPackedSeqParams:
    """Global layout metadata for one Qwen packed row.

    Parameters
    ----------
    cu_seqlens_q : torch.Tensor
        Real document boundaries. Documents are physically contiguous.
    cu_seqlens_q_padded : torch.Tensor
        Physical boundaries including the final CP/QSA alignment tail.
    max_seqlen_q, total_seqlen : int
        Largest physical segment and total physical row length.
    qkv_format : str
        Always ``"thd"``.
    """

    cu_seqlens_q: torch.Tensor
    cu_seqlens_q_padded: torch.Tensor
    max_seqlen_q: int
    total_seqlen: int
    qkv_format: str = "thd"

    def __post_init__(self) -> None:
        if self.cu_seqlens_q.ndim != 1 or self.cu_seqlens_q.numel() < 2:
            raise ValueError("cu_seqlens_q must be a rank-1 document-boundary tensor")
        if self.cu_seqlens_q.shape != self.cu_seqlens_q_padded.shape:
            raise ValueError("real and padded document boundaries must have the same shape")
        if self.cu_seqlens_q.dtype != torch.long or self.cu_seqlens_q_padded.dtype != torch.long:
            raise ValueError("packed document boundaries must use torch.long")
        if self.cu_seqlens_q.device != self.cu_seqlens_q_padded.device:
            raise ValueError("real and padded document boundaries must be on the same device")
        if self.qkv_format != "thd":
            raise ValueError(f"qkv_format must be 'thd', got {self.qkv_format!r}")
        if (
            int(self.cu_seqlens_q[0]) != 0
            or bool((self.cu_seqlens_q.diff() <= 0).any())
            or bool((self.cu_seqlens_q_padded.diff() <= 0).any())
            or not bool(torch.equal(self.cu_seqlens_q[:-1], self.cu_seqlens_q_padded[:-1]))
            or int(self.cu_seqlens_q[-1]) > self.total_seqlen
            or int(self.cu_seqlens_q_padded[-1]) != self.total_seqlen
        ):
            raise ValueError("invalid Qwen packed document boundaries")
        padded_lengths = self.cu_seqlens_q_padded.diff()
        if self.max_seqlen_q != int(padded_lengths.max()):
            raise ValueError("max_seqlen_q does not match the padded document layout")


def qwen4_exp_pack_sequences(
    input_ids: list[torch.Tensor],
    labels: list[torch.Tensor],
    *,
    cp_size: int,
    pad_multiple: int,
    pad_token_id: int,
    label_ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Qwen4ExpPackedSeqParams]:
    """Pack shifted SFT documents contiguously and add one alignment tail."""
    if not input_ids:
        raise ValueError("input_ids must be non-empty")
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must contain the same number of documents")
    if cp_size <= 0 or pad_multiple <= 0:
        raise ValueError(
            f"cp_size and pad_multiple must be positive, got {cp_size} and {pad_multiple}"
        )

    device = input_ids[0].device
    lengths = []
    for index, (document_ids, document_labels) in enumerate(zip(input_ids, labels)):
        if document_ids.ndim != 1 or document_labels.shape != document_ids.shape:
            raise ValueError(
                f"packed document {index} IDs/labels must be matching 1-D tensors, got "
                f"{tuple(document_ids.shape)} and {tuple(document_labels.shape)}"
            )
        if document_ids.numel() == 0:
            raise ValueError(f"packed document {index} is empty after next-token shifting")
        if document_ids.device != device or document_labels.device != device:
            raise ValueError("all packed IDs and labels must be on the same device")
        lengths.append(document_ids.numel())

    boundaries_host = [0]
    for length in lengths:
        boundaries_host.append(boundaries_host[-1] + length)
    total_real = boundaries_host[-1]
    total_alignment = cp_size * pad_multiple
    total_seqlen = (
        (total_real + total_alignment - 1) // total_alignment
    ) * total_alignment

    packed_ids = torch.full(
        (1, total_seqlen), pad_token_id, dtype=torch.long, device=device
    )
    packed_labels = torch.full(
        (1, total_seqlen), label_ignore_index, dtype=torch.long, device=device
    )
    position_ids = torch.zeros((1, total_seqlen), dtype=torch.long, device=device)
    for start, length, document_ids, document_labels in zip(
        boundaries_host[:-1], lengths, input_ids, labels
    ):
        end = start + length
        packed_ids[0, start:end] = document_ids.long()
        packed_labels[0, start:end] = document_labels.long()
        position_ids[0, start:end] = torch.arange(length, device=device)

    boundaries = torch.tensor(boundaries_host, dtype=torch.long, device=device)
    padded_boundaries = boundaries.clone()
    padded_boundaries[-1] = total_seqlen
    packed_seq_params = Qwen4ExpPackedSeqParams(
        cu_seqlens_q=boundaries,
        cu_seqlens_q_padded=padded_boundaries,
        max_seqlen_q=int(padded_boundaries.diff().max()),
        total_seqlen=total_seqlen,
    )
    return packed_ids, position_ids, packed_labels, packed_seq_params


@dataclass(frozen=True)
class Qwen4ExpCPContext:
    """Replicated metadata for one contiguous context-parallel sequence.

    Parameters
    ----------
    group : dist.ProcessGroup or None
        CP process group. ``None`` is valid only for the size-one identity path.
    rank, size : int
        Rank and world size inside ``group``.
    global_input_ids, global_padding_mask : torch.Tensor
        Replicated ``[batch, global_sequence]`` raw IDs and right-padding mask.
    local_sequence_start, local_sequence_length : int
        This rank's contiguous global interval.
    """

    group: Optional[dist.ProcessGroup]
    rank: int
    size: int
    global_input_ids: torch.Tensor
    global_padding_mask: torch.Tensor
    local_sequence_start: int
    local_sequence_length: int
    global_cu_seqlens: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.size <= 0 or not 0 <= self.rank < self.size:
            raise ValueError(f"invalid CP rank/size: rank={self.rank}, size={self.size}")
        if self.size > 1 and self.group is None:
            raise ValueError("CP size greater than one requires a process group")
        if self.global_input_ids.ndim != 2 or self.global_input_ids.dtype not in (
            torch.int32,
            torch.int64,
            torch.long,
        ):
            raise ValueError(
                "global_input_ids must be an int32/int64 [batch, sequence] tensor, "
                f"got shape={tuple(self.global_input_ids.shape)}, "
                f"dtype={self.global_input_ids.dtype}"
            )
        if self.global_padding_mask.shape != self.global_input_ids.shape:
            raise ValueError(
                "global padding mask and input IDs must have the same shape, got "
                f"{tuple(self.global_padding_mask.shape)} and "
                f"{tuple(self.global_input_ids.shape)}"
            )
        if self.global_padding_mask.dtype != torch.bool:
            raise ValueError(
                f"global_padding_mask must be bool, got {self.global_padding_mask.dtype}"
            )
        if self.global_padding_mask.device != self.global_input_ids.device:
            raise ValueError("global padding mask and input IDs must be on the same device")
        if self.local_sequence_length <= 0:
            raise ValueError(
                f"local_sequence_length must be positive, got {self.local_sequence_length}"
            )
        expected_global_length = self.local_sequence_length * self.size
        if self.global_input_ids.shape[1] != expected_global_length:
            raise ValueError(
                "global sequence length must equal local_sequence_length * CP size, "
                f"got global={self.global_input_ids.shape[1]}, "
                f"local={self.local_sequence_length}, size={self.size}"
            )
        expected_start = self.rank * self.local_sequence_length
        if self.local_sequence_start != expected_start:
            raise ValueError(
                "CP shards must use contiguous rank order, "
                f"got start={self.local_sequence_start}, expected={expected_start}"
            )
        if self.global_cu_seqlens is not None:
            boundaries = self.global_cu_seqlens
            if boundaries.ndim != 1 or boundaries.numel() < 2:
                raise ValueError("global_cu_seqlens must be a rank-1 boundary tensor")
            if self.global_input_ids.shape[0] != 1:
                raise ValueError("packed THD requires a batch of one row")
            if (
                int(boundaries[0]) != 0
                or int(boundaries[-1]) > expected_global_length
                or bool((boundaries.diff() <= 0).any())
            ):
                raise ValueError(
                    "packed boundaries must start at zero, increase strictly, and end "
                    f"by the global length {expected_global_length}"
                )
            expected_padding = torch.arange(
                expected_global_length, device=boundaries.device
            ).unsqueeze(0) >= int(boundaries[-1])
            if not bool(torch.equal(self.global_padding_mask, expected_padding)):
                raise ValueError("packed padding must be one right-tail after the final document")

    @property
    def global_sequence_length(self) -> int:
        return self.global_input_ids.shape[1]

    @property
    def local_sequence_end(self) -> int:
        return self.local_sequence_start + self.local_sequence_length

    @property
    def global_sequence_lengths(self) -> torch.Tensor:
        return self.global_padding_mask.logical_not().sum(dim=-1, dtype=torch.long)

    @property
    def local_attention_mask(self) -> torch.Tensor:
        return self.global_padding_mask[
            :, self.local_sequence_start : self.local_sequence_end
        ].logical_not()

    @property
    def global_segment_boundaries(self) -> torch.Tensor:
        """Return real document boundaries plus an isolated alignment tail."""
        if self.global_cu_seqlens is None:
            return torch.tensor(
                [0, self.global_sequence_length],
                dtype=torch.long,
                device=self.global_input_ids.device,
            )
        if int(self.global_cu_seqlens[-1]) == self.global_sequence_length:
            return self.global_cu_seqlens
        return torch.cat(
            (
                self.global_cu_seqlens,
                self.global_cu_seqlens.new_tensor([self.global_sequence_length]),
            )
        )


def qwen4_exp_cp_chunk_data(
    cp_rank: int,
    cp_size: int,
    cp_group: Optional[dist.ProcessGroup],
    *,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    position_ids: torch.Tensor,
    global_padding_mask: torch.Tensor,
    global_cu_seqlens: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Qwen4ExpCPContext,
]:
    """Contiguously slice SFT tensors and retain the replicated PLE metadata."""
    if tokens.ndim != 2:
        raise ValueError(f"CP tokens must be [batch, sequence], got {tuple(tokens.shape)}")
    for name, tensor in (
        ("labels", labels),
        ("loss_mask", loss_mask),
        ("position_ids", position_ids),
        ("global_padding_mask", global_padding_mask),
    ):
        if tensor.shape != tokens.shape:
            raise ValueError(
                f"CP {name} shape {tuple(tensor.shape)} does not match tokens "
                f"{tuple(tokens.shape)}"
            )
    if global_padding_mask.dtype != torch.bool:
        raise ValueError(
            f"global_padding_mask must be bool, got {global_padding_mask.dtype}"
        )
    if tokens.shape[1] % cp_size != 0:
        raise ValueError(
            f"sequence length {tokens.shape[1]} must be divisible by CP size {cp_size}"
        )
    local_sequence_length = tokens.shape[1] // cp_size
    valid = global_padding_mask.logical_not()
    lengths = valid.sum(dim=-1, dtype=torch.long)
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    if not bool(torch.equal(valid, positions < lengths.unsqueeze(1))):
        raise NotImplementedError(
            "qwen4_exp CP supports right-tail padding only; left and interior padding "
            "require THD segment metadata"
        )
    start = cp_rank * local_sequence_length
    sequence_slice = slice(start, start + local_sequence_length)
    context = Qwen4ExpCPContext(
        group=cp_group,
        rank=cp_rank,
        size=cp_size,
        global_input_ids=tokens,
        global_padding_mask=global_padding_mask,
        local_sequence_start=start,
        local_sequence_length=local_sequence_length,
        global_cu_seqlens=global_cu_seqlens,
    )
    return (
        tokens[:, sequence_slice].contiguous(),
        labels[:, sequence_slice].contiguous(),
        loss_mask[:, sequence_slice].contiguous(),
        position_ids[:, sequence_slice].contiguous(),
        context,
    )


def qwen4_exp_cp_all_gather(
    tensor: torch.Tensor,
    context: Qwen4ExpCPContext,
    *,
    sequence_dim: int,
    differentiable: bool,
) -> torch.Tensor:
    """Gather equal sequence shards in contiguous CP rank order."""
    if context.size == 1:
        return tensor
    if context.group is None:
        raise RuntimeError("CP context is missing its process group")
    if differentiable:
        parts = differentiable_all_gather(tensor.contiguous(), group=context.group)
    else:
        parts = [torch.empty_like(tensor) for _ in range(context.size)]
        dist.all_gather(parts, tensor.contiguous(), group=context.group)
    return torch.cat(tuple(parts), dim=sequence_dim)


def qwen4_exp_cp_left_halo(
    tensor: torch.Tensor,
    context: Qwen4ExpCPContext,
    *,
    history: int,
) -> torch.Tensor:
    """Return the differentiable causal history immediately before a local shard."""
    if tensor.ndim != 3 or tensor.shape[1] != context.local_sequence_length:
        raise ValueError(
            "CP halo input must be [batch, local_sequence, channels], got "
            f"{tuple(tensor.shape)}"
        )
    if history < 0:
        raise ValueError(f"history must be non-negative, got {history}")
    if history == 0:
        return tensor[:, :0]
    if context.size == 1:
        return tensor.new_zeros((tensor.shape[0], history, tensor.shape[2]))

    tail_length = min(history, context.local_sequence_length)
    gathered_tails = qwen4_exp_cp_all_gather(
        tensor[:, -tail_length:],
        context,
        sequence_dim=1,
        differentiable=True,
    ).unflatten(1, (context.size, tail_length))
    preceding = gathered_tails[:, : context.rank].flatten(1, 2)[:, -history:]
    missing = history - preceding.shape[1]
    if missing:
        preceding = torch.cat(
            (tensor.new_zeros((tensor.shape[0], missing, tensor.shape[2])), preceding),
            dim=1,
        )

    # Every rank must retain the collective in its backward graph.
    return preceding + gathered_tails[:, :, :0].sum()
