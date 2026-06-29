from __future__ import annotations

import torch
import torch.distributed as dist

from gpatch_v4.models.deepseek_v4.mtp import mtp_roll_tensor

# TODO: 这个文件需要重构，目前是 deepseek v4 强绑定的。


def calculate_mtp_loss(
    *,
    mtp_per_depth_h: list[torch.Tensor],
    labels: torch.Tensor,
    lm_head,
    loss_fct,
    loss_mask: torch.Tensor | None = None,
    cp_group=None,
    packed_seq_params=None,
    ignore_index: int = -100,
) -> list[torch.Tensor]:
    """Compute per-depth MTP masked-loss numerators using shared lm_head.

    Returns the **local** (this CP rank's) per-depth loss numerator
    ``sum(ce * mask)``. The caller is responsible for dividing by a global
    denominator (see ``mtp_per_depth_valid_count``).

    Parameters
    ----------
    labels : torch.Tensor
        Shape ``[bsz, s_local]``; shifted next-token labels (possibly CP-chunked).
    packed_seq_params : PackedSeqParams, optional
        When provided (THD mode), roll is segment-aware.

    Returns
    -------
    list[torch.Tensor]
        Per-depth masked loss numerators (length = ``len(mtp_per_depth_h)``).
    """
    num_depth = len(mtp_per_depth_h)
    if num_depth == 0:
        return []

    bsz, seq = labels.shape
    cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else dist.get_rank(cp_group)

    per_depth_num: list[torch.Tensor] = []
    rolled_labels = labels
    roll_kwargs = dict(cp_group=cp_group, packed_seq_params=packed_seq_params)
    rolled_mask = loss_mask
    for depth, hidden in enumerate(mtp_per_depth_h):
        logits = lm_head(hidden)
        trail = depth + 1
        rolled_labels = mtp_roll_tensor(
            rolled_labels,
            **roll_kwargs,
            trail=trail,
            fill_value=ignore_index,
        ).clone()

        depth_loss = loss_fct(
            logits.view(-1, logits.shape[-1]),
            rolled_labels.reshape(-1),
        ).view(bsz, seq)

        if rolled_mask is not None:
            rolled_mask = mtp_roll_tensor(
                rolled_mask,
                **roll_kwargs,
                trail=trail,
                fill_value=0,
            ).clone()
            valid_mask = (rolled_labels != ignore_index).to(depth_loss.dtype)
            dloss = (depth_loss * rolled_mask.to(depth_loss.dtype) * valid_mask).sum()
        else:
            valid = (rolled_labels != ignore_index).to(depth_loss.dtype)
            dloss = (depth_loss * valid).sum()

        per_depth_num.append(dloss)

    return per_depth_num


def mtp_per_depth_valid_count(
    labels_full: torch.Tensor,
    loss_mask_full: torch.Tensor,
    num_depth: int,
    cp_size: int,
    cp_rank: int,
    packed_seq_params=None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """This CP rank's per-depth MTP valid-token count, from the FULL labels.

    For BSHD: plain left-roll on the full sequence + contiguous CP chunk.
    For THD: segment-aware roll via ``mtp_roll_tensor`` on the full packed
    labels (no CP exchange needed since we operate on the unchunked tensor).

    Parameters
    ----------
    labels_full : torch.Tensor
        Shape ``[bsz, s_full]``; shifted next-token labels before CP chunking.
    loss_mask_full : torch.Tensor
        Shape ``[bsz, s_full]``; loss mask before CP chunking.
    packed_seq_params : PackedSeqParams, optional
        Packing metadata for THD mode. Either global or CP-sliced works
        because only ``cu_seqlens_q_padded`` is used, which stays global
        across CP chunks.
    """
    _, s_full = labels_full.shape
    assert s_full % cp_size == 0, f"{s_full=} not divisible by {cp_size=}"
    s_local = s_full // cp_size
    lo, hi = cp_rank * s_local, (cp_rank + 1) * s_local

    counts: list[torch.Tensor] = []
    rolled_labels = labels_full
    rolled_mask = loss_mask_full
    for depth in range(num_depth):
        trail = depth + 1
        rolled_labels = mtp_roll_tensor(
            rolled_labels,
            packed_seq_params=packed_seq_params,
            trail=trail,
            fill_value=ignore_index,
        )
        rolled_mask = mtp_roll_tensor(
            rolled_mask,
            packed_seq_params=packed_seq_params,
            trail=trail,
            fill_value=0,
        )
        valid = (rolled_labels != ignore_index) & (rolled_mask != 0)
        counts.append(valid[:, lo:hi].sum())
    return torch.stack(counts)
