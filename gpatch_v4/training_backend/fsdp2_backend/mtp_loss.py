from __future__ import annotations

import torch
import torch.distributed as dist

try:
    from gpatch_v4.models.deepseek_v4.mtp import mtp_roll_tensor_cp
except ImportError:
    mtp_roll_tensor_cp = None


def calculate_mtp_loss(
    *,
    mtp_per_depth_h: list[torch.Tensor],
    labels: torch.Tensor,
    lm_head,
    loss_fct,
    loss_mask: torch.Tensor | None = None,
    cp_group=None,
    scaling_factor: float = 0.1,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Compute per-depth MTP CE using shared lm_head.

    Notes
    -----
    `labels` is the shifted next-token label in gpatch_v4 SFT path.
    For depth `k`, MTP predicts one additional token into the future, so we
    roll labels by `-(k+1)` and mask the trailing `k+1` positions.
    """
    num_depth = len(mtp_per_depth_h)
    if num_depth == 0:
        z = labels.new_zeros((), dtype=torch.float32)
        return z, [], [], []

    bsz, seq = labels.shape
    cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else dist.get_rank(cp_group)

    per_depth_loss: list[torch.Tensor] = []
    per_depth_num: list[torch.Tensor] = []
    per_depth_den: list[torch.Tensor] = []
    rolled_labels = labels
    rolled_mask = loss_mask
    for depth, hidden in enumerate(mtp_per_depth_h):
        logits = lm_head(hidden)
        rolled_labels, _ = mtp_roll_tensor_cp(
            rolled_labels,
            shifts=-1,
            dim=1,
            cp_group=cp_group,
        )
        rolled_labels = rolled_labels.clone()
        trail = depth + 1
        mask_global_tailing(rolled_labels, trail, ignore_index, seq, cp_size, cp_rank)

        depth_loss = loss_fct(
            logits.view(-1, logits.shape[-1]),
            rolled_labels.reshape(-1),
        ).view(bsz, seq)

        if rolled_mask is not None:
            rolled_mask, _ = mtp_roll_tensor_cp(
                rolled_mask,
                shifts=-1,
                dim=1,
                cp_group=cp_group,
            )
            rolled_mask = rolled_mask.clone()
            mask_global_tailing(rolled_mask, trail, 0, seq, cp_size, cp_rank)
            valid_mask = (rolled_labels != ignore_index).to(depth_loss.dtype)
            depth_loss = depth_loss * rolled_mask.to(depth_loss.dtype) * valid_mask
            num = depth_loss.sum()
            den = valid_mask.mul(rolled_mask.to(valid_mask.dtype)).sum().clamp_min(1.0)
            depth_loss = num / den
        else:
            valid = (rolled_labels != ignore_index).to(depth_loss.dtype)
            num = (depth_loss * valid).sum()
            den = valid.sum().clamp_min(1.0)
            depth_loss = num / den

        per_depth_loss.append(depth_loss)
        per_depth_num.append(num)
        per_depth_den.append(den)

    total = torch.stack(per_depth_loss).sum() * (float(scaling_factor) / float(num_depth))
    return total, per_depth_loss, per_depth_num, per_depth_den


def mask_global_tailing(
    x: torch.Tensor, trail: int, fill_value: int | float, local_seq: int, cp_size: int, cp_rank: int
) -> None:
    if trail <= 0:
        return
    if cp_size == 1:
        x[:, -trail:] = fill_value
        return
    s_full = local_seq * cp_size
    g0 = s_full - trail
    local_from = max(0, g0 - cp_rank * local_seq)
    if local_from < local_seq:
        x[:, local_from:] = fill_value
