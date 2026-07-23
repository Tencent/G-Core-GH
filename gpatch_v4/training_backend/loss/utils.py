from typing import Dict, Optional

import torch

from gpatch_v4.utils import masked_mean


def agg(
    values: torch.Tensor,
    mask: torch.Tensor,
    calculate_per_token_loss: bool = False,
    sample_mask: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-token values into ``(bwd_sum, bwd_count)`` for one mode.
    In order to support fine-grained agentic sample custom weight, we support
    token-level weights. You can pre-compute the weights in rollout phase to
    customize various loss, like per-prompt / per-traj / per-sample etc.

    ``token_weights`` (if set) only reweights the numerator; the denominator
    stays mask / sample counts.
    """
    if token_weights is not None:
        b, s = values.shape[0], values.shape[-1]
        assert token_weights.shape == (b, 1) or token_weights.shape == (b, s), (
            f"token_weights shape {tuple(token_weights.shape)} must be None, "
            f"({b}, 1), or ({b}, {s})"
        )
        values = values * token_weights

    if calculate_per_token_loss:
        bwd_sum = (values * mask).sum()
        bwd_count = mask.sum()
        return bwd_sum, bwd_count

    per_seq_mean = (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    if sample_mask is not None:
        per_seq_mean = per_seq_mean * sample_mask
        bwd_count = sample_mask.sum()
    else:
        bwd_count = mask.new_tensor(float(mask.shape[0]))
    return per_seq_mean.sum(), bwd_count


def compute_clip_metrics(
    ratios: torch.Tensor,
    ratios_clamped: torch.Tensor,
    clip_ratio_high: float,
    clip_ratio_low: float,
    loss1: torch.Tensor,
    loss2: torch.Tensor,
    advantages: torch.Tensor,
    clip_max_loss: torch.Tensor,
    response_mask: torch.Tensor,
    numel: torch.Tensor,
    loss3: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    mask_bool = response_mask.bool()
    if ratios.dim() == 3:
        mask_bool = mask_bool.unsqueeze(-1)
    is_upper_clamped = (ratios > 1.0 + clip_ratio_high) & mask_bool
    is_lower_clamped = (ratios < 1.0 - clip_ratio_low) & mask_bool
    loss2_gt_loss1 = loss2 > loss1
    ppo_ratio_clamped = masked_mean(ratios_clamped.detach(), response_mask)

    metrics = {
        "ppo_ratio_clamped":
            torch.stack([ppo_ratio_clamped * numel, numel]),
        "ppo_ratio_clamped_upper_frac":
            torch.stack([is_upper_clamped.sum().float(), numel]),
        "ppo_ratio_clamped_lower_frac":
            torch.stack([is_lower_clamped.sum().float(), numel]),
        "ppo_ratio_clamped_effective_upper_frac":
            torch.stack([(is_upper_clamped & loss2_gt_loss1).sum().float(), numel]),
        "ppo_ratio_clamped_effective_lower_frac":
            torch.stack([(is_lower_clamped & loss2_gt_loss1).sum().float(), numel]),
    }
    if loss3 is not None:
        dual_clip_active = ((advantages < 0) & (clip_max_loss > loss3) & mask_bool).sum().float()
    else:
        dual_clip_active = torch.zeros(1, device=ratios.device).squeeze()
    metrics["ppo_dual_clip_frac"] = torch.stack([dual_clip_active, numel])
    return metrics
