from typing import Dict, Optional

import torch

from gpatch_v4.utils.training_utils import masked_mean, masked_sum, masked_sum_per_seq


def agg(
    values: torch.Tensor,
    mask: torch.Tensor,
    calculate_per_token_loss: bool = False,
    sample_mask: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
    cu_seqlens_padded: Optional[torch.Tensor] = None,
    local_cp_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-token values into ``(bwd_sum, bwd_count)`` for one mode.

    New loss expects plain 2D ``[B, S]`` tensors. Dyn-CP / THD packs must be
    reconstructed and response-padded before entering loss; pass
    ``cu_seqlens_padded=None`` and ``local_cp_size=1``.

    In order to support fine-grained agentic sample custom weight, we support
    token-level weights. You can pre-compute the weights in rollout phase to
    customize various loss, like per-prompt / per-traj / per-sample etc.

    ``token_weights`` (if set) only reweights the numerator; the denominator
    stays mask / sample counts.
    """
    assert cu_seqlens_padded is None, (
        "new loss agg expects [B, S] tensors; convert THD/dyn-CP to "
        "response-padded sequence before loss (cu_seqlens_padded must be None)"
    )
    assert local_cp_size == 1, (
        f"new loss agg expects local_cp_size=1 after response-pad, got {local_cp_size}"
    )

    if token_weights is not None:
        b, s = values.shape[0], values.shape[-1]
        assert token_weights.shape == (b, 1) or token_weights.shape == (b, s), (
            f"token_weights shape {tuple(token_weights.shape)} must be None, "
            f"({b}, 1), or ({b}, {s})"
        )
        values = values * token_weights

    if calculate_per_token_loss:
        return masked_sum(values, mask), mask.sum()

    bwd_sum = masked_sum_per_seq(values, mask, sample_mask)
    bwd_count = (
        sample_mask.sum() if sample_mask is not None else mask.new_tensor(float(mask.shape[0]))
    )
    return bwd_sum, bwd_count


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
