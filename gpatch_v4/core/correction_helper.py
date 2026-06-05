from typing import Dict, Optional, Tuple

import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils import masked_mean, masked_statistic, masked_sum


def masked_sum_expand(x: torch.Tensor, mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    result = (torch.where(mask.bool(), x, 0)).sum(dim=-1, keepdim=True)
    return result.expand_as(x) if expand else result


def masked_mean_expand(x: torch.Tensor, mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    result = masked_sum_expand(x, mask) / torch.clamp_min(mask.sum(dim=-1, keepdim=True), 1)
    return result.expand_as(x) if expand else result


def calculate_veto_mask(
    log_ratio: torch.Tensor,
    mask: torch.Tensor,
    veto_threshold: Optional[float],
    metrics: Dict[str, list[torch.Tensor]],
) -> torch.Tensor:
    if veto_threshold is None:
        return torch.ones_like(log_ratio)
    log_veto_threshold = torch.log(torch.tensor(veto_threshold, device=log_ratio.device))
    # For each sequence, if it has any catastrophic tokens, return 0 for the sequence
    catastrophic_tokens = ((log_ratio < log_veto_threshold)) & mask.bool()
    has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)
    veto_mask = (~has_catastrophic).float().expand_as(log_ratio)

    metrics["catastrophic_fraction"] = (catastrophic_tokens * mask
                                       ).sum().float() / torch.clamp_min(mask.sum().float(), 1)
    return veto_mask


def truncate_mode(
    weights: torch.Tensor,
    mask: torch.Tensor,
    metrics: Dict[str, list[torch.Tensor]],
    upper_bound: float,
) -> torch.Tensor:
    assert upper_bound is not None
    metrics["truncate_fraction"] = ((weights > upper_bound) *
                                    mask).sum().float() / torch.clamp_min(mask.sum().float(), 1)
    return weights.clamp(0, upper_bound) * mask


def clip_mode(
    weights: torch.Tensor,
    mask: torch.Tensor,
    metrics: Dict[str, list[torch.Tensor]],
    lower_bound: float,
    upper_bound: float,
) -> torch.Tensor:
    assert lower_bound is not None and upper_bound is not None and lower_bound < upper_bound
    metrics["clip_fraction_low"] = ((weights < lower_bound) *
                                    mask).sum().float() / torch.clamp_min(mask.sum().float(), 1)
    metrics["clip_fraction_high"] = ((weights > upper_bound) *
                                     mask).sum().float() / torch.clamp_min(mask.sum().float(), 1)
    return weights.clamp(lower_bound, upper_bound) * mask


def mask_mode(
    weights: torch.Tensor,
    mask: torch.Tensor,
    metrics: Dict[str, list[torch.Tensor]],
    lower_bound: float,
    upper_bound: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert lower_bound is not None and upper_bound is not None and lower_bound < upper_bound
    metrics["clip_fraction_low"] = ((weights < lower_bound) *
                                    mask).sum().float() / torch.clamp_min(mask.sum().float(), 1)
    metrics["clip_fraction_high"] = ((weights > upper_bound) *
                                     mask).sum().float() / torch.clamp_min(mask.sum().float(), 1)

    in_range = (weights >= lower_bound) & (weights <= upper_bound)
    modified_mask = mask * in_range.float()
    # Zero out padding in weights but preserve values at non-rejected positions
    weights = weights * mask
    return weights, modified_mask


def icepop_mode(
    weights: torch.Tensor,
    mask: torch.Tensor,
    metrics: Dict[str, list[torch.Tensor]],
    lower_bound: float,
    upper_bound: float,
) -> torch.Tensor:
    """IcePop: zero IS weights outside [lower, upper] while keeping mask unchanged.

    Unlike ``mask_mode`` which modifies the aggregation mask (so the denominator
    only counts in-range tokens), IcePop keeps the original mask intact and
    instead sets out-of-range weights to zero.  This means out-of-range tokens
    still count in the denominator of ``masked_mean``, effectively *diluting*
    the loss when many tokens are rejected.

    Reference: verl ``compute_rollout_correction_weights`` with IcePop threshold.
    """
    assert lower_bound is not None and upper_bound is not None and lower_bound < upper_bound
    oob_low = (weights < lower_bound) * mask
    oob_high = (weights > upper_bound) * mask
    mask_sum = torch.clamp_min(mask.sum().float(), 1)
    metrics["clip_fraction_low"] = oob_low.sum().float() / mask_sum
    metrics["clip_fraction_high"] = oob_high.sum().float() / mask_sum
    metrics["oob_ratio"] = (oob_low.bool() | oob_high.bool()).sum().float() / mask_sum

    in_range = (weights >= lower_bound) & (weights <= upper_bound)
    weights = torch.where(in_range, weights, torch.zeros_like(weights)) * mask
    return weights


def compute_off_policy_correction_weights(
    enable_off_policy_correction: bool,
    config: RlConfig,
    prev_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the importance sampling (IS) weights and metrics between the inference and training engine.
    Args:
        enable_off_policy_correction: if False, only retrun metric
        config:
            off_policy_correction_level: str = "token"
            off_policy_correction_mode: str = "truncate"
            off_policy_correction_upper_bound: float = 2.0
            off_policy_correction_lower_bound: float = None
            off_policy_correction_veto_threshold: float = None
        prev_log_probs: log probs from training backend. 1D tensor each. Lengths can be different.
        rollout_log_probs: log probs from inference backend. 1D tensor each.
        mask: masks. 1D tensor each.
            Note that for single turn RL, the mask is [1] * response_length tensor for each sequence
            For multi-turn RL, the tool response will be marked as 0 in the mask.

    Returns:
        weights: importance sampling weights (safety-bounded; zeroed at padding only). 1D tensor each.
        modified_response_masks: rejection masks to apply in aggregation (mask mode + veto). 1D tensor each.
        metrics: The metrics for the importance sampling
    """
    if rollout_log_probs is None:
        return None, mask, {}

    assert not prev_log_probs.requires_grad
    assert not rollout_log_probs.requires_grad
    assert not mask.requires_grad
    assert mask.dtype == torch.float32
    assert prev_log_probs.shape == rollout_log_probs.shape == mask.shape

    metrics: Dict[str, torch.Tensor] = add_correction_metrics(
        prev_log_probs,
        rollout_log_probs,
        mask,
    )
    if not enable_off_policy_correction:
        return None, mask, convert_metrics(metrics)

    level = config.ppo.off_policy_correction_level
    mode = config.ppo.off_policy_correction_mode
    lower_bound = config.ppo.off_policy_correction_lower_bound
    upper_bound = config.ppo.off_policy_correction_upper_bound
    if lower_bound is None:
        lower_bound = 1.0 / upper_bound

    SAFETY_BOUND = 20.0  # Add a safety bound to avoid exp overflow
    # handle each sequence independently
    raw_log_ratio_diff = prev_log_probs - rollout_log_probs

    # level: The aggregation level for the importance sampling weights.
    if level == "token":
        # Per-token ratio (biased)
        log_ratio_for_metrics = raw_log_ratio_diff
    elif level == "sequence":
        # Product of ratios (unbiased but high variance)
        log_ratio_for_metrics = masked_sum_expand(raw_log_ratio_diff, mask, expand=True)
    elif level == "geometric":
        # Geometric mean of ratios (biased but low variance)
        log_ratio_for_metrics = masked_mean_expand(raw_log_ratio_diff, mask, expand=True)
    else:
        raise ValueError(f"Invalid importance sampling level: {level}")

    log_ratio_safe = torch.clamp(log_ratio_for_metrics, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    weights = torch.exp(log_ratio_safe)
    ratio_before_clip_mean, ratio_before_clip_min, ratio_before_clip_max = masked_statistic(
        weights, mask
    )
    metrics["ratio_before_clip_mean"] = ratio_before_clip_mean
    metrics["ratio_before_clip_min"] = ratio_before_clip_min
    metrics["ratio_before_clip_max"] = ratio_before_clip_max

    modified_mask = mask.clone()

    # mode: how to handle the importance sampling weights exceeding the thresholds.
    if mode == "truncate":
        # Cap the importance sampling weights at the upper threshold
        # https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33
        weights = truncate_mode(weights, mask, metrics, upper_bound)
    elif mode == "mask":
        # Preserve safety-bounded weights; apply thresholds via modified_mask
        # https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
        weights, modified_mask = mask_mode(
            weights,
            mask,
            metrics,
            lower_bound,
            upper_bound,
        )
    elif mode == "icepop":
        # Zero IS weights outside [lower, upper]; keep mask unchanged so
        # out-of-range tokens still count in the masked_mean denominator.
        weights = icepop_mode(
            weights,
            mask,
            metrics,
            lower_bound,
            upper_bound,
        )
    elif mode == "clip":
        # Clip the importance sampling weights to the [lower, upper] range.
        # Original behavior in slime.
        weights = clip_mode(
            weights,
            mask,
            metrics,
            lower_bound,
            upper_bound,
        )
    else:
        raise ValueError(f"Unsupported mis_mode: {mode}")

    # Veto on raw per-token ratios (sequence-wise rejection)
    # Works independently of truncate/mask mode and does NOT modify IS weights
    if config.ppo.off_policy_correction_veto_threshold is not None:
        veto_mask = calculate_veto_mask(
            raw_log_ratio_diff,
            mask,
            config.ppo.off_policy_correction_veto_threshold,
            metrics,
        )
        modified_mask = modified_mask * veto_mask

    ratio_after_clip_mean, ratio_after_clip_min, ratio_after_clip_max = masked_statistic(
        weights, mask
    )
    metrics["ratio_after_clip_mean"] = ratio_after_clip_mean
    metrics["ratio_after_clip_min"] = ratio_after_clip_min
    metrics["ratio_after_clip_max"] = ratio_after_clip_max
    metrics["mask_fraction"] = modified_mask.sum().float() / modified_mask.numel()

    return weights.detach(), modified_mask.detach(), convert_metrics(metrics)


def add_correction_metrics(
    prev_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    metrics = {}
    # 1. actor perplexity metrics
    actor_log_mean = masked_mean(prev_log_probs, mask)
    metrics["actor_log_ppl"] = -actor_log_mean
    metrics["actor_ppl"] = torch.exp(-actor_log_mean)

    # 2. sampler perplexity metrics
    sampler_log_mean = masked_mean(rollout_log_probs, mask)
    metrics["sampler_log_ppl"] = -sampler_log_mean
    metrics["sampler_ppl"] = torch.exp(-sampler_log_mean)

    # 3a. kl: Direct estimator for KL(π_sampler || π_actor)
    log_ratio = rollout_log_probs - prev_log_probs
    k3_kl = torch.exp(log_ratio) - log_ratio - 1
    metrics["kl"] = masked_mean(log_ratio, mask)
    metrics["k3_kl"] = masked_mean(k3_kl, mask)

    # 3b. Log PPL difference (sequence-level perplexity difference)
    log_ppl_diff = sampler_log_mean - actor_log_mean
    metrics["log_ppl_diff"] = log_ppl_diff
    metrics["log_ppl_abs_diff"] = log_ppl_diff.abs()

    # 3c. PPL ratio (how much higher is training PPL vs rollout PPL)
    ppl_ratio = torch.exp(log_ppl_diff)
    metrics["ppl_ratio"] = ppl_ratio

    # 4a. Token-level chi-squared divergence
    # χ²(π_training || π_rollout) = E[ρ²] - 1, where ρ = π_training / π_rollout
    # This measures the second moment of the importance weights
    SAFETY_BOUND = 20.0
    log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    rho_token = torch.exp(log_ratio_safe)  # ρ = π_training / π_rollout
    rho_squared_token = rho_token.square()
    chi2_token = masked_mean(rho_squared_token, mask) - 1.0
    metrics["chi2_token"] = chi2_token

    # 4b. Sequence-level chi-squared divergence
    # Computes (Π ρ_t)² - 1 for the entire sequence
    # This captures the squared product of importance ratios
    log_ratio_sum = masked_sum(log_ratio, mask)
    log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)  # (Π ρ_t)²
    chi2_seq = rho_squared_seq - 1.0
    metrics["chi2_seq"] = chi2_seq

    # 4d. correction_ratio
    correction_ratio = torch.exp(-log_ratio_safe)
    correction_ratio_mean, correction_ratio_min, correction_ratio_max = masked_statistic(
        correction_ratio, mask
    )
    metrics["correction_ratio_mean"] = correction_ratio_mean
    metrics["correction_ratio_min"] = correction_ratio_min
    metrics["correction_ratio_max"] = correction_ratio_max

    return metrics


def convert_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefix = "off_policy_correction"
    res_metrics = {}
    for key, val in metrics.items():
        res_metrics[f"{prefix}/{key}"] = val.detach()

    return res_metrics
