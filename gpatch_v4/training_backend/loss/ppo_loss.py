"""Policy-loss path (sum-return + unified agg).

Enabled when ``ppo.use_legacy_loss=False``. Legacy implementations remain in
``loss_factory.py``, while entropy regularization is shared by both paths.

New loss expects plain 2D ``[B, S]`` tensors. Dyn-CP / THD packs must be
response-padded before entering this path (``cu_seqlens_padded=None``).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.core.adaptive_entropy import get_entropy_bonus_coef
from gpatch_v4.core.correction_helper import compute_off_policy_correction_weights
from gpatch_v4.training_backend.loss.metrics import compute_steer_histograms
from gpatch_v4.training_backend.loss.registry import register_loss
from gpatch_v4.training_backend.loss.utils import agg, compute_clip_metrics
from gpatch_v4.utils import (
    masked_mean,
    masked_sum,
    reduce_metrics_across_data_parallel_group,
)
from gpatch_v4.utils.ppo_utils import calculate_kl_loss


@dataclass
class PolicyLossInput:
    advantages: torch.Tensor
    prev_log_probs: Optional[torch.Tensor]
    ref_log_probs: Optional[torch.Tensor]
    curr_log_probs: torch.Tensor
    response_mask: torch.Tensor
    scaled_entropy: torch.Tensor
    rollout_log_probs: Optional[torch.Tensor] = None
    per_token_entropy: Optional[torch.Tensor] = None
    prev_per_token_entropy: Optional[torch.Tensor] = None
    parallel_logits: Optional[torch.Tensor] = None
    sample_mask: Optional[torch.Tensor] = None
    global_retention_ratio: Optional[torch.Tensor] = None
    entropy_aux_figures: Optional[torch.Tensor] = None
    teacher_log_probs: Optional[torch.Tensor] = None
    dumped_topk_logprobs: Optional[torch.Tensor] = None
    dumped_topk_token_ids: Optional[torch.Tensor] = None
    should_dump_metrics: bool = False
    # prev/curr_topk_logprobs: 来自topk模式时 student top-K ids 上的 ``[B, S-1, K]``
    # log-probs，用于 ``advantages.dim() == 3`` 时用于 3D PPO ratio 计算。
    prev_topk_logprobs: Optional[torch.Tensor] = None
    curr_topk_logprobs: Optional[torch.Tensor] = None
    # Kept for call-site compatibility; must stay None / 1. Dyn-CP THD packs
    # are response-padded to ``[B, S]`` before new loss (see mixin).
    cu_seqlens_padded: Optional[torch.Tensor] = None
    local_cp_size: int = 1
    calculate_per_token_loss: bool = False
    # Optional per-sample (shape ``[B, 1]``) or per-token (shape ``[B, S]``) reweight
    # for aggregation. ``None`` means all ones.
    token_weights: Optional[torch.Tensor] = None


@dataclass
class ActorLossResult:
    """Per-token surrogate + diagnostics from algorithm-specific code."""
    actor_loss: torch.Tensor
    ratios: torch.Tensor
    algo_metrics: Optional[Dict[str, torch.Tensor]] = None
    algo_dumps: Optional[Dict[str, Any]] = None


def compute_entropy_regularization_loss(
    ppo_config,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    pg_losses1: torch.Tensor,
    pg_losses2: torch.Tensor,
    entropy_aux_figures: Optional[torch.Tensor] = None,
    rollout_log_probs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Entropy regularization via clip-cov or kl-cov.

    Modifies the per-token policy gradient losses to control entropy collapse.
    ref: https://arxiv.org/abs/2505.22617

    Parameters
    ----------
    ppo_config : PpoConfig
        PPO configuration containing regularization hyperparameters.
    old_log_prob : torch.Tensor
        Previous policy log-probabilities, shape ``[B, S]``.
    log_prob : torch.Tensor
        Current policy log-probabilities, shape ``[B, S]``.
    advantages : torch.Tensor
        Per-token advantages, shape ``[B, S]``.
    response_mask : torch.Tensor
        Binary mask for valid response tokens, shape ``[B, S]``.
    pg_losses1 : torch.Tensor
        Unclipped surrogate loss ``-advantages * ratios``, shape ``[B, S]``.
    pg_losses2 : torch.Tensor
        Clipped surrogate loss ``-advantages * ratios_clamped``, shape ``[B, S]``.
    entropy_aux_figures : torch.Tensor, optional
        Global covariance figures ``[mean_adv, mean_logp, kl_cov_tau]`` (shape
        ``(3,)``) precomputed at collation across the global train_gbs / DP. When
        present and ``ppo_entropy_global_cov`` is on, the covariance is centered
        on the global means and kl-cov selects ``cov > kl_cov_tau`` (exact global
        top-rho%). When None, the per-micro-batch behavior is used.
    rollout_log_probs : torch.Tensor, optional
        Sampler log-probs, used as the covariance log-prob in the global path
        when ``skip_prev_logps`` (prev_log_probs is unavailable then).

    Returns
    -------
    pg_losses : torch.Tensor
        Modified per-token losses after entropy regularization, shape ``[B, S]``.
    metrics : dict
        Regularization-specific metrics for logging.
    """
    reg_type = ppo_config.ppo_entropy_regularization_type
    assert reg_type in ("clip-cov", "kl-cov"), \
        f"ppo_entropy_regularization_type must be 'clip-cov' or 'kl-cov', got '{reg_type}'"

    # Global covariance: center on global means (and, for kl-cov, select via the
    # global top-rho% threshold) using the same log-prob space as collation.
    use_global = ppo_config.ppo_entropy_global_cov and entropy_aux_figures is not None
    if use_global:
        cov_log_prob = rollout_log_probs if ppo_config.skip_prev_logps else old_log_prob
        assert cov_log_prob is not None, (
            "global entropy cov requires rollout_log_probs (skip_prev_logps) "
            "or prev_log_probs as the covariance log-prob"
        )
        cov_log_prob = cov_log_prob.detach()
        global_mean_adv = entropy_aux_figures[0]
        global_mean_logp = entropy_aux_figures[1]
        kl_cov_tau = entropy_aux_figures[2]

    metrics = {}
    cov_for_metrics = None
    pg_losses = None

    if reg_type == "clip-cov":
        corr = torch.ones_like(advantages)
        clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)
        if use_global:
            cov_all = (advantages - global_mean_adv) * (cov_log_prob - global_mean_logp)
        else:
            cov_all = (
                (advantages - masked_mean(advantages, response_mask)) *
                (log_prob - masked_mean(log_prob.detach(), response_mask))
            )
        cov_for_metrics = cov_all[response_mask > 0].clone().detach()
        cov_all[response_mask == 0] = -torch.inf
        cov_all[clip_by_origin] = -torch.inf
        clip_num = max(int(ppo_config.ppo_clip_cov_ratio * response_mask.sum().item()), 1)
        top_k_idx = (
            (cov_all < ppo_config.ppo_clip_cov_ub) & (cov_all > ppo_config.ppo_clip_cov_lb) &
            (response_mask > 0)
        )
        top_k_idx = torch.nonzero(top_k_idx)
        if len(top_k_idx) > 0:
            perm = torch.randperm(len(top_k_idx))
            top_k_idx = top_k_idx[perm[:min(clip_num, len(top_k_idx))]]
        else:
            top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)
        corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0
        pg_clipfrac = masked_mean((corr == 0).float(), response_mask)
        pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
        metrics["clip_cov_frac"] = pg_clipfrac

    elif reg_type == "kl-cov":
        clip_max_loss = torch.maximum(pg_losses1, pg_losses2)  # NOTE: avoid loss collapse
        negative_approx_kl = log_prob - old_log_prob
        if ppo_config.ppo_logps_ratio_clamp is not None:  # NOTE: avoid kl collapse
            negative_approx_kl = torch.clamp(
                negative_approx_kl,
                min=-ppo_config.ppo_logps_ratio_clamp,
                max=ppo_config.ppo_logps_ratio_clamp
            )
        abs_kl = negative_approx_kl.abs()
        pg_losses_kl = clip_max_loss + ppo_config.ppo_kl_cov_coef * abs_kl
        pg_losses = clip_max_loss.clone()

        if use_global:
            # Exact global top-rho%: select tokens whose globally-centered cov
            # exceeds the global threshold tau (tau = +inf disables selection).
            cov = (advantages - global_mean_adv) * (cov_log_prob - global_mean_logp)
            select = (cov > kl_cov_tau) & (response_mask > 0)
            cov_for_metrics = cov[response_mask > 0].clone().detach()
            pg_losses[select] = pg_losses_kl[select]
        else:
            all_valid = response_mask > 0
            all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
            all_valid_adv = advantages[all_valid].detach().reshape(-1)
            all_valid_logp = log_prob[all_valid].detach().reshape(-1)
            k = min(ppo_config.ppo_kl_cov_ratio, len(all_valid_adv))
            if k != 0:
                cov_lst_all = (
                    (all_valid_adv - all_valid_adv.mean()) *
                    (all_valid_logp - all_valid_logp.mean())
                )
                cov_for_metrics = cov_lst_all.clone().detach()
                k_percent_nums = max(1, int(len(cov_lst_all) * ppo_config.ppo_kl_cov_ratio))
                large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

                if len(large_cov_idxs) != 0:
                    large_cov_idxs = all_valid_idx[large_cov_idxs]
                    pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs %
                              advantages.shape[1]] = pg_losses_kl[large_cov_idxs //
                                                                  advantages.shape[1],
                                                                  large_cov_idxs %
                                                                  advantages.shape[1]]
        ppo_kl_abs = masked_mean(negative_approx_kl.abs(), response_mask)
        metrics["ppo_abs_kl"] = ppo_kl_abs

    if cov_for_metrics is not None and len(cov_for_metrics) > 0:
        # choose quantiles according to Table 1, https://arxiv.org/abs/2505.22617
        p50, p80, p98, p99_8, p99_98 = torch.quantile(
            cov_for_metrics,
            torch.tensor([0.5, 0.8, 0.98, 0.998, 0.9998], device=cov_for_metrics.device),
            interpolation='linear',
        )
        metrics["cov_p50"] = p50
        metrics["cov_p80"] = p80
        metrics["cov_p98"] = p98
        metrics["cov_p99_8"] = p99_8
        metrics["cov_p99_98"] = p99_98

    return pg_losses, metrics


def _token_level_ratios(ppo_config, curr_log_probs, effective_prev):
    log_ratio = curr_log_probs - effective_prev
    if ppo_config.ppo_logps_ratio_clamp is not None:
        log_ratio = torch.clamp(
            log_ratio,
            min=-ppo_config.ppo_logps_ratio_clamp,
            max=ppo_config.ppo_logps_ratio_clamp,
        )
    return log_ratio.exp()


def _seq_level_ratios(curr_log_probs, effective_prev, response_mask):
    negative_approx_kl = curr_log_probs - effective_prev
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = (torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths)
    log_seq_importance_ratio = (
        curr_log_probs - curr_log_probs.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    )
    return torch.exp(torch.clamp(log_seq_importance_ratio, max=10.0))


def _compute_steer_token_weights(
    advantages: torch.Tensor,
    prev_log_probs: torch.Tensor,
    curr_log_probs: torch.Tensor,
    prev_per_token_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    token_weight_min: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.no_grad():
        valid_mask = response_mask.bool()
        assert valid_mask.any(), "STEER requires at least one valid response token."

        curr_prob = curr_log_probs.detach().float().exp().clamp(min=1e-8, max=1.0 - 1e-8)
        x_one_minus_x_squared = curr_prob * (1.0 - curr_prob)
        ln_x_plus_h = torch.log(curr_prob) + prev_per_token_entropy.detach().float()
        metric = (
            advantages.float() / prev_log_probs.detach().float().exp().clamp(min=1e-8, max=1.0) *
            x_one_minus_x_squared * ln_x_plus_h
        )
        valid_metric = metric.abs()[valid_mask]
        metric_max = valid_metric.max()
        normalizer = torch.maximum(metric_max, metric_max.new_tensor(0.02))
        k = -metric_max.new_tensor(token_weight_min).log() / normalizer
        valid_weights = torch.exp(-k * valid_metric).clamp(min=token_weight_min, max=1.0)
        token_weights = torch.zeros_like(metric, dtype=torch.float)
        token_weights[valid_mask] = valid_weights
        token_weights = token_weights * response_mask.float()
        valid_token_weights = token_weights[valid_mask]
        valid_token_count = torch.tensor(
            valid_token_weights.numel(),
            dtype=valid_token_weights.dtype,
            device=valid_token_weights.device
        )
        metrics = {
            "steer/token_weight_mean":
                torch.stack([valid_token_weights.sum(), valid_token_count]),
            "steer/token_weight_min":
                valid_token_weights.min(),
            "steer/token_weight_max":
                valid_token_weights.max(),
            "steer/token_weight_lt_0_99_frac":
                torch.stack([(valid_token_weights < 0.99).sum().float(), valid_token_count]),
            "steer/token_weight_lt_0_95_frac":
                torch.stack([(valid_token_weights < 0.95).sum().float(), valid_token_count]),
            "steer/entropy_change_metric_mean":
                torch.stack([valid_metric.sum(), valid_token_count]),
            "steer/entropy_change_metric_max":
                metric_max,
        }
        metrics.update(
            compute_steer_histograms(
                valid_token_weights,
                valid_metric,
                token_weight_min,
            )
        )
    return token_weights, metrics


def _compute_grpo_actor_loss(config, loss_input, response_mask, effective_prev):
    ppo_config = config.ppo
    advantages = loss_input.advantages
    ratios = _token_level_ratios(ppo_config, loss_input.curr_log_probs, effective_prev)
    clip_ratio_low = (
        ppo_config.ppo_clip_ratio_low
        if ppo_config.ppo_clip_ratio_low is not None else ppo_config.ppo_ratio_eps
    )
    clip_ratio_high = (
        ppo_config.ppo_clip_ratio_high
        if ppo_config.ppo_clip_ratio_high is not None else ppo_config.ppo_ratio_eps
    )
    ratios_clamped = ratios.clamp(1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    loss1 = -advantages * ratios
    loss2 = -advantages * ratios_clamped
    entropy_reg_metrics = {}
    if ppo_config.ppo_entropy_regularization_type is not None:
        clip_max_loss, entropy_reg_metrics = compute_entropy_regularization_loss(
            ppo_config,
            old_log_prob=effective_prev,
            log_prob=loss_input.curr_log_probs,
            advantages=advantages,
            response_mask=response_mask,
            pg_losses1=loss1,
            pg_losses2=loss2,
            entropy_aux_figures=loss_input.entropy_aux_figures,
            rollout_log_probs=loss_input.rollout_log_probs,
        )
    else:
        clip_max_loss = torch.maximum(loss1, loss2)

    loss3 = None
    if ppo_config.ppo_dual_clip_ratio_c is not None:
        loss3 = -advantages * ppo_config.ppo_dual_clip_ratio_c
        actor_loss = torch.where(advantages < 0, torch.min(loss3, clip_max_loss), clip_max_loss)
    else:
        actor_loss = clip_max_loss
    # ============================ GRPO METRICS ================================
    with torch.no_grad():
        numel = response_mask.sum()
        algo_metrics = compute_clip_metrics(
            ratios,
            ratios_clamped,
            clip_ratio_high,
            clip_ratio_low,
            loss1,
            loss2,
            advantages,
            clip_max_loss,
            response_mask,
            numel,
            loss3=loss3,
        )
        algo_metrics.update(entropy_reg_metrics)
        algo_dumps = None
        if loss_input.should_dump_metrics:
            ratios_tmp = ratios.detach()
            algo_dumps = {
                "dump/ppo_ratio_unclamped":
                    ratios_tmp.to(dtype=torch.bfloat16, device="cpu"),
                "dump/is_ppo_ratio_clamped":
                    ((ratios_tmp == ratios_clamped.detach()) & response_mask.bool()).cpu(),
            }

    return ActorLossResult(
        actor_loss=actor_loss,
        ratios=ratios,
        algo_metrics=algo_metrics,
        algo_dumps=algo_dumps,
    )


def _compute_steer_actor_loss(config, loss_input, response_mask, effective_prev):
    assert not config.ppo.skip_prev_logps, "STEER requires prev_log_probs; set skip_prev_logps=False."
    assert loss_input.prev_log_probs is not None, "STEER requires prev_log_probs."
    assert loss_input.prev_per_token_entropy is not None, "STEER requires prev_per_token_entropy."
    if config.ppo.steer_policy_method == "grpo":
        actor_loss_result = _compute_grpo_actor_loss(
            config, loss_input, response_mask, effective_prev
        )
    elif config.ppo.steer_policy_method == "cispo":
        actor_loss_result = _compute_cispo_actor_loss(
            config, loss_input, response_mask, effective_prev
        )
    elif config.ppo.steer_policy_method == "gspo":
        actor_loss_result = _compute_gspo_actor_loss(
            config, loss_input, response_mask, effective_prev
        )
    elif config.ppo.steer_policy_method == "sapo":
        actor_loss_result = _compute_sapo_actor_loss(
            config, loss_input, response_mask, effective_prev
        )
    else:
        raise AssertionError(f"Unsupported steer_policy_method: {config.ppo.steer_policy_method}")

    # TODO: Derive method-specific entropy-change estimators from actual log-prob gradients.
    # 当前只是复用了 grpo 的公式
    token_weights, steer_metrics = _compute_steer_token_weights(
        advantages=loss_input.advantages,
        prev_log_probs=loss_input.prev_log_probs,
        curr_log_probs=loss_input.curr_log_probs,
        prev_per_token_entropy=loss_input.prev_per_token_entropy,
        response_mask=response_mask,
        token_weight_min=config.ppo.steer_token_weight_min,
    )
    actor_loss_result.actor_loss = actor_loss_result.actor_loss * token_weights
    actor_loss_result.algo_metrics = {**(actor_loss_result.algo_metrics or {}), **steer_metrics}
    return actor_loss_result


def _compute_cispo_actor_loss(config, loss_input, response_mask, effective_prev):
    ppo_config = config.ppo
    ratios = _token_level_ratios(ppo_config, loss_input.curr_log_probs, effective_prev)
    clip_ratio_low = (
        ppo_config.ppo_clip_ratio_low
        if ppo_config.ppo_clip_ratio_low is not None else ppo_config.ppo_ratio_eps
    )
    clip_ratio_high = (
        ppo_config.ppo_clip_ratio_high
        if ppo_config.ppo_clip_ratio_high is not None else ppo_config.ppo_ratio_eps
    )
    clipped_ratio = ratios.clamp(1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    actor_loss = -clipped_ratio.detach() * loss_input.advantages * loss_input.curr_log_probs

    with torch.no_grad():
        numel = response_mask.sum()
        mask_bool = response_mask.bool()
        is_clamped = (clipped_ratio.detach() != ratios.detach()) & mask_bool
        is_upper_clamped = (ratios.detach() > 1.0 + clip_ratio_high) & mask_bool
        is_lower_clamped = (ratios.detach() < 1.0 - clip_ratio_low) & mask_bool
        ppo_ratio_clamped = masked_mean(clipped_ratio.detach(), response_mask)
        algo_metrics = {
            "ppo_ratio_clamped": torch.stack([ppo_ratio_clamped * numel, numel]),
            "ppo_ratio_clamped_upper_frac": torch.stack([is_upper_clamped.sum().float(), numel]),
            "ppo_ratio_clamped_lower_frac": torch.stack([is_lower_clamped.sum().float(), numel]),
            "cispo/clipfrac": is_clamped.sum().float() / numel.clamp(min=1),
        }
        algo_dumps = None
        if loss_input.should_dump_metrics:
            ratios_tmp = ratios.detach()
            algo_dumps = {
                "dump/ppo_ratio_unclamped":
                    ratios_tmp.to(dtype=torch.bfloat16, device="cpu"),
                "dump/is_ppo_ratio_clamped":
                    ((ratios_tmp == clipped_ratio.detach()) & mask_bool).cpu(),
            }

    return ActorLossResult(
        actor_loss=actor_loss,
        ratios=ratios,
        algo_metrics=algo_metrics,
        algo_dumps=algo_dumps,
    )


def _compute_gspo_actor_loss(config, loss_input, response_mask, effective_prev):
    """GSPO actor loss with sequence-level importance ratios on ``[B, S]``.

    Dyn-CP / THD packs must already be response-padded before this path
    (``cu_seqlens_padded=None``, ``local_cp_size=1``); see
    ``mixin._rl_response_pad_dyn_cp_tensors``.
    """
    ppo_config = config.ppo
    assert ppo_config.ppo_entropy_regularization_type is None, (
        "new loss path does not support ppo_entropy_regularization_type yet"
    )
    advantages = loss_input.advantages
    ratios = _seq_level_ratios(loss_input.curr_log_probs, effective_prev, response_mask)
    clip_ratio_low = (
        ppo_config.ppo_clip_ratio_low
        if ppo_config.ppo_clip_ratio_low is not None else ppo_config.ppo_ratio_eps
    )
    clip_ratio_high = (
        ppo_config.ppo_clip_ratio_high
        if ppo_config.ppo_clip_ratio_high is not None else ppo_config.ppo_ratio_eps
    )
    ratios_clamped = ratios.clamp(1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    loss1 = -advantages * ratios
    loss2 = -advantages * ratios_clamped
    clip_max_loss = torch.maximum(loss1, loss2)

    loss3 = None
    if ppo_config.ppo_dual_clip_ratio_c is not None:
        loss3 = -advantages * ppo_config.ppo_dual_clip_ratio_c
        actor_loss = torch.where(advantages < 0, torch.min(loss3, clip_max_loss), clip_max_loss)
    else:
        actor_loss = clip_max_loss
    # ============================ GSPO METRICS ================================
    with torch.no_grad():
        numel = response_mask.sum()
        algo_metrics = compute_clip_metrics(
            ratios,
            ratios_clamped,
            clip_ratio_high,
            clip_ratio_low,
            loss1,
            loss2,
            advantages,
            clip_max_loss,
            response_mask,
            numel,
            loss3=loss3,
        )
        token_clipped = ((ratios.detach() != ratios_clamped.detach()) & response_mask.bool())
        seq_has_clip = token_clipped.any(dim=-1)
        num_seqs = torch.tensor(float(seq_has_clip.shape[0]), device=ratios.device)
        algo_metrics.update(
            {
                "ppo_ratio_clamped_seq_frac":
                    torch.stack([seq_has_clip.sum().float(), num_seqs]),
                "ppo_ratio_clamped_seq_effective_frac":
                    torch.stack(
                        [
                            (token_clipped & (loss2 > loss1)).any(dim=-1).sum().float(),
                            num_seqs,
                        ]
                    ),
            }
        )
        algo_dumps = None
        if loss_input.should_dump_metrics:
            ratios_tmp = ratios.detach()
            algo_dumps = {
                "dump/ppo_ratio_unclamped":
                    ratios_tmp.to(dtype=torch.bfloat16, device="cpu"),
                "dump/is_ppo_ratio_clamped":
                    ((ratios_tmp == ratios_clamped.detach()) & response_mask.bool()).cpu(),
            }

    return ActorLossResult(
        actor_loss=actor_loss,
        ratios=ratios,
        algo_metrics=algo_metrics,
        algo_dumps=algo_dumps,
    )


def _compute_sapo_actor_loss(config, loss_input, response_mask, effective_prev):
    ppo_config = config.ppo
    advantages = loss_input.advantages
    ratios = _token_level_ratios(ppo_config, loss_input.curr_log_probs, effective_prev)
    tau = torch.where(
        advantages > 0,
        torch.as_tensor(ppo_config.sapo_tau_pos, dtype=ratios.dtype, device=ratios.device),
        torch.as_tensor(ppo_config.sapo_tau_neg, dtype=ratios.dtype, device=ratios.device),
    )
    sigmoid_gate = torch.sigmoid(tau * (ratios - 1.0))
    gate = (4.0 / tau) * sigmoid_gate
    actor_loss = -advantages * gate
    # ============================ SAPO METRICS ================================
    with torch.no_grad():
        mask_bool = response_mask.bool()
        grad_kernel = 4.0 * sigmoid_gate.detach() * (1.0 - sigmoid_gate.detach())
        valid_gate = gate.detach()[mask_bool]
        valid_kernel = grad_kernel[mask_bool]
        if valid_gate.numel() > 0:
            gate_mean, gate_min, gate_max = valid_gate.mean(), valid_gate.min(), valid_gate.max()
            grad_kernel_mean = valid_kernel.mean()
            strong_attenuation_frac = (valid_kernel < 0.5).float().mean()
        else:
            zero = torch.tensor(0.0, device=ratios.device)
            gate_mean = gate_min = gate_max = grad_kernel_mean = strong_attenuation_frac = zero
        algo_metrics = {
            "sapo/gate_mean": gate_mean,
            "sapo/gate_min": gate_min,
            "sapo/gate_max": gate_max,
            "sapo/grad_kernel_mean": grad_kernel_mean,
            "sapo/strong_attenuation_frac": strong_attenuation_frac,
        }
        algo_dumps = None
        if loss_input.should_dump_metrics:
            algo_dumps = {
                "dump/ppo_ratio_unclamped": ratios.detach().to(dtype=torch.bfloat16, device="cpu"),
            }

    return ActorLossResult(
        actor_loss=actor_loss,
        ratios=ratios,
        algo_metrics=algo_metrics,
        algo_dumps=algo_dumps,
    )


def _compute_vespo_actor_loss(config, loss_input, response_mask, effective_prev):
    """VESPO: scale the REINFORCE gradient by a smooth sequence-level IS kernel.

    The kernel ``φ(W) = W^c1 · exp(c2 (1 - W))`` acts on the *product* sequence
    importance weight ``W = Π_t π_θ/π_rollout`` with no length normalization, and
    is detached so it only reweights ``∇log π_θ``. ``φ(1) = 1`` keeps on-policy
    samples at unit weight, so learning rates transfer from the GRPO path.
    See `arXiv:2602.10693 <https://arxiv.org/abs/2602.10693>`_ Eq. (20).

    Both sources of off-policyness are folded into that single ``W``: policy lag
    ``π_θ/π_prev`` and train/infer engine mismatch ``π_prev/π_rollout``. This is
    why ``enable_off_policy_correction`` must stay off — it would apply the
    latter a second time, outside the kernel.

    NOTE: THD and dynamic CP are not supported yet. Every reduction below folds
    dim -1 into one scalar per sequence, which assumes each ``[B, S]`` row holds
    exactly one sequence. Static CP is fine: log-probs are CP-all-gathered back
    to full length before the loss runs.
    """
    ppo_config = config.ppo
    assert loss_input.cu_seqlens_padded is None, (
        "VESPO does not support THD packing: reducing over dim -1 would turn the "
        "whole pack into a single W = Π_t ratio instead of one W per sequence."
    )
    advantages = loss_input.advantages
    assert advantages.dim() == 2, (
        "VESPO reweights whole sequences and cannot consume 3D (top-k OPD) advantages, "
        f"got {advantages.dim()}D."
    )
    assert not ppo_config.skip_prev_logps, (
        "VESPO requires prev_log_probs; set skip_prev_logps=False."
    )
    assert loss_input.rollout_log_probs is not None, (
        "VESPO folds the train/infer IS ratio into W and always requires "
        "rollout_log_probs."
    )
    assert ppo_config.ppo_logps_ratio_clamp is not None

    with torch.no_grad():
        token_log_ratio = (loss_input.curr_log_probs - effective_prev).float()
        if ppo_config.ppo_logps_ratio_clamp is not None:
            token_log_ratio = torch.clamp(
                token_log_ratio,
                min=-ppo_config.ppo_logps_ratio_clamp,
                max=ppo_config.ppo_logps_ratio_clamp,
            )
        ratios = token_log_ratio.exp()

        log_tis = torch.clamp(
            (effective_prev - loss_input.rollout_log_probs).float(),
            min=-ppo_config.ppo_logps_ratio_clamp,
            max=ppo_config.ppo_logps_ratio_clamp,
        )
        seq_log_tis = masked_sum(log_tis, response_mask, dim=-1)

        seq_log_w = masked_sum(token_log_ratio, response_mask, dim=-1) + seq_log_tis
        # Both saturation ends drive φ to 0 (W^c1 for W→0, exp(-c2 W) for W→∞),
        # so clamping only guards exp() from overflowing.
        seq_log_w = torch.clamp(
            seq_log_w, min=-ppo_config.ppo_logps_ratio_clamp, max=ppo_config.ppo_logps_ratio_clamp
        )
        w_seq = seq_log_w.exp()

        # GRPO advantages are constant within a sequence; the masked mean picks
        # that value without assuming any particular token position is valid.
        seq_advantages = (
            masked_sum(advantages, response_mask, dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
        )
        is_pos = (seq_advantages >= 0).float()
        c1 = is_pos * ppo_config.vespo_c1_pos + (1.0 - is_pos) * ppo_config.vespo_c1_neg
        c2 = is_pos * ppo_config.vespo_c2_pos + (1.0 - is_pos) * ppo_config.vespo_c2_neg
        c2 = torch.clamp(c2, min=1e-4)
        log_w_seq = torch.log(w_seq.clamp(min=1e-8))
        phi_seq = torch.exp(c2 + c1 * log_w_seq - c2 * w_seq)
        phi_seq = torch.nan_to_num(phi_seq, nan=0.0, posinf=0.0, neginf=0.0)

    actor_loss = -phi_seq.unsqueeze(-1) * advantages * loss_input.curr_log_probs

    # ============================ VESPO METRICS ===============================
    with torch.no_grad():
        num_seqs = torch.tensor(float(w_seq.shape[0]), device=w_seq.device)
        is_neg = 1.0 - is_pos
        algo_metrics = {
            "vespo/w_seq_mean":
                torch.stack([w_seq.sum(), num_seqs]),
            "vespo/w_seq_min":
                w_seq.min(),
            "vespo/w_seq_max":
                w_seq.max(),
            "vespo/log_w_seq_mean":
                torch.stack([seq_log_w.sum(), num_seqs]),
            "vespo/phi_mean":
                torch.stack([phi_seq.sum(), num_seqs]),
            "vespo/phi_min":
                phi_seq.min(),
            "vespo/phi_max":
                phi_seq.max(),
            "vespo/phi_pos_mean":
                torch.stack([(phi_seq * is_pos).sum(), is_pos.sum()]),
            "vespo/phi_neg_mean":
                torch.stack([(phi_seq * is_neg).sum(), is_neg.sum()]),
            # Negative-advantage sequences whose raw weight exploded: the paper's
            # early warning for off-policy noise dominating the update.
            "vespo/neg_noise_frac":
                torch.stack([(is_neg * (w_seq > 100.0).float()).sum(), num_seqs]),
        }
        algo_metrics["vespo/seq_log_tis_mean"] = torch.stack([seq_log_tis.sum(), num_seqs])
        algo_dumps = None
        if loss_input.should_dump_metrics:
            algo_dumps = {
                "dump/ppo_ratio_unclamped": ratios.to(dtype=torch.bfloat16, device="cpu"),
            }

    return ActorLossResult(
        actor_loss=actor_loss,
        ratios=ratios,
        algo_metrics=algo_metrics,
        algo_dumps=algo_dumps,
    )


def _policy_loss(
    config,
    loss_input: PolicyLossInput,
    compute_actor_loss: Callable,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Shared path: IS -> actor_loss(fn) -> entropy/KL agg -> metrics.

    Returns
    -------
    bwd_loss, bwd_count, metrics
        ``bwd_loss`` is the unnormalized sum; ``bwd_count`` is the matching
        denominator from ``agg`` (token count or sample count).
    """
    assert loss_input.cu_seqlens_padded is None, (
        "new loss expects [B, S] tensors; convert THD/dyn-CP to response-padded "
        "sequence before loss (cu_seqlens_padded must be None)"
    )
    assert loss_input.local_cp_size == 1, (
        "new loss expects local_cp_size=1; convert THD/dyn-CP to response-padded "
        "sequence before loss (local_cp_size must be 1)"
    )
    ppo_config = config.ppo
    per_token_entropy = loss_input.per_token_entropy
    response_mask = loss_input.response_mask

    # ============================ EFFECTIVE PREV ==============================
    # since we may skip prev logps computation under the on-policy case
    if ppo_config.skip_prev_logps:
        effective_prev = loss_input.curr_log_probs.detach()
    else:
        effective_prev = loss_input.prev_log_probs

    # ============================ COMPUTE ACTOR LOSS ==========================
    actor = compute_actor_loss(config, loss_input, response_mask, effective_prev)

    # ============================ IS CORRECTION ===============================
    correction_ratio, response_mask, is_metrics = compute_off_policy_correction_weights(
        ppo_config.enable_off_policy_correction,
        config,
        effective_prev,
        loss_input.rollout_log_probs,
        response_mask.float(),
    )
    if ppo_config.enable_off_policy_correction:
        actor.actor_loss = actor.actor_loss * correction_ratio

    # ============================ AGGREGATE ACTOR LOSS ========================
    agg_kw = dict(
        calculate_per_token_loss=loss_input.calculate_per_token_loss,
        sample_mask=loss_input.sample_mask,
        token_weights=loss_input.token_weights,
        cu_seqlens_padded=loss_input.cu_seqlens_padded,
        local_cp_size=loss_input.local_cp_size,
    )
    actor_bwd_sum, actor_bwd_count = agg(actor.actor_loss, response_mask, **agg_kw)

    # ============================ AGGREGATE ENTROPY LOSS ======================
    entropy_bwd_sum, entropy_bwd_count = agg(per_token_entropy, response_mask, **agg_kw)
    loss = actor_bwd_sum - entropy_bwd_sum * get_entropy_bonus_coef(ppo_config)

    # ============================ COMPUTE KL LOSS =============================
    use_absolute_kl = False
    use_low_var_kl = True
    if isinstance(config, OnPolicyDistillConfig):
        use_absolute_kl = True
        use_low_var_kl = False
    if loss_input.ref_log_probs is not None:
        per_token_kl = calculate_kl_loss(
            cur_log_probs=loss_input.curr_log_probs,
            ref_log_probs=loss_input.ref_log_probs,
            use_absolute_kl=use_absolute_kl,
            use_low_var_kl=use_low_var_kl,
            clamp_kl_loss=ppo_config.ppo_dual_clip_ratio_c is not None,
            clamp_kl_val=ppo_config.ppo_clamp_kl_val,
        )
        kl_bwd_sum, kl_bwd_count = agg(per_token_kl, response_mask, **agg_kw)
        loss = loss + kl_bwd_sum * ppo_config.grpo_kl_loss_beta
    else:
        kl_bwd_sum = torch.zeros_like(actor_bwd_sum)
        kl_bwd_count = actor_bwd_count

    # ============================ COMPUTE BWD LOSS ============================
    bwd_loss = loss.clone()

    # Token-level grads w.r.t. bwd_loss (before mixin GBS scale). retain_graph so
    # Megatron can still backward the same graph for the parameter update.
    dumped_actor_loss_grad = None
    dumped_curr_logprob_grad = None
    dump_gradient = (loss_input.should_dump_metrics and config.training.ppo_dump_gradient)
    if dump_gradient:
        actor_loss_grad, curr_logprob_grad = torch.autograd.grad(
            outputs=bwd_loss,
            inputs=(actor.actor_loss, loss_input.curr_log_probs),
            retain_graph=True,
            allow_unused=True,
        )
        if actor_loss_grad is not None:
            dumped_actor_loss_grad = actor_loss_grad.detach().to(dtype=torch.bfloat16, device="cpu")
        if curr_logprob_grad is not None:
            dumped_curr_logprob_grad = curr_logprob_grad.detach().to(
                dtype=torch.bfloat16, device="cpu"
            )

    # ============================ COLLECT METRICS =============================
    with torch.no_grad():
        numel = response_mask.sum()
        ppo_ratio = masked_mean(actor.ratios.detach(), response_mask)
        metrics = {
            "loss": torch.stack([loss.detach(), actor_bwd_count.detach()]),
            "policy_loss": torch.stack([actor_bwd_sum.detach(),
                                        actor_bwd_count.detach()]),
            "scaled_entropy": torch.stack([entropy_bwd_sum.detach(),
                                           entropy_bwd_count.detach()]),
            "grpo_kl_loss": torch.stack([kl_bwd_sum.detach(),
                                         kl_bwd_count.detach()]),
            "ppo_ratio": torch.stack([ppo_ratio * numel, numel]),
        }
    if is_metrics:
        metrics.update(is_metrics)
    if actor.algo_metrics:
        metrics.update(actor.algo_metrics)
    if loss_input.sample_mask is not None:
        sm = loss_input.sample_mask.float().detach()
        metrics["valid_sample_ratio"] = torch.stack([sm.sum(), sm.new_tensor(float(sm.numel()))])

    reduce_metrics_across_data_parallel_group(metrics)
    # ============================ DUMP METRICS ================================
    if loss_input.should_dump_metrics:
        dumped = {
            "dump/curr_logprobs":
                loss_input.curr_log_probs.clone().detach().to(dtype=torch.bfloat16, device="cpu"),
            "dump/per_token_entropy":
                per_token_entropy.clone().detach().to(dtype=torch.bfloat16, device="cpu"),
            "dump/topk_logprobs":
                loss_input.dumped_topk_logprobs,
            "dump/topk_token_ids":
                loss_input.dumped_topk_token_ids,
            "dump/mask":
                response_mask.detach().bool().to(device="cpu"),
        }
        if dump_gradient:
            dumped["dump/actor_loss_grad"] = dumped_actor_loss_grad
            dumped["dump/curr_logprob_grad"] = dumped_curr_logprob_grad
        if actor.algo_dumps is not None:
            dumped.update(actor.algo_dumps)
        metrics.update(dumped)

    return bwd_loss, actor_bwd_count, metrics


@register_loss(backends=("mcore", ), loss_name="grpo")
def grpo_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_grpo_actor_loss)


@register_loss(backends=("mcore", ), loss_name="steer")
def steer_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_steer_actor_loss)


@register_loss(backends=("mcore", ), loss_name="cispo")
def cispo_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_cispo_actor_loss)


@register_loss(backends=("mcore", ), loss_name="gspo")
def gspo_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_gspo_actor_loss)


@register_loss(backends=("mcore", ), loss_name="sapo")
def sapo_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_sapo_actor_loss)


@register_loss(backends=("mcore", ), loss_name="vespo")
def vespo_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_vespo_actor_loss)
