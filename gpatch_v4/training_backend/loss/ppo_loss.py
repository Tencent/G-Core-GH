"""Policy-loss path (sum-return + unified agg).

Enabled when ``ppo.use_legacy_loss=False``. Legacy implementations remain in
``loss_factory.py``.

dyn-CP is not supported yet.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.core.correction_helper import compute_off_policy_correction_weights
from gpatch_v4.training_backend.loss.metrics import compute_steer_histograms
from gpatch_v4.training_backend.loss.registry import register_loss
from gpatch_v4.training_backend.loss.utils import agg, compute_clip_metrics
from gpatch_v4.utils import masked_mean, reduce_metrics_across_data_parallel_group
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
    # cu_seqlens_padded: sample boundaries in THD packed format. When set,
    # loss aggregation uses per-sample mean instead of global per-token mean
    # to eliminate the length bias inherent in token-weighted averaging.
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
    assert ppo_config.ppo_entropy_regularization_type is None, (
        "new loss path does not support ppo_entropy_regularization_type yet"
    )
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
        algo_dumps = None
        if loss_input.should_dump_metrics:
            ratios_tmp = ratios.detach()
            algo_dumps = {
                "ppo_ratio_unclamped":
                    ratios_tmp.to(dtype=torch.bfloat16, device="cpu"),
                "is_ppo_ratio_clamped":
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
                "ppo_ratio_unclamped": ratios_tmp.to(dtype=torch.bfloat16, device="cpu"),
                "is_ppo_ratio_clamped": ((ratios_tmp == clipped_ratio.detach()) & mask_bool).cpu(),
            }

    return ActorLossResult(
        actor_loss=actor_loss,
        ratios=ratios,
        algo_metrics=algo_metrics,
        algo_dumps=algo_dumps,
    )


def _compute_gspo_actor_loss(config, loss_input, response_mask, effective_prev):
    """
    GSPO actor loss implementation.
    NTOE: THD and dynamic CP are not supported yet.

    Args:
        config: The configuration object.
        loss_input: The loss input object.
        response_mask: The response mask tensor.
        effective_prev: The effective previous log probabilities tensor.

    Returns:
        ActorLossResult: The actor loss result.
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
                "ppo_ratio_unclamped":
                    ratios_tmp.to(dtype=torch.bfloat16, device="cpu"),
                "is_ppo_ratio_clamped":
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
                "ppo_ratio_unclamped": ratios.detach().to(dtype=torch.bfloat16, device="cpu"),
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
        "new loss path does not support THD / dyn-CP yet "
        "(cu_seqlens_padded must be None)"
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
    )
    actor_bwd_sum, actor_bwd_count = agg(actor.actor_loss, response_mask, **agg_kw)

    # ============================ AGGREGATE ENTROPY LOSS ======================
    entropy_bwd_sum, entropy_bwd_count = agg(per_token_entropy, response_mask, **agg_kw)
    loss = actor_bwd_sum - entropy_bwd_sum * ppo_config.ppo_entropy_bonus

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
        sample_mask = loss_input.sample_mask
        metrics["valid_sample_ratio"] = masked_mean(
            sample_mask.float().detach(), torch.ones_like(sample_mask, dtype=torch.float)
        )

    reduce_metrics_across_data_parallel_group(metrics)
    # ============================ DUMP METRICS ================================
    if loss_input.should_dump_metrics:
        dumped = {
            "curr_logprobs":
                loss_input.curr_log_probs.clone().detach().to(dtype=torch.bfloat16, device="cpu"),
            "per_token_entropy":
                per_token_entropy.clone().detach().to(dtype=torch.bfloat16, device="cpu"),
            "topk_logprobs":
                loss_input.dumped_topk_logprobs,
            "topk_token_ids":
                loss_input.dumped_topk_token_ids,
            "mask":
                response_mask.detach().bool().to(device="cpu"),
        }
        if actor.algo_dumps is not None:
            dumped.update(actor.algo_dumps)
        metrics.update(dumped)

    return bwd_loss, actor_bwd_count, metrics


@register_loss(backends=("mcore", ), loss_name="grpo", log_registration=True)
def grpo_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_grpo_actor_loss)


@register_loss(backends=("mcore", ), loss_name="steer", log_registration=True)
def steer_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_steer_actor_loss)


@register_loss(backends=("mcore", ), loss_name="cispo", log_registration=True)
def cispo_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_cispo_actor_loss)


@register_loss(backends=("mcore", ), loss_name="gspo")
def gspo_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_gspo_actor_loss)


@register_loss(backends=("mcore", ), loss_name="sapo")
def sapo_loss(config, loss_input: PolicyLossInput):
    return _policy_loss(config, loss_input, _compute_sapo_actor_loss)
