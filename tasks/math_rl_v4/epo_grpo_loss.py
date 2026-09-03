"""EPO-lite GRPO loss (custom-registered).

Wraps builtin ``grpo_loss_func`` and applies entropy-band smoothing against
``PpoFeatureStore`` history under feature name ``\"epo\"``.
"""

from __future__ import annotations

import math

import torch

from gpatch_v4.core.ppo_feature_store import (
    feature_axis_total_key,
    get_ppo_feature_store,
)
from gpatch_v4.training_backend.loss_factory import (
    PolicyLossInput,
    grpo_loss_func,
    masked_mean_per_sample_or_token,
    reduce_metrics_across_data_parallel_group,
)

# Feature name used in PpoFeatureStore; not a core/builtin constant.
FEATURE_NAME = "epo"

_DEFAULT_ENTROPY_SMOOTH_COEFF = 1.0
_DEFAULT_MASK_MODE = "token"
_DEFAULT_MIN_RATIO = 0.8
_DEFAULT_MAX_RATIO = 1.2
_DEFAULT_OUT_RANGE_PENALTY = 0.1
_DEFAULT_ENABLE_SMOOTH_WEIGHTS = False


def _task_get(task, key: str, default):
    if task is None:
        return default
    if isinstance(task, dict):
        return task[key] if key in task else default
    if hasattr(task, key):
        return getattr(task, key)
    return default


def _task_epo_settings(config) -> dict:
    """Read EPO knobs from ``config.task`` (dict or namespace; not ``ppo_config``)."""
    task = getattr(config, "task", None)
    mask_mode = _task_get(task, "epo_mask_mode", _DEFAULT_MASK_MODE)
    min_ratio = float(_task_get(task, "epo_min_ratio", _DEFAULT_MIN_RATIO))
    max_ratio = float(_task_get(task, "epo_max_ratio", _DEFAULT_MAX_RATIO))
    out_range_penalty = float(_task_get(task, "epo_out_range_penalty", _DEFAULT_OUT_RANGE_PENALTY))
    entropy_smooth_coeff = float(
        _task_get(task, "epo_entropy_smooth_coeff", _DEFAULT_ENTROPY_SMOOTH_COEFF)
    )
    enable_smooth_weights = bool(
        _task_get(task, "epo_enable_smooth_weights", _DEFAULT_ENABLE_SMOOTH_WEIGHTS)
    )
    if mask_mode not in ("token", "seq"):
        raise ValueError(f"task.epo_mask_mode must be 'token' or 'seq', got {mask_mode!r}")
    if min_ratio > max_ratio:
        raise ValueError(
            f"task.epo_min_ratio ({min_ratio}) must be <= task.epo_max_ratio ({max_ratio})"
        )
    return {
        "mask_mode": mask_mode,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "out_range_penalty": out_range_penalty,
        "entropy_smooth_coeff": entropy_smooth_coeff,
        "enable_smooth_weights": enable_smooth_weights,
    }


def calculate_epo_phase_weight(current_step: int, total_steps: int) -> float:
    """Epoch-style phase weight from EPO; mapped to PPO step progress."""
    if total_steps <= 0:
        return 1.0
    current_step = max(0, min(current_step, total_steps))
    half = total_steps // 2
    if current_step <= half:
        if half == 0:
            return 1.0
        progress = current_step / half
        return 1.0 - 0.2 * (1 - math.exp(-2.0 * progress))
    remaining = total_steps - half
    if remaining == 0:
        return 0.8
    progress = (current_step - half) / remaining
    return 0.8 * math.exp(-3.0 * progress)


def generate_epo_entropy_mask(
    per_token_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    baseline_h: float,
    *,
    mask_mode: str = "token",
    min_ratio: float = 0.8,
    max_ratio: float = 1.2,
    out_range_penalty: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Build EPO-lite band penalty mask against a scalar history baseline.

    Parameters
    ----------
    per_token_entropy : torch.Tensor
        Shape ``[B, S]``.
    response_mask : torch.Tensor
        Shape ``[B, S]``; used for ``entropy_mask_ratio`` denominator.
    baseline_h : float
        Historical mean entropy baseline ``H``.
    mask_mode : str
        ``"token"`` or ``"seq"``.
    min_ratio, max_ratio : float
        Band ``[H * min_ratio, H * max_ratio]``.
    out_range_penalty : float
        Penalty for values outside the band; in-band is ``0``.

    Returns
    -------
    entropy_mask : torch.Tensor
        Shape ``[B, S]``, band-in ``0``, band-out ``out_range_penalty``.
    entropy_mask_ratio : float
        Fraction of valid positions that are inside the band.
    """
    lower = baseline_h * min_ratio
    upper = baseline_h * max_ratio
    if mask_mode == "token":
        within = (per_token_entropy > lower) & (per_token_entropy < upper)
        entropy_mask = torch.where(
            within,
            torch.zeros_like(per_token_entropy),
            torch.full_like(per_token_entropy, out_range_penalty),
        )
    elif mask_mode == "seq":
        seq_avg = per_token_entropy.mean(dim=-1)
        within_seq = (seq_avg > lower) & (seq_avg < upper)
        seq_penalty = torch.where(
            within_seq,
            torch.zeros_like(seq_avg),
            torch.full_like(seq_avg, out_range_penalty),
        )
        entropy_mask = seq_penalty.unsqueeze(-1).expand_as(per_token_entropy)
        within = within_seq.unsqueeze(-1).expand_as(per_token_entropy)
    else:
        raise ValueError(f"Invalid mask_mode: {mask_mode}. Must be 'token' or 'seq'")

    valid = response_mask.bool()
    valid_n = valid.sum().clamp(min=1)
    within_valid = (within & valid).sum()
    entropy_mask_ratio = (within_valid.float() / valid_n.float()).item()
    return entropy_mask, entropy_mask_ratio


def _apply_epo_penalty(
    config,
    loss_input: PolicyLossInput,
    bwd_loss: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Record entropy into feature store and optionally add band penalty to loss."""
    epo_cfg = _task_epo_settings(config)
    per_token_entropy = loss_input.per_token_entropy
    response_mask = loss_input.response_mask
    if per_token_entropy is None:
        return bwd_loss, {}

    store = get_ppo_feature_store()
    with torch.no_grad():
        ent = per_token_entropy.detach().float()
        m = response_mask.detach().float()
        entropy_sum = (ent * m).sum().item()
        token_count = m.sum().item()
        store.record(FEATURE_NAME, entropy_sum, weight=token_count, reduce="mean")

    baseline_h = store.get_history_mean(FEATURE_NAME)
    if baseline_h is None:
        return bwd_loss, {}

    entropy_mask, entropy_mask_ratio = generate_epo_entropy_mask(
        per_token_entropy.detach(),
        response_mask,
        float(baseline_h),
        mask_mode=epo_cfg["mask_mode"],
        min_ratio=epo_cfg["min_ratio"],
        max_ratio=epo_cfg["max_ratio"],
        out_range_penalty=epo_cfg["out_range_penalty"],
    )
    phase_weight = 1.0
    if epo_cfg["enable_smooth_weights"]:
        cur_step = store.get_feature_axis_step(FEATURE_NAME)
        training_cfg = getattr(config, "training", None)
        total_steps = int(store.get(feature_axis_total_key(FEATURE_NAME)) or 0)
        if total_steps <= 0:
            total_steps = int(getattr(training_cfg, "total_ppo_step", 0) or 0)
        if cur_step is not None and total_steps > 0:
            phase_weight = calculate_epo_phase_weight(int(cur_step), total_steps)
    entropy_mask = entropy_mask * phase_weight
    entropy_penalty = masked_mean_per_sample_or_token(
        entropy_mask,
        response_mask,
        loss_input.cu_seqlens_padded,
        config.policy.override_transformer_config.get("calculate_per_token_loss", False),
        loss_input.local_cp_size,
    )
    epo_term = (
        entropy_penalty * epo_cfg["entropy_smooth_coeff"] * config.ppo.ppo_entropy_bonus
    )

    global_retention_ratio = loss_input.global_retention_ratio
    if hasattr(config, "debug") and getattr(config.debug, "ignore_global_retention_ratio", False):
        global_retention_ratio = None
    addend = epo_term
    if global_retention_ratio is not None:
        grr = global_retention_ratio.flatten()[0].clamp(min=1e-6)
        addend = addend / grr
    bwd_loss = bwd_loss + addend

    epo_metrics = {
        "epo_entropy_mask_ratio":
            torch.tensor(entropy_mask_ratio, device=response_mask.device, dtype=torch.float32),
        "epo_baseline_H":
            torch.tensor(float(baseline_h), device=response_mask.device, dtype=torch.float32),
        "epo_phase_weight":
            torch.tensor(phase_weight, device=response_mask.device, dtype=torch.float32),
        "epo_entropy_penalty":
            entropy_penalty.detach(),
    }
    return bwd_loss, epo_metrics


def epo_grpo_loss_func(config, loss_input: PolicyLossInput):
    """GRPO + EPO-lite entropy smoothing; register via ``loss_func_py_path``."""
    bwd_loss, metrics = grpo_loss_func(config, loss_input)
    bwd_loss, epo_metrics = _apply_epo_penalty(config, loss_input, bwd_loss)
    if not epo_metrics:
        return bwd_loss, metrics

    numel = loss_input.response_mask.sum()
    epo_only = {}
    for epo_key, epo_val in epo_metrics.items():
        if epo_val.dim() == 0:
            epo_only[epo_key] = torch.stack([epo_val * numel, numel])
        else:
            epo_only[epo_key] = epo_val
    reduce_metrics_across_data_parallel_group(epo_only)
    metrics.update(epo_only)
    return bwd_loss, metrics
