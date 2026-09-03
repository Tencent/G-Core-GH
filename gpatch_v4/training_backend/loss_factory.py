# 此文件后续将逐渐废弃，替换到 training_backend/loss 目录下
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import mpu, parallel_state, tensor_parallel

try:
    from megatron.core.extensions.transformer_engine import te_parallel_cross_entropy
except:
    te_parallel_cross_entropy = None
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.core.adaptive_entropy import get_entropy_bonus_coef
from gpatch_v4.core.correction_helper import compute_off_policy_correction_weights
from gpatch_v4.core.mappings import (
    all_gather_from_context_parallel_region,
    all_gather_from_context_parallel_region_no_zigzag,
    reduce_from_context_parallel_region,
)
from gpatch_v4.training_backend.vocab_parallel_entropy import vocab_parallel_entropy
from gpatch_v4.utils import (
    all_reduce_autograd,
    average_losses_across_data_parallel_group,
    from_parallel_logits_to_logprobs,
    masked_mean,
    reduce_metrics_across_data_parallel_group,
)
from gpatch_v4.utils.common_utils import import_fn_from_path
from gpatch_v4.utils.ppo_utils import calculate_kl_loss
from gpatch_v4.utils.training_utils import (
    from_parallel_logits_to_topk_logprobs,
    masked_sum,
    masked_sum_per_seq,
)

try:
    from gpatch_v4.kernel import linear_cross_entropy, set_linear_ce_backend
except Exception:
    linear_cross_entropy = None
    set_linear_ce_backend = None

LOSS_FUNC_REGISTRY: Dict[str, Callable] = {}


def _compute_clip_metrics(
    ratios: torch.Tensor,
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
    """Compute PPO ratio clamp diagnostic metrics.

    Returns
    -------
    Dict[str, torch.Tensor]
        Metrics in [sum, count] pair format:
        - ppo_ratio_clamped_upper_frac: tokens clamped by upper bound
        - ppo_ratio_clamped_lower_frac: tokens clamped by lower bound
        - ppo_ratio_clamped_effective_upper_frac: upper-clamped AND affecting loss
        - ppo_ratio_clamped_effective_lower_frac: lower-clamped AND affecting loss
        - ppo_dual_clip_frac: tokens where dual-clip is active
    """
    mask_bool = response_mask.bool()
    if ratios.dim() == 3:
        mask_bool = mask_bool.unsqueeze(-1)
    is_upper_clamped = (ratios > 1.0 + clip_ratio_high) & mask_bool
    is_lower_clamped = (ratios < 1.0 - clip_ratio_low) & mask_bool

    loss2_gt_loss1 = loss2 > loss1

    metrics = {
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
    cp_partition_mode: str = "zigzag"


def _normalize_local_cp_size(local_cp_size: Any) -> int:
    if isinstance(local_cp_size, torch.Tensor):
        return int(local_cp_size.item())
    return int(local_cp_size)


def _get_local_cp_group(local_cp_size: Any):
    local_cp_size = _normalize_local_cp_size(local_cp_size)
    if local_cp_size <= 1:
        return None
    return mpu.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)


def _thd_cu_seqlens_for_values(
    cu_seqlens_padded: torch.Tensor,
    values_numel: int,
    local_cp_size: int,
) -> torch.Tensor:
    """Map global THD ``cu_seqlens_padded`` to boundaries that match ``values``.

    Dyn-CP / TE THD CP shards token tensors into a local compact buffer of length
    ``global_tokens / local_cp_size``.  ``cu_seqlens_padded`` on the batch stays
    global (needed by attention ``PackedSeqParams``), but per-sample loss
    aggregation must slice the *local* buffer.

    TE ``thd_partition_indices_kernel`` lays local tokens out with
    ``local_cu[i] = global_cu[i] / cp_size`` (each sequence length is required to
    be divisible by ``cp_size * 2``).  When ``local_cp_size == 1``, ``values`` is
    the full packed tensor and global boundaries apply unchanged.
    """
    local_cp_size = _normalize_local_cp_size(local_cp_size)
    if local_cp_size <= 1:
        return cu_seqlens_padded

    global_tokens = int(cu_seqlens_padded[-1].item())
    assert global_tokens % local_cp_size == 0, (
        f"cu_seqlens_padded[-1]={global_tokens} not divisible by "
        f"local_cp_size={local_cp_size}"
    )
    expected_local = global_tokens // local_cp_size
    assert values_numel == expected_local, (
        f"THD per-sample: local values length {values_numel} != "
        f"cu_seqlens_padded[-1]//local_cp_size ({global_tokens}//{local_cp_size}). "
        "After dyn-cp CP shard, slice with local boundaries (cu // cp_size)."
    )
    # Exact integer division: TE requires each cu boundary % (cp * 2) == 0.
    return cu_seqlens_padded // local_cp_size


def _thd_per_sample_sum_and_count_local(
    values_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    cu_seqlens_for_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample masked sum/count on a (possibly CP-local) packed tensor."""
    num_samples = cu_seqlens_for_values.shape[0] - 1
    n = values_flat.numel()
    assert int(
        cu_seqlens_for_values[0].item()
    ) == 0, (f"cu_seqlens_for_values[0] must be 0, got {int(cu_seqlens_for_values[0].item())}")
    assert int(cu_seqlens_for_values[-1].item()) == n, (
        f"cu_seqlens_for_values[-1]={int(cu_seqlens_for_values[-1].item())} "
        f"!= values length {n}; refusing silent OOB slice"
    )
    per_sample_sums = []
    per_sample_counts = []
    for i in range(num_samples):
        s = int(cu_seqlens_for_values[i].item())
        e = int(cu_seqlens_for_values[i + 1].item())
        assert 0 <= s <= e <= n, (
            f"sample {i} bounds [{s}, {e}) out of range for values length {n}"
        )
        per_sample_sums.append(values_flat[s:e].sum())
        per_sample_counts.append(mask_flat[s:e].sum())
    return torch.stack(per_sample_sums), torch.stack(per_sample_counts)


def masked_mean_per_sample_or_token(
    values: torch.Tensor,
    mask: torch.Tensor,
    cu_seqlens_padded: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    local_cp_size: int = 1,
) -> torch.Tensor:
    """Aggregate per-token loss values into a scalar.

    When ``cu_seqlens_padded`` is None (non-THD), falls back to global
    ``masked_mean`` (token-weighted average).

    When ``cu_seqlens_padded`` is provided (THD packed format), computes
    per-sample masked_mean first, then averages across samples. This gives
    each sample equal weight regardless of response length, eliminating the
    length bias of per-token averaging.

    When ``local_cp_size > 1``, ``values``/``mask`` are CP-local compact shards
    while ``cu_seqlens_padded`` remains global. Boundaries are remapped via
    ``cu // local_cp_size`` (TE local layout), then per-sample numerator and
    denominator are all-reduced across the local CP group before division.
    """
    if cu_seqlens_padded is None or calculate_per_token_loss:
        return masked_mean(values, mask)

    assert not calculate_per_token_loss, (
        "THD per-sample loss is incompatible with calculate_per_token_loss=True"
    )

    # THD packed format: per-sample
    values_flat = torch.where(mask > 0, values, 0.0).reshape(-1)
    mask_flat = mask.reshape(-1)
    local_cp_size = _normalize_local_cp_size(local_cp_size)
    cu_for_values = _thd_cu_seqlens_for_values(
        cu_seqlens_padded, values_flat.numel(), local_cp_size
    )
    sums, counts = _thd_per_sample_sum_and_count_local(values_flat, mask_flat, cu_for_values)

    if local_cp_size > 1:
        cp_group = mpu.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
        combined = torch.stack([sums, counts])
        combined = reduce_from_context_parallel_region(combined, group=cp_group)
        sums, counts = combined[0], combined[1].detach()

    return (sums / counts.clamp(min=1)).mean()


def _masked_sample_sum_and_count(
    values: torch.Tensor,
    mask: torch.Tensor,
    cu_seqlens_padded: Optional[torch.Tensor] = None,
    local_cp_size: int = 1,
    sample_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample ``(sum_of_token_means, sample_count)`` for one micro-batch."""
    if cu_seqlens_padded is not None:
        values_flat = torch.where(mask > 0, values, 0.0).reshape(-1)
        mask_flat = mask.reshape(-1)
        local_cp_size = _normalize_local_cp_size(local_cp_size)
        cu_for_values = _thd_cu_seqlens_for_values(
            cu_seqlens_padded, values_flat.numel(), local_cp_size
        )
        num_samples = cu_for_values.shape[0] - 1
        sums, counts = _thd_per_sample_sum_and_count_local(values_flat, mask_flat, cu_for_values)

        if local_cp_size > 1:
            cp_group = mpu.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
            combined = torch.stack([sums, counts])
            combined = reduce_from_context_parallel_region(combined, group=cp_group)
            sums, counts = combined[0], combined[1].detach()

        # Matches `masked_mean_per_sample_or_token`'s `.mean()`: every
        # cu_seqlens_padded segment counts once toward the denominator (an
        # empty/dead segment naturally contributes 0 to the numerator via
        # `sums=0 -> 0/clamp(1)=0`, exactly like the pre-existing THD path).
        per_sample_mean = sums / counts.clamp(min=1)
        return per_sample_mean.sum(), sums.new_tensor(float(num_samples))

    # Plain 2D per-sample (non-THD, non-dynamic-CP): each row is one sample.
    count = sample_mask.sum() if sample_mask is not None else mask.new_tensor(float(mask.shape[0]))
    return masked_sum_per_seq(values, mask, sample_mask), count


def masked_sum_and_count_per_sample_or_token(
    values: torch.Tensor,
    mask: torch.Tensor,
    cu_seqlens_padded: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    local_cp_size: int = 1,
    sample_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GBS-scope token/sample sums+counts for reporting and backward.

    Always computes both per-token and per-sample aggregates for the local
    micro-batch, then returns three ``(sum, count)`` pairs:

    1. Detached token-sum / token-count -- for metrics / logging.
    2. Detached sample-sum / sample-count -- for metrics / logging.
    3. With-grad backward pair -- token-sum/count when
       ``calculate_per_token_loss=True``, else sample-sum/count.

    The caller accumulates the chosen backward numerator across micro-batches
    and DP (/CP) ranks and divides **once** by the matching global count --
    same pattern as ``global_n`` in ``Fsdp2EngineMixin._finetune_step`` for
    SFT. In dynamic-CP per-sample mode, CP-collaborator ranks each receive the
    whole-sample mean after the subgroup all-reduce; caller-side normalizer
    plumbing must account for that CP participation consistently.

    Supports both THD/dynamic-CP packed batches (``cu_seqlens_padded`` set)
    and plain 2D ``[B, S]`` batches (``cu_seqlens_padded=None``).

    Parameters
    ----------
    values : Tensor
        Per-token values, ``[B, S]``.
    mask : Tensor
        Per-token validity mask, ``[B, S]``; ``1`` marks tokens that count.
    cu_seqlens_padded : Tensor, optional
        THD packed-format sample boundaries. When ``None``, ``values``/``mask``
        are treated as a plain 2D batch where each row is one sample.
    calculate_per_token_loss : bool
        Selects which aggregate keeps gradients in the third pair.
        ``True``: token flat sum (no CP-subgroup reduce needed).
        ``False``: sum of per-sample token-means.
    local_cp_size : int
        When > 1, ``values``/``mask`` are CP-local compact shards while
        ``cu_seqlens_padded`` stays global; boundaries are remapped with
        ``cu // local_cp_size`` (TE layout), then per-sample sum/count are
        all-reduced over the dynamic-CP subgroup before dividing.
    sample_mask : Tensor, optional
        ``[B]``; drops dead rows from the plain-2D per-sample path.

    Returns
    -------
    token_sum_det, token_count_det, sample_sum_det, sample_count_det, bwd_sum, bwd_count
        Six scalars. The first four are detached; ``bwd_sum``/``bwd_count``
        carry gradients for the selected aggregation mode.
    """
    # Always compute both aggregates (reporting needs both). Only the selected
    # backward target keeps the autograd graph; the other runs under no_grad.
    token_sum_ctx = nullcontext() if calculate_per_token_loss else torch.no_grad()
    sample_sum_ctx = torch.no_grad() if calculate_per_token_loss else nullcontext()
    with token_sum_ctx:
        token_sum = masked_sum(values, mask)
        token_count = mask.sum()

    with sample_sum_ctx:
        sample_sum, sample_count = _masked_sample_sum_and_count(
            values, mask, cu_seqlens_padded, local_cp_size, sample_mask
        )

    if calculate_per_token_loss:
        bwd_sum, bwd_count = token_sum, token_count
    else:
        bwd_sum, bwd_count = sample_sum, sample_count

    return (
        token_sum.detach().clone(),
        token_count.detach().clone(),
        sample_sum.detach().clone(),
        sample_count.detach().clone(),
        bwd_sum,
        bwd_count,
    )


def _masked_sum_and_count_contiguous_cp(
    values: torch.Tensor,
    mask: torch.Tensor,
    cu_seqlens_padded: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    local_cp_size: int = 1,
    sample_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate a contiguous THD shard while preserving global sample boundaries."""
    local_cp_size = _normalize_local_cp_size(local_cp_size)
    if cu_seqlens_padded is None or local_cp_size <= 1:
        return masked_sum_and_count_per_sample_or_token(
            values,
            mask,
            cu_seqlens_padded,
            calculate_per_token_loss,
            local_cp_size,
            sample_mask,
        )

    cp_group = mpu.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
    local_rows = values.numel()
    global_rows = int(cu_seqlens_padded[-1].item())
    assert global_rows == local_rows * local_cp_size, (
        f"contiguous THD rows mismatch: global={global_rows}, "
        f"local={local_rows}, cp={local_cp_size}"
    )

    global_start = cp_group.rank() * local_rows
    local_cu = (cu_seqlens_padded - global_start).clamp_(0, local_rows)
    values_flat = torch.where(mask > 0, values, 0.0).reshape(-1)
    mask_flat = mask.reshape(-1)

    with nullcontext() if calculate_per_token_loss else torch.no_grad():
        token_sum = masked_sum(values, mask)
        token_count = mask.sum()

    with torch.no_grad() if calculate_per_token_loss else nullcontext():
        sums, counts = _thd_per_sample_sum_and_count_local(values_flat, mask_flat, local_cu)
        combined = torch.stack([sums, counts])
        dist.all_reduce(combined, group=cp_group, op=dist.ReduceOp.SUM)
        sums, counts = combined[0], combined[1]
        sample_sum = (sums / counts.clamp(min=1)).sum()
        sample_count = sums.new_tensor(float(local_cu.numel() - 1))

    if calculate_per_token_loss:
        bwd_sum, bwd_count = token_sum, token_count
    else:
        bwd_sum, bwd_count = sample_sum, sample_count

    return (
        token_sum.detach().clone(),
        token_count.detach().clone(),
        sample_sum.detach().clone(),
        sample_count.detach().clone(),
        bwd_sum,
        bwd_count,
    )


@dataclass
class FinetuneLossInput:
    logits: torch.Tensor
    batch: Dict[str, torch.Tensor]
    unwrapped_model: Optional[torch.nn.Module] = None
    skip_cp_loss_reduce: bool = False
    """Dict from ``_postprocess`` when linear CE is enabled.
    Expected keys: ``hidden_states``, ``weight``, ``runtime_gather_output``.
    Used by ``linear_ce`` loss to compute fused linear+CE."""
    linear_ce_input: Optional[Dict[str, Any]] = None
    """Context-parallel process group for loss reduction.
    When not None, this group is used instead of ``mpu.get_context_parallel_group()``.
    Set by the dynamic-CP path to the per-microbatch dynamic CP subgroup."""
    cp_group: Optional[Any] = None
    cur_mbs_step: Optional[int] = None
    num_mbs: Optional[int] = None
    """Frozen Student-TP shard of the Teacher output-layer weight."""
    teacher_output_weight: Optional[torch.Tensor] = None


def register_loss(name: str):
    """Decorator to register a loss function.

    Usage::

        @register_loss("grpo")
        def grpo_loss_func(...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        if name in LOSS_FUNC_REGISTRY:
            raise ValueError(
                f"Loss type '{name}' already registered by "
                f"{LOSS_FUNC_REGISTRY[name].__module__}.{LOSS_FUNC_REGISTRY[name].__qualname__}"
            )
        LOSS_FUNC_REGISTRY[name] = fn
        return fn

    return decorator


def register_custom_loss_fn(name: str, py_path: str, fn_name: str):
    """Import a loss function from *py_path* and register it under *name*.

    Parameters
    ----------
    name : str
        Loss name used in config.
    py_path : str
        Absolute path to a ``.py`` file.
    fn_name : str
        Callable name to import.
    """
    fn = import_fn_from_path(py_path, fn_name)
    if name in LOSS_FUNC_REGISTRY:
        raise ValueError(f"Loss function '{name}' is already registered.")
    LOSS_FUNC_REGISTRY[name] = fn


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


def _compute_steer_token_weights(
    advantages: torch.Tensor,
    prev_log_probs: torch.Tensor,
    curr_log_probs: torch.Tensor,
    prev_per_token_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    token_weight_min: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert False, "please use use_legacy_loss:False"
    with torch.no_grad():
        valid_mask = response_mask.bool()
        assert valid_mask.any(), "STEER requires at least one valid response token."

        # Calculate \delta
        curr_prob = curr_log_probs.detach().float().exp().clamp(min=1e-8, max=1.0 - 1e-8)
        x_one_minus_x_squared = curr_prob * (1.0 - curr_prob)
        ln_x_plus_h = torch.log(curr_prob) + prev_per_token_entropy.detach().float()
        f_x = x_one_minus_x_squared * ln_x_plus_h

        # Calculate A / old_log_prob
        prev_prob = prev_log_probs.detach().float().exp().clamp(min=1e-8, max=1.0)

        # Calculate \Omega
        metric = advantages.float() / prev_prob * f_x

        # STEER 源码多了这个判断
        # if not torch.isfinite(metric).all():
        #     log(f"[Token Weighting] Warning: Found non-finite values in metric")
        #     metric = torch.where(torch.isfinite(metric), metric, torch.zeros_like(metric))

        # Calculate abs(\Omega)
        entropy_change_metric = torch.abs(metric)

        valid_metric = entropy_change_metric[valid_mask]
        if valid_metric.numel() == 0:
            return torch.zeros_like(response_mask, dtype=torch.float)

        metric_max = valid_metric.max()

        # Initialize token weights
        token_weights = torch.zeros_like(metric, dtype=torch.float)

        # Match STEER's reference implementation: avoid strong rescaling when
        # the whole micro-batch has negligible estimated entropy change.
        normalizer = torch.maximum(metric_max, metric_max.new_tensor(0.02))
        k = -metric_max.new_tensor(token_weight_min).log() / normalizer
        valid_weights = torch.exp(-k * valid_metric)
        valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=1.0)

        # Apply computed weights to valid token positions
        token_weights[valid_mask] = valid_weights

        token_weights = token_weights * response_mask.float()

        metrics = {
            "steer/token_weight_mean": token_weights[valid_mask].mean(),
            "steer/token_weight_min": token_weights[valid_mask].min(),
            "steer/token_weight_max": token_weights[valid_mask].max(),
            "steer/entropy_change_metric_mean": valid_metric.mean(),
            "steer/entropy_change_metric_max": metric_max,
        }

    return token_weights, metrics


@register_loss("steer")
@register_loss("grpo")
def grpo_loss_func(config, loss_input: PolicyLossInput):
    ppo_config = config.ppo
    advantages = loss_input.advantages
    prev_log_probs = loss_input.prev_log_probs
    ref_log_probs = loss_input.ref_log_probs
    curr_log_probs = loss_input.curr_log_probs
    response_mask = loss_input.response_mask
    scaled_entropy = loss_input.scaled_entropy
    rollout_log_probs = loss_input.rollout_log_probs
    per_token_entropy = loss_input.per_token_entropy
    dumped_topk_logprobs = loss_input.dumped_topk_logprobs
    dumped_topk_token_ids = loss_input.dumped_topk_token_ids
    sample_mask = loss_input.sample_mask
    calculate_per_token_loss = loss_input.calculate_per_token_loss

    if ppo_config.skip_prev_logps:
        log_ratio = curr_log_probs - curr_log_probs.detach()
        effective_prev = curr_log_probs.detach()
    else:
        log_ratio = curr_log_probs - prev_log_probs
        effective_prev = prev_log_probs

    if ppo_config.ppo_logps_ratio_clamp is not None:
        ratios = torch.clamp(
            log_ratio, min=-ppo_config.ppo_logps_ratio_clamp, max=ppo_config.ppo_logps_ratio_clamp
        ).exp()
    else:
        ratios = log_ratio.exp()

    correction_ratio, response_mask, extra_metrics = compute_off_policy_correction_weights(
        config.ppo.enable_off_policy_correction,
        config,
        effective_prev,
        rollout_log_probs,
        response_mask.float(),
        loss_input.cu_seqlens_padded,
        _get_local_cp_group(loss_input.local_cp_size),
    )

    # support dapo Clip-Higher https://arxiv.org/pdf/2503.14476
    clip_ratio_low = ppo_config.ppo_clip_ratio_low if ppo_config.ppo_clip_ratio_low is not None else ppo_config.ppo_ratio_eps
    clip_ratio_high = ppo_config.ppo_clip_ratio_high if ppo_config.ppo_clip_ratio_high is not None else ppo_config.ppo_ratio_eps
    ratios_clamped = ratios.clamp(1.0 - clip_ratio_low, 1.0 + clip_ratio_high)

    loss1 = -advantages * ratios
    loss2 = -advantages * ratios_clamped

    entropy_reg_metrics = {}
    if ppo_config.ppo_entropy_regularization_type is not None:
        clip_max_loss, entropy_reg_metrics = compute_entropy_regularization_loss(
            ppo_config,
            old_log_prob=prev_log_probs,
            log_prob=curr_log_probs,
            advantages=advantages,
            response_mask=response_mask,
            pg_losses1=loss1,
            pg_losses2=loss2,
            entropy_aux_figures=loss_input.entropy_aux_figures,
            rollout_log_probs=loss_input.rollout_log_probs,
        )
    else:
        clip_max_loss = torch.maximum(loss1, loss2)

    # ref from: https://arxiv.org/pdf/1912.09729
    if ppo_config.ppo_dual_clip_ratio_c is not None:
        loss3 = -advantages * ppo_config.ppo_dual_clip_ratio_c
        clip_min_loss = torch.min(loss3, clip_max_loss)
        actor_loss = torch.where(advantages < 0, clip_min_loss, clip_max_loss)
    else:
        actor_loss = clip_max_loss

    if config.ppo.enable_off_policy_correction:
        actor_loss = actor_loss * correction_ratio

    steer_metrics = {}
    if ppo_config.loss_func == "steer":
        assert prev_log_probs is not None, "STEER requires prev_log_probs."
        prev_per_token_entropy = loss_input.prev_per_token_entropy
        assert prev_per_token_entropy is not None, "STEER requires prev_per_token_entropy."
        steer_token_weights, steer_metrics = _compute_steer_token_weights(
            advantages=advantages,
            prev_log_probs=prev_log_probs,
            curr_log_probs=curr_log_probs,
            prev_per_token_entropy=prev_per_token_entropy,
            response_mask=response_mask,
            token_weight_min=ppo_config.steer_token_weight_min,
        )
        actor_loss = actor_loss * steer_token_weights

    cu_seqlens_padded = loss_input.cu_seqlens_padded
    local_cp_size = loss_input.local_cp_size
    aggregate = (
        _masked_sum_and_count_contiguous_cp if loss_input.cp_partition_mode == "contiguous" else
        masked_sum_and_count_per_sample_or_token
    )

    (
        actor_token_sum,
        actor_token_count,
        actor_sample_sum,
        actor_sample_count,
        actor_bwd_sum,
        actor_bwd_count,
    ) = aggregate(
        actor_loss, response_mask, cu_seqlens_padded, calculate_per_token_loss, local_cp_size,
        sample_mask
    )
    (
        entropy_token_sum,
        entropy_token_count,
        entropy_sample_sum,
        entropy_sample_count,
        entropy_bwd_sum,
        entropy_bwd_count,
    ) = aggregate(
        per_token_entropy, response_mask, cu_seqlens_padded, calculate_per_token_loss,
        local_cp_size, sample_mask
    )
    loss = actor_bwd_sum - entropy_bwd_sum * get_entropy_bonus_coef(ppo_config)

    with torch.no_grad():
        ppo_ratio = masked_mean(ratios.detach(), response_mask)
        ppo_ratio_clamped = masked_mean(ratios_clamped.detach(), response_mask)
        scaled_entropy = scaled_entropy.detach()

        # dump metrics
        if loss_input.should_dump_metrics:
            dumped_curr_logprobs = curr_log_probs.clone().detach().to(
                dtype=torch.bfloat16, device="cpu"
            )
            dumped_per_token_entropy = per_token_entropy.clone().detach().to(
                dtype=torch.bfloat16, device="cpu"
            )
            ratios_tmp = ratios.detach()
            ratios_clamped_tmp = ratios_clamped.detach()
            dumped_ppo_ratio_unclamped = ratios_tmp.to(dtype=torch.bfloat16, device="cpu")
            dumped_is_ppo_ratio_clamped = (
                (ratios_tmp == ratios_clamped_tmp) & (response_mask.bool())
            ).cpu()
            dumped_mask = response_mask.detach().bool().to(device="cpu")

    use_absolute_kl = False
    use_low_var_kl = True
    if isinstance(config, OnPolicyDistillConfig):
        use_absolute_kl = True
        use_low_var_kl = False
    if ref_log_probs is not None:
        per_token_kl = calculate_kl_loss(
            cur_log_probs=curr_log_probs,
            ref_log_probs=ref_log_probs,
            use_absolute_kl=use_absolute_kl,
            use_low_var_kl=use_low_var_kl,
            clamp_kl_loss=ppo_config.ppo_dual_clip_ratio_c is not None,
            clamp_kl_val=ppo_config.ppo_clamp_kl_val,
        )
        (
            kl_token_sum,
            kl_token_count,
            kl_sample_sum,
            kl_sample_count,
            kl_bwd_sum,
            kl_bwd_count,
        ) = aggregate(
            per_token_kl, response_mask, cu_seqlens_padded, calculate_per_token_loss, local_cp_size,
            sample_mask
        )
        loss = loss + kl_bwd_sum * ppo_config.grpo_kl_loss_beta
    else:
        per_token_kl = None
        zero = torch.zeros_like(actor_token_sum)
        kl_token_sum = kl_sample_sum = kl_bwd_sum = zero
        kl_token_count = actor_token_count
        kl_sample_count = actor_sample_count
        kl_bwd_count = actor_bwd_count

    bwd_loss = loss.clone()

    with torch.no_grad():
        numel = response_mask.sum()

        clip_metrics = _compute_clip_metrics(
            ratios,
            clip_ratio_high,
            clip_ratio_low,
            loss1,
            loss2,
            advantages,
            clip_max_loss,
            response_mask,
            numel,
            loss3=loss3 if ppo_config.ppo_dual_clip_ratio_c is not None else None,
        )

        metrics = {
            # Main loss metrics: same GBS-scope numerators as backward.
            "loss": torch.stack([loss.detach(), actor_bwd_count.detach()]),
            "policy_loss": torch.stack([actor_bwd_sum.detach(),
                                        actor_bwd_count.detach()]),
            "scaled_entropy": torch.stack([entropy_bwd_sum.detach(),
                                           entropy_bwd_count.detach()]),
            "grpo_kl_loss": torch.stack([kl_bwd_sum.detach(),
                                         kl_bwd_count.detach()]),
            "ppo_ratio": torch.stack([ppo_ratio * numel, numel]),
            "ppo_ratio_clamped": torch.stack([ppo_ratio_clamped * numel, numel]),
            # Extra token / sample pairs for reporting (detached).
            "policy_loss_token": torch.stack([actor_token_sum, actor_token_count]),
            "policy_loss_sample": torch.stack([actor_sample_sum, actor_sample_count]),
            "entropy_token": torch.stack([entropy_token_sum, entropy_token_count]),
            "entropy_sample": torch.stack([entropy_sample_sum, entropy_sample_count]),
            "grpo_kl_token": torch.stack([kl_token_sum, kl_token_count]),
            "grpo_kl_sample": torch.stack([kl_sample_sum, kl_sample_count]),
            **clip_metrics,
        }

    if extra_metrics:
        metrics.update(extra_metrics)
    if entropy_reg_metrics:
        metrics.update(entropy_reg_metrics)
    if steer_metrics:
        metrics.update(steer_metrics)
    if sample_mask is not None:
        metrics["valid_sample_ratio"] = masked_mean(
            sample_mask.float().detach(), torch.ones_like(sample_mask, dtype=torch.float)
        )

    reduce_metrics_across_data_parallel_group(metrics)

    if loss_input.should_dump_metrics:
        metrics.update(
            {
                "dump/curr_logprobs": dumped_curr_logprobs,
                "dump/per_token_entropy": dumped_per_token_entropy,
                "dump/topk_logprobs": dumped_topk_logprobs,
                "dump/topk_token_ids": dumped_topk_token_ids,
                "dump/ppo_ratio_unclamped": dumped_ppo_ratio_unclamped,
                "dump/is_ppo_ratio_clamped": dumped_is_ppo_ratio_clamped,
                "dump/mask": dumped_mask,
            }
        )

    return (bwd_loss, metrics)


@register_loss("opd")
def opd_loss_func(config, loss_input: PolicyLossInput):
    """On-Policy Distill loss: GRPO actor + policy-ref KL + teacher-student KL.

    Backward contract matches GRPO: returns the SUM of per-sample token-means
    (seq-mean-token-mean numerator). Listed in ``SEQ_MEAN_RL_LOSS_FNS``.

    When ``advantages.dim() == 3`` (``log_prob_top_k > 0``), PPO ratio / clip is
    computed in 3D on student top-K log-probs, summed over K before aggregate;
    ratio metrics use softmax-weighted collapse to 2D.
    """
    ppo_config = config.ppo
    advantages = loss_input.advantages
    prev_log_probs = loss_input.prev_log_probs
    ref_log_probs = loss_input.ref_log_probs
    curr_log_probs = loss_input.curr_log_probs
    teacher_log_probs = loss_input.teacher_log_probs
    response_mask = loss_input.response_mask
    scaled_entropy = loss_input.scaled_entropy
    rollout_log_probs = loss_input.rollout_log_probs
    per_token_entropy = loss_input.per_token_entropy
    sample_mask = loss_input.sample_mask
    calculate_per_token_loss = loss_input.calculate_per_token_loss

    assert teacher_log_probs is not None, ("opd_loss requires teacher_log_probs in PolicyLossInput")

    if per_token_entropy is None:
        per_token_entropy = torch.zeros_like(curr_log_probs)

    # ---- ratio computation (2D label-based or 3D top-K) ----
    if advantages.dim() == 3:
        assert loss_input.curr_topk_logprobs is not None and loss_input.prev_topk_logprobs is not None
        log_ratio = loss_input.curr_topk_logprobs - loss_input.prev_topk_logprobs
    else:
        log_ratio = curr_log_probs - prev_log_probs

    if ppo_config.ppo_logps_ratio_clamp is not None:
        ratios = torch.clamp(
            log_ratio,
            min=-ppo_config.ppo_logps_ratio_clamp,
            max=ppo_config.ppo_logps_ratio_clamp,
        ).exp()
    else:
        ratios = log_ratio.exp()

    correction_ratio, response_mask, extra_metrics = compute_off_policy_correction_weights(
        config.ppo.enable_off_policy_correction,
        config,
        prev_log_probs,
        rollout_log_probs,
        response_mask.float(),
        loss_input.cu_seqlens_padded,
        _get_local_cp_group(loss_input.local_cp_size),
    )

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
            old_log_prob=prev_log_probs,
            log_prob=curr_log_probs,
            advantages=advantages,
            response_mask=response_mask,
            pg_losses1=loss1,
            pg_losses2=loss2,
            entropy_aux_figures=loss_input.entropy_aux_figures,
            rollout_log_probs=loss_input.rollout_log_probs,
        )
    else:
        clip_max_loss = torch.maximum(loss1, loss2)

    if ppo_config.ppo_dual_clip_ratio_c is not None:
        loss3 = -advantages * ppo_config.ppo_dual_clip_ratio_c
        clip_min_loss = torch.min(loss3, clip_max_loss)
        actor_loss = torch.where(advantages < 0, clip_min_loss, clip_max_loss)
    else:
        actor_loss = clip_max_loss

    if config.ppo.enable_off_policy_correction:
        # correction_ratio 是 2D，top-K 时 broadcast 到 K 维。
        if advantages.dim() == 3:
            actor_loss = actor_loss * correction_ratio.unsqueeze(-1)
        else:
            actor_loss = actor_loss * correction_ratio

    if advantages.dim() == 3:
        # sum over K 压回 2D before aggregate.
        actor_loss = actor_loss.sum(dim=-1)
        # ratio metrics: softmax-weighted collapse to 2D (same scale as non-top-K).
        ratio_weights = torch.softmax(loss_input.prev_topk_logprobs.detach(), dim=-1)
        ratios_for_metric = (ratios.detach() * ratio_weights).sum(dim=-1)
        ratios_clamped_for_metric = (ratios_clamped.detach() * ratio_weights).sum(dim=-1)
    else:
        ratios_for_metric = ratios.detach()
        ratios_clamped_for_metric = ratios_clamped.detach()

    cu_seqlens_padded = loss_input.cu_seqlens_padded
    local_cp_size = loss_input.local_cp_size
    aggregate = (
        _masked_sum_and_count_contiguous_cp if loss_input.cp_partition_mode == "contiguous" else
        masked_sum_and_count_per_sample_or_token
    )

    (
        actor_token_sum,
        actor_token_count,
        actor_sample_sum,
        actor_sample_count,
        actor_bwd_sum,
        actor_bwd_count,
    ) = aggregate(
        actor_loss, response_mask, cu_seqlens_padded, calculate_per_token_loss, local_cp_size,
        sample_mask
    )
    (
        entropy_token_sum,
        entropy_token_count,
        entropy_sample_sum,
        entropy_sample_count,
        entropy_bwd_sum,
        entropy_bwd_count,
    ) = aggregate(
        per_token_entropy, response_mask, cu_seqlens_padded, calculate_per_token_loss,
        local_cp_size, sample_mask
    )
    loss = actor_bwd_sum - entropy_bwd_sum * get_entropy_bonus_coef(ppo_config)

    with torch.no_grad():
        ppo_ratio = masked_mean(ratios_for_metric, response_mask)
        ppo_ratio_clamped = masked_mean(ratios_clamped_for_metric, response_mask)
        scaled_entropy = scaled_entropy.detach()

    if ref_log_probs is not None:
        per_token_policy_ref_kl = calculate_kl_loss(
            cur_log_probs=curr_log_probs,
            ref_log_probs=ref_log_probs,
            use_absolute_kl=False,
            use_low_var_kl=True,
            clamp_kl_loss=ppo_config.ppo_dual_clip_ratio_c is not None,
            clamp_kl_val=ppo_config.ppo_clamp_kl_val,
        )
        (
            policy_ref_kl_token_sum,
            policy_ref_kl_token_count,
            policy_ref_kl_sample_sum,
            policy_ref_kl_sample_count,
            policy_ref_kl_bwd_sum,
            policy_ref_kl_bwd_count,
        ) = aggregate(
            per_token_policy_ref_kl, response_mask, cu_seqlens_padded, calculate_per_token_loss,
            local_cp_size, sample_mask
        )
        loss = loss + policy_ref_kl_bwd_sum * ppo_config.grpo_kl_loss_beta
    else:
        zero = torch.zeros_like(actor_token_sum)
        policy_ref_kl_token_sum = policy_ref_kl_sample_sum = policy_ref_kl_bwd_sum = zero
        policy_ref_kl_token_count = actor_token_count
        policy_ref_kl_sample_count = actor_sample_count
        policy_ref_kl_bwd_count = actor_bwd_count

    # teacher-student KL: always low-var (not absolute).
    per_token_teacher_kl = calculate_kl_loss(
        cur_log_probs=curr_log_probs,
        ref_log_probs=teacher_log_probs,
        use_absolute_kl=False,
        use_low_var_kl=True,
        clamp_kl_loss=ppo_config.ppo_dual_clip_ratio_c is not None,
        clamp_kl_val=ppo_config.ppo_clamp_kl_val,
    )
    (
        teacher_kl_token_sum,
        teacher_kl_token_count,
        teacher_kl_sample_sum,
        teacher_kl_sample_count,
        teacher_kl_bwd_sum,
        teacher_kl_bwd_count,
    ) = aggregate(
        per_token_teacher_kl, response_mask, cu_seqlens_padded, calculate_per_token_loss,
        local_cp_size, sample_mask
    )
    loss = loss + teacher_kl_bwd_sum * ppo_config.opd_teacher_kl_loss_beta

    bwd_loss = loss.clone()

    with torch.no_grad():
        numel = response_mask.sum()

        clip_metrics = _compute_clip_metrics(
            ratios,
            clip_ratio_high,
            clip_ratio_low,
            loss1,
            loss2,
            advantages,
            clip_max_loss,
            response_mask,
            numel,
            loss3=loss3 if ppo_config.ppo_dual_clip_ratio_c is not None else None,
        )

        metrics = {
            "loss":
                torch.stack([loss.detach(), actor_bwd_count.detach()]),
            "policy_loss":
                torch.stack([actor_bwd_sum.detach(),
                             actor_bwd_count.detach()]),
            "scaled_entropy":
                torch.stack([entropy_bwd_sum.detach(),
                             entropy_bwd_count.detach()]),
            "policy_ref_kl_loss":
                torch.stack([policy_ref_kl_bwd_sum.detach(),
                             policy_ref_kl_bwd_count.detach()]),
            "teacher_student_kl_loss":
                torch.stack([teacher_kl_bwd_sum.detach(),
                             teacher_kl_bwd_count.detach()]),
            "ppo_ratio":
                torch.stack([ppo_ratio * numel, numel]),
            "ppo_ratio_clamped":
                torch.stack([ppo_ratio_clamped * numel, numel]),
            "policy_loss_token":
                torch.stack([actor_token_sum, actor_token_count]),
            "policy_loss_sample":
                torch.stack([actor_sample_sum, actor_sample_count]),
            "entropy_token":
                torch.stack([entropy_token_sum, entropy_token_count]),
            "entropy_sample":
                torch.stack([entropy_sample_sum, entropy_sample_count]),
            "policy_ref_kl_token":
                torch.stack([policy_ref_kl_token_sum, policy_ref_kl_token_count]),
            "policy_ref_kl_sample":
                torch.stack([policy_ref_kl_sample_sum, policy_ref_kl_sample_count]),
            "teacher_student_kl_token":
                torch.stack([teacher_kl_token_sum, teacher_kl_token_count]),
            "teacher_student_kl_sample":
                torch.stack([teacher_kl_sample_sum, teacher_kl_sample_count]),
            **clip_metrics,
        }

    if extra_metrics:
        metrics.update(extra_metrics)
    if entropy_reg_metrics:
        metrics.update(entropy_reg_metrics)

    reduce_metrics_across_data_parallel_group(metrics)

    return (bwd_loss, metrics)


@register_loss("gspo")
def gspo_loss_func(config, loss_input: PolicyLossInput):
    """GSPO-token loss (arxiv 2507.18071 §4.3).

    Key differences from standard GRPO:
      1. Sequence-level importance ratio instead of token-level ratio.
      2. Sequence-level loss aggregation: mean over tokens per sequence, then
         mean over sequences (each sequence has equal weight regardless of length).
      3. Metrics use plain mean/min/max over the full ratio tensor.
    """
    assert loss_input.cu_seqlens_padded is None, (
        "gspo_loss_func expects response-padded [B, S] "
        "(cu_seqlens_padded=None). Convert dyn-CP THD packs before loss "
        "(mixin._rl_response_pad_dyn_cp_tensors)."
    )
    ppo_config = config.ppo
    advantages = loss_input.advantages
    prev_log_probs = loss_input.prev_log_probs
    ref_log_probs = loss_input.ref_log_probs
    curr_log_probs = loss_input.curr_log_probs
    response_mask = loss_input.response_mask
    scaled_entropy = loss_input.scaled_entropy
    rollout_log_probs = loss_input.rollout_log_probs
    per_token_entropy = loss_input.per_token_entropy
    dumped_topk_logprobs = loss_input.dumped_topk_logprobs
    dumped_topk_token_ids = loss_input.dumped_topk_token_ids
    sample_mask = loss_input.sample_mask

    # ------------------------------------------------------------------
    # 1. Sequence-level importance ratio
    # ------------------------------------------------------------------
    if ppo_config.skip_prev_logps:
        effective_prev = curr_log_probs.detach()
    else:
        effective_prev = prev_log_probs
    negative_approx_kl = curr_log_probs - effective_prev
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = (torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths)

    log_seq_importance_ratio = (
        curr_log_probs - curr_log_probs.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    )
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
    ratios = torch.exp(log_seq_importance_ratio)

    # ------------------------------------------------------------------
    # 2. Off-policy correction (shared with GRPO)
    # ------------------------------------------------------------------
    correction_ratio, response_mask, extra_metrics = compute_off_policy_correction_weights(
        config.ppo.enable_off_policy_correction,
        config,
        effective_prev,
        rollout_log_probs,
        response_mask.float(),
        loss_input.cu_seqlens_padded,
        _get_local_cp_group(loss_input.local_cp_size),
    )

    # ------------------------------------------------------------------
    # 3. Clipped surrogate loss (supports DAPO clip-higher & dual-clip)
    # ------------------------------------------------------------------
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
            old_log_prob=prev_log_probs,
            log_prob=curr_log_probs,
            advantages=advantages,
            response_mask=response_mask,
            pg_losses1=loss1,
            pg_losses2=loss2,
            entropy_aux_figures=loss_input.entropy_aux_figures,
            rollout_log_probs=loss_input.rollout_log_probs,
        )
    else:
        clip_max_loss = torch.maximum(loss1, loss2)

    # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
    if ppo_config.ppo_dual_clip_ratio_c is not None:
        loss3 = -advantages * ppo_config.ppo_dual_clip_ratio_c
        clip_min_loss = torch.min(loss3, clip_max_loss)
        actor_loss = torch.where(advantages < 0, clip_min_loss, clip_max_loss)
    else:
        actor_loss = clip_max_loss

    if config.ppo.enable_off_policy_correction:
        actor_loss = actor_loss * correction_ratio

    # ------------------------------------------------------------------
    # 4. Sequence-level loss aggregation
    #    L = (1/G) Σ_i (1/|y_i|) Σ_t l_{i,t}
    # ------------------------------------------------------------------
    seq_losses = (
        torch.sum(actor_loss * response_mask, dim=-1) /
        torch.sum(response_mask, dim=-1).clamp(min=1)
    )
    if sample_mask is not None:
        actor_loss = (seq_losses * sample_mask).sum() / sample_mask.sum().clamp(min=1)
    else:
        actor_loss = torch.mean(seq_losses)

    loss = actor_loss - scaled_entropy * get_entropy_bonus_coef(ppo_config)

    # ------------------------------------------------------------------
    # 5. KL regularization (token-level, same as GRPO)
    # ------------------------------------------------------------------
    use_absolute_kl = False
    use_low_var_kl = True
    if isinstance(config, OnPolicyDistillConfig):
        use_absolute_kl = True
        use_low_var_kl = False
    kl_loss = masked_mean(
        calculate_kl_loss(
            cur_log_probs=curr_log_probs,
            ref_log_probs=ref_log_probs,
            use_absolute_kl=use_absolute_kl,
            use_low_var_kl=use_low_var_kl,
            clamp_kl_loss=ppo_config.ppo_dual_clip_ratio_c is not None,
            clamp_kl_val=ppo_config.ppo_clamp_kl_val,
        ),
        response_mask,
    )
    loss = loss + kl_loss * ppo_config.grpo_kl_loss_beta

    # ------------------------------------------------------------------
    # 6. Metrics (token-level sum/count pairs)
    # ------------------------------------------------------------------
    bwd_loss = loss.clone()

    global_retention_ratio = loss_input.global_retention_ratio
    if hasattr(config, "debug") and getattr(config.debug, "ignore_global_retention_ratio", False):
        # DEBUG: skip the 1/global_retention_ratio compensation for grad-scaling experiments.
        global_retention_ratio = None
    if global_retention_ratio is not None:
        global_retention_ratio_scalar = global_retention_ratio.flatten()[0].clamp(min=1e-6)
        bwd_loss = bwd_loss / global_retention_ratio_scalar

    with torch.no_grad():
        numel = response_mask.sum()
        ones = torch.tensor(1.0, device=loss.device)

        ppo_ratio_sum = (ratios.detach() * response_mask).sum()
        ppo_ratio_clamped_sum = (ratios_clamped.detach() * response_mask).sum()

        clip_metrics = _compute_clip_metrics(
            ratios,
            clip_ratio_high,
            clip_ratio_low,
            loss1,
            loss2,
            advantages,
            clip_max_loss,
            response_mask,
            numel,
            loss3=loss3 if ppo_config.ppo_dual_clip_ratio_c is not None else None,
        )

        # Sequence-level clip fraction (GSPO-specific)
        mask_bool = response_mask.bool()
        token_clipped = (ratios.detach() != ratios_clamped.detach()) & mask_bool
        seq_has_clip = token_clipped.any(dim=-1)  # [batch]
        num_seqs = torch.tensor(float(seq_has_clip.shape[0]), device=loss.device)
        num_clipped_seqs = seq_has_clip.sum().float()
        token_clip_effective = token_clipped & (loss2 > loss1)
        num_clip_effective_seqs = token_clip_effective.any(dim=-1).sum().float()

        if loss_input.should_dump_metrics:
            ratios_tmp = ratios.detach()
            ratios_clamped_tmp = ratios_clamped.detach()
            dumped_curr_logprobs = curr_log_probs.clone().detach().to(
                dtype=torch.bfloat16, device="cpu"
            )
            dumped_per_token_entropy = per_token_entropy.clone().detach().to(
                dtype=torch.bfloat16, device="cpu"
            )
            dumped_ppo_ratio_unclamped = ratios_tmp.to(device="cpu")
            dumped_is_ppo_ratio_clamped = (
                (ratios_tmp == ratios_clamped_tmp) & (response_mask.bool())
            ).cpu()
            dumped_mask = response_mask.detach().bool().to(device="cpu")

    metrics = {
        "loss": torch.stack([loss.detach(), ones]),
        "policy_loss": torch.stack([actor_loss.detach(), ones]),
        "ppo_ratio": torch.stack([ppo_ratio_sum, numel]),
        "ppo_ratio_clamped": torch.stack([ppo_ratio_clamped_sum, numel]),
        "scaled_entropy": torch.stack([scaled_entropy.detach() * numel, numel]),
        "grpo_kl_loss": torch.stack([kl_loss.detach() * numel, numel]),
        **clip_metrics,
        "ppo_ratio_clamped_seq_frac": torch.stack([num_clipped_seqs, num_seqs]),
        "ppo_ratio_clamped_seq_effective_frac": torch.stack([num_clip_effective_seqs, num_seqs]),
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    if entropy_reg_metrics:
        metrics.update(entropy_reg_metrics)
    if sample_mask is not None:
        metrics["valid_sample_ratio"] = masked_mean(
            sample_mask.float().detach(), torch.ones_like(sample_mask, dtype=torch.float)
        )

    if global_retention_ratio is not None:
        grr_scalar = global_retention_ratio.flatten()[0].clamp(min=1e-6)
        metrics["loss_dead_aware"] = torch.stack([loss.detach() / grr_scalar, ones])
        metrics["policy_loss_dead_aware"] = torch.stack([actor_loss.detach() / grr_scalar, ones])
    reduce_metrics_across_data_parallel_group(metrics)

    if loss_input.should_dump_metrics:
        metrics.update(
            {
                "dump/curr_logprobs": dumped_curr_logprobs,
                "dump/per_token_entropy": dumped_per_token_entropy,
                "dump/topk_logprobs": dumped_topk_logprobs,
                "dump/topk_token_ids": dumped_topk_token_ids,
                "dump/ppo_ratio_unclamped": dumped_ppo_ratio_unclamped,
                "dump/is_ppo_ratio_clamped": dumped_is_ppo_ratio_clamped,
                "dump/mask": dumped_mask,
            }
        )

    return (bwd_loss, metrics)


@register_loss("cispo")
def cispo_loss_func(config, loss_input: PolicyLossInput):
    """CISPO loss (arXiv 2506.13585, MiniMax-M1 §3.1).

    Clipped IS-weight Policy Optimization: clips the importance sampling
    weight and stop-gradients it, so gradient flows only through ``log π``.
    All tokens contribute gradients regardless of IS ratio magnitude.

    ``L = -sg(clip(r, 1-ε_low, 1+ε_high)) · A · log π``

    Recommended: set ``ppo_clip_ratio_low`` large (e.g. 1.0) to disable the
    lower bound and only tune ``ppo_clip_ratio_high``.
    """
    ppo_config = config.ppo
    advantages = loss_input.advantages
    prev_log_probs = loss_input.prev_log_probs
    ref_log_probs = loss_input.ref_log_probs
    curr_log_probs = loss_input.curr_log_probs
    response_mask = loss_input.response_mask
    rollout_log_probs = loss_input.rollout_log_probs
    per_token_entropy = loss_input.per_token_entropy
    dumped_topk_logprobs = loss_input.dumped_topk_logprobs
    dumped_topk_token_ids = loss_input.dumped_topk_token_ids
    sample_mask = loss_input.sample_mask

    # ------------------------------------------------------------------
    # 1. Token-level importance ratio
    # ------------------------------------------------------------------
    if ppo_config.skip_prev_logps:
        log_ratio = curr_log_probs - curr_log_probs.detach()
        effective_prev = curr_log_probs.detach()
    else:
        log_ratio = curr_log_probs - prev_log_probs
        effective_prev = prev_log_probs

    if ppo_config.ppo_logps_ratio_clamp is not None:
        ratios = torch.clamp(
            log_ratio, min=-ppo_config.ppo_logps_ratio_clamp, max=ppo_config.ppo_logps_ratio_clamp
        ).exp()
    else:
        ratios = log_ratio.exp()

    # ------------------------------------------------------------------
    # 2. Off-policy correction (shared with GRPO)
    # ------------------------------------------------------------------
    correction_ratio, response_mask, extra_metrics = compute_off_policy_correction_weights(
        config.ppo.enable_off_policy_correction,
        config,
        effective_prev,
        rollout_log_probs,
        response_mask.float(),
        loss_input.cu_seqlens_padded,
        _get_local_cp_group(loss_input.local_cp_size),
    )

    # ------------------------------------------------------------------
    # 3. CISPO: clip IS weight, stop-gradient, multiply with A * log π
    # ------------------------------------------------------------------
    clip_ratio_low = ppo_config.ppo_clip_ratio_low if ppo_config.ppo_clip_ratio_low is not None else ppo_config.ppo_ratio_eps
    clip_ratio_high = ppo_config.ppo_clip_ratio_high if ppo_config.ppo_clip_ratio_high is not None else ppo_config.ppo_ratio_eps
    clipped_ratio = ratios.clamp(1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    # stop-gradient the clipped ratio
    clipped_ratio_sg = clipped_ratio.detach()

    actor_loss = -clipped_ratio_sg * advantages * curr_log_probs

    if config.ppo.enable_off_policy_correction:
        actor_loss = actor_loss * correction_ratio

    calculate_per_token_loss = loss_input.calculate_per_token_loss
    cu_seqlens_padded = loss_input.cu_seqlens_padded
    local_cp_size = loss_input.local_cp_size
    (
        actor_token_sum,
        actor_token_count,
        actor_sample_sum,
        actor_sample_count,
        actor_bwd_sum,
        actor_bwd_count,
    ) = masked_sum_and_count_per_sample_or_token(
        actor_loss,
        response_mask,
        cu_seqlens_padded,
        calculate_per_token_loss,
        local_cp_size,
        sample_mask,
    )
    (
        entropy_token_sum,
        entropy_token_count,
        entropy_sample_sum,
        entropy_sample_count,
        entropy_bwd_sum,
        entropy_bwd_count,
    ) = masked_sum_and_count_per_sample_or_token(
        per_token_entropy,
        response_mask,
        cu_seqlens_padded,
        calculate_per_token_loss,
        local_cp_size,
        sample_mask,
    )
    loss = actor_bwd_sum - entropy_bwd_sum * get_entropy_bonus_coef(ppo_config)

    # ------------------------------------------------------------------
    # 4. KL regularization (same as GRPO)
    # paper 里说不要 kl_loss 了，虽然代码这里保留了这一块的，但是建议 kl_loss_beta 设置为 0
    # ------------------------------------------------------------------
    use_absolute_kl = False
    use_low_var_kl = True
    if isinstance(config, OnPolicyDistillConfig):
        use_absolute_kl = True
        use_low_var_kl = False
    if ref_log_probs is not None:
        per_token_kl = calculate_kl_loss(
            cur_log_probs=curr_log_probs,
            ref_log_probs=ref_log_probs,
            use_absolute_kl=use_absolute_kl,
            use_low_var_kl=use_low_var_kl,
            clamp_kl_loss=ppo_config.ppo_dual_clip_ratio_c is not None,
            clamp_kl_val=ppo_config.ppo_clamp_kl_val,
        )
        (
            kl_token_sum,
            kl_token_count,
            kl_sample_sum,
            kl_sample_count,
            kl_bwd_sum,
            kl_bwd_count,
        ) = masked_sum_and_count_per_sample_or_token(
            per_token_kl,
            response_mask,
            cu_seqlens_padded,
            calculate_per_token_loss,
            local_cp_size,
            sample_mask,
        )
        loss = loss + kl_bwd_sum * ppo_config.grpo_kl_loss_beta
    else:
        zero = torch.zeros_like(actor_token_sum)
        kl_token_sum = kl_sample_sum = kl_bwd_sum = zero
        kl_token_count = actor_token_count
        kl_sample_count = actor_sample_count
        kl_bwd_count = actor_bwd_count

    # ------------------------------------------------------------------
    # 5. Backward loss
    # ------------------------------------------------------------------
    # Dead-sample retention is applied by the training backend's global
    # normalizer (per-sample: N = retention * gbs; per-token: mask.sum()
    # already ignores dead rows). Do NOT divide bwd_loss by retention here
    # or the compensation is applied twice. Keep the ratio only for
    # logging-only ``*_dead_aware`` metrics below.
    bwd_loss = loss.clone()

    global_retention_ratio = loss_input.global_retention_ratio
    if hasattr(config, "debug") and getattr(config.debug, "ignore_global_retention_ratio", False):
        global_retention_ratio = None

    # ------------------------------------------------------------------
    # 6. Metrics
    # ------------------------------------------------------------------
    with torch.no_grad():
        numel = response_mask.sum()
        ppo_ratio = masked_mean(ratios.detach(), response_mask)
        ppo_ratio_clamped = masked_mean(clipped_ratio_sg, response_mask)
        mask_bool = response_mask.bool()
        is_clamped = (clipped_ratio_sg != ratios.detach()) & mask_bool
        clipfrac = is_clamped.sum().float() / numel.clamp(min=1)
        is_upper_clamped = (ratios.detach() > 1.0 + clip_ratio_high) & mask_bool
        is_lower_clamped = (ratios.detach() < 1.0 - clip_ratio_low) & mask_bool

        if loss_input.should_dump_metrics:
            dumped_curr_logprobs = curr_log_probs.clone().detach().to(
                dtype=torch.bfloat16, device="cpu"
            )
            dumped_per_token_entropy = per_token_entropy.clone().detach().to(
                dtype=torch.bfloat16, device="cpu"
            )
            ratios_tmp = ratios.detach()
            ratios_clamped_tmp = clipped_ratio_sg
            dumped_ppo_ratio_unclamped = ratios_tmp.to(dtype=torch.bfloat16, device="cpu")
            dumped_is_ppo_ratio_clamped = ((ratios_tmp == ratios_clamped_tmp) & mask_bool).cpu()
            dumped_mask = response_mask.detach().bool().to(device="cpu")

    metrics = {
        "loss": torch.stack([loss.detach(), actor_bwd_count.detach()]),
        "policy_loss": torch.stack([actor_bwd_sum.detach(),
                                    actor_bwd_count.detach()]),
        "ppo_ratio": torch.stack([ppo_ratio * numel, numel]),
        "ppo_ratio_clamped": torch.stack([ppo_ratio_clamped * numel, numel]),
        "scaled_entropy": torch.stack([entropy_bwd_sum.detach(),
                                       entropy_bwd_count.detach()]),
        "grpo_kl_loss": torch.stack([kl_bwd_sum.detach(),
                                     kl_bwd_count.detach()]),
        "policy_loss_token": torch.stack([actor_token_sum.detach(),
                                          actor_token_count.detach()]),
        "policy_loss_sample": torch.stack([actor_sample_sum.detach(),
                                           actor_sample_count.detach()]),
        "entropy_token": torch.stack([entropy_token_sum.detach(),
                                      entropy_token_count.detach()]),
        "entropy_sample": torch.stack([entropy_sample_sum.detach(),
                                       entropy_sample_count.detach()]),
        "grpo_kl_token": torch.stack([kl_token_sum.detach(),
                                      kl_token_count.detach()]),
        "grpo_kl_sample": torch.stack([kl_sample_sum.detach(),
                                       kl_sample_count.detach()]),
        "ppo_ratio_clamped_upper_frac": torch.stack([is_upper_clamped.sum().float(), numel]),
        "ppo_ratio_clamped_lower_frac": torch.stack([is_lower_clamped.sum().float(), numel]),
        "cispo/clipfrac": clipfrac,
    }

    if extra_metrics:
        metrics.update(extra_metrics)
    if sample_mask is not None:
        metrics["valid_sample_ratio"] = masked_mean(
            sample_mask.float().detach(), torch.ones_like(sample_mask, dtype=torch.float)
        )

    if global_retention_ratio is not None:
        grr_scalar = global_retention_ratio.flatten()[0].clamp(min=1e-6)
        metrics["loss_dead_aware"] = torch.stack(
            [loss.detach() / grr_scalar, actor_bwd_count.detach()]
        )
        metrics["policy_loss_dead_aware"] = torch.stack(
            [actor_bwd_sum.detach() / grr_scalar,
             actor_bwd_count.detach()]
        )

    reduce_metrics_across_data_parallel_group(metrics)

    if loss_input.should_dump_metrics:
        metrics.update(
            {
                "dump/curr_logprobs": dumped_curr_logprobs,
                "dump/per_token_entropy": dumped_per_token_entropy,
                "dump/topk_logprobs": dumped_topk_logprobs,
                "dump/topk_token_ids": dumped_topk_token_ids,
                "dump/ppo_ratio_unclamped": dumped_ppo_ratio_unclamped,
                "dump/is_ppo_ratio_clamped": dumped_is_ppo_ratio_clamped,
                "dump/mask": dumped_mask,
            }
        )

    return (bwd_loss, metrics)


@register_loss("sapo")
def sapo_loss_func(config, loss_input: PolicyLossInput):
    """SAPO loss (arXiv 2511.20347): Soft Adaptive Policy Optimization.

    Replaces the GRPO/GSPO hard clip with a smooth, temperature-controlled soft
    gate on the token-level importance ratio ``r = exp(curr - prev)``::

        f(r) = (4 / τ) * sigmoid(τ * (r - 1)),   τ = τ_pos if A > 0 else τ_neg

    The per-token surrogate is ``f(r) * A`` (``r`` carries gradient). Its gradient
    kernel ``sech^2(τ/2 * (r - 1)) = 4·σ·(1-σ)`` preserves gradients near the
    on-policy point (r=1) and attenuates smoothly as the ratio deviates, instead
    of truncating like a hard clip. Asymmetric temperatures (``τ_neg > τ_pos``)
    make negative-token gradients decay faster, improving stability.

    Aggregation defaults to the paper Eq.(5) seq-mean-token-mean when
    ``calculate_per_token_loss=False`` (each sequence weighted equally). Set
    ``calculate_per_token_loss=True`` for GRPO-style global token-mean / token
    sum. Both aggregation modes support Dynamic CP: sequence means reduce each
    sample's token sum/count across its CP subgroup before normalization, while
    token mean aggregates local token sums globally. There is no hard / dual
    clipping; ``ppo_clip_ratio_*`` and ``ppo_dual_clip_ratio_c`` are inert here.
    """
    ppo_config = config.ppo
    advantages = loss_input.advantages
    prev_log_probs = loss_input.prev_log_probs
    ref_log_probs = loss_input.ref_log_probs
    curr_log_probs = loss_input.curr_log_probs
    response_mask = loss_input.response_mask
    scaled_entropy = loss_input.scaled_entropy
    rollout_log_probs = loss_input.rollout_log_probs
    per_token_entropy = loss_input.per_token_entropy
    dumped_topk_logprobs = loss_input.dumped_topk_logprobs
    dumped_topk_token_ids = loss_input.dumped_topk_token_ids
    sample_mask = loss_input.sample_mask
    cu_seqlens_padded = loss_input.cu_seqlens_padded
    local_cp_size = loss_input.local_cp_size
    calculate_per_token_loss = loss_input.calculate_per_token_loss

    # ------------------------------------------------------------------
    # 1. Token-level importance ratio (supports skip_prev_logps)
    # ------------------------------------------------------------------
    if ppo_config.skip_prev_logps:
        effective_prev = curr_log_probs.detach()
    else:
        effective_prev = prev_log_probs
    log_ratio = curr_log_probs - effective_prev
    if ppo_config.ppo_logps_ratio_clamp is not None:
        log_ratio = torch.clamp(
            log_ratio,
            min=-ppo_config.ppo_logps_ratio_clamp,
            max=ppo_config.ppo_logps_ratio_clamp,
        )
    ratios = log_ratio.exp()

    # ------------------------------------------------------------------
    # 2. Off-policy correction (shared with GRPO/GSPO)
    # ------------------------------------------------------------------
    correction_ratio, response_mask, extra_metrics = compute_off_policy_correction_weights(
        config.ppo.enable_off_policy_correction,
        config,
        effective_prev,
        rollout_log_probs,
        response_mask.float(),
        loss_input.cu_seqlens_padded,
        _get_local_cp_group(loss_input.local_cp_size),
    )

    # ------------------------------------------------------------------
    # 3. SAPO soft gate (no hard clip)
    #    f(r) = (4 / τ) * sigmoid(τ * (r - 1)),  τ chosen by advantage sign
    # ------------------------------------------------------------------
    # TODO: gate 计算独立出来，后面可能有其他 gate 方法？
    tau = torch.where(
        advantages > 0,
        torch.as_tensor(ppo_config.sapo_tau_pos, dtype=ratios.dtype, device=ratios.device),
        torch.as_tensor(ppo_config.sapo_tau_neg, dtype=ratios.dtype, device=ratios.device),
    )
    sigmoid_gate = torch.sigmoid(tau * (ratios - 1.0))
    gate = (4.0 / tau) * sigmoid_gate
    actor_loss = -advantages * gate

    if config.ppo.enable_off_policy_correction:
        actor_loss = actor_loss * correction_ratio

    # ------------------------------------------------------------------
    # 4. Aggregate the token-level SAPO surrogate.
    #
    # Shared helper selects token-sum (``calculate_per_token_loss=True``) or
    # seq-mean-token-mean (paper Eq.(5)).  For the latter under Dynamic CP it
    # remaps THD boundaries onto the compact local CP shard and all-reduces
    # each sample's sum/count before forming per-sample means.  Returns a
    # numerator/count pair so Megatron can normalize once over the PPO step.
    # ------------------------------------------------------------------
    (
        actor_token_sum,
        actor_token_count,
        actor_sample_sum,
        actor_sample_count,
        actor_bwd_sum,
        actor_bwd_count,
    ) = masked_sum_and_count_per_sample_or_token(
        actor_loss,
        response_mask,
        cu_seqlens_padded,
        calculate_per_token_loss,
        local_cp_size,
        sample_mask,
    )
    (
        entropy_token_sum,
        entropy_token_count,
        entropy_sample_sum,
        entropy_sample_count,
        entropy_bwd_sum,
        entropy_bwd_count,
    ) = masked_sum_and_count_per_sample_or_token(
        per_token_entropy,
        response_mask,
        cu_seqlens_padded,
        calculate_per_token_loss,
        local_cp_size,
        sample_mask,
    )
    loss = actor_bwd_sum - entropy_bwd_sum * get_entropy_bonus_coef(ppo_config)

    # ------------------------------------------------------------------
    # 5. KL regularization.  Use the same aggregation as the actor term so a
    # single global denominator remains valid for dynamic CP.
    # ------------------------------------------------------------------
    use_absolute_kl = False
    use_low_var_kl = True
    if isinstance(config, OnPolicyDistillConfig):
        use_absolute_kl = True
        use_low_var_kl = False
    if ref_log_probs is not None:
        per_token_kl = calculate_kl_loss(
            cur_log_probs=curr_log_probs,
            ref_log_probs=ref_log_probs,
            use_absolute_kl=use_absolute_kl,
            use_low_var_kl=use_low_var_kl,
            clamp_kl_loss=ppo_config.ppo_dual_clip_ratio_c is not None,
            clamp_kl_val=ppo_config.ppo_clamp_kl_val,
        )
        (
            kl_token_sum,
            kl_token_count,
            kl_sample_sum,
            kl_sample_count,
            kl_bwd_sum,
            kl_bwd_count,
        ) = masked_sum_and_count_per_sample_or_token(
            per_token_kl,
            response_mask,
            cu_seqlens_padded,
            calculate_per_token_loss,
            local_cp_size,
            sample_mask,
        )
        loss = loss + kl_bwd_sum * ppo_config.grpo_kl_loss_beta
    else:
        per_token_kl = None
        zero = torch.zeros_like(actor_token_sum)
        kl_token_sum = kl_sample_sum = kl_bwd_sum = zero
        kl_token_count = actor_token_count
        kl_sample_count = actor_sample_count
        kl_bwd_count = actor_bwd_count

    # ------------------------------------------------------------------
    # 6. Metrics (gspo-style seq-level loss + SAPO gate diagnostics)
    # ------------------------------------------------------------------
    bwd_loss = loss.clone()  # TODO: 为什么需要clone

    with torch.no_grad():
        numel = response_mask.sum()
        mask_bool = response_mask.bool()

        ppo_ratio_sum = (ratios.detach() * response_mask).sum()

        # SAPO gate diagnostics (0-dim scalars; DP-reduced by key-name convention).
        gate_det = gate.detach()
        # Gradient kernel sech^2(τ/2·(r-1)) = 4·σ·(1-σ); 1.0 at r=1, → 0 off-policy.
        grad_kernel = 4.0 * sigmoid_gate.detach() * (1.0 - sigmoid_gate.detach())
        valid_gate = gate_det[mask_bool]
        valid_kernel = grad_kernel[mask_bool]
        if valid_gate.numel() > 0:
            gate_mean = valid_gate.mean()
            gate_min = valid_gate.min()
            gate_max = valid_gate.max()
            grad_kernel_mean = valid_kernel.mean()
            strong_attenuation_frac = (valid_kernel < 0.5).float().mean()
        else:
            zero = torch.tensor(0.0, device=loss.device)
            gate_mean = gate_min = gate_max = grad_kernel_mean = strong_attenuation_frac = zero

        if loss_input.should_dump_metrics:
            dumped_curr_logprobs = curr_log_probs.clone().detach().to(
                dtype=torch.bfloat16, device="cpu"
            )
            dumped_per_token_entropy = per_token_entropy.clone().detach().to(
                dtype=torch.bfloat16, device="cpu"
            )
            dumped_ratios = ratios.detach().to(dtype=torch.bfloat16, device="cpu")
            dumped_mask = response_mask.detach().bool().to(device="cpu")

    metrics = {
        "loss": torch.stack([loss.detach(), actor_bwd_count.detach()]),
        "policy_loss": torch.stack([actor_bwd_sum.detach(),
                                    actor_bwd_count.detach()]),
        "ppo_ratio": torch.stack([ppo_ratio_sum, numel]),
        "scaled_entropy": torch.stack([entropy_bwd_sum.detach(),
                                       entropy_bwd_count.detach()]),
        "grpo_kl_loss": torch.stack([kl_bwd_sum.detach(),
                                     kl_bwd_count.detach()]),
        "policy_loss_token": torch.stack([actor_token_sum, actor_token_count]),
        "policy_loss_sample": torch.stack([actor_sample_sum, actor_sample_count]),
        "entropy_token": torch.stack([entropy_token_sum, entropy_token_count]),
        "entropy_sample": torch.stack([entropy_sample_sum, entropy_sample_count]),
        "grpo_kl_token": torch.stack([kl_token_sum, kl_token_count]),
        "grpo_kl_sample": torch.stack([kl_sample_sum, kl_sample_count]),
        # on-policy 理论 =2/sapo_tau_pos (adv>=0) =2/sapo_tau_neg (adv<0)，越小 ratio 差别越大
        "sapo/gate_mean": gate_mean,
        # >= 0 | 越偏离 2/sapo_tau_pos(neg) ratio 差别越大
        "sapo/gate_min": gate_min,
        # <= 4/sapo_tau_pos(eng)，越小 ratio 差别越大
        "sapo/gate_max": gate_max,
        # <=1 | on-policy 理论=1 | 越小 ratio 差别越大
        "sapo/grad_kernel_mean": grad_kernel_mean,
        # <=1 | on-policy 理论=0 | 越大 ratio 差别越大
        "sapo/strong_attenuation_frac": strong_attenuation_frac,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    if sample_mask is not None:
        metrics["valid_sample_ratio"] = masked_mean(
            sample_mask.float().detach(), torch.ones_like(sample_mask, dtype=torch.float)
        )
    reduce_metrics_across_data_parallel_group(metrics)

    if loss_input.should_dump_metrics:
        metrics.update(
            {
                "dump/curr_logprobs": dumped_curr_logprobs,
                "dump/per_token_entropy": dumped_per_token_entropy,
                "dump/topk_logprobs": dumped_topk_logprobs,
                "dump/topk_token_ids": dumped_topk_token_ids,
                "dump/ppo_ratio_unclamped": dumped_ratios,
                "dump/mask": dumped_mask,
            }
        )

    return (bwd_loss, metrics)


@register_loss("fipo")
def fipo_loss_func(config, loss_input: PolicyLossInput):
    """FIPO loss (arXiv 2603.19835): Future-KL Influenced Policy Optimization.

    Uses exponentially-decayed Future-KL to compute per-token influence weights,
    enabling dense token-level credit assignment that breaks the length-stagnation
    bottleneck of standard GRPO.

    Algorithm
    ---------
    1. Compute importance ratio and delta_logp.
    2. Filter extreme IS tokens (> dual-clip threshold).
    3. Chunked matmul to accumulate exponentially-decayed Future-KL.
    4. Convert Future-KL to influence weights with clipping.
    5. Safety mechanism for negative-advantage + high IS tokens.
    6. Standard PPO clipped surrogate loss with weighted advantages.
    7. Sequence-level filtering (discard sequences with >1 dual-clip token).
    8. KL regularisation via ``grpo_kl_loss_beta``.
    """
    ppo_config = config.ppo
    advantages = loss_input.advantages
    prev_log_probs = loss_input.prev_log_probs
    ref_log_probs = loss_input.ref_log_probs
    curr_log_probs = loss_input.curr_log_probs
    response_mask = loss_input.response_mask
    scaled_entropy = loss_input.scaled_entropy
    rollout_log_probs = loss_input.rollout_log_probs

    # ------------------------------------------------------------------
    # 1. Importance ratio
    # ------------------------------------------------------------------
    assert ppo_config.ppo_logps_ratio_clamp is not None
    negative_approx_kl = curr_log_probs - prev_log_probs
    negative_approx_kl = torch.clamp(
        negative_approx_kl,
        min=-ppo_config.ppo_logps_ratio_clamp,
        max=ppo_config.ppo_logps_ratio_clamp,
    )
    ratios = torch.exp(negative_approx_kl)

    # ------------------------------------------------------------------
    # 2. Off-policy correction (shared infrastructure)
    # ------------------------------------------------------------------
    correction_ratio, response_mask, extra_metrics = compute_off_policy_correction_weights(
        config.ppo.enable_off_policy_correction,
        config,
        prev_log_probs,
        rollout_log_probs,
        response_mask.float(),
        loss_input.cu_seqlens_padded,
        _get_local_cp_group(loss_input.local_cp_size),
    )

    # ------------------------------------------------------------------
    # 3. Delta logp & extreme-value filtering
    # ------------------------------------------------------------------
    delta_logp = negative_approx_kl

    # Use dual-clip ratio c as the IS filter threshold (log-space)
    filter_threshold = None
    assert ppo_config.ppo_dual_clip_ratio_c is not None and ppo_config.ppo_dual_clip_ratio_c > 1.0, \
        f"ppo_dual_clip_ratio_c must be set and greater than 1.0, but got {ppo_config.ppo_dual_clip_ratio_c}"

    filter_threshold = torch.log(
        torch.tensor(
            ppo_config.ppo_dual_clip_ratio_c,
            device=curr_log_probs.device,
            dtype=curr_log_probs.dtype,
        )
    )

    kl_response_premask = delta_logp * response_mask.to(curr_log_probs.dtype)
    participation_mask = ~(delta_logp > filter_threshold)
    kl_response = kl_response_premask * participation_mask.to(curr_log_probs.dtype)

    # ------------------------------------------------------------------
    # 4. Chunked Future-KL computation
    #    future_kl[b, i] = Σ_{j>=i} γ^(j-i) * kl_response[b, j]
    # ------------------------------------------------------------------
    batch_size, response_len = curr_log_probs.shape
    device = curr_log_probs.device
    dtype = curr_log_probs.dtype

    decay_rate = ppo_config.fipo_decay_rate
    gamma = 2.0**(-1.0 / decay_rate)
    gamma_t = torch.tensor(gamma, dtype=dtype, device=device)
    chunk_size = ppo_config.fipo_chunk_size

    future_kl = torch.zeros((batch_size, response_len), device=device, dtype=dtype)
    pos_i = torch.arange(response_len, device=device).unsqueeze(1)  # (L, 1)

    for j_start in range(0, response_len, chunk_size):
        j_end = min(response_len, j_start + chunk_size)
        j_idx = torch.arange(j_start, j_end, device=device).unsqueeze(0)  # (1, K)
        distance = j_idx - pos_i  # (L, K)
        causal_mask = (distance >= 0).to(dtype)
        decay_block = torch.pow(gamma_t, distance.clamp(min=0)) * causal_mask  # (L, K)
        kl_block = kl_response[:, j_start:j_end]  # (B, K)
        # (B, K) @ (K, L) -> (B, L)
        contrib = torch.matmul(kl_block, decay_block.t())
        future_kl += contrib

    # ------------------------------------------------------------------
    # 5. Convert Future-KL to influence weights
    # ------------------------------------------------------------------
    fipo_clip_ratio = ppo_config.fipo_clip_ratio
    if ppo_config.fipo_clip_high_only:
        upper_bound = 1.0 + fipo_clip_ratio
        lower_bound = 1.0
        influence_weights = torch.clamp(torch.exp(future_kl), min=1.0, max=upper_bound).detach()
    else:
        upper_bound = 1.0 + fipo_clip_ratio
        lower_bound = 1.0 - fipo_clip_ratio
        influence_weights = torch.clamp(torch.exp(future_kl), min=lower_bound,
                                        max=upper_bound).detach()

    # Safety: cap influence weight for negative-advantage + high IS tokens
    safe_threshold = ppo_config.fipo_safety_thresh
    mask_neg_high_is = (advantages < 0) & (ratios > safe_threshold)
    influence_weights = torch.where(
        mask_neg_high_is,
        torch.clamp(influence_weights, min=0.8, max=1.0),
        influence_weights,
    )

    # ------------------------------------------------------------------
    # 6. Weighted advantages + standard PPO clipped surrogate loss
    # ------------------------------------------------------------------
    weighted_advantages = advantages * influence_weights

    clip_ratio_low = (
        ppo_config.ppo_clip_ratio_low
        if ppo_config.ppo_clip_ratio_low is not None else ppo_config.ppo_ratio_eps
    )
    clip_ratio_high = (
        ppo_config.ppo_clip_ratio_high
        if ppo_config.ppo_clip_ratio_high is not None else ppo_config.ppo_ratio_eps
    )
    ratios_clamped = ratios.clamp(1.0 - clip_ratio_low, 1.0 + clip_ratio_high)

    loss1 = -weighted_advantages * ratios
    loss2 = -weighted_advantages * ratios_clamped

    entropy_reg_metrics = {}
    if ppo_config.ppo_entropy_regularization_type is not None:
        clip_max_loss, entropy_reg_metrics = compute_entropy_regularization_loss(
            ppo_config,
            old_log_prob=prev_log_probs,
            log_prob=curr_log_probs,
            advantages=advantages,  # use original advantages for covariance
            response_mask=response_mask,
            pg_losses1=loss1,
            pg_losses2=loss2,
            entropy_aux_figures=loss_input.entropy_aux_figures,
            rollout_log_probs=loss_input.rollout_log_probs,
        )
    else:
        clip_max_loss = torch.maximum(loss1, loss2)

    # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
    loss3 = -weighted_advantages * ppo_config.ppo_dual_clip_ratio_c
    clip_min_loss = torch.min(loss3, clip_max_loss)
    actor_loss_per_token = torch.where(weighted_advantages < 0, clip_min_loss, clip_max_loss)

    # ------------------------------------------------------------------
    # 7. Sequence-level filtering: discard sequences with >1 dual-clip token
    # ------------------------------------------------------------------
    active_mask = response_mask.bool()
    if ppo_config.fipo_correction_aware_filter and correction_ratio is not None:
        active_mask = active_mask & (correction_ratio > 0)
    lower_clip_mask = ((advantages < 0) & (clip_max_loss > loss3) & active_mask)
    low_clip_token_counts = lower_clip_mask.sum(dim=1)
    seq_valid = (low_clip_token_counts <= 1).unsqueeze(1)
    final_mask = (response_mask.bool() & seq_valid).to(dtype)

    if config.ppo.enable_off_policy_correction:
        actor_loss_per_token = actor_loss_per_token * correction_ratio

    actor_loss = masked_mean_per_sample_or_token(
        actor_loss_per_token,
        final_mask,
        loss_input.cu_seqlens_padded,
        config.policy.override_transformer_config.get("calculate_per_token_loss", False),
        loss_input.local_cp_size,
    )
    loss = actor_loss - scaled_entropy * get_entropy_bonus_coef(ppo_config)

    # ------------------------------------------------------------------
    # 8. KL regularisation
    # ------------------------------------------------------------------
    kl_loss = masked_mean(
        calculate_kl_loss(
            cur_log_probs=curr_log_probs,
            ref_log_probs=ref_log_probs,
            use_absolute_kl=False,
            use_low_var_kl=True,
            clamp_kl_loss=ppo_config.ppo_dual_clip_ratio_c is not None,
            clamp_kl_val=ppo_config.ppo_clamp_kl_val,
        ),
        response_mask,
    )
    loss = loss + kl_loss * ppo_config.grpo_kl_loss_beta

    # ------------------------------------------------------------------
    # 9. Metrics (token-level sum/count pairs for base metrics)
    # ------------------------------------------------------------------
    bwd_loss = loss.clone()

    with torch.no_grad():
        numel = response_mask.sum()
        final_numel = final_mask.sum()

        ppo_ratio_sum = (ratios.detach() * response_mask).sum()
        ppo_ratio_clamped_sum = (ratios_clamped.detach() * response_mask).sum()

        # FIPO-specific metrics (scalar, aggregated via .mean() in consumer)
        iw_mean = masked_mean(influence_weights, response_mask)
        valid_iw = influence_weights[response_mask.bool()]
        iw_min = valid_iw.min() if valid_iw.numel() > 0 else torch.tensor(0.0, device=device)
        iw_max = valid_iw.max() if valid_iw.numel() > 0 else torch.tensor(0.0, device=device)

        clip_frac_upper = masked_mean(
            (influence_weights >= upper_bound - 1e-7).float(), response_mask
        )
        clip_frac_lower = masked_mean(
            (influence_weights <= lower_bound + 1e-7).float(), response_mask
        )
        clip_frac_total = clip_frac_upper + clip_frac_lower

        # Raw (un-clipped) influence weight stats
        raw_iw = torch.exp(future_kl)
        iw_mean_raw = masked_mean(raw_iw, response_mask)
        valid_raw = raw_iw[response_mask.bool()]
        iw_min_raw = valid_raw.min() if valid_raw.numel() > 0 else torch.tensor(0.0, device=device)
        iw_max_raw = valid_raw.max() if valid_raw.numel() > 0 else torch.tensor(0.0, device=device)

        # Importance ratio tail diagnostics (quantiles are DP-averaged as approximation)
        neg_valid = ratios[(advantages < 0) & response_mask.bool()]
        if neg_valid.numel() > 0:
            neg_is_p995 = torch.quantile(neg_valid, 0.995)
            neg_is_p999 = torch.quantile(neg_valid, 0.999)
        else:
            neg_is_p995 = torch.tensor(0.0, device=device)
            neg_is_p999 = torch.tensor(0.0, device=device)

        pos_valid = ratios[(advantages > 0) & response_mask.bool()]
        if pos_valid.numel() > 0:
            pos_is_p999 = torch.quantile(pos_valid, 0.999)
        else:
            pos_is_p999 = torch.tensor(0.0, device=device)

        pos_mini_frac = masked_mean(((ratios < 1e-3) & (advantages > 0)).float(), response_mask)

    metrics = {
        "loss": torch.stack([loss.detach() * final_numel, final_numel]),
        "policy_loss": torch.stack([actor_loss.detach() * final_numel, final_numel]),
        "ppo_ratio": torch.stack([ppo_ratio_sum, numel]),
        "ppo_ratio_clamped": torch.stack([ppo_ratio_clamped_sum, numel]),
        "scaled_entropy": torch.stack([scaled_entropy.detach() * numel, numel]),
        "grpo_kl_loss": torch.stack([kl_loss.detach() * numel, numel]),
        # FIPO-specific (scalar)
        "fipo/influence_weights_mean": iw_mean,
        "fipo/influence_weights_min": iw_min,
        "fipo/influence_weights_max": iw_max,
        "fipo/clip_frac_total": clip_frac_total,
        "fipo/clip_frac_upper": clip_frac_upper,
        "fipo/clip_frac_lower": clip_frac_lower,
        "fipo/influence_weights_mean_raw": iw_mean_raw,
        "fipo/influence_weights_min_raw": iw_min_raw,
        "fipo/influence_weights_max_raw": iw_max_raw,
        "fipo/neg_is_p995": neg_is_p995,
        "fipo/neg_is_p999": neg_is_p999,
        "fipo/pos_is_p999": pos_is_p999,
        "fipo/pos_mini_frac": pos_mini_frac,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    if entropy_reg_metrics:
        metrics.update(entropy_reg_metrics)
    reduce_metrics_across_data_parallel_group(metrics)

    return (bwd_loss, metrics)


def ce_loss(
    config,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    kl_alpha_mask: torch.Tensor = None,
    return_src_loss: bool = False,
    skip_cp_reduce: bool = False,
    cp_group=None,
    loss_weights: Optional[torch.Tensor] = None,
):
    labels = labels.transpose(0, 1).contiguous()
    logits = logits.transpose(0, 1).clone(memory_format=torch.contiguous_format)

    if config.training.cross_entropy_loss_fusion:
        if config.training.cross_entropy_fusion_impl == 'te':
            if te_parallel_cross_entropy is not None:
                labels = torch.as_strided(labels, labels.size(), (labels.size()[1], 1))
                loss = te_parallel_cross_entropy(
                    logits, labels, parallel_state.get_tensor_model_parallel_group()
                )
            else:
                raise RuntimeError("Trying to use a TE block when it's not present.")
        elif config.training.cross_entropy_fusion_impl == 'native':
            loss = fused_vocab_parallel_cross_entropy(
                logits, labels, parallel_state.get_tensor_model_parallel_group()
            )
    else:
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)

    # [s b] => [b, s]
    loss = loss.transpose(0, 1).contiguous()
    if return_src_loss:
        return loss

    if kl_alpha_mask is not None:
        ce_weight = (1 - kl_alpha_mask).view(-1, 1).float()
        loss = loss * ce_weight
        loss_mask = loss_mask * (ce_weight != 0)

    losses = loss.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    if loss_weights is not None:
        weighted_mask = loss_mask * loss_weights.view(-1).float()
    else:
        weighted_mask = loss_mask

    loss = torch.sum(losses * weighted_mask)
    total_tokens = loss_mask.sum()
    loss = torch.cat([loss.view(1), total_tokens.view(1)])

    # When calculate_per_token_loss=True the pipeline schedule + finalize_model_grads
    # handle CP gradient aggregation.  Skipping the AVG here keeps each CP rank's
    # local loss_sum/tokens so finalize_model_grads produces correctly scaled gradients.
    _cp_group = cp_group if cp_group is not None else mpu.get_context_parallel_group()
    if not skip_cp_reduce and dist.get_world_size(_cp_group) > 1:
        torch.distributed.all_reduce(loss, group=_cp_group, op=torch.distributed.ReduceOp.AVG)
    return loss


@register_loss("cross_entropy")
def cross_entroy_loss_func(
    config,
    loss_input: FinetuneLossInput,
):
    """LM cross-entropy loss.

    Args:
        labels (Tensor): ``[B, S]``.
        logits (Tensor): ``[B, S, V]`` from the output layer.
        loss_mask (Tensor): ``[B, S]``.
    Returns:
        Tensor: ``[B, S]``.
    """
    batch = loss_input.batch
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    loss_weights = batch.get("loss_weights", None)

    if loss_input.linear_ce_input is not None:
        assert config.training.return_hidden_states_for_ce
        hidden_ce_loss = (
            linear_ce_loss if config.training.use_linear_ce else compact_ce_loss
        )
        loss = hidden_ce_loss(
            config,
            loss_input.linear_ce_input,
            labels,
            loss_mask,
            skip_cp_reduce=loss_input.skip_cp_loss_reduce,
            cp_group=loss_input.cp_group,
            loss_weights=loss_weights,
        )
    else:
        logits = loss_input.logits.float()
        loss = ce_loss(
            config,
            logits,
            labels,
            loss_mask,
            skip_cp_reduce=loss_input.skip_cp_loss_reduce,
            cp_group=loss_input.cp_group,
            loss_weights=loss_weights,
        )
    #TODO: check loss nan or not

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    # metrics: [sum, count] 对，由消费端统一做 DP all-reduce 再除
    metrics = {"lm_loss": loss.clone().detach()}

    # dump metrics
    if batch.get("should_dump_metrics", False):
        assert not config.training.return_hidden_states_for_ce, "暂不支持，因为没有 logits"
        metrics.update(
            {"dump/mask": batch["full_loss_mask"].detach().bool().to(device="cpu")}
        )

        if config.training.dump_metrics_logprobs_topk > 0:
            topk_logprobs, topk_token_ids = from_parallel_logits_to_topk_logprobs(
                vocab_parallel_logits=logits,
                topk=config.training.dump_metrics_logprobs_topk,
                temperature=1.0,
            )
            topk_logprobs = topk_logprobs.to(dtype=torch.bfloat16, device="cpu")
            topk_token_ids = topk_token_ids.to(dtype=torch.int32, device="cpu")
            metrics.update(
                {
                    "dump/topk_logprobs": topk_logprobs,
                    "dump/topk_token_ids": topk_token_ids,
                }
            )

        if config.training.ppo_dump_per_token_entropy:
            _, per_token_entropy_unmasked = vocab_parallel_entropy(logits.detach().clone())
            metrics.update(
                {
                    "dump/per_token_entropy":
                        per_token_entropy_unmasked.to(dtype=torch.bfloat16, device="cpu")
                }
            )

    return (loss[0].clone(), local_num_tokens, metrics)


def compact_ce_loss(
    config,
    linear_ce_input: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    kl_alpha_mask: torch.Tensor = None,
    return_src_loss: bool = False,
    skip_cp_reduce: bool = False,
    cp_group=None,
    loss_weights: Optional[torch.Tensor] = None,
):
    """Ordinary vocab-parallel CE on compacted hidden states.

    Returns:
        If ``return_src_loss``: per-token loss ``[b, s]``.
        Otherwise: ``torch.Tensor`` ``[loss_sum, total_tokens]``.
    """
    assert config.training.ce_compaction
    assert not config.training.use_linear_ce
    labels = labels.transpose(0, 1).contiguous()

    hidden_states = linear_ce_input["hidden_states"]
    output_layer = linear_ce_input["output_layer"]
    tp_group = output_layer.tp_group
    if output_layer.sequence_parallel:
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=False,
        )
    weight = linear_ce_input["weight"]
    if weight is None:
        weight = output_layer.weight
    assert output_layer.bias is None, (
        "CE compaction requires an output projection without bias"
    )
    assert not output_layer.gather_output
    assert not linear_ce_input["runtime_gather_output"]

    loss_mask_sb = loss_mask.transpose(0, 1).contiguous()
    assert hidden_states.shape[:-1] == labels.shape == loss_mask_sb.shape, (
        "compact CE shape mismatch: "
        f"{hidden_states.shape=}, {labels.shape=}, {loss_mask_sb.shape=}"
    )
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1)
    keep_indices = (loss_mask_sb.reshape(-1) != 0).nonzero(as_tuple=False).flatten()
    num_valid_tokens = keep_indices.numel()
    if num_valid_tokens == 0:
        zero_loss = (flat_hidden[:0].sum() + weight[:0].sum()).float()
        loss = zero_loss.expand(labels.numel()).view_as(labels)
    else:
        compact_hidden = flat_hidden.index_select(0, keep_indices).unsqueeze(1)
        compact_labels = flat_labels.index_select(0, keep_indices)
        assert compact_hidden.shape[-1] == weight.shape[-1]
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            # Enter TP region so vocab-parallel CE all-reduces d_hidden.
            compact_hidden = tensor_parallel.copy_to_tensor_model_parallel_region(
                compact_hidden,
                group=tp_group,
            )
        compact_logits = F.linear(compact_hidden, weight).float()
        compact_labels = compact_labels.unsqueeze(1)
        if config.training.cross_entropy_loss_fusion:
            if config.training.cross_entropy_fusion_impl == 'te':
                if te_parallel_cross_entropy is not None:
                    compact_labels = torch.as_strided(
                        compact_labels,
                        compact_labels.size(),
                        (compact_labels.size()[1], 1),
                    )
                    compact_loss = te_parallel_cross_entropy(
                        compact_logits,
                        compact_labels,
                        parallel_state.get_tensor_model_parallel_group(),
                    )
                else:
                    raise RuntimeError("Trying to use a TE block when it's not present.")
            elif config.training.cross_entropy_fusion_impl == 'native':
                compact_loss = fused_vocab_parallel_cross_entropy(
                    compact_logits,
                    compact_labels,
                    parallel_state.get_tensor_model_parallel_group(),
                )
        else:
            compact_loss = tensor_parallel.vocab_parallel_cross_entropy(
                compact_logits,
                compact_labels,
            )
        loss = compact_loss.new_zeros(labels.numel())
        loss = loss.scatter(0, keep_indices, compact_loss.reshape(-1))
        loss = loss.view_as(labels)

    loss = loss.transpose(0, 1).contiguous()
    if return_src_loss:
        return loss

    if kl_alpha_mask is not None:
        ce_weight = (1 - kl_alpha_mask).view(-1, 1).float()
        loss = loss * ce_weight
        loss_mask = loss_mask * (ce_weight != 0)

    losses = loss.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    if loss_weights is not None:
        weighted_mask = loss_mask * loss_weights.view(-1).float()
    else:
        weighted_mask = loss_mask

    loss = torch.sum(losses * weighted_mask)
    total_tokens = loss_mask.sum()
    loss = torch.cat([loss.view(1), total_tokens.view(1)])

    # When calculate_per_token_loss=True the pipeline schedule + finalize_model_grads
    # handle CP gradient aggregation.  Skipping the AVG here keeps each CP rank's
    # local loss_sum/tokens so finalize_model_grads produces correctly scaled gradients.
    _cp_group = cp_group if cp_group is not None else mpu.get_context_parallel_group()
    if not skip_cp_reduce and dist.get_world_size(_cp_group) > 1:
        torch.distributed.all_reduce(loss, group=_cp_group, op=torch.distributed.ReduceOp.AVG)
    return loss


def linear_ce_loss(
    config,
    linear_ce_input: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    kl_alpha_mask: torch.Tensor = None,
    return_src_loss: bool = False,
    skip_cp_reduce: bool = False,
    cp_group=None,
    loss_weights: Optional[torch.Tensor] = None,
):
    """Compute linear (fused) cross-entropy loss from hidden states.

    Returns:
        If ``return_src_loss``: per-token loss ``[b, s]``.
        Otherwise: ``torch.Tensor`` ``[loss_sum, total_tokens]``.
    """
    set_linear_ce_backend(config.training.linear_ce_backend)
    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()

    hidden_states = linear_ce_input["hidden_states"]
    output_layer = linear_ce_input['output_layer']
    tp_group = output_layer.tp_group
    if output_layer.sequence_parallel:
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=True,
        )
    else:
        assert tp_group is None or dist.get_world_size(tp_group) <= 1, (
            "linear_cross_entropy backward does not all-reduce d_hidden across TP ranks. "
            "With TP > 1 and sequence_parallel=False there is no reduce-scatter to compensate, "
            "so d_hidden would be incorrect. Enable sequence_parallel or use vocab_parallel_cross_entropy."
        )
    weight = linear_ce_input["weight"]
    if weight is None:
        weight = output_layer.weight

    if not config.training.ce_compaction:
        loss = linear_cross_entropy(
            hidden_states,
            weight,
            labels,
            1.0,
            "none",
            tp_group,
        )
    else:
        loss_mask_sb = loss_mask.transpose(0, 1).contiguous()
        assert hidden_states.shape[:-1] == labels.shape == loss_mask_sb.shape, (
            "linear CE shape mismatch: "
            f"{hidden_states.shape=}, {labels.shape=}, {loss_mask_sb.shape=}"
        )

        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_labels = labels.reshape(-1)
        keep_indices = (loss_mask_sb.reshape(-1) != 0).nonzero(as_tuple=False).flatten()
        num_valid_tokens = keep_indices.numel()
        if num_valid_tokens == 0:
            # Preserve zero gradients for hidden states and output weights.
            zero_loss = flat_hidden[:0].sum() + weight[:0].sum()
            loss = zero_loss.expand(labels.numel()).view_as(labels)
        else:
            linear_ce_hidden = flat_hidden.index_select(0, keep_indices)
            linear_ce_labels = flat_labels.index_select(0, keep_indices)
            # Triton specializes one token to a scalar, so pad it to two rows.
            if num_valid_tokens == 1:
                linear_ce_hidden = torch.cat((linear_ce_hidden, linear_ce_hidden), dim=0)
                linear_ce_labels = torch.cat((linear_ce_labels, linear_ce_labels))
            loss = linear_cross_entropy(
                linear_ce_hidden,
                weight,
                linear_ce_labels,
                1.0,
                "none",
                tp_group,
            )
            dense_loss = loss.new_zeros(labels.numel())
            loss = dense_loss.scatter(0, keep_indices, loss[:num_valid_tokens])
            loss = loss.view_as(labels)

    # [s b] => [b, s]
    loss = loss.transpose(0, 1).contiguous()
    if return_src_loss:
        return loss

    if kl_alpha_mask is not None:
        ce_weight = (1 - kl_alpha_mask).view(-1, 1).float()
        loss = loss * ce_weight
        loss_mask = loss_mask * (ce_weight != 0)

    losses = loss.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    if loss_weights is not None:
        weighted_mask = loss_mask * loss_weights.view(-1).float()
    else:
        weighted_mask = loss_mask

    loss = torch.sum(losses * weighted_mask)
    total_tokens = loss_mask.sum()
    loss = torch.cat([loss.view(1), total_tokens.view(1)])

    # When calculate_per_token_loss=True the pipeline schedule + finalize_model_grads
    # handle CP gradient aggregation.  Skipping the AVG here keeps each CP rank's
    # local loss_sum/tokens so finalize_model_grads produces correctly scaled gradients.
    _cp_group = cp_group if cp_group is not None else mpu.get_context_parallel_group()
    if not skip_cp_reduce and dist.get_world_size(_cp_group) > 1:
        torch.distributed.all_reduce(loss, group=_cp_group, op=torch.distributed.ReduceOp.AVG)
    return loss


def logits_kl_loss(
    teacher_logits, student_logits, loss_mask, temperature, kl_alpha_mask: torch.Tensor = None
):
    torch.cuda.synchronize()
    teacher_logits = teacher_logits.float()
    student_logits = student_logits.float()

    if mpu.get_tensor_model_parallel_world_size() > 1:
        teacher_logits = teacher_logits / temperature
        student_logits = student_logits / temperature
        teacher_logits_max, _ = torch.max(teacher_logits, dim=-1)
        torch.distributed.all_reduce(
            teacher_logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_tensor_model_parallel_group(),
        )
        teacher_logits = teacher_logits - teacher_logits_max.unsqueeze(dim=-1)

        denom_teacher = torch.sum(torch.exp(teacher_logits), dim=-1)
        # We can't use standard reduction function here since the computation
        # that follows it isn't identical across TP ranks.
        denom_teacher = all_reduce_autograd(
            denom_teacher, group=mpu.get_tensor_model_parallel_group()
        )
        # Maximum value along vocab dimension across all GPUs.
        student_logits_max, _ = torch.max(student_logits, dim=-1)
        torch.distributed.all_reduce(
            student_logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_tensor_model_parallel_group(),
        )
        student_logits = student_logits - student_logits_max.unsqueeze(dim=-1).detach()

        denom_student = torch.sum(torch.exp(student_logits), dim=-1)
        denom_student = all_reduce_autograd(
            denom_student, group=mpu.get_tensor_model_parallel_group()
        )

        bsz, slen, sharded_vocab_size = student_logits.shape
        student_log_prob = student_logits - torch.log(denom_student).view(bsz, slen, 1).expand(
            bsz, slen, sharded_vocab_size
        )
        teacher_log_prob = teacher_logits - torch.log(denom_teacher).view(bsz, slen, 1).expand(
            bsz, slen, sharded_vocab_size
        )
        kl_loss = torch.sum(
            F.kl_div(student_log_prob, teacher_log_prob, reduction="none", log_target=True),
            dim=-1,
        ).float()
        if kl_alpha_mask is not None:
            kl_weight = kl_alpha_mask.view(-1, 1).float()
            kl_loss = kl_loss * kl_weight
            loss_mask = loss_mask * (kl_weight != 0)

        kl_loss = kl_loss.view(-1)
        local_mask = loss_mask.view(-1).float()
        kl_loss = torch.sum(kl_loss * local_mask / (temperature**2))
        torch.distributed.all_reduce(kl_loss, group=mpu.get_tensor_model_parallel_group())
    else:
        kl_loss = torch.sum(
            F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="none",
            ),
            dim=-1,
        ).float()
        if kl_alpha_mask is not None:
            kl_weight = kl_alpha_mask.view(-1, 1).float()
            kl_loss = kl_loss * kl_weight
            loss_mask = loss_mask * (kl_weight != 0)

        kl_loss = kl_loss.view(-1)
        local_mask = loss_mask.view(-1).float()
        kl_loss = torch.sum(kl_loss * local_mask / (temperature**2))

    reduce_kl_tensor = torch.cat([kl_loss.view(1), local_mask.sum().view(1)])
    # Match ``ce_loss`` / ``linear_ce_loss``: AVG so d(mean)/d(local_sum)
    # is ``cp / global_tokens``. A SUM here would make KL grads ``1/cp``
    # of CE after DP×CP DDP averaging.
    if dist.get_world_size(mpu.get_context_parallel_group()) > 1:
        torch.distributed.all_reduce(
            reduce_kl_tensor,
            group=mpu.get_context_parallel_group(),
            op=torch.distributed.ReduceOp.AVG,
        )
    return reduce_kl_tensor


def _offpd_megatron_loss_outputs(
    loss: torch.Tensor,
    num_tokens: torch.Tensor,
    metrics: Dict[str, Any],
):
    """Pack a Megatron 3-tuple so CP does not apply the legacy ``*cp_size``.

    A 2-tuple return is treated as an already-averaged loss and then
    multiplied by ``cp_group_size`` in ``forward_step_calc_loss``. That is
    wrong for ``ce_with_kl``, whose CE/KL means are already CP-reduced.
    Returning ``(loss_sum, num_tokens, metrics)`` matches ``cross_entropy``
    so TP/PP/CP configs share the same backward scale.
    """
    num_tokens = num_tokens.detach().to(dtype=torch.int)
    token_scale = num_tokens.clamp(min=1).to(dtype=loss.dtype)
    return (loss * token_scale, num_tokens, metrics)


@register_loss("ce_with_kl")
def off_policy_loss_func(
    config,
    loss_input: FinetuneLossInput,
):
    # [b s] => [s b]
    kl_loss_alpha = config.distill.offpd_loss_alpha
    temperature = config.distill.offpd_temperature

    batch = loss_input.batch
    parallel_logits = loss_input.logits

    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    loss_weights = batch.get("loss_weights", None)
    kl_alpha_mask = batch.get("kl_alpha_mask", None)
    if kl_alpha_mask is not None:
        assert kl_alpha_mask.shape[0] == labels.shape[
            0], f"shape mismatch {kl_alpha_mask.shape} {labels.shape}"

    ce_kl_alpha_mask = (
        kl_alpha_mask if
        (config.training.enable_teacher_kl_loss and config.distill.enable_data_with_alpha) else None
    )
    if config.training.use_linear_ce:
        loss = linear_ce_loss(
            config,
            loss_input.linear_ce_input,
            labels,
            loss_mask,
            kl_alpha_mask=ce_kl_alpha_mask,
            skip_cp_reduce=loss_input.skip_cp_loss_reduce,
            cp_group=loss_input.cp_group,
            loss_weights=loss_weights,
        )

        linear_ce_input = loss_input.linear_ce_input
        student_hidden_states = linear_ce_input["hidden_states"]
        student_output_layer = linear_ce_input["output_layer"]
        assert not getattr(student_output_layer.config, "use_mup", False), (
            "Teacher hidden-state transfer does not yet support MuP logit scaling"
        )
        student_weight = linear_ce_input["weight"]
        if student_weight is None:
            student_weight = student_output_layer.weight
        student_output = student_output_layer(
            student_hidden_states,
            weight=student_weight,
            runtime_gather_output=linear_ce_input["runtime_gather_output"],
        )
        if isinstance(student_output, tuple):
            student_output = student_output[0]
        parallel_logits = student_output.transpose(0, 1).contiguous()
    else:
        loss = ce_loss(
            config,
            parallel_logits,
            labels,
            loss_mask,
            kl_alpha_mask=ce_kl_alpha_mask,
            loss_weights=loss_weights,
        )
    ce_numel = loss[1].detach()  # token count for metrics
    if loss[1] != 0:
        lm_loss = loss[0] / loss[1]
    else:
        lm_loss = loss[0]

    if config.training.enable_teacher_kl_loss:
        teacher_hidden_states = batch["teacher_hidden_states"]
        teacher_output_weight = loss_input.teacher_output_weight
        assert teacher_output_weight is not None
        assert teacher_hidden_states.shape[-1] == teacher_output_weight.shape[-1], (
            f"Teacher hidden/head mismatch: {teacher_hidden_states.shape} vs "
            f"{teacher_output_weight.shape}"
        )
        with torch.no_grad():
            teacher_logits = F.linear(
                teacher_hidden_states.to(dtype=teacher_output_weight.dtype),
                teacher_output_weight,
            )
        assert teacher_logits.shape == parallel_logits.shape, (
            f"Teacher/Student logits mismatch: {teacher_logits.shape} vs "
            f"{parallel_logits.shape}"
        )
        # logits_kl_loss 函数中，TP > 1 时需要手动实现 all-reduce 来跨 TP ranks
        # 计算全局 softmax 和 KL 散度（因为每个 rank 只持有 vocab 的一部分 logits）。
        reduce_kl_loss = logits_kl_loss(
            teacher_logits,  # Teacher [B, S, V//TP]
            parallel_logits,  # Student [B, S, V//TP]
            loss_mask,
            temperature,
            kl_alpha_mask=kl_alpha_mask if config.distill.enable_data_with_alpha else None
        )
        kl_numel = reduce_kl_loss[1].detach()
        if reduce_kl_loss[1] != 0:
            kl_loss = reduce_kl_loss[0] / reduce_kl_loss[1]
        else:
            kl_loss = reduce_kl_loss[0]

        if config.distill.enable_data_with_alpha:
            loss = kl_loss + lm_loss
        else:
            loss = kl_loss_alpha * kl_loss + (1 - kl_loss_alpha) * lm_loss

        # 各 metric 必须使用各自分支的 token 计数加权：
        #   - lm_loss 仅在 (1-alpha)>0 的 token 上有贡献 -> ce_numel
        #   - kl_loss 仅在 alpha>0 的 token 上有贡献   -> kl_numel
        # 否则 alpha 接近 0 或 1 时 DP 聚合 sum/count 会得到 0（loss 仍生效）。
        # 总 loss 用原始 loss_mask 的 token 数加权，使其与 alpha 取值
        # （包括 (0,1) 的中间值）无关，避免重复计数。
        total_numel = loss_mask.sum().detach()
        if dist.get_world_size(mpu.get_context_parallel_group()) > 1:
            torch.distributed.all_reduce(total_numel, group=mpu.get_context_parallel_group())
        with torch.no_grad():
            metrics = {
                "loss": torch.stack([loss.detach() * total_numel, total_numel]),
                "lm_loss": torch.stack([lm_loss.detach() * ce_numel, ce_numel]),
                "kl_loss": torch.stack([kl_loss.detach() * kl_numel, kl_numel]),
            }
        return _offpd_megatron_loss_outputs(loss, total_numel, metrics)
    else:
        loss = lm_loss
        with torch.no_grad():
            metrics = {
                "loss": torch.stack([loss.detach() * ce_numel, ce_numel]),
                "lm_loss": torch.stack([lm_loss.detach() * ce_numel, ce_numel]),
            }
        return _offpd_megatron_loss_outputs(loss, ce_numel, metrics)


def compute_dpo_loss_core(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
):
    """Backend-agnostic DPO loss from pre-computed per-sequence log-probs.

    Parameters
    ----------
    policy_chosen_logps, policy_rejected_logps : Tensor ``[rbs]``
    ref_chosen_logps, ref_rejected_logps : Tensor ``[rbs]``
    beta : float
    label_smoothing : float
    loss_type : ``"sigmoid"``

    Returns
    -------
    losses : Tensor ``[rbs]``
    chosen_rewards : Tensor ``[rbs]``  (detached)
    rejected_rewards : Tensor ``[rbs]``  (detached)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_rewards - rejected_rewards

    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        raise ValueError(f"unknown DPO loss type {loss_type}")

    return losses, chosen_rewards.detach(), rejected_rewards.detach()


def _dpo_loss_from_sequence_logprobs(
    config,
    policy_seq_logps: torch.Tensor,
    ref_seq_logps: torch.Tensor,
    chosen_token_counts: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply pairwise DPO and optional FTX to aligned sequence log-probs."""
    assert policy_seq_logps.numel() % 2 == 0, "DPO requires chosen/rejected pairs"
    rbs = policy_seq_logps.numel() // 2
    policy_chosen_logps, policy_rejected_logps = policy_seq_logps.split(rbs)
    ref_chosen_logps, ref_rejected_logps = ref_seq_logps.split(rbs)
    losses, chosen_rewards, rejected_rewards = compute_dpo_loss_core(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=config.training.dpo_beta,
        label_smoothing=config.training.dpo_label_smoothing,
        loss_type=config.training.dpo_loss_type,
    )
    if config.training.dpo_ftx_gamma > 1e-6:
        assert chosen_token_counts is not None
        losses -= (
            config.training.dpo_ftx_gamma * policy_chosen_logps /
            chosen_token_counts.clamp_min(1.0)
        )
    return (
        losses,
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )


def _contiguous_thd_pair_logits_mean(
    logits: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BSHD-equivalent chosen/rejected logit means for contiguous THD."""
    assert cu_seqlens_padded.numel() % 2 == 1, "DPO requires chosen/rejected pairs"
    cp_group = mpu.get_context_parallel_group()
    local_rows = logits.shape[1]
    global_start = cp_group.rank() * local_rows
    global_rows = torch.arange(
        global_start,
        global_start + local_rows,
        device=logits.device,
        dtype=cu_seqlens_padded.dtype,
    )
    sequence_ids = torch.bucketize(global_rows, cu_seqlens_padded[1:], right=True)
    chosen_rows = sequence_ids < (cu_seqlens_padded.numel() - 1) // 2

    # Match BSHD's ``logits.mean()`` semantics while avoiding a per-sequence
    # slice/reduction and host synchronization for every packed boundary.
    row_sums = logits.detach().sum(dim=(0, 2))
    elements_per_row = logits.shape[0] * logits.shape[2]
    stats = torch.stack(
        (
            row_sums[chosen_rows].sum(),
            chosen_rows.sum().to(logits.dtype) * elements_per_row,
            row_sums[~chosen_rows].sum(),
            (~chosen_rows).sum().to(logits.dtype) * elements_per_row,
        )
    )

    dist.all_reduce(stats, group=cp_group, op=dist.ReduceOp.SUM)
    return stats[0] / stats[1].clamp_min(1), stats[2] / stats[3].clamp_min(1)


def contiguous_thd_dpo_loss(
    config,
    loss_input: FinetuneLossInput,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute DPO over DSv4 contiguous-CP THD shards and return pair-level values.

    Log-probs are gathered back into physical packed-buffer order before
    sequence aggregation. ``cu_seqlens_q_padded`` is required because its
    boundaries include the inter-sequence CP alignment padding.
    """
    logits = loss_input.logits.float()
    batch = loss_input.batch
    assert "ref_logprobs" in batch, "THD DPO requires packed reference log-probs"
    packed_seq_params = batch["full_packed_seq_params"]
    assert packed_seq_params.cp_partition_mode == "contiguous", (
        "THD DPO currently supports only DSv4 contiguous CP partitioning"
    )
    local_logps = from_parallel_logits_to_logprobs(
        vocab_parallel_logits=logits,
        target=batch["labels"],
        inference_only=False,
        ignore_cp=True,
        pre_shifted=True,
    )
    local_mask = batch["loss_mask"].to(dtype=local_logps.dtype)
    local_ref_logps = batch["ref_logprobs"].to(dtype=local_logps.dtype)
    assert local_logps.shape == local_mask.shape == local_ref_logps.shape

    policy_logps = all_gather_from_context_parallel_region_no_zigzag(local_logps)
    loss_mask = all_gather_from_context_parallel_region_no_zigzag(local_mask)
    ref_logps = all_gather_from_context_parallel_region_no_zigzag(local_ref_logps)
    cu_seqlens = packed_seq_params.cu_seqlens_q_padded
    policy_chosen_logits_mean, policy_rejected_logits_mean = _contiguous_thd_pair_logits_mean(
        logits, cu_seqlens
    )
    policy_seq_logps = []
    ref_seq_logps = []
    sequence_token_counts = []
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        seq_slice = slice(int(seq_start.item()), int(seq_end.item()))
        mask = loss_mask[0, seq_slice]
        policy_seq_logps.append((policy_logps[0, seq_slice] * mask).sum())
        ref_seq_logps.append((ref_logps[0, seq_slice] * mask).sum())
        sequence_token_counts.append(mask.sum())

    policy_seq_logps = torch.stack(policy_seq_logps)
    ref_seq_logps = torch.stack(ref_seq_logps)
    sequence_token_counts = torch.stack(sequence_token_counts)
    chosen_token_counts = sequence_token_counts[:sequence_token_counts.numel() // 2]
    (
        losses,
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    ) = _dpo_loss_from_sequence_logprobs(
        config,
        policy_seq_logps,
        ref_seq_logps,
        chosen_token_counts,
    )

    return (
        losses,
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        policy_chosen_logits_mean,
        policy_rejected_logits_mean,
    )


def bshd_dpo_loss(
    config,
    loss_input: FinetuneLossInput,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute DPO for standard BSHD, including Megatron zigzag CP."""
    logits = loss_input.logits.float()
    batch = loss_input.batch
    labels = batch["labels"]
    loss_mask = batch["full_loss_mask"][:, :-1]
    target = batch["full_tokens"]

    assert logits.shape[0] % 2 == 0, "mbs must be 2*n"
    rbs = logits.shape[0] // 2
    policy_chosen_logits, policy_rejected_logits = logits.split(rbs, dim=0)
    logps = from_parallel_logits_to_logprobs(
        vocab_parallel_logits=logits, target=target, inference_only=False
    )
    logps = (logps * loss_mask).sum(-1)
    ref_logps = (batch["ref_logprobs"] * loss_mask).sum(-1)
    chosen_token_counts = None
    if config.training.dpo_ftx_gamma > 1e-6:
        chosen_labels = labels.split(rbs, dim=0)[0]
        chosen_token_counts = (chosen_labels != -100).sum(-1)
        if config.policy.dist_config.context_parallel_size > 1:
            chosen_token_counts = reduce_from_context_parallel_region(chosen_token_counts)
    (
        losses,
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    ) = _dpo_loss_from_sequence_logprobs(
        config,
        logps,
        ref_logps,
        chosen_token_counts,
    )

    return (
        losses,
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        policy_chosen_logits.detach().mean().float(),
        policy_rejected_logits.detach().mean().float(),
    )


@register_loss("dpo")
def dpo_loss_func(
    config,
    loss_input: FinetuneLossInput,
):
    """Megatron-backend DPO loss (uses ``from_parallel_logits_to_logprobs``)."""
    # TODO: support ``logprobs_from_linear_ce`` so DPO can avoid materializing
    # vocab logits when ``training.use_linear_ce`` is enabled.
    batch = loss_input.batch
    is_thd = batch.get("full_packed_seq_params") is not None

    # Standard BSHD, including Megatron zigzag CP, uses the non-THD helper.
    # Only DSv4 contiguous THD CP is supported here. Generic THD + CP uses
    # different partition/reorder semantics and needs its own aggregation path.
    if is_thd:
        (
            losses,
            chosen_rewards,
            rejected_rewards,
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            policy_chosen_logits_mean,
            policy_rejected_logits_mean,
        ) = contiguous_thd_dpo_loss(config, loss_input)
    else:
        (
            losses,
            chosen_rewards,
            rejected_rewards,
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            policy_chosen_logits_mean,
            policy_rejected_logits_mean,
        ) = bshd_dpo_loss(config, loss_input)

    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    reported_loss = losses.mean()
    losses = reported_loss
    metrics = {
        "dpo-metrics/rewards-accuracies": reward_accuracies.mean().float(),
        "dpo-metrics/rewards-chosen": chosen_rewards.mean().float(),
        "dpo-metrics/rewards-rejected": rejected_rewards.mean().float(),
        "dpo-metrics/rewards-margins": (chosen_rewards - rejected_rewards).mean().float(),
        "dpo-metrics/logps-rejected": policy_rejected_logps.detach().mean().float(),
        "dpo-metrics/logps-chosen": policy_chosen_logps.detach().mean().float(),
        "dpo-metrics/ref-logps-rejected": ref_rejected_logps.detach().mean().float(),
        "dpo-metrics/ref-logps-chosen": ref_chosen_logps.detach().mean().float(),
        "dpo-metrics/logits-rejected": policy_rejected_logits_mean.detach().mean().float(),
        "dpo-metrics/logits-chosen": policy_chosen_logits_mean.detach().mean().float(),
        "dpo-metrics/loss": reported_loss.clone().detach().float(),
    }
    keys = sorted(list(metrics.keys()))
    avg_metrics = average_losses_across_data_parallel_group([metrics[k] for k in keys])
    metrics = {k: v for k, v in zip(keys, avg_metrics)}

    return (losses, metrics)


@register_loss("square_averaging_cross_entropy")
def square_averaging_cross_entroy_loss_func(
    config,
    loss_input: FinetuneLossInput,
):
    """LM cross-entropy loss with square-averaging weighting.

    Each sample is weighted by ``1 / sqrt(num_answer_tokens)`` so that
    short-answer samples are not overwhelmed by long-answer ones.

    When ``calculate_per_token_loss=True`` (required by dynamic CP), this
    function returns 3 elements ``(loss_sum, effective_num_tokens, metrics)``
    so that ``forward_step_calc_loss`` takes the per-token-loss branch and
    avoids the ``output_tensor *= cp_group_size`` multiplication that would
    otherwise double the gradient under CP=2.

    To preserve the exact square-averaging normalisation while keeping
    ``num_tokens`` as an integer (required by
    ``torch.distributed.broadcast``), both ``loss_sum`` and
    ``effective_num_tokens`` are scaled by ``_SA_SCALE = 10_000``:

    - ``loss_sum = weighted_sum * _SA_SCALE``
    - ``effective_num_tokens = int(W / cp_size * _SA_SCALE)``

    where ``W = square_averaging_weights.sum()``.  After
    ``finalize_model_grads`` divides by ``total_num_tokens``, the
    ``_SA_SCALE`` factor cancels and the gradient equals the correct
    square-averaging gradient ``Σ(weighted_sum) / Σ(W)``.

    Args:
        labels (Tensor): ``[B, S]``.
        logits (Tensor): ``[B, S, V]`` from the output layer.
        loss_mask (Tensor): ``[B, S]``.
    Returns:
        When ``calculate_per_token_loss=True``:
            ``(loss_sum, effective_num_tokens, metrics)`` – 3-element tuple.
        Otherwise:
            ``(loss, metrics)`` – 2-element tuple (legacy behaviour).
    """
    batch = loss_input.batch
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    # shape = [b,] 都是一样的
    square_averaging_weights = batch["square_averaging_weights"]
    assert square_averaging_weights is not None
    assert square_averaging_weights.requires_grad is False

    if loss_input.linear_ce_input is not None:
        hidden_ce_loss = (
            linear_ce_loss if config.training.use_linear_ce else compact_ce_loss
        )
        losses = hidden_ce_loss(
            config,
            loss_input.linear_ce_input,
            labels,
            loss_mask,
            return_src_loss=True,
        )
    else:
        logits = loss_input.logits.float()
        losses = ce_loss(config, logits, labels, loss_mask, return_src_loss=True)
    loss_weight = loss_mask.sum(dim=-1).float()
    _cp_group = loss_input.cp_group if loss_input.cp_group is not None else mpu.get_context_parallel_group(
    )
    if dist.get_world_size(_cp_group) > 1:
        torch.distributed.all_reduce(loss_weight, group=_cp_group)
    loss_weight = 1 / torch.clamp_min(loss_weight.sqrt(), 1)
    loss_weight = torch.where(
        loss_mask == 1, loss_weight.unsqueeze(1),
        torch.tensor(0.0, dtype=loss_weight.dtype, device=loss_weight.device)
    )

    weighted_losses = losses.float() * loss_weight
    #TODO: check loss nan or not

    # ``skip_cp_loss_reduce`` is set to ``calc_per_token_loss`` by the
    # finetune backend, so it serves as a reliable proxy for whether
    # per-token-loss normalisation is active.
    calculate_per_token_loss = loss_input.skip_cp_loss_reduce

    if calculate_per_token_loss:
        # --- Per-token-loss path (required by dynamic CP) ---
        # Return (loss_sum, effective_num_tokens, metrics) so that
        # forward_step_calc_loss takes the 3-element branch and avoids the
        # ``output_tensor *= cp_group_size`` bug.
        #
        # To preserve the square-averaging normalisation semantics
        # (divide by Σ 1/√N_i instead of by total token count), we encode
        # the square-averaging weight sum into ``num_tokens``.
        #
        # Because ``num_tokens`` must be an integer (``forward_step_calc_loss``
        # initialises it as ``torch.tensor(0, dtype=torch.int)`` and
        # ``torch.distributed.broadcast`` requires matching dtypes), we scale
        # both ``loss_sum`` and ``effective_num_tokens`` by a constant
        # ``_SA_SCALE`` so that the weight sum can be represented as an
        # integer:
        #
        #   effective_num_tokens = int(W / cp_size * _SA_SCALE)   (int)
        #   loss_sum            = weighted_sum * _SA_SCALE        (float tensor)
        #
        # After ``finalize_model_grads`` all-reduces num_tokens across the
        # dp_cp_group and calls ``scale_gradients(1 / total_num_tokens)``,
        # the gradient becomes:
        #
        #   _SA_SCALE * Σ_dp grad(weighted_sum_dp)
        #   ─────────────────────────────────────── = Σ_dp grad(weighted_sum_dp) / Σ_dp W_dp
        #   _SA_SCALE * Σ_dp W_dp
        #
        # which is exactly the correct global square-averaging gradient.
        # The _SA_SCALE factor cancels out perfectly.
        #
        # Dividing by cp_size avoids double-counting because
        # square_averaging_weights is per-sample (identical across CP ranks)
        # whereas loss_mask is per-token (different across CP ranks).
        _SA_SCALE = 10_000

        cp_size = dist.get_world_size(_cp_group)

        loss_sum = weighted_losses.sum().view(1).clone() * _SA_SCALE

        effective_num_tokens_float = square_averaging_weights.sum() / cp_size * _SA_SCALE
        effective_num_tokens = torch.tensor(int(effective_num_tokens_float.item()), dtype=torch.int)

        # Do NOT do CP all-reduce on loss_sum/effective_num_tokens when
        # skip_cp_loss_reduce is True (the dynamic-CP case).  Each CP rank
        # keeps its local values and finalize_model_grads aggregates
        # num_tokens across the dp_cp_group.
        if not loss_input.skip_cp_loss_reduce and cp_size > 1:
            # Non-dynamic-CP fallback: CP AVG on [loss_sum, effective_num_tokens]
            loss_for_reduce = torch.stack([loss_sum.float(), effective_num_tokens.float()])
            torch.distributed.all_reduce(
                loss_for_reduce, group=_cp_group, op=torch.distributed.ReduceOp.AVG
            )
            loss_sum = loss_for_reduce[0].view(1)
            effective_num_tokens = loss_for_reduce[1].to(torch.int)

        # Reporting loss: weighted mean for display purposes (unscaled)
        reporting_loss = (weighted_losses.sum() / square_averaging_weights.sum()).clone().detach()
        reporting_loss = average_losses_across_data_parallel_group([reporting_loss])
        metrics = {"lm_loss": reporting_loss}

        return (loss_sum, effective_num_tokens, metrics)
    else:
        # --- Legacy path (calculate_per_token_loss=False) ---
        # losses.sum() 包括了 mbs 序列之和，计算 loss 时应该求 mean
        # 所以 square_averaging_weights 直接多个求 sum 就可以了
        loss = weighted_losses.sum().view(1).clone() / square_averaging_weights.sum().view(1)
        if (not loss_input.skip_cp_loss_reduce and dist.get_world_size(_cp_group) > 1):
            torch.distributed.all_reduce(loss, group=_cp_group)

        reporting_loss = loss.clone().detach()
        reporting_loss = average_losses_across_data_parallel_group([reporting_loss])
        return (loss, {"lm_loss": reporting_loss})


@register_loss("ppo_value_loss")
def value_loss_func(ppo_config, old_values, values, returns, mask):
    if ppo_config.ppo_value_clip is not None:
        values_clipped = old_values + (values - old_values).clamp(
            -ppo_config.ppo_value_clip, ppo_config.ppo_value_clip
        )
        surr1 = (values_clipped - returns)**2
        surr2 = (values - returns)**2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns)**2

    with torch.no_grad():
        numel = mask.sum()
    loss = masked_mean(0.5 * loss, mask)
    metrics = {"value_loss": torch.stack([loss.detach() * numel, numel])}

    return loss, metrics


@register_loss("rm_bt")
def rm_bt_loss_func(
    config,
    loss_input: FinetuneLossInput,
):
    """Bradley-Terry pairwise reward-model loss (sequence-level scalar reward).

    The model carries a scalar value head (``LinearForLastLayer`` -> 1), so
    ``logits`` here is the per-token scalar ``[B, S, 1]``. The sequence-level
    reward is read at each sequence's last valid token (index
    ``sequence_lengths - 1``). Within a micro-batch the first half are the
    chosen sequences and the second half the rejected ones.

    Parameters
    ----------
    loss_input : FinetuneLossInput
        ``logits`` shape ``[B, S, 1]``; ``batch["sequence_lengths"]`` shape
        ``[B]`` holds the valid-token count per sequence (last index + 1).
    """
    batch = loss_input.batch
    scores = loss_input.logits
    assert scores is not None, "rm_bt loss requires the scalar head output as logits"
    assert scores.dim() == 3 and scores.shape[-1] == 1, \
        f"rm_bt expects [B, S, 1] scalar head output, got {tuple(scores.shape)}"

    sequence_lengths = batch["sequence_lengths"]
    bs = scores.shape[0]
    assert bs % 2 == 0, f"reward model micro-batch must be 2*n (chosen|rejected), got {bs}"

    # last-token pooling: hidden_state at index (sequencxe_lengths - 1)
    last_idx = (sequence_lengths.long() - 1).clamp(min=0)
    rewards = scores[torch.arange(bs, device=scores.device), last_idx, 0].float()  # [B]

    r_chosen, r_rejected = rewards.split(bs // 2, dim=0)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()

    with torch.no_grad():
        acc = (r_chosen > r_rejected).float().mean()
        margin = (r_chosen - r_rejected).mean()
        metrics = {
            "rm-metrics/loss": loss.detach().float(),
            "rm-metrics/acc": acc.float(),
            "rm-metrics/reward_margin": margin.float(),
            "rm-metrics/reward_chosen": r_chosen.mean().float(),
            "rm-metrics/reward_rejected": r_rejected.mean().float(),
        }
    keys = sorted(metrics.keys())
    avg_metrics = average_losses_across_data_parallel_group([metrics[k] for k in keys])
    metrics = {k: v for k, v in zip(keys, avg_metrics)}

    return (loss, metrics)


def get_policy_loss_fn(name: str) -> Callable:
    if name not in LOSS_FUNC_REGISTRY:
        available = ", ".join(sorted(LOSS_FUNC_REGISTRY.keys())) or "(none)"
        raise ValueError(f"Unknown loss type: '{name}'. Available: [{available}]")
    return LOSS_FUNC_REGISTRY[name]


class CpMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, cp_group):
        ctx.local_nel = tensor.shape[dim]
        nel = torch.tensor(tensor.shape[dim], device=tensor.device)
        dist.all_reduce(nel, group=cp_group)
        ctx.nel = int(nel.item())
        ctx.dim = dim
        out = tensor.sum(dim=dim)
        dist.all_reduce(out, group=cp_group)
        out = out / ctx.nel
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output / ctx.nel
        grad_output = grad_output.unsqueeze(
            ctx.dim
        ).expand(*grad_output.shape[:ctx.dim], ctx.local_nel,
                 *grad_output.shape[ctx.dim:]).contiguous()
        return grad_output, None, None


def load_balancing_loss_func(
    *,
    gate_logits: tuple[torch.Tensor, ...],
    num_experts: int,
    top_k: int,
    cp_group=None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """

    # [t, e], t=l*b*s
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
    )

    # [t, e]
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    # [t, top_k]
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    # [t, top_k, e]
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if cp_group is None or dist.get_world_size(cp_group) == 1:
        # Compute the percentage of tokens routed to each experts
        # [top_k, e]
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        # [e]
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        # tokens_per_expert comes from topk/one_hot and carries no gradient,
        # so a plain (detached) CP all-reduce global mean is enough.
        # [top_k, e]
        t_local = expert_mask.shape[0]
        tokens_per_expert = expert_mask.float().sum(dim=0)
        dist.all_reduce(tokens_per_expert, group=cp_group)
        t_global = torch.tensor(t_local, device=tokens_per_expert.device)
        dist.all_reduce(t_global, group=cp_group)
        tokens_per_expert = tokens_per_expert / t_global.item()

        # router_prob_per_expert is a global average over CP tokens; route it
        # through the autograd-aware mean so gradients reach the local
        # router_logits with the correct 1/T scale.
        # [e]
        router_prob_per_expert = CpMean.apply(routing_weights, 0, cp_group)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# RL loss functions whose backward value is a sum of per-sequence token-means
# (seq-mean-token-mean numerator) rather than a micro-batch token-mean. The
# training backend divides this by the global active-sequence count. Loss funcs
# not listed here keep the legacy micro-batch-mean backward semantics.
SEQ_MEAN_RL_LOSS_FNS = {grpo_loss_func, cispo_loss_func, sapo_loss_func, opd_loss_func}


def is_seq_mean_rl_loss_fn(fn: Callable) -> bool:
    """Whether ``fn`` returns the seq-mean-token-mean sum as its backward loss."""
    return fn in SEQ_MEAN_RL_LOSS_FNS


@register_loss("grad_cache_loss")
def grad_cache_loss_replay_func(
    config,
    loss_input: FinetuneLossInput,
):

    batch = loss_input.batch
    linear_ce_input = loss_input.linear_ce_input
    hidden_states = linear_ce_input['hidden_states']
    output_layer = linear_ce_input['output_layer']
    if output_layer is not None and getattr(output_layer, "sequence_parallel", False):
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False
        )
    elif output_layer is not None:
        tp_group = getattr(output_layer, "tp_group", None)
        if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
            assert not hidden_states.requires_grad, (
                "tensor_model_parallel_size > 1 且 sequence_parallel=False 时, "
                "hidden_states 在各 tp rank 间是复制而不是切分的, 但反向不会自动 "
                "all-reduce d_hidden, 直接用会导致梯度错误。请开启 sequence_parallel "
                "(通常和 tp>1 一起自动启用), 或改用其它方式处理。"
            )

    # [seqlen, batch, hidden] -> [batch, seqlen, hidden]
    last_hidden_state = hidden_states.transpose(0, 1).contiguous()
    if "eos_positions" in batch and batch["eos_positions"] is not None:
        eos_positions = batch["eos_positions"].to(device=last_hidden_state.device)
        assert last_hidden_state.size(0) == 1, (
            f"packed THD expects batch dim 1, got {last_hidden_state.shape}"
        )
        eos_embeddings = last_hidden_state[0, eos_positions]
    else:
        attention_mask = batch["attention_mask"]  # (batch_size, seq_len)
        eos_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        eos_embeddings = last_hidden_state[batch_indices, eos_positions]

    logit_scale = loss_input.unwrapped_model.logit_scale

    embeddings_grad = batch.get('embeddings_grad')
    assert embeddings_grad is not None, (
        "batch 里没有 embeddings_grad: GradCache 回放需要 "
        "EmbeddingActor 在 gather_embs 之后调用 "
        "engine.embedding_cache_backward (阶段2), 且 "
        "wemm3_embedding._sft_train_packed 需要透传 embeddings_grad。"
    )
    assert embeddings_grad.size(0) == eos_embeddings.size(0), (
        f"grad_buf {embeddings_grad.shape} 与 eos_embeddings {eos_embeddings.shape} 不一致"
    )

    grad_logit_scale = batch.get('logit_scale_grad')

    surrogate = (eos_embeddings * embeddings_grad).sum()
    if grad_logit_scale is not None:
        surrogate = surrogate + logit_scale * grad_logit_scale

    # metrics 是阶段2 算好的真值: 每个 mbs 返回同一份, Megatron 跨 mbs 平均后
    # 数值不变; 浅拷贝避免被 schedule 原地修改。
    metrics = batch['cache_metrics']
    if metrics is not None:
        metrics = dict(metrics)
    return surrogate, metrics
