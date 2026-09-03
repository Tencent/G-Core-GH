from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed

from gpatch_v4.core.advantage_impl import (
    calculate_g_opd_advantages,
    calculate_gdpo_advantages,
    calculate_gdpo_sample_bn_advantages,
    calculate_grpo_advantages,
    calculate_identity_advantages,
    calculate_ppo_advantages_and_returns,
    calculate_ppo_rewards,
    calculate_reinforce_advantages,
    calculate_reverse_kl_advantages,
    calculate_topk_advantages,
    compute_gdpo_combined_advantages,
)
from gpatch_v4.utils import log
from gpatch_v4.utils.common_utils import import_fn_from_path
from gpatch_v4.utils.ppo_utils import get_advantage_clip_bounds


@dataclass
class AdvantageContext:
    """Unified input for all advantage computation functions.

    Common fields are pre-extracted for convenience; each registered function
    can also reach into ``rollout_batch`` or ``config`` for type-specific data
    (e.g. teacher logprobs, routing fields).
    """

    # ---- raw sources (always present) ----
    rollout_batch: Dict[str, Any]
    config: Any

    # ---- pre-extracted common fields ----
    mask: List[torch.Tensor]
    logprobs: List[torch.Tensor]
    rewards: Optional[List[torch.Tensor]] = None
    sample_mask: Optional[List[torch.Tensor]] = None

    # ---- PPO-specific (None for non-PPO types) ----
    values: Optional[List[torch.Tensor]] = None
    per_token_rewards: Optional[List[torch.Tensor]] = None
    init_policy_kl: Optional[List[torch.Tensor]] = None
    sequence_lengths: Optional[List[Any]] = None

    prompt_lengths: Optional[List[Any]] = None
    gdpo_rewards: Optional[Dict[str, List[torch.Tensor]]] = None


@dataclass
class AdvantageResult:
    advantages: List[torch.Tensor]
    returns: Optional[List[torch.Tensor]] = None
    metrics: Optional[Dict[str, float]] = field(default_factory=dict)
    metrics_prefix: Optional[str] = None
    # When set, signals that ``advantages`` are NOT final — the caller must
    # invoke the registered post-advantage function to finalize them.
    pre_bn_advantages: Optional[torch.Tensor] = None


@dataclass
class PostAdvantageContext:
    """Unified input for post-advantage processing (e.g. global BN across DP ranks).

    Registered post-advantage functions receive this after the per-micro-batch
    advantage computation loop completes.
    """

    rollout_batches: List[Dict[str, Any]]
    config: Any
    dp_group: Any
    num_samples: int


@dataclass
class PostAdvantageResult:
    """Output of a post-advantage processing function."""

    rollout_batches: List[Dict[str, Any]]
    metrics: Dict[str, float] = field(default_factory=dict)


ADVANTAGE_DISPATCH: Dict[str, Callable[[AdvantageContext], AdvantageResult]] = {}
POST_ADVANTAGE_DISPATCH: Dict[str, Callable[[PostAdvantageContext], PostAdvantageResult]] = {}


def register_advantage(name: str):
    """Decorator to register an advantage computation function.

    Usage::

        @register_advantage("grpo")
        def compute_grpo_advantage(ctx: AdvantageContext) -> AdvantageResult:
            ...
    """
    def decorator(
        fn: Callable[[AdvantageContext], AdvantageResult]
    ) -> Callable[[AdvantageContext], AdvantageResult]:
        if name in ADVANTAGE_DISPATCH:
            raise ValueError(
                f"Advantage type '{name}' already registered by "
                f"{ADVANTAGE_DISPATCH[name].__module__}.{ADVANTAGE_DISPATCH[name].__qualname__}"
            )
        ADVANTAGE_DISPATCH[name] = fn
        return fn

    return decorator


def register_post_advantage(*names: str):
    """Decorator to register a post-advantage processing function.

    Usage::

        @register_post_advantage("gdpo", "gdpo_sample_bn")
        def gdpo_global_bn(ctx: PostAdvantageContext) -> PostAdvantageResult:
            ...
    """
    def decorator(
        fn: Callable[[PostAdvantageContext], PostAdvantageResult]
    ) -> Callable[[PostAdvantageContext], PostAdvantageResult]:
        for name in names:
            if name in POST_ADVANTAGE_DISPATCH:
                raise ValueError(
                    f"Post-advantage '{name}' already registered by "
                    f"{POST_ADVANTAGE_DISPATCH[name].__module__}."
                    f"{POST_ADVANTAGE_DISPATCH[name].__qualname__}"
                )
            POST_ADVANTAGE_DISPATCH[name] = fn
        return fn

    return decorator


def get_advantage_fn(name: str) -> Callable[[AdvantageContext], AdvantageResult]:
    if name not in ADVANTAGE_DISPATCH:
        available = ", ".join(sorted(ADVANTAGE_DISPATCH.keys())) or "(none)"
        raise ValueError(f"Unknown advantage type: '{name}'. Available: [{available}]")
    return ADVANTAGE_DISPATCH[name]


def get_post_advantage_fn(
    name: str,
) -> Optional[Callable[[PostAdvantageContext], PostAdvantageResult]]:
    """Return the post-advantage function for *name*, or None if not registered."""
    return POST_ADVANTAGE_DISPATCH.get(name, None)


def register_custom_advantage(name: str, py_path: str, fn_name: str):
    """Import an advantage function from *py_path* and register it under *name*.

    Parameters
    ----------
    name : str
        Used in ``ppo.advantage_type``.
    py_path : str
        Absolute ``.py`` path.
    fn_name : str
        Callable signature: ``(AdvantageContext) -> AdvantageResult``.
    """
    fn = import_fn_from_path(py_path, fn_name)
    if name in ADVANTAGE_DISPATCH:
        log(
            f"Advantage type '{name}' already registered — overwriting with "
            f"custom function {fn_name} from {py_path}",
            rank=0,
        )
    ADVANTAGE_DISPATCH[name] = fn
    log(f"Registered custom advantage '{name}' from {py_path}::{fn_name}", rank=0)


def register_custom_post_advantage(name: str, py_path: str, fn_name: str):
    """Import a post-advantage function from *py_path* and register it under *name*.

    Parameters
    ----------
    name : str
        Used in ``ppo.advantage_type``.
    py_path : str
        Absolute ``.py`` path.
    fn_name : str
        Callable signature: ``(PostAdvantageContext) -> PostAdvantageResult``.
    """
    fn = import_fn_from_path(py_path, fn_name)
    if name in POST_ADVANTAGE_DISPATCH:
        log(
            f"Post-advantage '{name}' already registered — overwriting with "
            f"custom function {fn_name} from {py_path}",
            rank=0,
        )
    POST_ADVANTAGE_DISPATCH[name] = fn
    log(f"Registered custom post-advantage '{name}' from {py_path}::{fn_name}", rank=0)


# ============================================================================
# Built-in advantage functions
# ============================================================================


@register_advantage("grpo")
def prepare_and_compute_grpo_advantages(ctx: AdvantageContext) -> AdvantageResult:
    assert ctx.rewards is not None
    advantages, returns = calculate_grpo_advantages(
        rewards=ctx.rewards,
        mask=ctx.mask,
        grpo_sampling_times=ctx.config.training.sampling_keep_n,
        grpo_advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
        sample_mask=ctx.sample_mask,
        norm_adv_by_std_in_grpo=ctx.config.ppo.norm_adv_by_std_in_grpo,
    )
    return AdvantageResult(advantages=advantages, returns=returns)


@register_advantage("gdpo")
def prepare_and_compute_gdpo_advantages(ctx: AdvantageContext) -> AdvantageResult:
    """Return pre-BN combined advantages; post-processing does global token-level BN."""
    assert ctx.gdpo_rewards is not None
    combined = compute_gdpo_combined_advantages(
        rewards_dict=ctx.gdpo_rewards,
        mask=ctx.mask,
        grpo_sampling_times=ctx.config.training.sampling_keep_n,
        grpo_advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
        gdpo_reward_weights=ctx.config.ppo.gdpo_reward_weights,
        sample_mask=ctx.sample_mask,
    )
    return AdvantageResult(advantages=[], pre_bn_advantages=combined)


@register_advantage("gdpo_sample_bn")
def prepare_and_compute_gdpo_sample_bn_advantages(ctx: AdvantageContext) -> AdvantageResult:
    """Return pre-BN combined advantages; post-processing does global sample-level BN."""
    assert ctx.gdpo_rewards is not None
    combined = compute_gdpo_combined_advantages(
        rewards_dict=ctx.gdpo_rewards,
        mask=ctx.mask,
        grpo_sampling_times=ctx.config.training.sampling_keep_n,
        grpo_advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
        gdpo_reward_weights=ctx.config.ppo.gdpo_reward_weights,
        sample_mask=ctx.sample_mask,
    )
    return AdvantageResult(advantages=[], pre_bn_advantages=combined)


@register_advantage("group_gdpo")
def prepare_and_compute_group_gdpo_advantages(ctx: AdvantageContext) -> AdvantageResult:
    assert ctx.gdpo_rewards is not None
    advantages, returns = calculate_gdpo_advantages(
        rewards_dict=ctx.gdpo_rewards,
        mask=ctx.mask,
        grpo_sampling_times=ctx.config.training.sampling_keep_n,
        grpo_advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
        gdpo_reward_weights=ctx.config.ppo.gdpo_reward_weights,
        sample_mask=ctx.sample_mask,
    )
    return AdvantageResult(advantages=advantages, returns=returns)


@register_advantage("group_gdpo_sample_bn")
def prepare_and_compute_group_gdpo_sample_bn_advantages(ctx: AdvantageContext, ) -> AdvantageResult:
    assert ctx.gdpo_rewards is not None
    advantages, returns, metrics = calculate_gdpo_sample_bn_advantages(
        rewards_dict=ctx.gdpo_rewards,
        mask=ctx.mask,
        grpo_sampling_times=ctx.config.training.sampling_keep_n,
        grpo_advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
        gdpo_reward_weights=ctx.config.ppo.gdpo_reward_weights,
        sample_mask=ctx.sample_mask,
    )
    return AdvantageResult(
        advantages=advantages,
        returns=returns,
        metrics=metrics,
        metrics_prefix="ppo-metrics",
    )


@register_advantage("ppo")
def prepare_and_compute_ppo_advantages(ctx: AdvantageContext) -> AdvantageResult:
    rewards = ctx.rewards
    values = ctx.values
    init_policy_kl = ctx.init_policy_kl
    sequence_lengths = ctx.sequence_lengths
    mask = ctx.mask
    per_token_rewards = ctx.per_token_rewards

    assert rewards is not None
    advantages = []
    returns = []
    for i in range(len(values)):
        values_i = values[i]
        rewards_i = rewards[i]
        init_policy_kl_i = init_policy_kl[i]
        sequence_lengths_i = sequence_lengths[i]
        mask_i = mask[i]
        per_token_rewards_i = None
        if per_token_rewards is not None:
            per_token_rewards_i = per_token_rewards[i]

        # TODO(by astrachang): 这里没有实现类型变化的计算
        rewards_with_kl = calculate_ppo_rewards(
            values_i,
            rewards_i,
            None,
            sequence_lengths_i,
            init_policy_kl_i,
            ctx.config.ppo.ppo_initial_policy_kl_penalty,
        )
        advantages_i, returns_i = calculate_ppo_advantages_and_returns(
            values=values_i,
            rewards=rewards_with_kl,
            discount_factor=ctx.config.ppo.ppo_discount_factor,
            gae_lambda=ctx.config.ppo.ppo_gae_lambda,
            mask=mask_i,
            per_token_rewards=per_token_rewards_i,
        )
        advantages.append(advantages_i)
        returns.append(returns_i)
    assert returns[0].dtype == torch.float32
    return AdvantageResult(advantages=advantages, returns=returns)


@register_advantage("on_policy_distill")
def prepare_and_compute_opd_advantages(ctx: AdvantageContext) -> AdvantageResult:
    prompt_lengths = ctx.prompt_lengths
    assert prompt_lengths is not None and isinstance(prompt_lengths, list)

    teacher_logprobs = None
    # OPD 只有一个 teacher，所以直接取第一个
    teacher_name = list(ctx.config.teachers.keys())[0]
    for key in ctx.rollout_batch:
        if key == f"teacher_logprobs_{teacher_name}":
            teacher_logprobs = ctx.rollout_batch[key]
            break
    assert teacher_logprobs is not None, f"teacher_logprobs_{teacher_name} not found in rollout_batch"

    log_prob_top_k = ctx.config.ppo.log_prob_top_k

    # Pure-OPD switch: drop reward before KL-only advantage so the env reward
    # (still recorded in the rollout batch for metrics) does not contribute to
    # the policy gradient. Without this flag, calculate_reverse_kl_advantages
    # adds a GRPO reward term whenever ``rewards`` is non-None — fine for
    # PG-loss OPD, but not for "pure OPD".
    rewards_for_adv = (
        None if getattr(ctx.config.ppo, "opd_ignore_env_reward", False) else ctx.rewards
    )

    if log_prob_top_k > 0:
        # Top-K path: 3D ``[S-1, K]`` advantages replace the 2D label-based KL advantages.
        # The loss layer detects this via ``advantages.dim() == 3``.
        #
        # NOTE: rewards set to None for the top-K path.
        # Mixing GRPO outcome rewards into 3D topk advantages causes KL divergence
        advantages, distill_metrics = _compute_topk_advantages(
            ctx, teacher_name, has_base=False, rewards=None
        )
        returns = None
    else:
        advantages, returns, distill_metrics = calculate_reverse_kl_advantages(
            rewards=rewards_for_adv,
            mask_lst=ctx.mask,
            logprobs=ctx.logprobs,
            teacher_logprobs=teacher_logprobs,
            sampling_repeat_n=ctx.config.training.sampling_keep_n,
            advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
            kl_penalty_coef=ctx.config.ppo.distill_kl_penalty_coef,
            kl_discount_factor=ctx.config.ppo.distill_kl_discount_factor,
        )

    for k in distill_metrics.keys():
        # 内部已经除以过 len(rollout_batch) 了，为了兼容 ppo_rollout_metrics 下面的写法，先乘以 len(rollout_batch)
        if isinstance(distill_metrics[k], float):
            distill_metrics[k] = distill_metrics[k] * len(prompt_lengths)

    return AdvantageResult(
        advantages=advantages,
        returns=returns,
        metrics=distill_metrics,
        metrics_prefix="on-policy-distillation",
    )


@register_advantage("g_opd")
def prepare_and_compute_g_opd_advantages(ctx: AdvantageContext) -> AdvantageResult:
    prompt_lengths = ctx.prompt_lengths
    assert prompt_lengths is not None and isinstance(prompt_lengths, list)

    rollout_batch = ctx.rollout_batch
    base_logprobs = rollout_batch.get("ref_logprobs", None)
    g_opd_lambda = getattr(ctx.config.ppo, "g_opd_lambda", 1.0)
    g_opd_mix_reward = getattr(ctx.config.ppo, "g_opd_mix_reward_advantage", False)
    routing_field = getattr(ctx.config.ppo, "g_opd_teacher_routing_field", "teacher_type")

    # Collect all teacher logprobs (stored as teacher_logprobs_{name})
    multi_teacher_logprobs = {}
    for key in rollout_batch:
        if key.startswith("teacher_logprobs_"):
            t_name = key[len("teacher_logprobs_"):]
            multi_teacher_logprobs[t_name] = rollout_batch[key]

    # check number of teachers and teacher logprobs
    assert len(list(ctx.config.teachers.keys())) == len(list(multi_teacher_logprobs.keys())), \
        f"number of teachers {len(list(ctx.config.teachers.keys()))} != number of teacher logprobs {len(list(multi_teacher_logprobs.keys()))}"

    teacher_names = list(multi_teacher_logprobs.keys())
    is_single_teacher = len(teacher_names) == 1

    # Per-sample routing: only needed when multiple teachers
    teacher_types = None
    if not is_single_teacher:
        teacher_types = rollout_batch.get(routing_field, None)
        if teacher_types is not None and not isinstance(teacher_types, list):
            teacher_types = list(teacher_types)

    # For single teacher, use it directly as the default
    default_teacher_name = teacher_names[0] if is_single_teacher else teacher_names[0]

    # Pick any teacher's logprobs as "primary" (first one)
    primary_teacher_logprobs = multi_teacher_logprobs.pop(default_teacher_name)

    log_prob_top_k = ctx.config.ppo.log_prob_top_k

    if log_prob_top_k > 0:
        advantages, distill_metrics = _compute_topk_advantages(
            ctx,
            default_teacher_name,
            has_base=base_logprobs is not None,
            rewards=ctx.rewards if g_opd_mix_reward else None,
        )
        returns = None
    else:
        advantages, returns, distill_metrics = calculate_g_opd_advantages(
            mask_lst=ctx.mask,
            logprobs=ctx.logprobs,
            teacher_logprobs=primary_teacher_logprobs,
            base_logprobs=base_logprobs,
            g_opd_lambda=g_opd_lambda,
            rewards=ctx.rewards if g_opd_mix_reward else None,
            sampling_repeat_n=ctx.config.training.sampling_keep_n,
            advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
            multi_teacher_logprobs=multi_teacher_logprobs if multi_teacher_logprobs else None,
            teacher_types=teacher_types,
            default_teacher_name=default_teacher_name,
        )

    for k in distill_metrics.keys():
        # 内部已经除以过 len(rollout_batch) 了，为了兼容 ppo_rollout_metrics 下面的写法，先乘以 len(rollout_batch)
        if isinstance(distill_metrics[k], float):
            distill_metrics[k] = distill_metrics[k] * len(prompt_lengths)

    return AdvantageResult(
        advantages=advantages,
        returns=returns,
        metrics=distill_metrics,
        metrics_prefix="g-opd",
    )


def _compute_topk_advantages(
    ctx: AdvantageContext,
    default_teacher_name: str,
    has_base: bool,
    rewards: Optional[List[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """Compute 3D ``[S-1, K]`` advantages for the OPD / G-OPD top-K logits path.

    Returns ``(advantages_3d, metrics)`` where each advantage tensor has shape
    ``[S-1, K]``.  When ``rewards`` is provided, they are broadcast to
    ``[S-1, K]`` and added to the KL-based 3D advantages.
    """
    rollout_batch = ctx.rollout_batch
    strategy = ctx.config.ppo.opd_top_k_strategy

    # Resolve student and teacher topk logprobs based on strategy.
    if strategy == "only_tch":
        stu_topk_lp = rollout_batch.get("stu_on_tch_topk_logprobs", None)
        assert stu_topk_lp is not None, "stu_on_tch_topk_logprobs required for only_tch"
        primary_teacher_topk_key = f"teacher_topk_logprobs_{default_teacher_name}"
    else:
        stu_topk_lp = rollout_batch.get("prev_topk_logprobs", None)
        assert stu_topk_lp is not None, "prev_topk_logprobs required for top-K advantage"
        primary_teacher_topk_key = f"teacher_on_stu_topk_logprobs_{default_teacher_name}"

    primary_teacher_topk_lp = rollout_batch.get(primary_teacher_topk_key, None)
    assert primary_teacher_topk_lp is not None, (
        f"{primary_teacher_topk_key} not found in rollout_batch; "
        f"available keys: {list(rollout_batch.keys())}"
    )

    multi_teacher_topk_lp: Dict[str, List[torch.Tensor]] = {}
    prefix = (
        "teacher_topk_logprobs_" if strategy == "only_tch" else "teacher_on_stu_topk_logprobs_"
    )
    for key, val in rollout_batch.items():
        if key.startswith(prefix):
            t_name = key[len(prefix):]
            if t_name != default_teacher_name:
                multi_teacher_topk_lp[t_name] = val

    base_topk_lp = (rollout_batch.get("base_on_topk_logprobs", None) if has_base else None)

    routing_field = getattr(ctx.config.ppo, "g_opd_teacher_routing_field", "teacher_type")
    teacher_types = rollout_batch.get(routing_field, None) if multi_teacher_topk_lp else None
    if teacher_types is not None and not isinstance(teacher_types, list):
        teacher_types = list(teacher_types)

    g_opd_lambda = getattr(ctx.config.ppo, "g_opd_lambda", 1.0)

    # Intersection: pass overlap_mask to restrict softmax to overlapping tokens.
    valid_mask = None
    if strategy == "intersection":
        valid_mask = rollout_batch.get(f"overlap_mask_{default_teacher_name}", None)

    topk_advantages, topk_metrics = calculate_topk_advantages(
        mask_lst=ctx.mask,
        stu_topk_logprobs=stu_topk_lp,
        teacher_topk_logprobs=primary_teacher_topk_lp,
        base_topk_logprobs=base_topk_lp,
        g_opd_lambda=g_opd_lambda,
        rewards=rewards,
        sampling_repeat_n=ctx.config.training.sampling_keep_n,
        advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
        multi_teacher_topk_logprobs=multi_teacher_topk_lp or None,
        teacher_types=teacher_types,
        default_teacher_name=default_teacher_name,
        topk_valid_mask=valid_mask,
    )
    return topk_advantages, topk_metrics


@register_advantage("identity")
def prepare_and_compute_identity_advantages(ctx: AdvantageContext) -> AdvantageResult:
    """Identity advantage — raw reward tiled to each token position, no normalization."""
    assert ctx.rewards is not None
    advantages, returns = calculate_identity_advantages(
        rewards=ctx.rewards,
        mask=ctx.mask,
    )
    return AdvantageResult(advantages=advantages, returns=returns)


@register_advantage("reinforce")
def prepare_and_compute_reinforce_advantages(ctx: AdvantageContext) -> AdvantageResult:
    assert ctx.rewards is not None
    advantages, returns = calculate_reinforce_advantages(
        rewards=ctx.rewards,
        mask=ctx.mask,
        gamma=ctx.config.ppo.reinforce_gamma,
    )
    return AdvantageResult(advantages=advantages, returns=returns)


# ============================================================================
# Built-in post-advantage functions
# ============================================================================


@register_post_advantage("gdpo")
def gdpo_token_bn_post_advantage(ctx: PostAdvantageContext) -> PostAdvantageResult:
    """Global BN at token level: expand to tokens first, then normalize across all valid tokens."""
    rollout_batches = ctx.rollout_batches
    assert "pre_bn_advantages" in rollout_batches[
        0], "gdpo_sample_bn_post_advantage requires pre_bn_advantages"
    epsilon = ctx.config.ppo.grpo_advantage_epsilon

    # Expand pre-BN sample scalars to token level, collect all valid token values
    all_token_values = []
    for rb in rollout_batches:
        pre_bn = rb["pre_bn_advantages"]
        mask = rb["mask"]
        for adv, m in zip(pre_bn, mask):
            expanded = adv.expand(m.shape[-1]) * m
            valid = expanded[m.bool()]
            if valid.numel() > 0:
                all_token_values.append(valid)

    if len(all_token_values) > 0:
        all_valid = torch.cat(all_token_values)
        local_sum = all_valid.sum()
        local_count = torch.tensor(all_valid.numel(), dtype=torch.float64)
    else:
        all_valid = None
        local_sum = torch.zeros((), dtype=torch.float64)
        local_count = torch.zeros((), dtype=torch.float64)

    # All-reduce to get global mean across DP ranks
    sum_and_count = torch.tensor(
        [local_sum.item(), local_count.item()],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_reduce(sum_and_count, group=ctx.dp_group)
    global_count = sum_and_count[1].item()

    if global_count > 0:
        global_mean = sum_and_count[0].item() / global_count
    else:
        global_mean = 0.0

    # All-reduce to get global std across DP ranks
    if all_valid is not None:
        local_var_sum = ((all_valid - global_mean)**2).sum()
    else:
        local_var_sum = torch.zeros((), dtype=torch.float64)

    var_sum_tensor = torch.tensor(
        [local_var_sum.item()],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_reduce(var_sum_tensor, group=ctx.dp_group)

    if global_count > 1:
        global_std = (var_sum_tensor[0].item() / (global_count - 1))**0.5
    else:
        global_std = 1.0

    # Apply token-level BN per rollout_batch
    advantage_clip_bounds = get_advantage_clip_bounds(
        ctx.config.ppo.advantage_clip,
        ctx.config.ppo.advantage_clip_lower_bound,
        ctx.config.ppo.advantage_clip_upper_bound,
    )
    for rollout_batch in rollout_batches:
        mask = rollout_batch["mask"]
        pre_bn = rollout_batch.pop("pre_bn_advantages")
        expanded = [adv.expand(m.shape[-1]) * m for adv, m in zip(pre_bn, mask)]
        advantages = [
            ((t - global_mean) / (global_std + epsilon)) * m for t, m in zip(expanded, mask)
        ]

        assert advantages[0].dtype == torch.float32
        if advantage_clip_bounds is not None:
            clip_lo, clip_hi = advantage_clip_bounds
            rollout_batch["original_advantages"] = advantages
            advantages = [a.clamp(min=clip_lo, max=clip_hi) for a in advantages]
        rollout_batch["advantages"] = advantages
        rollout_batch["returns"] = advantages

    # Global stats of final advantages (token-level, post-clip)
    all_adv_values = torch.cat(
        [a[m.bool()] for rb in rollout_batches for a, m in zip(rb["advantages"], rb["mask"])]
    )
    local_adv_sum = all_adv_values.sum().item() if all_adv_values.numel() > 0 else 0.0
    local_adv_count = float(all_adv_values.numel())
    adv_sc = torch.tensor(
        [local_adv_sum, local_adv_count],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_reduce(adv_sc, group=ctx.dp_group)
    global_adv_count = adv_sc[1].item()
    adv_mean = adv_sc[0].item() / global_adv_count if global_adv_count > 0 else 0.0

    local_adv_var_sum = ((all_adv_values -
                          adv_mean)**2).sum().item() if all_adv_values.numel() > 0 else 0.0
    adv_var_t = torch.tensor(
        [local_adv_var_sum], dtype=torch.float64, device=torch.cuda.current_device()
    )
    torch.distributed.all_reduce(adv_var_t, group=ctx.dp_group)
    adv_std = (adv_var_t[0].item() / (global_adv_count - 1))**0.5 if global_adv_count > 1 else 0.0

    metrics = {
        "ppo-metrics/global_bn_mean": global_mean * ctx.num_samples,
        "ppo-metrics/global_bn_std": global_std * ctx.num_samples,
        "ppo-metrics/advantages_mean": adv_mean * ctx.num_samples,
        "ppo-metrics/advantages_std": adv_std * ctx.num_samples,
    }
    return PostAdvantageResult(rollout_batches=rollout_batches, metrics=metrics)


@register_post_advantage("gdpo_sample_bn")
def gdpo_sample_bn_post_advantage(ctx: PostAdvantageContext) -> PostAdvantageResult:
    """Global BN at sample level: normalize sample scalars, then expand to tokens."""
    rollout_batches = ctx.rollout_batches
    assert "pre_bn_advantages" in rollout_batches[
        0], "gdpo_sample_bn_post_advantage requires pre_bn_advantages"
    epsilon = ctx.config.ppo.grpo_advantage_epsilon

    all_pre_bn = torch.cat([rb["pre_bn_advantages"] for rb in rollout_batches])
    all_sample_masks = []
    for rb in rollout_batches:
        sm = rb.get("sample_mask", None)
        if sm is not None:
            all_sample_masks.extend(sm)
    has_sample_mask = len(all_sample_masks) > 0

    if has_sample_mask:
        valid_mask = torch.stack(all_sample_masks).to(
            device=all_pre_bn.device, dtype=all_pre_bn.dtype
        )
    else:
        valid_mask = torch.ones_like(all_pre_bn)

    # All-reduce to get global mean across DP ranks
    local_sum = (all_pre_bn * valid_mask).sum()
    local_count = valid_mask.sum()
    sum_and_count = torch.tensor(
        [local_sum.item(), local_count.item()],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_reduce(sum_and_count, group=ctx.dp_group)
    global_count = sum_and_count[1].item()

    if global_count > 0:
        global_mean = sum_and_count[0].item() / global_count
    else:
        global_mean = 0.0

    # All-reduce to get global variance across DP ranks
    local_var_sum = (((all_pre_bn - global_mean)**2) * valid_mask).sum()
    var_sum_tensor = torch.tensor(
        [local_var_sum.item()],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_reduce(var_sum_tensor, group=ctx.dp_group)

    if global_count > 1:
        global_std = (var_sum_tensor[0].item() / (global_count - 1))**0.5
    else:
        global_std = 1.0

    # Normalize at sample level, then expand to token level
    advantage_clip_bounds = get_advantage_clip_bounds(
        ctx.config.ppo.advantage_clip,
        ctx.config.ppo.advantage_clip_lower_bound,
        ctx.config.ppo.advantage_clip_upper_bound,
    )
    all_normalized = []
    all_norm_masks = []
    for rollout_batch in rollout_batches:
        mask = rollout_batch["mask"]
        n = len(mask)
        pre_bn = rollout_batch.pop("pre_bn_advantages")
        assert pre_bn.shape[0] == n, f"pre_bn.shape[0] {pre_bn.shape[0]} != n {n}"

        normalized = (pre_bn - global_mean) / (global_std + epsilon)

        sm = rollout_batch.get("sample_mask", None)
        if sm is not None:
            sm_tensor = torch.stack(sm).to(device=normalized.device, dtype=normalized.dtype)
            normalized = normalized * sm_tensor
            all_norm_masks.append(sm_tensor)
        else:
            all_norm_masks.append(torch.ones(n, device=normalized.device, dtype=normalized.dtype))
        all_normalized.append(normalized)

        advantages = [adv.expand(m.shape[-1]) * m for adv, m in zip(normalized, mask)]

        assert advantages[0].dtype == torch.float32
        if advantage_clip_bounds is not None:
            clip_lo, clip_hi = advantage_clip_bounds
            rollout_batch["original_advantages"] = advantages
            advantages = [a.clamp(min=clip_lo, max=clip_hi) for a in advantages]
        rollout_batch["advantages"] = advantages
        rollout_batch["returns"] = advantages

    # Global stats of normalized sample-level advantages (pre-expand, pre-clip)
    all_norm = torch.cat(all_normalized)
    all_nm = torch.cat(all_norm_masks)
    local_norm_sum = (all_norm * all_nm).sum().item()
    local_norm_count = all_nm.sum().item()
    norm_sc = torch.tensor(
        [local_norm_sum, local_norm_count],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_reduce(norm_sc, group=ctx.dp_group)
    global_norm_count = norm_sc[1].item()
    norm_mean = norm_sc[0].item() / global_norm_count if global_norm_count > 0 else 0.0

    local_norm_var_sum = (((all_norm - norm_mean)**2) * all_nm).sum().item()
    norm_var_t = torch.tensor(
        [local_norm_var_sum], dtype=torch.float64, device=torch.cuda.current_device()
    )
    torch.distributed.all_reduce(norm_var_t, group=ctx.dp_group)
    norm_std = (
        norm_var_t[0].item() / (global_norm_count - 1)
    )**0.5 if global_norm_count > 1 else 0.0

    metrics = {
        "ppo-metrics/global_bn_mean": global_mean * ctx.num_samples,
        "ppo-metrics/global_bn_std": global_std * ctx.num_samples,
        "ppo-metrics/normalized_advantages_mean": norm_mean * ctx.num_samples,
        "ppo-metrics/normalized_advantages_std": norm_std * ctx.num_samples,
    }
    return PostAdvantageResult(rollout_batches=rollout_batches, metrics=metrics)
