from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from gpatch_v4.core.advantage_impl import (
    calculate_identity_advantages,
    calculate_ppo_advantages_and_returns,
    calculate_ppo_rewards,
    calculate_reinforce_advantages,
)
from gpatch_v4.utils import log, masked_global_statistics_list, masked_mean_list
from gpatch_v4.utils.common_utils import import_fn_from_path
from gpatch_v4.utils.ppo_utils import (
    align_token_level_tensors_to_logprobs,
    calculate_kl_penalty,
    count_advantage_clip_samples,
    get_advantage_clip_bounds,
)
from gpatch_v4.utils.training_utils import whiten_advantages_cross_dp

DynamicAdvantageResult = Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    Optional[List[torch.Tensor]],
]
DynamicAdvantageFn = Callable[
    [Any, List[Dict[str, Any]]],
    DynamicAdvantageResult,
]
DYNAMIC_ADVANTAGE_DISPATCH: Dict[str, DynamicAdvantageFn] = {}


def register_dynamic_advantage(*names: str):
    def decorator(fn: DynamicAdvantageFn) -> DynamicAdvantageFn:
        for name in names:
            assert name not in DYNAMIC_ADVANTAGE_DISPATCH, (
                f"dynamic advantage type {name!r} is already registered"
            )
            DYNAMIC_ADVANTAGE_DISPATCH[name] = fn
        return fn

    return decorator


def get_dynamic_advantage_fn(name: str) -> DynamicAdvantageFn:
    assert name in DYNAMIC_ADVANTAGE_DISPATCH, (f"unsupported dynamic advantage_type: {name!r}")
    return DYNAMIC_ADVANTAGE_DISPATCH[name]


def register_custom_dynamic_advantage(name: str, py_path: str, fn_name: str) -> None:
    """Import ``(config, samples) -> (advantages, returns, init_policy_kl)`` and register it."""
    fn = import_fn_from_path(py_path, fn_name)
    if name in DYNAMIC_ADVANTAGE_DISPATCH:
        log(
            f"Dynamic advantage type '{name}' already registered — overwriting with "
            f"custom function {fn_name} from {py_path}",
            rank=0,
        )
    DYNAMIC_ADVANTAGE_DISPATCH[name] = fn
    log(f"Registered custom dynamic advantage '{name}' from {py_path}::{fn_name}", rank=0)


def _reward_tensor(value: Any, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return reference.new_tensor(value, dtype=torch.float32)


@register_dynamic_advantage("ppo")
def _compute_ppo_advantages(
    config,
    samples: List[Dict[str, Any]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    logprobs = [sample["logprobs"] for sample in samples]
    values = [sample["values"] for sample in samples]
    assert all(
        value.shape == logprob.shape for value, logprob in zip(values, logprobs, strict=True)
    )

    if all(
        "per_token_rewards" in sample and sample["per_token_rewards"] is not None
        for sample in samples
    ):
        per_token_rewards = align_token_level_tensors_to_logprobs(
            [sample["per_token_rewards"] for sample in samples],
            logprobs,
            [sample["sequence_lengths"] for sample in samples],
            config.ppo.ppo_value_truncate_head,
        )
        for sample, reward in zip(samples, per_token_rewards, strict=True):
            sample["per_token_rewards"] = reward
    else:
        assert all(
            "per_token_rewards" not in sample or sample["per_token_rewards"] is None
            for sample in samples
        )
        per_token_rewards = [None] * len(samples)

    if config.ppo.ppo_initial_policy_kl_penalty > 0:
        ref_logprobs = [sample["ref_logprobs"] for sample in samples]
        init_policy_kl = calculate_kl_penalty(
            logprobs,
            ref_logprobs,
            use_absolute_kl=config.ppo.ppo_use_absolute_kl,
        )
    else:
        init_policy_kl = [torch.zeros_like(logprob) for logprob in logprobs]

    advantages = []
    returns = []
    for sample, value, kl, token_reward in zip(
        samples, values, init_policy_kl, per_token_rewards, strict=True
    ):
        rewards_with_kl = calculate_ppo_rewards(
            value,
            _reward_tensor(sample["rewards"], value),
            None,
            sample["sequence_lengths"],
            kl,
            config.ppo.ppo_initial_policy_kl_penalty,
        )
        advantage, value_return = calculate_ppo_advantages_and_returns(
            values=value,
            rewards=rewards_with_kl,
            discount_factor=config.ppo.ppo_discount_factor,
            gae_lambda=config.ppo.ppo_gae_lambda,
            mask=sample["mask"],
            per_token_rewards=token_reward,
        )
        advantages.append(advantage)
        returns.append(value_return)
    return advantages, returns, init_policy_kl


@register_dynamic_advantage(
    "grpo",
    "gdpo",
    "gdpo_sample_bn",
    "group_gdpo",
    "group_gdpo_sample_bn",
)
def _compute_normalized_reward_advantages(
    _config,
    samples: List[Dict[str, Any]],
) -> DynamicAdvantageResult:
    masks = [sample["mask"] for sample in samples]
    advantages, returns = calculate_identity_advantages(
        [sample["normalized_rewards"] for sample in samples],
        masks,
    )
    return advantages, returns, None


@register_dynamic_advantage("identity")
def _compute_identity_advantages(
    _config,
    samples: List[Dict[str, Any]],
) -> DynamicAdvantageResult:
    masks = [sample["mask"] for sample in samples]
    advantages, returns = calculate_identity_advantages(
        [_reward_tensor(sample["rewards"], sample["mask"]) for sample in samples],
        masks,
    )
    return advantages, returns, None


@register_dynamic_advantage("reinforce")
def _compute_reinforce_advantages(
    config,
    samples: List[Dict[str, Any]],
) -> DynamicAdvantageResult:
    masks = [sample["mask"] for sample in samples]
    advantages, returns = calculate_reinforce_advantages(
        [_reward_tensor(sample["rewards"], sample["mask"]) for sample in samples],
        masks,
        gamma=config.ppo.reinforce_gamma,
    )
    return advantages, returns, None


DYNAMIC_BATCH_ADVANTAGE_TYPES = set(DYNAMIC_ADVANTAGE_DISPATCH)


def _compute_global_metrics(
    samples: List[Dict[str, Any]],
    dp_group,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    masks = [sample["mask"] for sample in samples]
    for key in ["advantages", "original_advantages", "returns", "values", "per_token_rewards"]:
        if key not in samples[0] or samples[0][key] is None:
            continue
        values = [sample[key] for sample in samples]
        mean, var, min_value, max_value = masked_global_statistics_list(
            values,
            masks,
            key_name=key,
            group=dp_group,
        )
        metrics[f"ppo-metrics/global_{key}_mean"] = mean.item()
        metrics[f"ppo-metrics/global_{key}_std"] = var.sqrt().item()
        metrics[f"ppo-metrics/global_{key}_min"] = min_value.item()
        metrics[f"ppo-metrics/global_{key}_max"] = max_value.item()

    if "sample_mask" in samples[0]:
        sample_masks = torch.stack([sample["sample_mask"] for sample in samples]).float()
        sum_and_count = torch.tensor(
            [sample_masks.sum(), sample_masks.numel()],
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(sum_and_count, group=dp_group)
        retention_ratio = (sum_and_count[0] / sum_and_count[1]).item()
        metrics["ppo-metrics/global_sample_mask_mean"] = retention_ratio

    if "original_advantages" in samples[0]:
        n_lower, n_upper, n_samples = count_advantage_clip_samples(
            [sample["original_advantages"] for sample in samples],
            [sample["advantages"] for sample in samples],
            masks,
        )
        counts = torch.tensor(
            [n_lower, n_upper, n_samples],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(counts, group=dp_group)
        global_n = counts[2].item()
        lower_fraction = counts[0].item() / global_n if global_n > 0 else 0.0
        upper_fraction = counts[1].item() / global_n if global_n > 0 else 0.0
        metrics["ppo-metrics/advantage_clip_lower_sample_frac"] = lower_fraction
        metrics["ppo-metrics/advantage_clip_upper_sample_frac"] = upper_fraction
        metrics["ppo-metrics/advantage_clip_sample_frac"] = lower_fraction + upper_fraction
    return metrics


def compute_dynamic_batch_advantages(
    config,
    samples: List[Dict[str, Any]],
    dp_group,
) -> Dict[str, float]:
    """Compute final token advantages for one dynamic train step."""
    assert samples
    masks = [sample["mask"] for sample in samples]
    advantage_type = config.ppo.advantage_type
    advantage_fn = get_dynamic_advantage_fn(advantage_type)
    advantages, returns, init_policy_kl = advantage_fn(config, samples)

    assert all(advantage.dtype == torch.float32 for advantage in advantages)
    for sample, value_return in zip(samples, returns, strict=True):
        sample["returns"] = value_return

    metrics: Dict[str, float] = {}
    if config.ppo.whiten_advantages:
        # GDPO variants already normalize rewards before advantage computation;
        # whitening their derived advantages would apply normalization twice.
        assert config.ppo.advantage_type in ("identity", "reinforce", "ppo", "grpo"), (
            "whiten_advantages only support identity, reinforce, ppo and grpo, "
            f"got {config.ppo.advantage_type!r}"
        )
        advantages, whiten_metrics = whiten_advantages_cross_dp(
            advantages,
            masks,
            dp_group=dp_group,
        )
        metrics.update(whiten_metrics)

    clip_bounds = get_advantage_clip_bounds(
        config.ppo.advantage_clip,
        config.ppo.advantage_clip_lower_bound,
        config.ppo.advantage_clip_upper_bound,
    )
    if clip_bounds is not None:
        lower, upper = clip_bounds
        original_advantages = advantages
        advantages = [advantage.clamp(min=lower, max=upper) for advantage in advantages]
        for sample, original in zip(samples, original_advantages, strict=True):
            sample["original_advantages"] = original
    for sample, advantage in zip(samples, advantages, strict=True):
        sample["advantages"] = advantage

    if init_policy_kl is not None and config.ppo.ppo_initial_policy_kl_penalty > 0:
        metrics["ppo-metrics/init_policy_kl"] = (
            masked_mean_list(init_policy_kl, masks, dim=-1).mean().item()
        )
    metrics.update(_compute_global_metrics(samples, dp_group))
    return metrics
