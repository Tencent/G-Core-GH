from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

from gpatch_v4.core.advantage_impl import (
    _calc_grpo_advantages_func,
    compute_gdpo_combined_advantages,
)
from gpatch_v4.utils.dynamic_batch_utils import group_trajectories, scalar, traj_key

RewardNormalizeFn = Callable[[Any, List[Dict[str, Any]]], Dict[str, float]]
REWARD_NORMALIZE_DISPATCH: Dict[str, RewardNormalizeFn] = {}


def register_reward_normalize(name: str):
    def decorator(fn: RewardNormalizeFn) -> RewardNormalizeFn:
        assert name not in REWARD_NORMALIZE_DISPATCH, (
            f"reward normalize type {name!r} is already registered"
        )
        REWARD_NORMALIZE_DISPATCH[name] = fn
        return fn

    return decorator


def get_reward_normalize_fn(name: str) -> Optional[RewardNormalizeFn]:
    return REWARD_NORMALIZE_DISPATCH.get(name)


def prepare_sample_masks(samples: List[Dict[str, Any]]) -> None:
    """Create or align response masks on the next-token axis."""
    assert samples
    for sample in samples:
        target_length = len(sample["tokens"]) - 1
        assert target_length >= 0
        if "mask" not in sample or sample["mask"] is None:
            prompt_length = int(scalar(sample["prompt_lengths"]))
            sequence_length = int(scalar(sample["sequence_lengths"]))
            assert 0 < prompt_length <= sequence_length <= len(sample["tokens"])
            mask = torch.zeros(target_length, dtype=torch.float32)
            mask[prompt_length - 1:sequence_length - 1] = 1.0
        else:
            mask = sample["mask"]
            assert torch.is_tensor(mask) and mask.ndim == 1
            assert mask.numel() <= target_length, (
                f"mask length {mask.numel()} exceeds next-token length {target_length}"
            )
            mask = torch.nn.functional.pad(mask, (0, target_length - mask.numel()), value=0)
        sample_mask = sample.get("sample_mask")
        if sample_mask is None:
            sample_mask = torch.tensor(True)
            sample["sample_mask"] = sample_mask
        else:
            assert torch.is_tensor(sample_mask) and sample_mask.numel() == 1
            if not bool(sample_mask.item()):
                mask.zero_()
        sample["mask"] = mask


def _prepare_normalizable_trajectories(
    trajectories: Dict[Tuple[Any, Any], List[Dict[str, Any]]],
) -> Dict[Tuple[Any, Any], List[Dict[str, Any]]]:
    valid = {}
    for key, traj_samples in trajectories.items():
        sample_masks = [bool(scalar(sample.get("sample_mask", True))) for sample in traj_samples]
        assert len(
            set(sample_masks)
        ) == 1, (f"trajectory {key!r} has inconsistent sample_mask values: {sample_masks}")
        if sample_masks[0]:
            valid[key] = traj_samples
    if len(valid) <= 1:
        for trajectory_samples in trajectories.values():
            for sample in trajectory_samples:
                sample["sample_mask"] = torch.tensor(False)
                sample["mask"].zero_()
                sample["normalized_rewards"] = torch.tensor(0.0, dtype=torch.float32)
        return {}
    return valid


def _aggregate_trajectory_rewards(
    trajectories: Dict[Tuple[Any, Any], List[Dict[str, Any]]],
    reward_names: Iterable[str],
) -> Tuple[List[Tuple[Any, Any]], Dict[str, List[torch.Tensor]], List[torch.Tensor]]:
    keys = list(trajectories)
    rewards_dict = {
        reward_name:
            [
                torch.tensor(
                    sum(float(scalar(sample[reward_name])) for sample in trajectories[key]),
                    dtype=torch.float32,
                ) for key in keys
            ]
        for reward_name in reward_names
    }
    masks = [torch.ones(1, dtype=torch.float32) for _ in keys]
    return keys, rewards_dict, masks


def _write_normalized_rewards(
    trajectories: Dict[Tuple[Any, Any], List[Dict[str, Any]]],
    combined: Dict[Tuple[Any, Any], float],
) -> None:
    for key, traj_samples in trajectories.items():
        value = combined.get(key, 0.0)
        for sample in traj_samples:
            sample["normalized_rewards"] = torch.tensor(value, dtype=torch.float32)


def _compute_gdpo_normalized_rewards(
    config,
    samples: List[Dict[str, Any]],
) -> Dict[Tuple[Any, Any], float]:
    reward_weights = config.ppo.gdpo_reward_weights
    assert reward_weights
    combined: Dict[Tuple[Any, Any], float] = {}
    for trajectories in group_trajectories(samples).values():
        valid = _prepare_normalizable_trajectories(trajectories)
        if not valid:
            continue
        keys, rewards_dict, masks = _aggregate_trajectory_rewards(valid, reward_weights)
        values = compute_gdpo_combined_advantages(
            rewards_dict=rewards_dict,
            mask=masks,
            grpo_sampling_times=len(keys),
            grpo_advantage_epsilon=config.ppo.grpo_advantage_epsilon,
            gdpo_reward_weights=reward_weights,
        )
        combined.update({key: float(value.item()) for key, value in zip(keys, values, strict=True)})
        _write_normalized_rewards(trajectories, combined)
    return combined


def _weighted_mean_std(values: List[float], counts: List[int]) -> Tuple[float, float]:
    assert values and len(values) == len(counts)
    count = sum(counts)
    assert count > 0
    values_tensor = torch.tensor(values, dtype=torch.float64)
    counts_tensor = torch.tensor(counts, dtype=torch.float64)
    mean = (values_tensor * counts_tensor).sum() / count
    if count == 1:
        return mean.item(), 1.0
    variance = ((values_tensor - mean).square() * counts_tensor).sum() / (count - 1)
    return mean.item(), variance.clamp(min=0).sqrt().item()


def _apply_gdpo_bn(
    config,
    partitions: List[List[Dict[str, Any]]],
    combined: Dict[Tuple[Any, Any], float],
    sample_bn: bool,
) -> Tuple[float, float, float, float]:
    epsilon = config.ppo.grpo_advantage_epsilon
    weighted_means = []
    weighted_stds = []
    all_normalized = []
    all_normalized_counts = []
    total_weight = 0
    for partition in partitions:
        counts_by_key = {}
        for sample in partition:
            key = traj_key(sample)
            if key not in combined:
                continue
            if sample_bn:
                counts_by_key[key] = 1
            else:
                counts_by_key[key] = (
                    counts_by_key.get(key, 0) + int(sample["mask"].bool().sum().item())
                )
        counts_by_key = {key: count for key, count in counts_by_key.items() if count > 0}
        values = [combined[key] for key in counts_by_key]
        counts = list(counts_by_key.values())
        if values:
            mean, std = _weighted_mean_std(values, counts)
            all_normalized.extend((value - mean) / (std + epsilon) for value in values)
            all_normalized_counts.extend(counts)
        else:
            mean, std = 0.0, 1.0
        for sample in partition:
            key = traj_key(sample)
            value = (combined[key] - mean) / (std + epsilon) if key in combined else 0.0
            sample["normalized_rewards"] = torch.tensor(value, dtype=torch.float32)
        weight = sum(counts)
        weighted_means.append(mean * weight)
        weighted_stds.append(std * weight)
        total_weight += weight

    assert total_weight > 0
    bn_mean = sum(weighted_means) / total_weight
    bn_std = sum(weighted_stds) / total_weight
    norm_mean, norm_std = (
        _weighted_mean_std(all_normalized, all_normalized_counts) if all_normalized else (0.0, 0.0)
    )
    return bn_mean, bn_std, norm_mean, norm_std


@register_reward_normalize("grpo")
def _reward_normalize_grpo(config, samples: List[Dict[str, Any]]) -> Dict[str, float]:
    for trajectories in group_trajectories(samples).values():
        valid = _prepare_normalizable_trajectories(trajectories)
        if not valid:
            continue
        keys, rewards_dict, masks = _aggregate_trajectory_rewards(valid, ["rewards"])
        values = _calc_grpo_advantages_func(
            rewards_dict["rewards"],
            masks,
            grpo_sampling_times=len(keys),
            grpo_advantage_epsilon=config.ppo.grpo_advantage_epsilon,
        )
        normalized_rewards = {
            key: float(value.item())
            for key, value in zip(keys, values, strict=True)
        }
        _write_normalized_rewards(trajectories, normalized_rewards)
    return {}


@register_reward_normalize("gdpo")
def _reward_normalize_gdpo(
    config,
    samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    combined = _compute_gdpo_normalized_rewards(config, samples)
    bn_mean, bn_std, norm_mean, norm_std = _apply_gdpo_bn(
        config,
        [samples],
        combined,
        sample_bn=False,
    )
    return {
        "ppo-metrics/global_bn_mean": bn_mean,
        "ppo-metrics/global_bn_std": bn_std,
        "ppo-metrics/advantages_mean": norm_mean,
        "ppo-metrics/advantages_std": norm_std,
    }


@register_reward_normalize("gdpo_sample_bn")
def _reward_normalize_gdpo_sample_bn(
    config,
    samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    combined = _compute_gdpo_normalized_rewards(config, samples)
    bn_mean, bn_std, norm_mean, norm_std = _apply_gdpo_bn(
        config,
        [samples],
        combined,
        sample_bn=True,
    )
    return {
        "ppo-metrics/global_bn_mean": bn_mean,
        "ppo-metrics/global_bn_std": bn_std,
        "ppo-metrics/normalized_advantages_mean": norm_mean,
        "ppo-metrics/normalized_advantages_std": norm_std,
    }


@register_reward_normalize("group_gdpo")
def _reward_normalize_group_gdpo(
    config,
    samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    combined = _compute_gdpo_normalized_rewards(config, samples)
    grouped_samples: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped_samples[scalar(sample["group_id"])].append(sample)
    _apply_gdpo_bn(
        config,
        list(grouped_samples.values()),
        combined,
        sample_bn=False,
    )
    return {}


@register_reward_normalize("group_gdpo_sample_bn")
def _reward_normalize_group_gdpo_sample_bn(
    config,
    samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    combined = _compute_gdpo_normalized_rewards(config, samples)
    grouped_samples: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped_samples[scalar(sample["group_id"])].append(sample)
    bn_mean, bn_std, norm_mean, norm_std = _apply_gdpo_bn(
        config,
        list(grouped_samples.values()),
        combined,
        sample_bn=True,
    )
    return {
        "ppo-metrics/sample_normalized_advantages_mean":
            norm_mean,
        "ppo-metrics/sample_normalized_advantages_std":
            norm_std,
        "ppo-metrics/sample_bn_mean":
            bn_mean,
        "ppo-metrics/sample_bn_std":
            bn_std,
        "ppo-metrics/num_zero_combined_adv":
            float(sum(value == 0.0 for value in combined.values())),
        "ppo-metrics/bn_std_is_zero":
            float(bn_std == 0.0),
    }


def reward_normalize(
    config,
    samples: List[Dict[str, Any]],
    custom_reward_normalize_fn: Optional[RewardNormalizeFn] = None,
) -> Dict[str, float]:
    """Prepare normalized rewards before DP partitioning."""
    if custom_reward_normalize_fn is not None:
        metrics = custom_reward_normalize_fn(config, samples)
    else:
        normalize_fn = get_reward_normalize_fn(config.ppo.advantage_type)
        if normalize_fn is None:
            return {}
        metrics = normalize_fn(config, samples)
    assert isinstance(metrics, dict)
    return metrics


def uses_group_reward_normalization(config) -> bool:
    return (
        config.ppo.custom_reward_normalize_py_path is not None or
        config.ppo.advantage_type in REWARD_NORMALIZE_DISPATCH
    )
