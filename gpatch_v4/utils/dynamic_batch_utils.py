from typing import Any, Callable, Dict, List, Optional

import torch

TRAIN_STEP_ID_KEY = "_train_step_id"


def validate_token_weights_mode(samples: List[Dict[str, Any]]) -> None:
    """Require one token-weight shape mode within a train batch."""
    assert samples
    has_token_weights = ["token_weights" in sample for sample in samples]
    assert all(has_token_weights) or not any(has_token_weights), (
        "token_weights must be present for every sample or absent for every sample"
    )
    if not has_token_weights[0]:
        return

    token_weights = [sample["token_weights"] for sample in samples]
    assert all(
        torch.is_tensor(weight) and weight.ndim == 1 and weight.numel() > 0
        for weight in token_weights
    ), ("token_weights must be non-empty 1D tensors")
    scalar_modes = [weight.numel() == 1 for weight in token_weights]
    assert len(set(scalar_modes)) == 1, (
        "scalar and token-level token_weights cannot be mixed in one train batch; "
        f"got shapes {[tuple(weight.shape) for weight in token_weights]}"
    )


def scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        assert value.numel() == 1, f"hierarchy id must be scalar, got {value.shape}"
        return value.item()
    return value


def traj_key(sample: Dict[str, Any]) -> tuple[Any, Any]:
    return scalar(sample["group_id"]), scalar(sample["traj_id"])


def group_trajectories(
    samples: List[Dict[str, Any]],
) -> Dict[Any, Dict[tuple[Any, Any], List[Dict[str, Any]]]]:
    groups: Dict[Any, Dict[tuple[Any, Any], List[Dict[str, Any]]]] = {}
    for sample in samples:
        group_id = scalar(sample["group_id"])
        trajectories = groups.setdefault(group_id, {})
        trajectory = trajectories.setdefault(traj_key(sample), [])
        trajectory.append(sample)
    return groups


def group_samples_by_train_step(samples: List[Dict[str, Any]], ) -> List[List[Dict[str, Any]]]:
    samples_by_step: Dict[int, List[Dict[str, Any]]] = {}
    for sample in samples:
        step_id = sample[TRAIN_STEP_ID_KEY]
        samples_by_step.setdefault(step_id, []).append(sample)
    assert samples_by_step
    assert set(samples_by_step) == set(
        range(len(samples_by_step))
    ), (f"train step ids must be contiguous from 0, got {sorted(samples_by_step)}")
    return [samples_by_step[step_id] for step_id in range(len(samples_by_step))]


def assign_train_steps(
    samples: List[Dict[str, Any]],
    training_config,
) -> List[Dict[str, Any]]:
    """Assign complete prompts to a fixed number of train steps. The principle of
    assigning train steps is to group prompts in this training step by prompt_idx,
    but gcore train_gbs config indicates the number of samples to be processed in
    each train step. So, the number of prompts in each train step is
    train_gbs / (sampling_keep_n * rb_multiplier).

    Returns flat samples with ``_train_step_id`` stamped on each sample.
    """
    assert samples
    assert all(TRAIN_STEP_ID_KEY not in sample for sample in samples
              ), (f"{TRAIN_STEP_ID_KEY} must not exist before train-step assignment")
    samples_by_prompt: Dict[int, List[Dict[str, Any]]] = {}
    for sample in samples:
        prompt_idx = scalar(sample["prompt_idx"])
        samples_by_prompt.setdefault(prompt_idx, []).append(sample)

    prompt_indices = sorted(samples_by_prompt)
    num_prompts = len(prompt_indices)
    assert num_prompts == training_config.rollout_gbs, (
        f"expected {training_config.rollout_gbs} prompts, got {num_prompts}"
    )
    samples_per_prompt = training_config.sampling_keep_n * training_config.rb_multiplier
    assert training_config.train_gbs % samples_per_prompt == 0, (
        f"train_gbs ({training_config.train_gbs}) must be divisible by nominal samples "
        f"per prompt ({samples_per_prompt})"
    )
    num_prompts_per_step = training_config.train_gbs // samples_per_prompt
    assert num_prompts % num_prompts_per_step == 0, (
        f"prompts ({num_prompts}) must be divisible by prompts per step "
        f"({num_prompts_per_step})"
    )
    num_train_steps = num_prompts // num_prompts_per_step

    assigned: List[Dict[str, Any]] = []
    for prompt_position, prompt_idx in enumerate(prompt_indices):
        step_id = prompt_position // num_prompts_per_step
        prompt_samples = samples_by_prompt[prompt_idx]
        for sample in prompt_samples:
            sample[TRAIN_STEP_ID_KEY] = step_id
        assigned.extend(prompt_samples)

    assert len(assigned) == len(samples)
    assert {sample[TRAIN_STEP_ID_KEY] for sample in assigned} == set(range(num_train_steps))
    return assigned


def validate_train_steps(samples: List[Dict[str, Any]]) -> None:
    """Validate hierarchy members never cross a train-step boundary."""
    assert samples
    train_steps = group_samples_by_train_step(samples)
    prompt_steps: Dict[int, int] = {}
    group_steps: Dict[Any, int] = {}
    for step_id, step_samples in enumerate(train_steps):
        assert step_samples, f"train step {step_id} is empty"
        for sample in step_samples:
            assert sample[TRAIN_STEP_ID_KEY] == step_id
            prompt_idx = scalar(sample["prompt_idx"])
            group_id = scalar(sample["group_id"])
            assert prompt_steps.setdefault(prompt_idx, step_id) == step_id
            assert group_steps.setdefault(group_id, step_id) == step_id


def validate_dynamic_batch_samples(
    samples: List[Dict[str, Any]],
    require_normalized_rewards: bool,
) -> None:
    """Validate samples after dynamic-batch conversion."""
    assert samples
    has_per_token_rewards = [
        "per_token_rewards" in sample and sample["per_token_rewards"] is not None
        for sample in samples
    ]
    assert len(
        set(has_per_token_rewards)
    ) == 1, ("per_token_rewards must be present for every sample or absent for every sample")

    valid_by_step: Dict[int, int] = {}
    for sample in samples:
        mask = sample["mask"]
        assert torch.is_tensor(mask) and mask.ndim == 1
        assert mask.numel() == len(sample["tokens"]) - 1
        if require_normalized_rewards:
            reward = sample["normalized_rewards"]
            assert torch.is_tensor(reward) and reward.numel() == 1
        assert torch.is_tensor(sample["sample_mask"]) and sample["sample_mask"].numel() == 1
        step_id = int(sample[TRAIN_STEP_ID_KEY])
        valid_by_step[step_id] = valid_by_step.get(step_id, 0) + int(mask.bool().any().item())

    for step_id, valid_count in valid_by_step.items():
        assert valid_count > 0, f"train step {step_id} has no valid samples"
    validate_train_steps(samples)
    for step_samples in group_samples_by_train_step(samples):
        validate_token_weights_mode(step_samples)


def compute_rollout_metrics(
    config,
    samples: List[Dict[str, Any]],
    metrics_report: List[str],
    custom_metrics_fn: Optional[Callable[[Any, List[Dict[str, Any]]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute filtered rollout metrics on the controller.

    Returns scalars plus two ``*_histogram`` entries (response lengths as
    a sample list, prompt lengths as 64-bin ``count``/``edges``). A custom
    ``custom_metrics_fn`` replaces this function entirely. See
    ``docs/source/metrics.md``.
    """
    if custom_metrics_fn is not None:
        metrics = custom_metrics_fn(config, samples)
        assert isinstance(metrics, dict)
        return metrics

    assert samples
    prompt_lengths = [float(scalar(sample["prompt_lengths"])) for sample in samples]
    sequence_lengths = [float(scalar(sample["sequence_lengths"])) for sample in samples]
    response_lengths = sorted(
        sequence_length - prompt_length
        for prompt_length, sequence_length in zip(prompt_lengths, sequence_lengths, strict=True)
    )
    metrics = {
        "rollout-metrics/global_response_lengths_mean":
            sum(response_lengths) / len(response_lengths),
        "rollout-metrics/global_prompt_lengths":
            sum(prompt_lengths) / len(prompt_lengths),
        "rollout-metrics/global_response_lengths_max":
            max(response_lengths),
        "rollout-metrics/global_response_lengths_min":
            min(response_lengths),
        "rollout-metrics/global_prompt_lengths_max":
            max(prompt_lengths),
        "rollout-metrics/global_prompt_lengths_min":
            min(prompt_lengths),
    }
    rewards = [float(scalar(sample["rewards"])) for sample in samples if "rewards" in sample]
    metrics["rollout-rewards/global_rewards"] = sum(rewards) / len(samples) if rewards else 0.0
    for metric_name in metrics_report:
        if metric_name not in samples[0]:
            continue
        values = [float(scalar(sample[metric_name])) for sample in samples]
        prefix = "rollout-rewards" if "reward" in metric_name else "rollout-metrics"
        metrics[f"{prefix}/global_{metric_name}"] = sum(values) / len(values)

    n = len(response_lengths)
    for percentile in (10, 50, 90):
        metrics[f"rollout-metrics/global_response_lengths_p{percentile}"] = response_lengths[min(
            n - 1, int(percentile / 100 * n)
        )]
    metrics["rollout-metrics/global_response_lengths_max_cnt"] = sum(
        length == response_lengths[-1] for length in response_lengths
    )
    metrics["rollout-metrics/global_response_lengths_min_cnt"] = sum(
        length == response_lengths[0] for length in response_lengths
    )
    n_bins = 64
    value_min = min(prompt_lengths)
    value_max = max(prompt_lengths)
    if value_min == value_max:
        value_max = value_min + 1.0
    bin_width = (value_max - value_min) / n_bins
    prompt_counts = [0.0] * n_bins
    for length in prompt_lengths:
        idx = min(max(int((length - value_min) / bin_width), 0), n_bins - 1)
        prompt_counts[idx] += 1.0
    metrics["rollout-metrics/global_response_lengths_histogram"] = response_lengths
    metrics["rollout-metrics/global_prompt_lengths_histogram"] = {
        "count": prompt_counts,
        "edges": [value_min + i * bin_width for i in range(n_bins + 1)],
    }
    return metrics


def convert_samples_to_train_data(
    config,
    samples: List[Dict[str, Any]],
    custom_convert_fn: Optional[Callable[[Any, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Apply the optional dynamic-batch train-data conversion hook.

    The hook receives a flat ``list[sample]`` whose entries already carry
    ``_train_step_id``. Step-aware logic should group by that field.
    """
    if custom_convert_fn is None:
        return samples
    converted = custom_convert_fn(config, samples)
    assert isinstance(converted, list) and converted
    return converted


def split_train_steps_by_dp(
    samples: List[Dict[str, Any]],
    dp_size: int,
    train_mbs: int,
    dynamic_context_parallel: bool,
) -> List[List[List[Dict[str, Any]]]]:
    """Balance each train step by token length at sample granularity.

    If dynamic_context_parallel is False, the number of samples in each train
    step must be divisible by dp_size * train_mbs.
    """
    train_steps = group_samples_by_train_step(samples)
    dp_steps: List[List[List[Dict[str, Any]]]] = [[] for _ in range(dp_size)]
    for step_id, step_samples in enumerate(train_steps):
        assert len(step_samples) >= dp_size, (
            f"train step {step_id} has {len(step_samples)} samples for {dp_size} DP ranks"
        )
        if not dynamic_context_parallel:
            divisor = dp_size * train_mbs
            assert len(step_samples) % divisor == 0, (
                f"train step {step_id} has {len(step_samples)} samples, not divisible by "
                f"dp_size * train_mbs ({dp_size} * {train_mbs}); enable dynamic CP"
            )
            capacity = len(step_samples) // dp_size
        else:
            capacity = None

        partitions: List[List[Dict[str, Any]]] = [[] for _ in range(dp_size)]
        token_sums = [0] * dp_size
        sorted_samples = sorted(
            step_samples,
            key=lambda sample: int(scalar(sample["sequence_lengths"])),
            reverse=True,
        )
        for sample in sorted_samples:
            candidates = [
                rank
                for rank in range(dp_size) if capacity is None or len(partitions[rank]) < capacity
            ]
            assert candidates
            rank = min(candidates, key=lambda candidate: (token_sums[candidate], candidate))
            partitions[rank].append(sample)
            token_sums[rank] += int(scalar(sample["sequence_lengths"]))

        assert all(partitions)
        if not dynamic_context_parallel:
            local_counts = {len(partition) for partition in partitions}
            assert local_counts == {capacity}
            assert capacity % train_mbs == 0
        for rank, partition in enumerate(partitions):
            dp_steps[rank].append(partition)
    return dp_steps
