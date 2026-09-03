from typing import Any, Callable, Dict, List, Optional

import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils.common_utils import import_fn_from_path
from gpatch_v4.utils.dynamic_batch_utils import group_trajectories, scalar


class FilterSamplingRegistry:
    """Registry for filter-sampling strategy functions.

    Built-in strategies are pre-registered at module load time.
    Custom strategies can be registered at actor init time via
    :func:`register_custom_filter_sampling`.
    """

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, fn: Callable):
        cls._registry[name] = fn

    @classmethod
    def get(cls, name: str) -> Callable:
        assert name in cls._registry, (
            f"Unknown filter sampling strategy '{name}'. "
            f"Available: {list(cls._registry.keys())}. "
            f"Use FilterSamplingRegistry.register() to add custom strategies."
        )
        return cls._registry[name]


def register_custom_filter_sampling(name: str, py_path: str, fn_name: str = "ppo_filter_samplings"):
    """Import a function from ``py_path`` and register it under ``name``.

    Parameters
    ----------
    name : str
    py_path : str
    fn_name : str
    """
    fn = import_fn_from_path(py_path, fn_name)
    FilterSamplingRegistry.register(name, fn)


# ---------------------------------------------------------------------------
# Built-in strategies (legacy rollout-batch path)
# ---------------------------------------------------------------------------


def keep_all(
    config: RlConfig, rollout_batches: List[Dict[str, List[Any]]], sampling_repeat_n: int,
    sampling_keep_n: int
):
    """Identity strategy — keep all samples unchanged."""
    return rollout_batches


def truncated_test(
    config: RlConfig, rollout_batches: List[Dict[str, List[Any]]], sampling_repeat_n: int,
    sampling_keep_n: int
):
    """Truncate rollout batches to keep only the first ``sampling_keep_n`` samples.

    Parameters
    ----------
    rollout_batches : list of dict
    sampling_repeat_n : int
    sampling_keep_n : int
    """
    for rb in rollout_batches:
        # assert len(rb) == 8
        for k in rb.keys():
            rb[k] = rb[k][:sampling_keep_n]


def best_and_worst(
    config: RlConfig, rollout_batches: List[Dict[str, List[Any]]], sampling_repeat_n: int,
    sampling_keep_n: int
):
    """Keep only the best and worst samples per rollout batch.

    Parameters
    ----------
    rollout_batches : list of dict
        Must have ``'rewards'`` key.
    sampling_repeat_n : int
    sampling_keep_n : int
        Must be 2.
    """
    assert sampling_keep_n == 2, f"sampling_keep_n must be 2 for best-and-worst strategy"
    for rb in rollout_batches:
        rewards = torch.cat(rb["rewards"]).view(-1)
        max_ind = torch.argmax(rewards).item()
        min_ind = torch.argmin(rewards).item()
        for k in rb.keys():
            rb[k] = [rb[k][max_ind], rb[k][min_ind]]


FilterSamplingRegistry.register("all", keep_all)
FilterSamplingRegistry.register("test", truncated_test)
FilterSamplingRegistry.register("best-and-worst", best_and_worst)

BUILDIN_FILTER_SAMPLING_STRATEGIES = ['all', 'test', 'best-and-worst']

# ---------------------------------------------------------------------------
# Dynamic-batch sample-level filter strategies
# ---------------------------------------------------------------------------

DynamicBatchFilterFn = Callable[[Any, List[Dict[str, Any]]], List[Dict[str, Any]]]


class DynamicBatchFilterRegistry:
    """Registry for sample-level dynamic-batch filter strategies."""

    _registry: Dict[str, DynamicBatchFilterFn] = {}

    @classmethod
    def register(cls, name: str, fn: DynamicBatchFilterFn) -> None:
        assert name not in cls._registry, f"filter strategy {name!r} already registered"
        cls._registry[name] = fn

    @classmethod
    def get(cls, name: str) -> DynamicBatchFilterFn:
        assert name in cls._registry, (
            f"Unknown dynamic-batch filter strategy {name!r}. "
            f"Available: {list(cls._registry.keys())}"
        )
        return cls._registry[name]


def _flatten_selected_trajectories(
    selected_by_group: List[List[tuple[Any, List[Dict[str, Any]]]]],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for selected in selected_by_group:
        for _, traj_samples in selected:
            filtered.extend(traj_samples)
    return filtered


def filter_best_and_worst(config, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the lowest- and highest-reward trajectories in each group.

    Groups with fewer than 2 trajectories are left unchanged so a later
    ``valid_group`` strategy can drop them explicitly.
    """
    keep_n = config.training.sampling_keep_n
    assert keep_n == 2, f"sampling_keep_n must be 2 for best-and-worst, got {keep_n}"
    selected_by_group = []
    for trajectories in group_trajectories(samples).values():
        traj_items = list(trajectories.items())
        if len(traj_items) < 2:
            selected_by_group.append(traj_items)
            continue
        scored = []
        for key, traj_samples in traj_items:
            reward = sum(float(scalar(sample["rewards"])) for sample in traj_samples)
            scored.append((reward, key, traj_samples))
        scored.sort(key=lambda item: item[0])
        selected_by_group.append([
            (scored[0][1], scored[0][2]),
            (scored[-1][1], scored[-1][2]),
        ])
    return _flatten_selected_trajectories(selected_by_group)


def filter_sample_mask(config, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop trajectories whose sample_mask is False."""
    selected_by_group = []
    for trajectories in group_trajectories(samples).values():
        selected = []
        for key, traj_samples in trajectories.items():
            sample_masks = [
                bool(scalar(sample.get("sample_mask", True))) for sample in traj_samples
            ]
            assert len(
                set(sample_masks)
            ) == 1, (f"trajectory {key!r} has inconsistent sample_mask values: {sample_masks}")
            if sample_masks[0]:
                selected.append((key, traj_samples))
        selected_by_group.append(selected)
    return _flatten_selected_trajectories(selected_by_group)


def filter_valid_group(config, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop groups that have fewer than 2 trajectories."""
    selected_by_group = []
    for trajectories in group_trajectories(samples).values():
        traj_items = list(trajectories.items())
        if len(traj_items) < 2:
            continue
        selected_by_group.append(traj_items)
    return _flatten_selected_trajectories(selected_by_group)


DynamicBatchFilterRegistry.register("best-and-worst", filter_best_and_worst)
DynamicBatchFilterRegistry.register("sample-mask", filter_sample_mask)
DynamicBatchFilterRegistry.register("valid_group", filter_valid_group)


def filter_rollout_samples(
    config,
    samples: List[Dict[str, Any]],
    custom_filter_fn: Optional[DynamicBatchFilterFn] = None,
) -> List[Dict[str, Any]]:
    """Apply configured dynamic-batch filter strategies, then optional custom hook.

    Empty ``dynamic_batch_rollout_filter_strategy`` keeps all samples.
    ``custom_filter_fn`` must be ``(config, samples) -> list[sample]``
    (not the legacy rollout-batch signature).
    """
    assert samples
    strategies = config.training.dynamic_batch_rollout_filter_strategy or []
    filtered = samples
    for strategy in strategies:
        filtered = DynamicBatchFilterRegistry.get(strategy)(config, filtered)
        assert filtered, f"filter strategy {strategy!r} removed every sample"
    if custom_filter_fn is not None:
        filtered = custom_filter_fn(config, filtered)
        assert isinstance(filtered, list) and filtered
    return filtered
