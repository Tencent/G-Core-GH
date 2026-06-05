from typing import Any, Callable, Dict, List

import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils.common_utils import import_fn_from_path


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
# Built-in strategies
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
