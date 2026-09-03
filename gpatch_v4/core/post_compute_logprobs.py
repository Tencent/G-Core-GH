# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
"""Registry for post-``compute_logprobs`` hooks on rollout batches."""

from typing import Any, Callable, Dict, List

from gpatch_v4.utils.common_utils import import_fn_from_path

PostComputeLogprobsFn = Callable[[Any, List[Dict[str, Any]]], None]


class PostComputeLogprobsRegistry:
    """Named hooks run after prev/ref logprobs are written into rollout batches.

    Built-in ``\"none\"`` is a no-op. Custom hooks register at actor init via
    :func:`register_custom_post_compute_logprobs`.
    """

    _registry: Dict[str, PostComputeLogprobsFn] = {}

    @classmethod
    def register(cls, name: str, fn: PostComputeLogprobsFn) -> None:
        cls._registry[name] = fn

    @classmethod
    def get(cls, name: str) -> PostComputeLogprobsFn:
        assert name in cls._registry, (
            f"Unknown post_compute_logprobs '{name}'. "
            f"Available: {list(cls._registry.keys())}. "
            f"Use register_custom_post_compute_logprobs() for custom hooks."
        )
        return cls._registry[name]

    @classmethod
    def clear_for_test(cls) -> None:
        cls._registry.clear()
        cls.register("none", noop_post_compute_logprobs)


def noop_post_compute_logprobs(config: Any, rollout_batches: List[Dict[str, Any]]) -> None:
    return


def register_custom_post_compute_logprobs(
    name: str,
    py_path: str,
    fn_name: str,
) -> None:
    """Import ``fn_name`` from ``py_path`` and register under ``name``."""
    fn = import_fn_from_path(py_path, fn_name)
    if name in PostComputeLogprobsRegistry._registry:
        raise ValueError(f"post_compute_logprobs '{name}' is already registered.")
    PostComputeLogprobsRegistry.register(name, fn)


PostComputeLogprobsRegistry.register("none", noop_post_compute_logprobs)

BUILDIN_POST_COMPUTE_LOGPROBS = ["none"]
