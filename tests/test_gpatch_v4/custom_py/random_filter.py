import random
from typing import Any, Dict, List

from gpatch_v4.configs.config import RlConfig


def random_filter(
    config: RlConfig,
    rollout_batches: List[Dict[str, List[Any]]],
    sampling_repeat_n: int,
    sampling_keep_n: int,
):
    """Randomly select *sampling_keep_n* samples per prompt group.

    Each rollout batch contains ``rollout_mbs * sampling_repeat_n`` samples,
    grouped as ``rollout_mbs`` prompts each with ``sampling_repeat_n``
    responses.  For every prompt group, we randomly pick
    ``sampling_keep_n`` responses.

    A local RNG seeded deterministically ensures all PP/TP ranks select
    identical indices (the global ``random`` state may differ across ranks).
    """
    rng = random.Random(42)

    for rb in rollout_batches:
        num_samples = len(next(iter(rb.values())))
        assert num_samples % sampling_repeat_n == 0
        num_prompts = num_samples // sampling_repeat_n

        keep_indices = []
        for p in range(num_prompts):
            group_start = p * sampling_repeat_n
            group_indices = list(range(group_start, group_start + sampling_repeat_n))
            keep_indices.extend(sorted(rng.sample(group_indices, sampling_keep_n)))

        for k in rb.keys():
            rb[k] = [rb[k][i] for i in keep_indices]

    return rollout_batches
