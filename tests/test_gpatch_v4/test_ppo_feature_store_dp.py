"""Multi-rank gloo tests for PpoFeatureStore DP reduce (mean/min/max)."""
import os

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from gpatch_v4.core.ppo_feature_store import (
    feature_history_key,
    feature_pending_key,
    get_ppo_feature_store,
    reset_ppo_feature_store_for_test,
)

_WORLD = 2
# Unequal weights so naive avg-of-means would be wrong: (1+100)/2=50.5 vs 110/11=10.
_RANK0_MEAN = (10.0, 10.0)  # sum, weight
_RANK1_MEAN = (100.0, 1.0)
_EXPECT_MEAN = 110.0 / 11.0
_RANK0_MIN, _RANK1_MIN = 1.0, -2.0
_EXPECT_MIN = -2.0
_RANK0_MAX, _RANK1_MAX = 3.0, 9.0
_EXPECT_MAX = 9.0


def _dp_worker(rank: int, world: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        store.begin_ppo_step_interval(1)
        group = dist.group.WORLD

        if rank == 0:
            store.record("a", _RANK0_MEAN[0], weight=_RANK0_MEAN[1], reduce="mean")
            store.record("b", _RANK0_MIN, reduce="min")
            store.record("c", _RANK0_MAX, reduce="max")
        else:
            store.record("a", _RANK1_MEAN[0], weight=_RANK1_MEAN[1], reduce="mean")
            store.record("b", _RANK1_MIN, reduce="min")
            store.record("c", _RANK1_MAX, reduce="max")

        values = store.finalize_all_pending_features(group=group)
        assert values["a"] == pytest.approx(_EXPECT_MEAN)
        assert values["b"] == pytest.approx(_EXPECT_MIN)
        assert values["c"] == pytest.approx(_EXPECT_MAX)
        assert store.get(feature_history_key("a"))[0] == pytest.approx(_EXPECT_MEAN)
        assert store.get(feature_history_key("b"))[0] == pytest.approx(_EXPECT_MIN)
        assert store.get(feature_history_key("c"))[0] == pytest.approx(_EXPECT_MAX)
        assert store.get(feature_pending_key("a"), []) == []
    finally:
        reset_ppo_feature_store_for_test()
        dist.destroy_process_group()


def test_dp_mean_min_max_across_ranks():
    port = str(29580 + os.getpid() % 1000)
    mp.spawn(_dp_worker, args=(_WORLD, port), nprocs=_WORLD, join=True)
