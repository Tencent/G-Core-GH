"""Unit tests for tasks/math_rl_v4/custom_advantage adv_* feature_store hooks."""
from types import SimpleNamespace

import pytest
import torch

from gpatch_v4.core.advantage_helper import AdvantageContext
from gpatch_v4.core.ppo_feature_store import (
    feature_history_key,
    feature_pending_key,
    feature_reduce_key,
    get_ppo_feature_store,
    ppo_step_interval,
    reset_ppo_feature_store_for_test,
)
from tasks.math_rl_v4.custom_advantage import custom_grpo_advantage_with_adv_stats

_ADV_MEAN = "adv_mean"
_ADV_MIN = "adv_min"
_ADV_MAX = "adv_max"


def _make_ctx(rewards, masks, sampling_keep_n: int = 2):
    config = SimpleNamespace(
        training=SimpleNamespace(sampling_keep_n=sampling_keep_n),
        ppo=SimpleNamespace(grpo_advantage_epsilon=1e-4),
    )
    return AdvantageContext(
        rollout_batch={},
        config=config,
        mask=masks,
        logprobs=[],
        rewards=rewards,
        sample_mask=None,
    )


def _valid_vals(advantages, masks):
    return torch.cat([a[m.bool()] for a, m in zip(advantages, masks) if m.bool().any()]).float()


class TestCustomGrpoAdvantageWithAdvStats:
    def test_records_mean_min_max_and_reduces(self):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        rewards = [torch.tensor(1.0), torch.tensor(3.0)]
        masks = [torch.ones(4), torch.ones(4)]
        ctx = _make_ctx(rewards, masks)

        with ppo_step_interval(ppo_step=0, enabled=True) as iv:
            result = custom_grpo_advantage_with_adv_stats(ctx)
            assert len(result.advantages) == 2
            vals = _valid_vals(result.advantages, masks)
            expect_mean = float(vals.mean().item())
            expect_min = float(vals.min().item())
            expect_max = float(vals.max().item())

            assert store.get(feature_reduce_key(_ADV_MEAN)) == "mean"
            assert store.get(feature_reduce_key(_ADV_MIN)) == "min"
            assert store.get(feature_reduce_key(_ADV_MAX)) == "max"

            mean_sum, mean_w = store.get(feature_pending_key(_ADV_MEAN))[0]
            assert mean_w == pytest.approx(float(vals.numel()))
            assert mean_sum / mean_w == pytest.approx(expect_mean)
            assert store.get(feature_pending_key(_ADV_MIN))[0][0] == pytest.approx(expect_min)
            assert store.get(feature_pending_key(_ADV_MAX))[0][0] == pytest.approx(expect_max)

        assert iv.final_values[_ADV_MEAN] == pytest.approx(expect_mean)
        assert iv.final_values[_ADV_MIN] == pytest.approx(expect_min)
        assert iv.final_values[_ADV_MAX] == pytest.approx(expect_max)
        assert store.get(feature_history_key(_ADV_MEAN)) == [expect_mean]
        assert store.get(feature_history_key(_ADV_MIN)) == [expect_min]
        assert store.get(feature_history_key(_ADV_MAX)) == [expect_max]
        reset_ppo_feature_store_for_test()

    def test_reads_prior_history(self):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        store.set(feature_history_key(_ADV_MEAN), [0.5])
        store.set(feature_history_key(_ADV_MIN), [-1.0])
        store.set(feature_history_key(_ADV_MAX), [2.0])

        rewards = [torch.tensor(0.0), torch.tensor(2.0)]
        masks = [torch.ones(2), torch.ones(2)]
        ctx = _make_ctx(rewards, masks)

        with ppo_step_interval(ppo_step=1, enabled=True) as iv:
            assert store.get(feature_history_key(_ADV_MEAN)) == [0.5]
            custom_grpo_advantage_with_adv_stats(ctx)

        assert store.get(feature_history_key(_ADV_MEAN))[0] == pytest.approx(0.5)
        assert len(store.get(feature_history_key(_ADV_MEAN))) == 2
        assert len(store.get(feature_history_key(_ADV_MIN))) == 2
        assert len(store.get(feature_history_key(_ADV_MAX))) == 2
        assert iv.final_values[_ADV_MEAN] is not None
        reset_ppo_feature_store_for_test()

    def test_ckpt_save_load_preserves_adv_stats_history(self, tmp_path):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        rewards = [torch.tensor(1.0), torch.tensor(3.0)]
        masks = [torch.ones(4), torch.ones(4)]
        ctx = _make_ctx(rewards, masks)

        with ppo_step_interval(ppo_step=2, enabled=True) as iv:
            custom_grpo_advantage_with_adv_stats(ctx)
        iter_dir = str(tmp_path / "iter_0000002")
        store.save_to_ckpt(iter_dir, 2)

        reset_ppo_feature_store_for_test()
        loaded = get_ppo_feature_store()
        assert loaded.load_from_ckpt(iter_dir, 2)
        assert loaded.get(feature_reduce_key(_ADV_MEAN)) == "mean"
        assert loaded.get(feature_reduce_key(_ADV_MIN)) == "min"
        assert loaded.get(feature_reduce_key(_ADV_MAX)) == "max"
        assert loaded.get(feature_history_key(_ADV_MEAN)) == [iv.final_values[_ADV_MEAN]]
        assert loaded.get(feature_history_key(_ADV_MIN)) == [iv.final_values[_ADV_MIN]]
        assert loaded.get(feature_history_key(_ADV_MAX)) == [iv.final_values[_ADV_MAX]]
        assert loaded.get(feature_pending_key(_ADV_MEAN), []) == []
        reset_ppo_feature_store_for_test()
