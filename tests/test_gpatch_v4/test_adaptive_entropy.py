# coding=utf-8
"""Adaptive entropy: L1 controller / L2 metric wiring / L3 multi-step smoke."""

from __future__ import annotations

import pytest
import torch

from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.core.adaptive_entropy import (
    ENTROPY_COEF_KEY,
    LAST_WORLD_ENTROPY_KEY,
    get_entropy_bonus_coef,
    update_adaptive_entropy_after_train_step,
)
from gpatch_v4.core.ppo_feature_store import (
    get_ppo_feature_store,
    reset_ppo_feature_store_for_test,
)
from gpatch_v4.utils.training_utils import extend_value_to_dict


@pytest.fixture(autouse=True)
def _reset_store():
    reset_ppo_feature_store_for_test()
    yield
    reset_ppo_feature_store_for_test()


def _adaptive_cfg(**kwargs) -> PpoConfig:
    defaults = dict(
        use_adaptive_entropy=True,
        feature_store_enable=True,
        ppo_entropy_bonus=0.01,
        entropy_target=5.0,
        entropy_coef_delta=0.005,
        entropy_coef_min=0.0,
        entropy_coef_max=1.0,
    )
    defaults.update(kwargs)
    return PpoConfig(**defaults)


def _token_weighted_mean(mb_stats: list[tuple[float, float]]) -> float:
    """Mirror mixin: sum [sum,count] over MBs then mean (single-rank)."""
    total = torch.tensor([0.0, 0.0])
    for s, c in mb_stats:
        total = total + torch.tensor([s, c])
    return (total[0] / total[1].clamp(min=1)).item()


# ---------------------------------------------------------------------------
# L1: controller (TRL lag-1)
# ---------------------------------------------------------------------------


class TestAdaptiveEntropyL1Controller:
    def test_first_step_gating_is_zero(self):
        cfg = _adaptive_cfg()
        assert get_entropy_bonus_coef(cfg) == 0.0

    def test_below_target_increments_coef_and_applies(self):
        cfg = _adaptive_cfg()
        metrics = update_adaptive_entropy_after_train_step(cfg, 3.0)
        assert metrics["policy/last_world_entropy"] == pytest.approx(3.0)
        assert metrics["policy/entropy_coef"] == pytest.approx(0.015)
        store = get_ppo_feature_store()
        assert store.get(LAST_WORLD_ENTROPY_KEY) == pytest.approx(3.0)
        assert store.get(ENTROPY_COEF_KEY) == pytest.approx(0.015)
        assert get_entropy_bonus_coef(cfg) == pytest.approx(0.015)

    def test_above_target_decrements_and_gates_off(self):
        cfg = _adaptive_cfg()
        update_adaptive_entropy_after_train_step(cfg, 3.0)
        metrics = update_adaptive_entropy_after_train_step(cfg, 6.0)
        assert metrics["policy/last_world_entropy"] == pytest.approx(6.0)
        assert metrics["policy/entropy_coef"] == pytest.approx(0.01)
        assert get_entropy_bonus_coef(cfg) == 0.0

    def test_coef_clamped_to_max(self):
        cfg = _adaptive_cfg(ppo_entropy_bonus=0.99, entropy_coef_delta=0.05, entropy_coef_max=1.0)
        update_adaptive_entropy_after_train_step(cfg, 1.0)
        assert get_ppo_feature_store().get(ENTROPY_COEF_KEY) == pytest.approx(1.0)

    def test_coef_clamped_to_min(self):
        cfg = _adaptive_cfg(ppo_entropy_bonus=0.01, entropy_coef_delta=0.05, entropy_coef_min=0.0)
        update_adaptive_entropy_after_train_step(cfg, 9.0)
        assert get_ppo_feature_store().get(ENTROPY_COEF_KEY) == pytest.approx(0.0)

    def test_disabled_returns_static_bonus(self):
        cfg = _adaptive_cfg(use_adaptive_entropy=False, feature_store_enable=False)
        assert get_entropy_bonus_coef(cfg) == pytest.approx(0.01)
        assert update_adaptive_entropy_after_train_step(cfg, 3.0) == {}

    def test_ckpt_round_trip(self, tmp_path):
        cfg = _adaptive_cfg()
        update_adaptive_entropy_after_train_step(cfg, 2.5)
        store = get_ppo_feature_store()
        store.save_to_ckpt(str(tmp_path), ppo_step=7)

        reset_ppo_feature_store_for_test()
        store2 = get_ppo_feature_store()
        assert store2.load_from_ckpt(str(tmp_path), ppo_step=7)
        assert store2.get(LAST_WORLD_ENTROPY_KEY) == pytest.approx(2.5)
        assert store2.get(ENTROPY_COEF_KEY) == pytest.approx(0.015)
        assert get_entropy_bonus_coef(cfg) == pytest.approx(0.015)

    def test_config_requires_feature_store(self):
        with pytest.raises(AssertionError, match="feature_store_enable"):
            PpoConfig(use_adaptive_entropy=True, feature_store_enable=False)


# ---------------------------------------------------------------------------
# L2: train_step metric shape + no duplicate scaled_entropy
# ---------------------------------------------------------------------------


class TestAdaptiveEntropyL2Wiring:
    def test_world_entropy_is_token_weighted_over_mbs(self):
        # 模拟 loss 每 MB 报 [sum, count]；_update_policy 先 sum 再 /count
        mb_stats = [(10.0, 2.0), (30.0, 6.0)]  # means 5 and 5; weighted = 40/8 = 5
        h = _token_weighted_mean(mb_stats)
        assert h == pytest.approx(5.0)
        # 若错误地对 MB 均值再平均，这里也会是 5；换不均 token 数验证
        mb_uneven = [(2.0, 1.0), (20.0, 4.0)]  # weighted 22/5=4.4；naive mean of means=(2+5)/2=3.5
        h2 = _token_weighted_mean(mb_uneven)
        assert h2 == pytest.approx(4.4)
        naive = (2.0 / 1.0 + 20.0 / 4.0) / 2.0
        assert h2 != pytest.approx(naive)

    def test_after_train_step_extend_does_not_duplicate_scaled_entropy(self):
        cfg = _adaptive_cfg()
        metrics: dict = {}
        # 模拟 _update_policy 已写入本步 scaled_entropy
        _metric = {"policy/scaled_entropy": 3.0, "policy/loss": 0.1}
        extend_value_to_dict(metrics, _metric)
        assert metrics["policy/scaled_entropy"] == [3.0]

        extend_value_to_dict(
            metrics,
            update_adaptive_entropy_after_train_step(cfg, float(_metric["policy/scaled_entropy"])),
        )
        assert metrics["policy/scaled_entropy"] == [3.0]
        assert metrics["policy/entropy_coef"][0] == pytest.approx(0.015)
        assert metrics["policy/last_world_entropy"][0] == pytest.approx(3.0)

    def test_two_train_steps_scaled_entropy_list_len_equals_steps(self):
        cfg = _adaptive_cfg()
        metrics: dict = {}
        for h in (3.0, 6.0):
            _metric = {"policy/scaled_entropy": h}
            extend_value_to_dict(metrics, _metric)
            extend_value_to_dict(
                metrics,
                update_adaptive_entropy_after_train_step(cfg, float(h)),
            )
        assert metrics["policy/scaled_entropy"] == [3.0, 6.0]
        assert len(metrics["policy/entropy_coef"]) == 2
        assert len(metrics["policy/last_world_entropy"]) == 2


# ---------------------------------------------------------------------------
# L3: multi-step TRL-like behavior smoke (+ interval 不清 set keys)
# ---------------------------------------------------------------------------


class TestAdaptiveEntropyL3Smoke:
    def test_lag1_apply_coef_across_steps(self):
        cfg = _adaptive_cfg(entropy_target=5.0, ppo_entropy_bonus=0.01, entropy_coef_delta=0.005)
        # step0: last_H=inf → apply 0；结束后 H=3 → coef=0.015
        assert get_entropy_bonus_coef(cfg) == 0.0
        update_adaptive_entropy_after_train_step(cfg, 3.0)
        # step1: apply 0.015；H=6 → coef down, next apply 0
        assert get_entropy_bonus_coef(cfg) == pytest.approx(0.015)
        update_adaptive_entropy_after_train_step(cfg, 6.0)
        assert get_entropy_bonus_coef(cfg) == 0.0
        # step2: H=4 ≤ target → coef 0.01+0.005；下一步 apply 0.015
        update_adaptive_entropy_after_train_step(cfg, 4.0)
        assert get_entropy_bonus_coef(cfg) == pytest.approx(0.015)

    def test_ppo_step_interval_does_not_clear_controller_keys(self):
        from gpatch_v4.core.ppo_feature_store import ppo_step_interval

        cfg = _adaptive_cfg()
        update_adaptive_entropy_after_train_step(cfg, 2.0)
        with ppo_step_interval(ppo_step=1, enabled=True):
            pass
        store = get_ppo_feature_store()
        assert store.get(LAST_WORLD_ENTROPY_KEY) == pytest.approx(2.0)
        assert store.get(ENTROPY_COEF_KEY) == pytest.approx(0.015)
        assert get_entropy_bonus_coef(cfg) == pytest.approx(0.015)

    def test_high_target_monotonic_coef_growth(self):
        cfg = _adaptive_cfg(entropy_target=100.0, ppo_entropy_bonus=0.01, entropy_coef_delta=0.005)
        coefs = []
        for _ in range(5):
            m = update_adaptive_entropy_after_train_step(cfg, 1.0)
            coefs.append(m["policy/entropy_coef"])
        assert coefs == pytest.approx([0.015, 0.02, 0.025, 0.03, 0.035])
