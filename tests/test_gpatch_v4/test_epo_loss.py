"""Tests for EPO-lite custom GRPO loss."""
import types
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.core.ppo_feature_store import (
    feature_axis_total_key,
    feature_history_key,
    feature_pending_key,
    feature_reduce_key,
    get_ppo_feature_store,
    ppo_step_interval,
    reset_ppo_feature_store_for_test,
)
from tasks.math_rl_v4.epo_grpo_loss import (
    FEATURE_NAME,
    calculate_epo_phase_weight,
    epo_grpo_loss_func,
    generate_epo_entropy_mask,
)
from gpatch_v4.training_backend.loss_factory import PolicyLossInput, grpo_loss_func

_REDUCE_METRICS_PATH = (
    "gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group"
)
_EPO_REDUCE_METRICS_PATH = (
    "tasks.math_rl_v4.epo_grpo_loss.reduce_metrics_across_data_parallel_group"
)


def _make_config(
    epo_entropy_smooth_coeff: float = 1.0,
    epo_mask_mode: str = "token",
    epo_min_ratio: float = 0.8,
    epo_max_ratio: float = 1.2,
    epo_out_range_penalty: float = 0.1,
    ppo_entropy_bonus: float = 0.01,
    epo_enable_smooth_weights: bool = False,
    total_ppo_step: int | None = None,
):
    ppo = PpoConfig(
        ppo_entropy_bonus=ppo_entropy_bonus,
        feature_store_enable=True,
    )
    task = types.SimpleNamespace(
        epo_entropy_smooth_coeff=epo_entropy_smooth_coeff,
        epo_mask_mode=epo_mask_mode,
        epo_min_ratio=epo_min_ratio,
        epo_max_ratio=epo_max_ratio,
        epo_out_range_penalty=epo_out_range_penalty,
        epo_enable_smooth_weights=epo_enable_smooth_weights,
    )
    policy = types.SimpleNamespace(override_transformer_config={})
    cfg = types.SimpleNamespace(ppo=ppo, debug=DebugConfig(), policy=policy, task=task)
    if total_ppo_step is not None:
        cfg.training = types.SimpleNamespace(total_ppo_step=total_ppo_step)
    return cfg


def _make_loss_input(
    batch_size: int = 2,
    seq_len: int = 4,
    per_token_entropy: torch.Tensor | None = None,
):
    torch.manual_seed(0)
    curr = torch.randn(batch_size, seq_len, requires_grad=True)
    prev = curr.detach().clone()
    advantages = torch.ones(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)
    if per_token_entropy is None:
        per_token_entropy = torch.ones(batch_size, seq_len)
    return PolicyLossInput(
        advantages=advantages,
        prev_log_probs=prev,
        ref_log_probs=None,
        curr_log_probs=curr,
        response_mask=mask,
        scaled_entropy=torch.tensor(0.5),
        per_token_entropy=per_token_entropy,
    )


def _seed_history(store, values: list[float], ppo_step: int = 1) -> None:
    store.begin_ppo_step_interval(ppo_step)
    store.configure(FEATURE_NAME, reduce="mean")
    store.set(feature_history_key(FEATURE_NAME), list(values))


class TestGenerateEpoEntropyMask:
    def test_token_mode_in_band_zero(self):
        entropy = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        mask = torch.ones_like(entropy)
        out, ratio = generate_epo_entropy_mask(
            entropy, mask, baseline_h=1.0, mask_mode="token", min_ratio=0.8, max_ratio=1.2
        )
        assert torch.all(out == 0)
        assert ratio == 1.0

    def test_token_mode_out_of_band_penalty(self):
        entropy = torch.tensor([[2.0, 0.1], [1.0, 1.0]])
        mask = torch.ones_like(entropy)
        out, ratio = generate_epo_entropy_mask(
            entropy,
            mask,
            baseline_h=1.0,
            mask_mode="token",
            min_ratio=0.8,
            max_ratio=1.2,
            out_range_penalty=0.1,
        )
        assert out[0, 0].item() == pytest.approx(0.1)
        assert out[0, 1].item() == pytest.approx(0.1)
        assert out[1, 0].item() == pytest.approx(0.0)
        assert ratio == pytest.approx(0.5)

    def test_seq_mode(self):
        entropy = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
        mask = torch.ones_like(entropy)
        out, ratio = generate_epo_entropy_mask(
            entropy,
            mask,
            baseline_h=1.0,
            mask_mode="seq",
            min_ratio=0.8,
            max_ratio=1.2,
            out_range_penalty=0.1,
        )
        assert torch.allclose(out[0], torch.full_like(out[0], 0.1))
        assert torch.allclose(out[1], torch.zeros_like(out[1]))
        assert ratio == pytest.approx(0.5)


class TestEpoPhaseWeight:
    def test_decreases(self):
        assert calculate_epo_phase_weight(0, 100) == 1.0
        assert calculate_epo_phase_weight(100, 100) < calculate_epo_phase_weight(50, 100)


class TestEpoGrpoLoss:
    @patch(_EPO_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    @patch(_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    def test_records_pending_and_uses_history(self, _mock_a, _mock_b):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        _seed_history(store, [1.0], ppo_step=1)

        config = _make_config(ppo_entropy_bonus=1.0, epo_out_range_penalty=0.5)
        entropy = torch.full((2, 4), 10.0)
        li = _make_loss_input(per_token_entropy=entropy)
        loss, metrics = epo_grpo_loss_func(config, li)
        assert not torch.isnan(loss)
        assert "epo_baseline_H" in metrics
        pending = store.get(feature_pending_key(FEATURE_NAME))
        assert len(pending) == 1
        assert pending[0][1] == 8.0
        reset_ppo_feature_store_for_test()

    @patch(_EPO_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    @patch(_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    def test_empty_history_no_penalty_still_records(self, _mock_a, _mock_b):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        store.begin_ppo_step_interval(0)

        config = _make_config(ppo_entropy_bonus=1.0)
        li = _make_loss_input()
        loss, metrics = epo_grpo_loss_func(config, li)
        assert not torch.isnan(loss)
        assert "epo_baseline_H" not in metrics
        assert len(store.get(feature_pending_key(FEATURE_NAME))) == 1
        reset_ppo_feature_store_for_test()

    @patch(_EPO_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    @patch(_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    def test_out_of_band_increases_loss_vs_plain_grpo(self, _mock_a, _mock_b):
        reset_ppo_feature_store_for_test()
        entropy = torch.full((2, 4), 10.0)

        store = get_ppo_feature_store()
        _seed_history(store, [1.0], ppo_step=1)
        li_on = _make_loss_input(per_token_entropy=entropy.clone())
        loss_on, metrics_on = epo_grpo_loss_func(
            _make_config(
                ppo_entropy_bonus=1.0,
                epo_entropy_smooth_coeff=1.0,
                epo_out_range_penalty=0.5,
            ),
            li_on,
        )

        reset_ppo_feature_store_for_test()
        li_off = _make_loss_input(per_token_entropy=entropy.clone())
        loss_off, metrics_off = grpo_loss_func(
            _make_config(ppo_entropy_bonus=1.0),
            li_off,
        )

        assert "epo_baseline_H" in metrics_on
        assert "epo_baseline_H" not in metrics_off
        assert loss_on.item() > loss_off.item()
        reset_ppo_feature_store_for_test()

    @patch(_EPO_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    @patch(_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    def test_baseline_ignores_current_pending(self, _mock_a, _mock_b):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        _seed_history(store, [1.0], ppo_step=1)
        store.record(FEATURE_NAME, 100.0, weight=1.0, reduce="mean")
        assert store.get_history_mean(FEATURE_NAME) == pytest.approx(1.0)

        config = _make_config(ppo_entropy_bonus=1.0)
        li = _make_loss_input(per_token_entropy=torch.full((2, 4), 10.0))
        _, metrics = epo_grpo_loss_func(config, li)
        assert "epo_baseline_H" in metrics
        assert store.get_history_mean(FEATURE_NAME) == pytest.approx(1.0)
        assert len(store.get(feature_pending_key(FEATURE_NAME))) == 2
        reset_ppo_feature_store_for_test()

    @patch(_EPO_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    @patch(_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    def test_interval_finalize_and_ckpt_resume(self, _mock_a, _mock_b, tmp_path):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        config = _make_config(ppo_entropy_bonus=1.0)
        li0 = _make_loss_input(per_token_entropy=torch.full((2, 4), 2.0))
        li1 = _make_loss_input(per_token_entropy=torch.full((2, 4), 4.0))

        with ppo_step_interval(ppo_step=0, enabled=True) as iv0:
            epo_grpo_loss_func(config, li0)
        assert iv0.final_values[FEATURE_NAME] == pytest.approx(2.0)
        assert store.get(feature_reduce_key(FEATURE_NAME)) == "mean"
        assert store.get(feature_history_key(FEATURE_NAME)) == [2.0]

        with ppo_step_interval(ppo_step=1, enabled=True) as iv1:
            assert store.get_history_mean(FEATURE_NAME) == pytest.approx(2.0)
            loss, metrics = epo_grpo_loss_func(config, li1)
            assert "epo_baseline_H" in metrics
            baseline = metrics["epo_baseline_H"]
            assert baseline[0] / baseline[1] == pytest.approx(2.0)
        assert iv1.final_values[FEATURE_NAME] == pytest.approx(4.0)
        assert store.get(feature_history_key(FEATURE_NAME)) == [2.0, 4.0]

        iter_dir = str(tmp_path / "iter_0000001")
        store.save_to_ckpt(iter_dir, 1)

        reset_ppo_feature_store_for_test()
        loaded = get_ppo_feature_store()
        assert loaded.load_from_ckpt(iter_dir, 1)
        assert loaded.get(feature_history_key(FEATURE_NAME)) == [2.0, 4.0]
        assert loaded.get_history_mean(FEATURE_NAME) == pytest.approx(3.0)
        assert loaded.get(feature_reduce_key(FEATURE_NAME)) == "mean"
        assert loaded.get(feature_pending_key(FEATURE_NAME), []) == []
        reset_ppo_feature_store_for_test()

    @patch(_EPO_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    @patch(_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    def test_smooth_weights_off_phase_weight_is_one(self, _mock_a, _mock_b):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        _seed_history(store, [1.0], ppo_step=50)
        store.set(feature_axis_total_key(FEATURE_NAME), 100)

        config = _make_config(
            ppo_entropy_bonus=1.0,
            epo_out_range_penalty=0.5,
            epo_enable_smooth_weights=False,
            total_ppo_step=100,
        )
        li = _make_loss_input(per_token_entropy=torch.full((2, 4), 10.0))
        _, metrics = epo_grpo_loss_func(config, li)
        assert "epo_phase_weight" in metrics
        pw = metrics["epo_phase_weight"]
        assert pw[0] / pw[1] == pytest.approx(1.0)
        reset_ppo_feature_store_for_test()

    @patch(_EPO_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    @patch(_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    def test_smooth_weights_scales_phase_weight_and_loss(self, _mock_a, _mock_b):
        reset_ppo_feature_store_for_test()
        entropy = torch.full((2, 4), 10.0)
        cur_step, total = 50, 100
        expect_pw = calculate_epo_phase_weight(cur_step, total)
        assert expect_pw < 1.0

        store = get_ppo_feature_store()
        _seed_history(store, [1.0], ppo_step=cur_step)
        store.set(feature_axis_total_key(FEATURE_NAME), total)
        loss_on, metrics_on = epo_grpo_loss_func(
            _make_config(
                ppo_entropy_bonus=1.0,
                epo_entropy_smooth_coeff=1.0,
                epo_out_range_penalty=0.5,
                epo_enable_smooth_weights=True,
                total_ppo_step=total,
            ),
            _make_loss_input(per_token_entropy=entropy.clone()),
        )
        pw = metrics_on["epo_phase_weight"]
        assert pw[0] / pw[1] == pytest.approx(expect_pw)

        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        _seed_history(store, [1.0], ppo_step=cur_step)
        store.set(feature_axis_total_key(FEATURE_NAME), total)
        loss_off, metrics_off = epo_grpo_loss_func(
            _make_config(
                ppo_entropy_bonus=1.0,
                epo_entropy_smooth_coeff=1.0,
                epo_out_range_penalty=0.5,
                epo_enable_smooth_weights=False,
                total_ppo_step=total,
            ),
            _make_loss_input(per_token_entropy=entropy.clone()),
        )
        pw_off = metrics_off["epo_phase_weight"]
        assert pw_off[0] / pw_off[1] == pytest.approx(1.0)
        assert loss_on.item() < loss_off.item()
        reset_ppo_feature_store_for_test()
