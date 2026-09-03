"""Unit tests for PpoFeatureStore."""
import os

import pytest
import torch

from gpatch_v4.core.ppo_feature_store import (
    LEGACY_TRAIN_EXTRA_STATE_FILENAME,
    PPO_FEATURE_STORE_FILENAME,
    PPO_STEP_KEY,
    PpoFeatureStore,
    feature_history_key,
    feature_pending_key,
    feature_reduce_key,
    get_ppo_feature_store,
    iter_checkpoint_dir,
    ppo_step_interval,
    register_feature_reduce,
    reset_ppo_feature_store_for_test,
)

_FEAT = "feat"


class TestPpoFeatureStoreBasic:
    def test_get_set_append(self):
        store = PpoFeatureStore()
        assert store.get("missing", 3) == 3
        store.set("k", 1)
        assert store.get("k") == 1
        hist = feature_history_key(_FEAT)
        store.append(hist, 0.5)
        store.append(hist, 0.7)
        assert store.get(hist) == [0.5, 0.7]

    def test_begin_and_record(self):
        store = PpoFeatureStore()
        store.begin_feature_interval(_FEAT, 3)
        store.configure(_FEAT, reduce="mean")
        assert store.get(PPO_STEP_KEY) == 3
        store.record(_FEAT, 2.0, weight=4.0)
        store.record(_FEAT, 1.0, weight=1.0)
        assert store.get(feature_pending_key(_FEAT)) == [(2.0, 4.0), (1.0, 1.0)]

    def test_baseline_from_history_only(self):
        store = PpoFeatureStore()
        hist = feature_history_key(_FEAT)
        assert store.get_history_mean(_FEAT) is None
        store.append(hist, 1.0)
        store.append(hist, 3.0)
        assert store.get_history_mean(_FEAT) == 2.0

    def test_finalize_without_dist(self):
        store = PpoFeatureStore()
        store.begin_feature_interval(_FEAT, 0)
        store.configure(_FEAT, reduce="mean")
        store.record(_FEAT, 4.0, weight=2.0)
        store.record(_FEAT, 6.0, weight=2.0)
        value = store.finalize_feature_interval(_FEAT)
        assert value == pytest.approx(2.5)
        assert store.get(feature_history_key(_FEAT)) == [2.5]
        assert store.get(feature_pending_key(_FEAT)) == []

    def test_train_step_axis_assert(self):
        store = PpoFeatureStore()
        with pytest.raises(AssertionError, match="train_step"):
            store.begin_feature_interval("x", 0, axis="train_step")
        with pytest.raises(AssertionError, match="begin_train_step"):
            store.begin_train_step(0)


class TestPerKeyReduce:
    def test_mean_sum_max_min(self):
        store = PpoFeatureStore()
        store.configure("a", reduce="mean")
        store.configure("b", reduce="sum")
        store.configure("c", reduce="max")
        store.configure("d", reduce="min")
        store.record("a", 4.0, weight=2.0)
        store.record("a", 6.0, weight=2.0)
        store.record("b", 1.0)
        store.record("b", 3.0)
        store.record("c", 1.0)
        store.record("c", 9.0)
        store.record("d", 2.0)
        store.record("d", -1.0)
        values = store.finalize_all_pending_features()
        assert values["a"] == pytest.approx(2.5)
        assert values["b"] == pytest.approx(4.0)
        assert values["c"] == pytest.approx(9.0)
        assert values["d"] == pytest.approx(-1.0)

    def test_configure_conflict(self):
        store = PpoFeatureStore()
        store.configure("x", reduce="mean")
        with pytest.raises(AssertionError, match="already set"):
            store.configure("x", reduce="max")

    def test_register_custom_reduce(self):
        @register_feature_reduce("twice_sum", override=True)
        def twice_sum(local_values, local_weights, group):
            del local_weights, group
            return 2.0 * float(sum(local_values))

        store = PpoFeatureStore()
        store.configure("z", reduce="twice_sum")
        store.record("z", 1.5)
        store.record("z", 2.5)
        assert store.finalize_feature_interval("z") == pytest.approx(8.0)
        assert store.get(feature_history_key("z")) == [8.0]


class TestPpoFeatureStoreCkpt:
    def test_round_trip(self, tmp_path):
        store = PpoFeatureStore()
        store.begin_feature_interval(_FEAT, 3)
        store.append(feature_history_key(_FEAT), 1.0)
        store.append(feature_history_key(_FEAT), 2.0)
        store.configure(_FEAT, reduce="mean")
        store.record(_FEAT, 9.0, weight=1.0)
        iter_dir = str(tmp_path / "iter_0000003")
        store.save_to_ckpt(iter_dir, 3)
        assert os.path.isfile(os.path.join(iter_dir, PPO_FEATURE_STORE_FILENAME))

        loaded = PpoFeatureStore()
        ok = loaded.load_from_ckpt(iter_dir, 3, strict_step=True)
        assert ok
        assert loaded.get(feature_history_key(_FEAT)) == [1.0, 2.0]
        assert loaded.get(feature_reduce_key(_FEAT)) == "mean"
        assert loaded.get(feature_pending_key(_FEAT), []) == []

    def test_round_trip_mean_min_max_reduces(self, tmp_path):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        with ppo_step_interval(ppo_step=1, enabled=True) as iv:
            store.record("a", 4.0, weight=2.0, reduce="mean")
            store.record("a", 6.0, weight=2.0)
            store.record("b", 1.0, reduce="min")
            store.record("b", -2.0)
            store.record("c", 3.0, reduce="max")
            store.record("c", 9.0)
        assert iv.final_values["a"] == pytest.approx(2.5)
        assert iv.final_values["b"] == pytest.approx(-2.0)
        assert iv.final_values["c"] == pytest.approx(9.0)

        iter_dir = str(tmp_path / "iter_0000001")
        store.save_to_ckpt(iter_dir, 1)

        reset_ppo_feature_store_for_test()
        loaded = get_ppo_feature_store()
        assert loaded.load_from_ckpt(iter_dir, 1)
        assert loaded.get(feature_reduce_key("a")) == "mean"
        assert loaded.get(feature_reduce_key("b")) == "min"
        assert loaded.get(feature_reduce_key("c")) == "max"
        assert loaded.get(feature_history_key("a")) == [2.5]
        assert loaded.get(feature_history_key("b")) == [-2.0]
        assert loaded.get(feature_history_key("c")) == [9.0]
        assert loaded.get(feature_pending_key("a"), []) == []
        assert loaded.get(feature_pending_key("b"), []) == []
        assert loaded.get(feature_pending_key("c"), []) == []
        reset_ppo_feature_store_for_test()

    def test_load_legacy_filename(self, tmp_path):
        store = PpoFeatureStore()
        store.append(feature_history_key(_FEAT), 0.5)
        iter_dir = str(tmp_path / "iter_0000001")
        os.makedirs(iter_dir, exist_ok=True)
        legacy_path = os.path.join(iter_dir, LEGACY_TRAIN_EXTRA_STATE_FILENAME)
        torch.save(store.state_dict(), legacy_path)

        loaded = PpoFeatureStore()
        ok = loaded.load_from_ckpt(iter_dir, 1, strict_step=False)
        assert ok
        assert loaded.get(feature_history_key(_FEAT)) == [0.5]

    def test_missing_file(self, tmp_path):
        loaded = PpoFeatureStore()
        ok = loaded.load_from_ckpt(str(tmp_path / "iter_0000001"), 1)
        assert not ok

    def test_iter_checkpoint_dir(self):
        assert iter_checkpoint_dir("/ckpt", 12) == os.path.join("/ckpt", "iter_0000012")


class TestPpoFeatureStoreSingleton:
    def test_get_ppo_feature_store(self):
        reset_ppo_feature_store_for_test()
        a = get_ppo_feature_store()
        b = get_ppo_feature_store()
        assert a is b
        reset_ppo_feature_store_for_test()

    def test_disabled_refuses_lazy_init(self):
        from gpatch_v4.core.ppo_feature_store import set_ppo_feature_store_enabled

        set_ppo_feature_store_enabled(False)
        with pytest.raises(RuntimeError, match="disabled"):
            get_ppo_feature_store()
        reset_ppo_feature_store_for_test()
        assert get_ppo_feature_store() is not None
        reset_ppo_feature_store_for_test()


class TestCustomFeature:
    def test_generic_feature_interval(self):
        feature = "my_adv"
        store = PpoFeatureStore()
        store.begin_feature_interval(feature, 2, axis="ppo_step")
        assert store.get(PPO_STEP_KEY) == 2
        store.record_weighted_sum(feature, 3.0, 2.0)
        value = store.finalize_feature_interval(feature)
        assert value == pytest.approx(1.5)
        assert store.get(feature_history_key(feature)) == [1.5]
        assert store.get(feature_pending_key(feature)) == []

    def test_custom_keys_persist_in_ckpt(self, tmp_path):
        store = PpoFeatureStore()
        store.set("my_adv.group_mean", 0.42)
        store.begin_feature_interval("my_adv", 1, axis="ppo_step")
        store.append(feature_history_key("my_adv"), 0.1)
        store.record_weighted_sum("my_adv", 1.0, 1.0)
        iter_dir = str(tmp_path / "iter_0000001")
        store.save_to_ckpt(iter_dir, 1)

        loaded = PpoFeatureStore()
        loaded.load_from_ckpt(iter_dir, 1)
        assert loaded.get("my_adv.group_mean") == 0.42
        assert loaded.get(feature_history_key("my_adv")) == [0.1]
        assert loaded.get(feature_pending_key("my_adv"), []) == []


class TestPpoStepIntervalContext:
    def test_multi_feature_flexible_insert(self):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        with ppo_step_interval(ppo_step=3, enabled=True) as iv:
            assert store.get(PPO_STEP_KEY) == 3
            store.record(_FEAT, 4.0, weight=2.0, reduce="mean")
            store.record_weighted_sum("my_adv", 6.0, 3.0)
            store.set("my_adv.group_mean", 0.5)
        assert iv.final_values[_FEAT] == pytest.approx(2.0)
        assert iv.final_values["my_adv"] == pytest.approx(2.0)
        assert store.get(feature_history_key(_FEAT)) == [2.0]
        assert store.get(feature_history_key("my_adv")) == [2.0]
        assert store.get("my_adv.group_mean") == 0.5
        reset_ppo_feature_store_for_test()

    def test_feature_interval_closed_loop(self):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        with ppo_step_interval(ppo_step=0, enabled=True) as iv0:
            assert store.get_history_mean(_FEAT) is None
            store.record(_FEAT, 4.0, weight=2.0, reduce="mean")
            store.record(_FEAT, 6.0, weight=2.0)
        assert iv0.final_values[_FEAT] == pytest.approx(2.5)
        assert store.get(feature_history_key(_FEAT)) == [2.5]
        assert store.get(feature_pending_key(_FEAT), []) == []

        with ppo_step_interval(ppo_step=1, enabled=True) as iv1:
            assert store.get_history_mean(_FEAT) == pytest.approx(2.5)
            store.record(_FEAT, 1.0, weight=1.0, reduce="mean")
        assert iv1.final_values[_FEAT] == pytest.approx(1.0)
        assert store.get(feature_history_key(_FEAT)) == [2.5, 1.0]
        assert store.get_history_mean(_FEAT) == pytest.approx(1.75)
        reset_ppo_feature_store_for_test()

    def test_ckpt_resume_restores_baseline(self, tmp_path):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        with ppo_step_interval(ppo_step=2, enabled=True):
            store.record(_FEAT, 2.0, weight=1.0, reduce="mean")
            store.record(_FEAT, 4.0, weight=1.0)
        iter_dir = str(tmp_path / "iter_0000002")
        store.save_to_ckpt(iter_dir, 2)
        assert store.get_history_mean(_FEAT) == pytest.approx(3.0)

        reset_ppo_feature_store_for_test()
        loaded = get_ppo_feature_store()
        assert loaded.get_history_mean(_FEAT) is None
        ok = loaded.load_from_ckpt(iter_dir, 2)
        assert ok
        assert loaded.get(feature_history_key(_FEAT)) == [3.0]
        assert loaded.get_history_mean(_FEAT) == pytest.approx(3.0)
        assert loaded.get(feature_pending_key(_FEAT), []) == []
        reset_ppo_feature_store_for_test()

    def test_disabled_is_noop(self):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        with ppo_step_interval(ppo_step=1, enabled=False) as iv:
            store.record(_FEAT, 1.0, weight=1.0, reduce="mean")
        assert iv.final_values == {}
        assert store.get(feature_history_key(_FEAT), []) == []
        reset_ppo_feature_store_for_test()

    def test_skip_finalize_on_exception(self):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        with pytest.raises(RuntimeError):
            with ppo_step_interval(ppo_step=0, enabled=True) as iv:
                store.record(_FEAT, 1.0, weight=1.0, reduce="mean")
                raise RuntimeError("boom")
        assert iv.final_values == {}
        assert store.get(feature_history_key(_FEAT), []) == []
        reset_ppo_feature_store_for_test()


class TestPpoFeatureStoreStepLocal:
    def test_set_get_clone_detach(self):
        store = PpoFeatureStore()
        src = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        store.set_step_local("w", src)
        got = store.get_step_local("w")
        assert torch.equal(got, torch.tensor([1.0, 2.0, 3.0]))
        assert got is not src
        assert not got.requires_grad
        src.detach().add_(1)
        assert got.tolist() == [1.0, 2.0, 3.0]

    def test_double_set_asserts(self):
        store = PpoFeatureStore()
        store.set_step_local("w", torch.tensor([1.0]))
        with pytest.raises(AssertionError, match="already set"):
            store.set_step_local("w", torch.tensor([2.0]))

    def test_get_missing_raises(self):
        store = PpoFeatureStore()
        with pytest.raises(KeyError, match="was not set"):
            store.get_step_local("missing")

    def test_non_tensor_raises(self):
        store = PpoFeatureStore()
        with pytest.raises(TypeError, match="torch.Tensor"):
            store.set_step_local("w", [1.0, 2.0])

    def test_isolated_from_record(self):
        store = PpoFeatureStore()
        store.record("x", 1.0, reduce="mean")
        store.set_step_local("x", torch.tensor([9.0, 8.0]))
        assert store.get_step_local("x").tolist() == [9.0, 8.0]
        assert store.get(feature_pending_key("x")) == [(1.0, 1.0)]

    def test_not_in_state_dict_or_ckpt(self, tmp_path):
        store = PpoFeatureStore()
        store.set_step_local("w", torch.tensor([1.0, 2.0]))
        store.append(feature_history_key(_FEAT), 1.0)
        state = store.state_dict()
        assert "w" not in state["data"]
        assert "_step_local" not in state
        assert store.has_persisted_features()

        store_only_step_local = PpoFeatureStore()
        store_only_step_local.set_step_local("w", torch.tensor([1.0]))
        assert not store_only_step_local.has_persisted_features()

        iter_dir = str(tmp_path / "iter_0000001")
        store.save_to_ckpt(iter_dir, 1)
        loaded = PpoFeatureStore()
        assert loaded.load_from_ckpt(iter_dir, 1)
        with pytest.raises(KeyError):
            loaded.get_step_local("w")
        assert loaded.get(feature_history_key(_FEAT)) == [1.0]

    def test_interval_clears_on_exit(self):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        with ppo_step_interval(ppo_step=0, enabled=True):
            store.set_step_local("w", torch.tensor([1.0, 2.0]))
            assert store.get_step_local("w").tolist() == [1.0, 2.0]
        with pytest.raises(KeyError):
            store.get_step_local("w")
        reset_ppo_feature_store_for_test()

    def test_interval_clears_on_exception(self):
        reset_ppo_feature_store_for_test()
        store = get_ppo_feature_store()
        with pytest.raises(RuntimeError):
            with ppo_step_interval(ppo_step=0, enabled=True):
                store.set_step_local("w", torch.tensor([1.0]))
                raise RuntimeError("boom")
        with pytest.raises(KeyError):
            store.get_step_local("w")
        reset_ppo_feature_store_for_test()

    def test_begin_clears_and_allows_reset(self):
        store = PpoFeatureStore()
        store.set_step_local("w", torch.tensor([1.0]))
        store.begin_ppo_step_interval(1)
        with pytest.raises(KeyError):
            store.get_step_local("w")
        store.set_step_local("w", torch.tensor([2.0, 3.0]))
        assert store.get_step_local("w").tolist() == [2.0, 3.0]

    def test_full_clear(self):
        store = PpoFeatureStore()
        store.set_step_local("w", torch.tensor([1.0]))
        store.clear()
        with pytest.raises(KeyError):
            store.get_step_local("w")

