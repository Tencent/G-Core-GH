"""Process-global PPO feature store (pending → reduce → history) + ckpt sidecar."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.distributed as dist

from gpatch_v4.core.ppo_feature_store.keys import (
    DEFAULT_REDUCE,
    HISTORY_AXIS_SUFFIX,
    LEGACY_TRAIN_EXTRA_STATE_FILENAME,
    PPO_FEATURE_STORE_FILENAME,
    PPO_STEP_KEY,
    REDUCE_SUFFIX,
    STATE_VERSION,
    TRAIN_STEP_KEY,
    assert_ppo_step_axis_only,
    feature_from_pending_key,
    feature_history_axis_key,
    feature_history_key,
    feature_pending_key,
    feature_reduce_key,
    is_pending_key,
)
from gpatch_v4.core.ppo_feature_store.reduce import default_dp_group, get_feature_reduce

_instance: Optional["PpoFeatureStore"] = None
# False: refuse lazy init. True: allow get() to create singleton.
# Tests call ``set_ppo_feature_store_enabled(True)`` via reset helper.
_enabled: bool = False


def set_ppo_feature_store_enabled(enabled: bool) -> None:
    """Enable/disable the process-local store; disable clears the singleton."""
    global _instance, _enabled
    _enabled = bool(enabled)
    if not _enabled:
        _instance = None


def is_ppo_feature_store_enabled() -> bool:
    return _enabled


def get_ppo_feature_store() -> "PpoFeatureStore":
    global _instance
    if not _enabled:
        raise RuntimeError(
            "PpoFeatureStore is disabled (ppo.feature_store_enable=False). "
            "Enable it before calling get_ppo_feature_store()."
        )
    if _instance is None:
        _instance = PpoFeatureStore()
    return _instance


def reset_ppo_feature_store_for_test() -> None:
    """Clear singleton and enable store for unit tests."""
    global _instance
    _instance = None
    set_ppo_feature_store_enabled(True)


class PpoFeatureStore:
    """Process-local global KV for RL side-car scalars and step-local tensors.

    Interval model
    --------------
    One outermost ``ppo_step`` window (see ``PpoStepInterval``). Callers
    ``configure`` / ``record`` / ``get_history_mean`` any feature name inside
    the window; on exit all non-empty ``*.pending`` features are reduced then
    broadcast once.

    ``set_step_local`` / ``get_step_local`` hold a per-step tensor that is not
    reduced, not persisted, and cleared on interval enter/exit.
    """
    def __init__(self) -> None:
        self._data: dict[str, Any] = {
            PPO_STEP_KEY: None,
            TRAIN_STEP_KEY: None,
        }
        self._step_local: dict[str, torch.Tensor] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def append(self, key: str, item: Any) -> None:
        cur = self._data.get(key)
        if cur is None:
            self._data[key] = [item]
            return
        if not isinstance(cur, list):
            raise TypeError(
                f"PpoFeatureStore.append requires list value for key={key!r}, got {type(cur)}"
            )
        cur.append(item)

    def has(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def has_persisted_features(self) -> bool:
        for key in self._data:
            if key in (PPO_STEP_KEY, TRAIN_STEP_KEY):
                continue
            if is_pending_key(key):
                continue
            return True
        return False

    def clear(self, key: Optional[str] = None) -> None:
        if key is None:
            self._data = {
                PPO_STEP_KEY: None,
                TRAIN_STEP_KEY: None,
            }
            self.clear_step_local()
            return
        if is_pending_key(key):
            self._data[key] = []
        else:
            self._data.pop(key, None)

    def set_step_local(self, name: str, value: torch.Tensor) -> None:
        """Store a step-local tensor; not persisted and not DP-reduced.

        Parameters
        ----------
        name : str
            One write per ppo_step; a second write raises ``AssertionError``.
        value : torch.Tensor
            Stored as ``detach().clone()`` on the same device. Store does no
            DP/PP comm; the caller MUST already have reduced across DP if
            every rank will read the same value.
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"set_step_local requires torch.Tensor, got {type(value)} for name={name!r}"
            )
        if name in self._step_local:
            raise AssertionError(f"step-local feature {name!r} already set in this ppo_step")
        self._step_local[name] = value.detach().clone()

    def get_step_local(self, name: str) -> torch.Tensor:
        """Return the step-local tensor written by ``set_step_local``.

        Parameters
        ----------
        name : str

        Returns
        -------
        torch.Tensor
            Clone from ``set_step_local``; same device as the original write.

        Raises
        ------
        KeyError
            If ``name`` was not set in this ppo_step.
        """
        if name not in self._step_local:
            raise KeyError(f"step-local feature {name!r} was not set in this ppo_step")
        return self._step_local[name]

    def clear_step_local(self) -> None:
        self._step_local.clear()

    def configure(self, feature: str, *, reduce: str = DEFAULT_REDUCE) -> None:
        get_feature_reduce(reduce)  # raise if unknown
        existing = self.get(feature_reduce_key(feature))
        if existing is not None and str(existing) != reduce:
            raise AssertionError(
                f"feature {feature!r} reduce already set to {existing!r}, "
                f"cannot reconfigure as {reduce!r}"
            )
        self.set(feature_reduce_key(feature), reduce)
        self.set(feature_history_axis_key(feature), "ppo_step")

    def begin_ppo_step_interval(self, ppo_step: int) -> None:
        self._data[PPO_STEP_KEY] = int(ppo_step)
        self._clear_all_pending()
        self.clear_step_local()

    def begin_feature_interval(self, feature: str, step_id: int, *, axis: str = "ppo_step") -> None:
        assert_ppo_step_axis_only(axis)
        self.set(feature_history_axis_key(feature), "ppo_step")
        self._data[PPO_STEP_KEY] = int(step_id)
        self._data[feature_pending_key(feature)] = []

    def record(
        self,
        feature: str,
        value: float,
        *,
        weight: float = 1.0,
        reduce: Optional[str] = None,
    ) -> None:
        """Append one local contribution for ``feature``.

        For ``reduce='mean'``, ``value`` is a sum contribution and ``weight`` is
        the count (same as historical ``record_weighted_sum``).
        For ``sum`` / ``max`` / ``min``, ``weight`` is ignored.
        """
        if reduce is not None:
            self.configure(feature, reduce=reduce)
        elif self.get(feature_reduce_key(feature)) is None:
            self.configure(feature, reduce=DEFAULT_REDUCE)

        reduce_name = str(self.get(feature_reduce_key(feature)))
        if reduce_name == "mean":
            if weight <= 0:
                return
            self.append(feature_pending_key(feature), (float(value), float(weight)))
            return
        self.append(feature_pending_key(feature), (float(value), 1.0))

    def record_weighted_sum(self, feature: str, value_sum: float, count: float) -> None:
        self.record(feature, value_sum, weight=count, reduce="mean")

    def get_history_mean(self, feature: str) -> Optional[float]:
        hist = self.get(feature_history_key(feature), [])
        if not hist:
            return None
        return float(sum(hist) / len(hist))

    def get_feature_axis_step(self, feature: str) -> Optional[int]:
        axis = self.get(feature_history_axis_key(feature), "ppo_step")
        assert_ppo_step_axis_only(str(axis))
        step = self.get(PPO_STEP_KEY)
        return None if step is None else int(step)

    def list_pending_features(self) -> list[str]:
        names = []
        for key, pending in self._data.items():
            if not is_pending_key(key):
                continue
            if pending:
                names.append(feature_from_pending_key(key))
        return sorted(names)

    def finalize_feature_interval(self, feature: str, group=None) -> Optional[float]:
        axis = self.get(feature_history_axis_key(feature), "ppo_step")
        assert_ppo_step_axis_only(str(axis))
        pending_key = feature_pending_key(feature)
        history_key = feature_history_key(feature)
        pending = self.get(pending_key, [])
        reduce_name = str(self.get(feature_reduce_key(feature), DEFAULT_REDUCE))
        reduce_fn = get_feature_reduce(reduce_name)

        local_values = [float(part[0]) for part in pending]
        local_weights = [float(part[1]) for part in pending]
        if group is None:
            group = default_dp_group()
        value = reduce_fn(
            local_values,
            local_weights if reduce_name == "mean" else None,
            group,
        )
        self._data[pending_key] = []
        if value is None:
            return None
        self.append(history_key, float(value))
        return float(value)

    def finalize_all_pending_features(self, group=None) -> dict[str, Optional[float]]:
        features = self._collect_pending_features_union(group=group)
        values: dict[str, Optional[float]] = {}
        for feature in features:
            values[feature] = self.finalize_feature_interval(feature, group=group)
        return values

    def _collect_pending_features_union(self, group=None) -> list[str]:
        local = self.list_pending_features()
        if not (dist.is_available() and dist.is_initialized()):
            return local
        if group is None:
            group = default_dp_group()
        if group is None or dist.get_world_size(group=group) <= 1:
            return local
        gathered = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(gathered, local, group=group)
        names: set[str] = set()
        for item in gathered:
            if item:
                names.update(item)
        return sorted(names)

    def begin_train_step(self, train_step: int) -> None:
        raise AssertionError(
            "begin_train_step is removed; use PpoStepInterval / begin_ppo_step_interval"
        )

    def _persisted_data(self) -> dict[str, Any]:
        return {key: value for key, value in self._data.items() if not is_pending_key(key)}

    def _clear_all_pending(self) -> None:
        for key in list(self._data.keys()):
            if is_pending_key(key):
                self._data[key] = []

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "ppo_step": self._data.get(PPO_STEP_KEY),
            "data": self._persisted_data(),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        expected_ppo_step: Optional[int] = None,
        strict_step: bool = True,
    ) -> None:
        if "data" not in state:
            raise KeyError("PpoFeatureStore.load_state_dict requires 'data' key")
        version = int(state.get("version", 1))
        if version not in (1, 2, 3, 4, STATE_VERSION):
            raise ValueError(
                f"Unsupported PpoFeatureStore version={version}, "
                f"expected 1/2/3/4/{STATE_VERSION}"
            )
        loaded_step = state.get("ppo_step")
        if (
            strict_step and expected_ppo_step is not None and loaded_step is not None and
            int(loaded_step) != int(expected_ppo_step)
        ):
            raise ValueError(
                f"PpoFeatureStore ppo_step mismatch: file={loaded_step}, "
                f"expected={expected_ppo_step}"
            )
        data = dict(state["data"])
        for key, value in data.items():
            if key.endswith(HISTORY_AXIS_SUFFIX) and value == "train_step":
                raise AssertionError(
                    f"Loaded train_step axis for {key}; only ppo_step is supported"
                )
            if key.endswith(REDUCE_SUFFIX):
                get_feature_reduce(str(value))
        self._data = {
            PPO_STEP_KEY: data.get(PPO_STEP_KEY, loaded_step),
            TRAIN_STEP_KEY: data.get(TRAIN_STEP_KEY),
        }
        for key, value in data.items():
            if key in (PPO_STEP_KEY, TRAIN_STEP_KEY):
                continue
            if is_pending_key(key):
                continue
            self._data[key] = value
        self._clear_all_pending()

    def save_to_ckpt(self, iter_dir: str, ppo_step: int) -> None:
        self._data[PPO_STEP_KEY] = int(ppo_step)
        path = os.path.join(iter_dir, PPO_FEATURE_STORE_FILENAME)
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        os.makedirs(iter_dir, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_from_ckpt(self, iter_dir: str, ppo_step: int, *, strict_step: bool = True) -> bool:
        path = os.path.join(iter_dir, PPO_FEATURE_STORE_FILENAME)
        if not os.path.isfile(path):
            path = os.path.join(iter_dir, LEGACY_TRAIN_EXTRA_STATE_FILENAME)
        if not os.path.isfile(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state, expected_ppo_step=ppo_step, strict_step=strict_step)
        return True

    def broadcast(self, group=None, src: int = 0) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        if dist.get_world_size(group=group) <= 1:
            return
        payload = [self.state_dict() if dist.get_rank(group=group) == src else None]
        dist.broadcast_object_list(payload, src=src, group=group)
        self.load_state_dict(payload[0], strict_step=False)


def finalize_all_pending_features_and_sync() -> dict[str, Optional[float]]:
    """Finalize all pending features on last PP, then broadcast state once."""
    store = get_ppo_feature_store()
    values: dict[str, Optional[float]] = {}
    payload = None
    try:
        from megatron.core import mpu
        is_last = mpu.is_pipeline_last_stage(ignore_virtual=True)
    except Exception:
        is_last = True
    if is_last:
        values = store.finalize_all_pending_features()
        payload = {"state": store.state_dict(), "values": values}
    if dist.is_available() and dist.is_initialized():
        from gpatch_v4.core.parallel_state import cpu_group, get_last_rank
        obj = [payload]
        dist.broadcast_object_list(obj, src=get_last_rank(cpu_group()), group=cpu_group())
        if obj[0] is not None:
            store.load_state_dict(obj[0]["state"], strict_step=False)
            values = dict(obj[0]["values"])
    elif payload is not None:
        store.load_state_dict(payload["state"], strict_step=False)
        values = dict(payload["values"])
    return values


def finalize_feature_interval_and_sync(
    feature: str,
    *,
    history_key: Optional[str] = None,
) -> Optional[float]:
    del history_key
    values = finalize_all_pending_features_and_sync()
    return values.get(feature)
