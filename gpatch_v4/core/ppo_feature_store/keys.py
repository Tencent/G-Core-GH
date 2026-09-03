"""Key helpers for ``PpoFeatureStore``."""

from __future__ import annotations

import os

PPO_FEATURE_STORE_FILENAME = "ppo_feature_store.pt"
# Legacy sidecar name; still accepted on load for resume compatibility.
LEGACY_TRAIN_EXTRA_STATE_FILENAME = "train_extra_state.pt"

PPO_STEP_KEY = "ppo_step"
TRAIN_STEP_KEY = "train_step"

STATE_VERSION = 5
PENDING_SUFFIX = ".pending"
REDUCE_SUFFIX = ".reduce"
HISTORY_AXIS_SUFFIX = ".history_axis"
AXIS_TOTAL_SUFFIX = ".axis_total"
DEFAULT_REDUCE = "mean"


def feature_history_key(feature: str) -> str:
    return f"{feature}.history"


def feature_pending_key(feature: str) -> str:
    return f"{feature}.pending"


def feature_reduce_key(feature: str) -> str:
    return f"{feature}.reduce"


def feature_history_axis_key(feature: str) -> str:
    return f"{feature}.history_axis"


def feature_axis_total_key(feature: str) -> str:
    return f"{feature}.axis_total"


def feature_from_pending_key(key: str) -> str:
    assert key.endswith(PENDING_SUFFIX), key
    return key[:-len(PENDING_SUFFIX)]


def is_pending_key(key: str) -> bool:
    return key.endswith(PENDING_SUFFIX)


def assert_ppo_step_axis_only(axis: str) -> None:
    assert axis != "train_step", (
        "train_step history axis is not supported; "
        "use the outermost ppo_step interval only."
    )
    assert axis == "ppo_step", f"Invalid history axis: {axis!r} (only 'ppo_step' is supported)"


def iter_checkpoint_dir(ckpt_root: str, step: int) -> str:
    return os.path.join(ckpt_root, f"iter_{step:07d}")
