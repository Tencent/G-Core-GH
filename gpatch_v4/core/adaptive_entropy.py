"""TRL-style adaptive entropy control state in ``PpoFeatureStore``.

Lag-1 controller (see HuggingFace TRL GRPO ``use_adaptive_entropy``):
after each train / optimizer step, write ``last_world_entropy`` and
``entropy_coef`` via ``store.set`` (persisted; not cleared by
``ppo_step_interval``). The next step's loss reads them for gating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpatch_v4.core.ppo_feature_store import get_ppo_feature_store

if TYPE_CHECKING:
    from gpatch_v4.configs.ppo_config import PpoConfig

LAST_WORLD_ENTROPY_KEY = "adaptive_entropy.last_world_entropy"
ENTROPY_COEF_KEY = "adaptive_entropy.coef"


def get_entropy_bonus_coef(ppo_config: "PpoConfig") -> float:
    """Return the entropy bonus coefficient for the current train step.

    Static mode returns ``ppo_entropy_bonus``. Adaptive mode freezes the
    previous step's coefficient and applies it only when
    ``last_world_entropy <= entropy_target`` (else ``0.0``).
    """
    if not ppo_config.use_adaptive_entropy:
        return float(ppo_config.ppo_entropy_bonus)

    store = get_ppo_feature_store()
    last_world_entropy = store.get(LAST_WORLD_ENTROPY_KEY, float("inf"))
    coef = store.get(ENTROPY_COEF_KEY, None)
    if coef is None:
        coef = float(ppo_config.ppo_entropy_bonus)
    else:
        coef = float(coef)
    if float(last_world_entropy) <= float(ppo_config.entropy_target):
        return coef
    return 0.0


def update_adaptive_entropy_after_train_step(
    ppo_config: "PpoConfig",
    world_entropy: float,
) -> dict[str, float]:
    """Update and ``set`` adaptive entropy state after one optimizer step.

    Parameters
    ----------
    ppo_config : PpoConfig
    world_entropy : float
        Token-weighted mean per-token entropy over the train step (already
        reduced across micro-batches and DP), matching TRL ``H_world``.

    Returns
    -------
    dict[str, float]
        Metrics for logging (``policy/entropy_coef``,
        ``policy/last_world_entropy``). Empty when adaptive entropy is off.
    """
    if not ppo_config.use_adaptive_entropy:
        return {}

    store = get_ppo_feature_store()
    coef = store.get(ENTROPY_COEF_KEY, None)
    if coef is None:
        coef = float(ppo_config.ppo_entropy_bonus)
    else:
        coef = float(coef)

    world_entropy = float(world_entropy)
    if world_entropy <= float(ppo_config.entropy_target):
        coef = min(coef + float(ppo_config.entropy_coef_delta), float(ppo_config.entropy_coef_max))
    else:
        coef = max(coef - float(ppo_config.entropy_coef_delta), float(ppo_config.entropy_coef_min))

    store.set(LAST_WORLD_ENTROPY_KEY, world_entropy)
    store.set(ENTROPY_COEF_KEY, coef)
    return {
        "policy/entropy_coef": coef,
        "policy/last_world_entropy": world_entropy,
    }
