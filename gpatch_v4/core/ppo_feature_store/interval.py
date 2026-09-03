"""Outermost ``ppo_step`` interval context for ``PpoFeatureStore``."""

from __future__ import annotations

from types import TracebackType
from typing import Optional, Type

from gpatch_v4.core.ppo_feature_store.state import (
    finalize_all_pending_features_and_sync,
    get_ppo_feature_store,
)


class PpoStepInterval:
    """Outermost PPO-step context: clear pending/step-local on enter; finalize history on success; always drop step-local on exit."""
    def __init__(self, *, ppo_step: int, enabled: bool = True) -> None:
        self.ppo_step = int(ppo_step)
        self.enabled = enabled
        self.final_values: dict[str, Optional[float]] = {}
        self._entered = False

    def __enter__(self) -> "PpoStepInterval":
        if not self.enabled:
            return self
        get_ppo_feature_store().begin_ppo_step_interval(self.ppo_step)
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        if not self.enabled or not self._entered:
            return False
        try:
            if exc_type is None:
                self.final_values = finalize_all_pending_features_and_sync()
        finally:
            get_ppo_feature_store().clear_step_local()
        return False


def ppo_step_interval(*, ppo_step: int, enabled: bool = True) -> PpoStepInterval:
    return PpoStepInterval(ppo_step=ppo_step, enabled=enabled)
