from types import SimpleNamespace

import pytest

import gpatch_v4.actor.finetune_actor as finetune_actor_module
from gpatch_v4.actor.finetune_actor import FinetuneActor
from gpatch_v4.configs.training_config import FinetuneTrainingConfig


def test_finetune_manual_gc_defaults_disabled_and_validates_interval():
    config = FinetuneTrainingConfig()

    assert config.manual_gc is False
    assert config.manual_gc_interval == 20
    with pytest.raises(ValueError, match="manual_gc_interval must be positive"):
        FinetuneTrainingConfig(manual_gc=True, manual_gc_interval=0)


def test_finetune_actor_collects_only_at_manual_gc_interval(monkeypatch):
    actor = FinetuneActor.__new__(FinetuneActor)
    actor.config = SimpleNamespace(
        training=FinetuneTrainingConfig(manual_gc=True, manual_gc_interval=20)
    )
    calls = []
    monkeypatch.setattr(finetune_actor_module.gc, "disable", lambda: calls.append("disable"))
    monkeypatch.setattr(finetune_actor_module.gc, "collect", lambda: calls.append("collect"))

    actor._setup_manual_gc()
    actor._maybe_manual_gc(18)
    actor._maybe_manual_gc(19)

    assert calls == ["disable", "collect", "collect"]
