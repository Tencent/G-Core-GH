from types import SimpleNamespace

from gpatch_v4.agentic.env_manager.traj_env_manager import TrajEnvManager
from gpatch_v4.utils.constants import GenerateStopReason
from tasks.retool.multi_segment.dup_traj_env_manager import DupTrajEnvManager


def _make_manager(manager_cls, stop_reasons):
    manager = manager_cls.__new__(manager_cls)
    manager.env_config = {"env_id": 0}
    cache = SimpleNamespace(terminated=False)
    decisions = iter(stop_reasons)
    step_calls = []

    manager._reset = lambda seed, ppo_step, data: cache
    manager._make_decision = lambda rollout_cache: {
        "stop_reason": next(decisions),
        "response_ids": [1],
    }

    def step(lm_output):
        step_calls.append(lm_output["stop_reason"])
        if len(step_calls) == len(stop_reasons):
            cache.terminated = True
        return cache

    manager._step = step
    return manager, cache, step_calls


def test_traj_manager_steps_generation_length_before_environment_termination():
    manager, cache, step_calls = _make_manager(
        TrajEnvManager,
        [GenerateStopReason.MAX_GEN_LENGTH, GenerateStopReason.FINISH],
    )
    manager._formulate_rollout_batch = lambda rollout_cache: {"cache": rollout_cache}

    result = manager.run(seed=1, ppo_step=0, data={})

    assert result == {"cache": cache}
    assert step_calls == [
        GenerateStopReason.MAX_GEN_LENGTH,
        GenerateStopReason.FINISH,
    ]


def test_traj_manager_does_not_step_when_prompt_exceeds_context_length():
    manager, cache, step_calls = _make_manager(
        TrajEnvManager,
        [GenerateStopReason.MAX_LENGTH],
    )
    manager._formulate_rollout_batch = lambda rollout_cache: {"cache": rollout_cache}

    result = manager.run(seed=1, ppo_step=0, data={})

    assert result == {"cache": cache}
    assert step_calls == []


def test_dup_traj_manager_steps_generation_length(monkeypatch):
    manager, cache, step_calls = _make_manager(
        DupTrajEnvManager,
        [GenerateStopReason.MAX_GEN_LENGTH],
    )
    manager._align_traj_path = lambda seed: None
    manager._cfg_get = lambda key: None
    manager._apply_dup = lambda batch: batch
    monkeypatch.setattr(
        TrajEnvManager,
        "_formulate_rollout_batch",
        lambda self, rollout_cache: {"cache": rollout_cache},
    )

    result = manager.run(seed=1, ppo_step=0, data={})

    assert result == {"cache": cache}
    assert step_calls == [GenerateStopReason.MAX_GEN_LENGTH]
