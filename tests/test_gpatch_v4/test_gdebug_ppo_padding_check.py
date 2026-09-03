"""Tests for gpatch_v4.gdebug.ppo_padding_check."""
from dataclasses import fields
from types import SimpleNamespace

import pytest

import gpatch_v4.gdebug.ppo_padding_check as ppo_padding_check
from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.gdebug.ppo_padding_check import (
    _find_risky_padding_settings,
    maybe_check_ppo_padding,
)


@pytest.fixture(autouse=True)
def logged(monkeypatch):
    lines = []
    monkeypatch.setattr(ppo_padding_check, "logging_rank0", lines.append)
    return lines


def make_config(mode="warn", advantage_type="ppo", policy_smart_pad=False, critic_smart_pad=False):
    return SimpleNamespace(
        debug=SimpleNamespace(ppo_padding_check_mode=mode),
        ppo=SimpleNamespace(advantage_type=advantage_type, skip_prev_logps=False),
        training=SimpleNamespace(training_backend="mcore"),
        policy=SimpleNamespace(
            smart_pad_infer=policy_smart_pad,
            forward_only_mbs=1,
            override_transformer_config={},
            dist_config=SimpleNamespace(dynamic_context_parallel=False),
        ),
        critic=SimpleNamespace(
            smart_pad_infer=critic_smart_pad,
            forward_only_mbs=1,
            override_transformer_config={},
        ),
    )


def with_skip_prev_logps(cfg):
    cfg.ppo.skip_prev_logps = True
    return cfg


def with_dynamic_cp(cfg):
    cfg.policy.dist_config.dynamic_context_parallel = True
    return cfg


PER_SAMPLE_LEN_ROUTES = (with_skip_prev_logps, with_dynamic_cp)


def test_clean_when_padding_matches():
    assert _find_risky_padding_settings(make_config(policy_smart_pad=True,
                                                    critic_smart_pad=True)) == []
    assert _find_risky_padding_settings(
        make_config(policy_smart_pad=False, critic_smart_pad=False)
    ) == []


def test_flags_the_mismatch_that_can_leave_the_values_short():
    violations = _find_risky_padding_settings(
        make_config(policy_smart_pad=False, critic_smart_pad=True)
    )
    assert any("smart_pad_infer" in v for v in violations)


def test_the_other_mismatch_is_not_reported():
    """A non-smart critic pads to the global width, so its values are never short."""
    assert _find_risky_padding_settings(make_config(policy_smart_pad=True,
                                                    critic_smart_pad=False)) == []


def test_matching_smart_pad_still_needs_the_same_microbatch_size():
    cfg = make_config(policy_smart_pad=True, critic_smart_pad=True)
    cfg.critic.forward_only_mbs = 4
    assert any("forward_only_mbs" in v for v in _find_risky_padding_settings(cfg))

    cfg.critic.forward_only_mbs = 1
    assert _find_risky_padding_settings(cfg) == []


def test_microbatch_size_only_matters_when_both_engines_bucket():
    cfg = make_config(policy_smart_pad=True, critic_smart_pad=False)
    cfg.critic.forward_only_mbs = 4
    assert _find_risky_padding_settings(cfg) == []


@pytest.mark.parametrize("route", PER_SAMPLE_LEN_ROUTES, ids=["skip_prev_logps", "dynamic_cp"])
def test_padding_checks_skip_per_sample_len_routes(route):
    mismatch = route(make_config(policy_smart_pad=False, critic_smart_pad=True))
    assert _find_risky_padding_settings(mismatch) == []

    same_flag_different_mbs = route(make_config(policy_smart_pad=True, critic_smart_pad=True))
    same_flag_different_mbs.critic.forward_only_mbs = 4
    assert _find_risky_padding_settings(same_flag_different_mbs) == []


def test_route_exemption_reads_the_flags_the_way_the_host_does():
    cfg = make_config(policy_smart_pad=False, critic_smart_pad=True)
    assert _find_risky_padding_settings(cfg) != []

    # yaml hands this over as a string; the host reads it truthily, so must we
    cfg.ppo.skip_prev_logps = "true"
    assert _find_risky_padding_settings(cfg) == []


def test_skips_non_ppo_advantage_type():
    cfg = make_config(advantage_type="grpo", policy_smart_pad=False, critic_smart_pad=True)
    assert _find_risky_padding_settings(cfg) == []


def test_skips_backends_without_the_aligner():
    cfg = make_config(policy_smart_pad=False, critic_smart_pad=True)
    cfg.training.training_backend = "fsdp2"
    assert _find_risky_padding_settings(cfg) == []


def test_warn_logs_once_and_returns(logged):
    cfg = make_config(mode="warn", policy_smart_pad=False, critic_smart_pad=True)
    assert maybe_check_ppo_padding(cfg) is None
    assert len(logged) == 1
    assert "WARN:" in logged[0] and "the run continues" in logged[0]
    assert "gdebug.ppo_padding" in logged[0] and "smart_pad_infer" in logged[0]


def test_warn_reports_the_microbatch_mismatch_too(logged):
    """The second diagnostic has its own wording, so it needs its own path to a reader."""
    cfg = make_config(mode="warn", policy_smart_pad=True, critic_smart_pad=True)
    cfg.critic.forward_only_mbs = 4
    assert maybe_check_ppo_padding(cfg) is None
    assert len(logged) == 1
    assert "forward_only_mbs" in logged[0] and "gdebug.ppo_padding" in logged[0]


def test_warn_says_nothing_about_a_clean_config(logged):
    cfg = make_config(mode="warn", policy_smart_pad=True, critic_smart_pad=True)
    assert maybe_check_ppo_padding(cfg) is None
    assert logged == []


def test_abort_raises():
    cfg = make_config(mode="abort", policy_smart_pad=False, critic_smart_pad=True)
    with pytest.raises(RuntimeError, match="smart_pad_infer"):
        maybe_check_ppo_padding(cfg)


def test_clean_config_says_nothing(logged):
    cfg = make_config(mode="abort", policy_smart_pad=True, critic_smart_pad=True)
    assert maybe_check_ppo_padding(cfg) is None
    assert logged == []


def test_off_does_not_read_the_config(logged):
    class _MustNotRead:
        def __init__(self, name):
            self._name = name

        def __getattr__(self, item):
            raise AssertionError(f"off mode touched {self._name}.{item}")

    cfg = SimpleNamespace(
        debug=SimpleNamespace(ppo_padding_check_mode="off"),
        ppo=_MustNotRead("ppo"),
        policy=_MustNotRead("policy"),
        critic=_MustNotRead("critic"),
        training=_MustNotRead("training"),
    )
    assert maybe_check_ppo_padding(cfg) is None
    assert logged == []


def test_a_value_that_is_not_a_mode_is_rejected():
    cfg = make_config(mode="wanr", policy_smart_pad=False, critic_smart_pad=True)
    with pytest.raises(ValueError, match="ppo_padding_check_mode"):
        maybe_check_ppo_padding(cfg)


def test_the_check_reads_a_field_debug_config_declares():
    """Renaming the field on either side has to break something, so drive a real DebugConfig."""
    cfg = make_config(policy_smart_pad=False, critic_smart_pad=True)
    cfg.debug = DebugConfig(ppo_padding_check_mode="abort")
    with pytest.raises(RuntimeError, match="smart_pad_infer"):
        maybe_check_ppo_padding(cfg)


def test_debug_config_rejects_an_unknown_mode():
    assert DebugConfig().ppo_padding_check_mode == "off"
    for mode in ("off", "warn", "abort"):
        assert DebugConfig(ppo_padding_check_mode=mode).ppo_padding_check_mode == mode
    with pytest.raises(AssertionError, match="ppo_padding_check_mode"):
        DebugConfig(ppo_padding_check_mode="wanr")


def test_debug_config_still_validates_every_check_mode_it_declares():
    """A second __post_init__ would silently replace this one, taking the checks with it."""
    modes = [f.name for f in fields(DebugConfig) if f.name.endswith("_check_mode")]
    assert modes

    for name in modes:
        with pytest.raises(AssertionError, match=name):
            DebugConfig(**{name: "not-a-mode"})
