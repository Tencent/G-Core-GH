# coding=utf-8

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gpatch_v4.core.ppo_feature_store import (
    get_ppo_feature_store,
    reset_ppo_feature_store_for_test,
    set_ppo_feature_store_enabled,
)
from gpatch_v4.training_backend.loss_factory import PolicyLossInput
from gpatch_v4.core.post_compute_logprobs import (
    PostComputeLogprobsRegistry,
    noop_post_compute_logprobs,
    register_custom_post_compute_logprobs,
)

_GCORE_ROOT = Path(__file__).resolve().parents[2]
if str(_GCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_GCORE_ROOT))

from tasks.math_rl_v4.pre_entropy_quantile import (  # noqa: E402
    PREV_ENTROPY_QUANTILE_CUTS_KEY,
    collect_valid_prev_entropies,
    compute_quantile_cuts,
    prev_entropy_quantile_grpo_loss_func,
    prev_entropy_quantile_postprocess,
)


@pytest.fixture(autouse=True)
def _reset_feature_store_and_registry():
    PostComputeLogprobsRegistry.clear_for_test()
    reset_ppo_feature_store_for_test()
    yield
    PostComputeLogprobsRegistry.clear_for_test()
    reset_ppo_feature_store_for_test()


def test_noop_post_compute_logprobs_is_default():
    fn = PostComputeLogprobsRegistry.get("none")
    assert fn is noop_post_compute_logprobs
    batches = [{"tokens": [torch.zeros(3)]}]
    fn(SimpleNamespace(), batches)
    assert "prev_per_token_entropies" not in batches[0]


def test_register_custom_post_compute_logprobs(tmp_path):
    py = tmp_path / "hook.py"
    py.write_text(
        "def my_hook(config, rollout_batches):\n"
        "    rollout_batches[0]['marked'] = True\n"
    )
    register_custom_post_compute_logprobs("my_hook", str(py), "my_hook")
    batches = [{}]
    PostComputeLogprobsRegistry.get("my_hook")(None, batches)
    assert batches[0]["marked"] is True


def test_compute_quantile_cuts_matches_torch_quantile():
    entropy = torch.arange(1, 21, dtype=torch.float32)
    cuts = compute_quantile_cuts(entropy, num_bins=10)
    qs = torch.linspace(0.0, 1.0, 11)[1:-1]
    expected = torch.quantile(entropy, qs)
    assert cuts.shape == (9,)
    assert torch.allclose(cuts, expected)


def test_collect_valid_prev_entropies_drops_prompt():
    # logprob axis length 5; prompt_length=3 → response mask [2:4] on S-1 axis
    # (create_response_mask: prompt_length-1 : sequence_length-1)
    ent = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    rb = {
        "prev_per_token_entropies": [ent],
        "prompt_lengths": [torch.tensor(3)],
        "sequence_lengths": [torch.tensor(5)],
    }
    valid = collect_valid_prev_entropies([rb])
    assert torch.allclose(valid, torch.tensor([30.0, 40.0]))


def test_postprocess_sets_step_local_cuts():
    set_ppo_feature_store_enabled(True)
    store = get_ppo_feature_store()
    store.begin_ppo_step_interval(0)

    ent = torch.linspace(0.1, 2.0, 8)
    rb = {
        "prev_per_token_entropies": [ent],
        "prompt_lengths": [torch.tensor(1)],
        "sequence_lengths": [torch.tensor(9)],
    }
    config = SimpleNamespace(
        task={"prev_entropy_quantile_num_bins": 5},
        ppo=SimpleNamespace(feature_store_enable=True),
    )
    prev_entropy_quantile_postprocess(config, [rb])
    cuts = store.get_step_local(PREV_ENTROPY_QUANTILE_CUTS_KEY)
    assert cuts.device.type == "cpu"
    assert cuts.dtype == torch.float32
    assert cuts.shape == (4,)
    expected = compute_quantile_cuts(ent.float(), 5)
    assert torch.allclose(cuts, expected)


def test_custom_loss_gets_cuts_and_delegates_to_grpo(monkeypatch):
    set_ppo_feature_store_enabled(True)
    store = get_ppo_feature_store()
    store.begin_ppo_step_interval(0)

    b, s = 2, 4
    entropy = torch.tensor([[0.2, 1.0, 1.8, 0.0], [0.4, 0.6, 2.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    cuts = compute_quantile_cuts(entropy[mask.bool()], num_bins=4)
    store.set_step_local(PREV_ENTROPY_QUANTILE_CUTS_KEY, cuts)

    loss_input = PolicyLossInput(
        advantages=torch.ones(b, s),
        prev_log_probs=torch.zeros(b, s),
        ref_log_probs=None,
        curr_log_probs=torch.zeros(b, s),
        response_mask=mask,
        scaled_entropy=torch.zeros(()),
        per_token_entropy=torch.zeros(b, s),
        prev_per_token_entropy=entropy,
        calculate_per_token_loss=True,
    )
    config = SimpleNamespace(
        task={"prev_entropy_quantile_num_bins": 4},
        ppo=SimpleNamespace(
            skip_prev_logps=False,
            ppo_logps_ratio_clamp=None,
            enable_off_policy_correction=False,
            ppo_clip_ratio_low=None,
            ppo_clip_ratio_high=None,
            ppo_ratio_eps=0.2,
            ppo_entropy_regularization_type=None,
            ppo_dual_clip_ratio_c=None,
            loss_func="prev_entropy_quantile_grpo",
            ppo_entropy_bonus=0.0,
            grpo_kl_loss_beta=0.0,
            ppo_clamp_kl_val=None,
        ),
        policy=SimpleNamespace(override_transformer_config={}),
    )

    called = {}

    def _fake_grpo(cfg, inp):
        called["loss_input"] = inp
        numel = inp.response_mask.sum()
        return torch.zeros((), requires_grad=True), {
            "loss": torch.stack([torch.zeros(()), numel]),
        }

    monkeypatch.setattr(
        "tasks.math_rl_v4.pre_entropy_quantile.grpo_loss_func",
        _fake_grpo,
    )
    monkeypatch.setattr(
        "tasks.math_rl_v4.pre_entropy_quantile.logging_rank0",
        lambda msg: called.setdefault("logs", []).append(msg),
    )

    bwd, metrics = prev_entropy_quantile_grpo_loss_func(config, loss_input)
    assert bwd.ndim == 0
    assert "loss" in metrics
    assert called["loss_input"] is loss_input
    assert any("prev_entropy_quantile_cuts=" in m for m in called["logs"])
