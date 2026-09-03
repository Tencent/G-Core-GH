"""Dynamic import registration for custom loss / advantage into registries."""
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.core.advantage_helper import (
    ADVANTAGE_DISPATCH,
    AdvantageContext,
    get_advantage_fn,
    register_custom_advantage,
)
from gpatch_v4.core.ppo_feature_store import (
    feature_history_key,
    feature_pending_key,
    feature_reduce_key,
    get_ppo_feature_store,
    ppo_step_interval,
    reset_ppo_feature_store_for_test,
)
from gpatch_v4.training_backend.loss_factory import (
    LOSS_FUNC_REGISTRY,
    PolicyLossInput,
    get_policy_loss_fn,
    register_custom_loss_fn,
)

_REPO = Path(__file__).resolve().parents[2]
_EPO_LOSS_PY = str(_REPO / "tasks" / "math_rl_v4" / "epo_grpo_loss.py")
_CUSTOM_ADV_PY = str(_REPO / "tasks" / "math_rl_v4" / "custom_advantage.py")

_REDUCE_METRICS_PATH = (
    "gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group"
)
_EPO_REDUCE_METRICS_PATH = (
    "tasks.math_rl_v4.epo_grpo_loss.reduce_metrics_across_data_parallel_group"
)


class TestDynamicRegisterCustomLoss:
    @patch(_EPO_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    @patch(_REDUCE_METRICS_PATH, side_effect=lambda m: m)
    def test_register_epo_loss_from_py_path(self, _mock_a, _mock_b):
        name = f"epo_grpo_dyn_{uuid.uuid4().hex[:8]}"
        register_custom_loss_fn(name, _EPO_LOSS_PY, "epo_grpo_loss_func")
        try:
            fn = get_policy_loss_fn(name)
            reset_ppo_feature_store_for_test()
            store = get_ppo_feature_store()
            store.begin_ppo_step_interval(0)
            store.set(feature_history_key("epo"), [1.0])

            ppo = PpoConfig(
                ppo_entropy_bonus=1.0,
                feature_store_enable=True,
            )
            config = SimpleNamespace(
                ppo=ppo,
                debug=DebugConfig(),
                policy=SimpleNamespace(override_transformer_config={}),
                task=SimpleNamespace(
                    epo_out_range_penalty=0.5,
                    epo_entropy_smooth_coeff=1.0,
                    epo_mask_mode="token",
                    epo_min_ratio=0.8,
                    epo_max_ratio=1.2,
                    epo_enable_smooth_weights=False,
                ),
            )
            curr = torch.randn(2, 4, requires_grad=True)
            li = PolicyLossInput(
                advantages=torch.ones(2, 4),
                prev_log_probs=curr.detach().clone(),
                ref_log_probs=None,
                curr_log_probs=curr,
                response_mask=torch.ones(2, 4),
                scaled_entropy=torch.tensor(0.5),
                per_token_entropy=torch.full((2, 4), 10.0),
            )
            loss, metrics = fn(config, li)
            assert not torch.isnan(loss)
            assert "epo_baseline_H" in metrics
            assert len(store.get(feature_pending_key("epo"))) == 1
            reset_ppo_feature_store_for_test()
        finally:
            LOSS_FUNC_REGISTRY.pop(name, None)


class TestDynamicRegisterCustomAdvantage:
    def test_register_adv_stats_from_py_path(self):
        name = f"custom_adv_dyn_{uuid.uuid4().hex[:8]}"
        register_custom_advantage(name, _CUSTOM_ADV_PY, "custom_grpo_advantage_with_adv_stats")
        try:
            fn = get_advantage_fn(name)
            reset_ppo_feature_store_for_test()
            store = get_ppo_feature_store()
            with ppo_step_interval(ppo_step=0, enabled=True):
                ctx = AdvantageContext(
                    rollout_batch={},
                    config=SimpleNamespace(
                        training=SimpleNamespace(sampling_keep_n=2),
                        ppo=SimpleNamespace(grpo_advantage_epsilon=1e-4),
                    ),
                    mask=[torch.ones(4), torch.ones(4)],
                    logprobs=[],
                    rewards=[torch.tensor(1.0), torch.tensor(3.0)],
                    sample_mask=None,
                )
                result = fn(ctx)
                assert len(result.advantages) == 2
                assert store.get(feature_reduce_key("adv_mean")) == "mean"
                assert store.get(feature_reduce_key("adv_min")) == "min"
                assert store.get(feature_reduce_key("adv_max")) == "max"
                assert len(store.get(feature_pending_key("adv_mean"))) == 1
            reset_ppo_feature_store_for_test()
        finally:
            ADVANTAGE_DISPATCH.pop(name, None)
