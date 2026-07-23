from types import SimpleNamespace
from unittest.mock import patch

import torch

from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.training_backend.loss import PolicyLossInput, get_loss_fn

_REDUCE_METRICS_PATH = (
    "gpatch_v4.training_backend.loss.ppo_loss.reduce_metrics_across_data_parallel_group"
)


@patch(_REDUCE_METRICS_PATH)
def test_cispo_backend_loss_clips_ratios_without_dropping_gradients(_mock_reduce):
    curr_log_probs = torch.randn(2, 4, requires_grad=True)
    loss_input = PolicyLossInput(
        advantages=torch.ones_like(curr_log_probs),
        prev_log_probs=curr_log_probs.detach() - 2.0,
        ref_log_probs=None,
        curr_log_probs=curr_log_probs,
        response_mask=torch.ones_like(curr_log_probs),
        scaled_entropy=torch.tensor(0.0),
        per_token_entropy=torch.zeros_like(curr_log_probs),
    )
    config = SimpleNamespace(ppo=PpoConfig(grpo_kl_loss_beta=0.0))

    bwd_loss, bwd_count, metrics = get_loss_fn("mcore", "cispo")(config, loss_input)

    assert bwd_loss.ndim == 0
    assert bwd_count == 2
    assert metrics["cispo/clipfrac"] > 0
    bwd_loss.backward()
    assert torch.all(loss_input.curr_log_probs.grad != 0)
