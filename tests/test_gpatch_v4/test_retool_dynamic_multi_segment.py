import random
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.training_backend.loss import PolicyLossInput, get_loss_fn
from gpatch_v4.training_backend.loss.utils import agg
from tasks.retool.multi_segment.dup_traj_env_manager import DupTrajEnvManager
from tasks.retool.multi_segment.dynamic_batch_hooks import (
    compute_rollout_metrics,
    convert_samples_to_train_data,
)


def _config():
    return SimpleNamespace(
        training=SimpleNamespace(
            train_gbs=2,
            metrics_report=[],
        ),
    )


def _sample(
    traj_id: int,
    segment_id: int,
    reward: float,
    normalized_reward: float,
):
    return {
        "_train_step_id": 0,
        "group_id": 0,
        "traj_id": traj_id,
        "segment_id": segment_id,
        "tokens": torch.tensor([1, 2, 3]),
        "prompt_lengths": torch.tensor(1),
        "sequence_lengths": torch.tensor(3),
        "rewards": torch.tensor(reward),
        "normalized_rewards": torch.tensor(normalized_reward),
        "sample_mask": torch.tensor(True),
        "mask": torch.ones(2),
    }


def test_dup_traj_keeps_reward_only_on_last_segment():
    manager = DupTrajEnvManager.__new__(DupTrajEnvManager)
    manager.env_config = {"config": {"dup_num_segments": 3}}
    base = {
        "tokens": [torch.tensor([1, 2, 3])],
        "rewards": [torch.tensor(7.0)],
    }

    duplicated = manager._apply_dup(base)

    assert len(duplicated["tokens"]) == 3
    assert [reward.item() for reward in duplicated["rewards"]] == [0.0, 0.0, 7.0]
    assert all(
        duplicated["tokens"][idx].data_ptr() != duplicated["tokens"][0].data_ptr()
        for idx in (1, 2)
    )


def test_dup_traj_random_segment_count_is_seeded():
    manager = DupTrajEnvManager.__new__(DupTrajEnvManager)
    manager.env_config = {"config": {"dup_max_segments": 3}}
    manager.sampling_seed_offset = 17

    assert manager._resolve_dup_k() == random.Random(17).randint(1, 3)


def test_dynamic_multi_segment_hooks_use_trajectory_equal_weights_and_metrics():
    samples = [
        _sample(0, 0, 1.0, -1.0),
        _sample(1, 0, 0.0, 1.0),
        _sample(1, 1, 0.0, 1.0),
        _sample(1, 2, 3.0, 1.0),
    ]

    metrics = compute_rollout_metrics(_config(), samples)
    converted = convert_samples_to_train_data(_config(), samples)

    weights = [sample["token_weights"].item() for sample in converted]
    assert torch.allclose(
        torch.tensor(weights),
        torch.tensor([2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
    )
    assert metrics["rollout-rewards/global_rewards"] == 2.0
    assert metrics["multi_segment/actual_segment_gbs_mean"] == 4.0
    assert metrics["multi_segment/segments_per_trajectory_mean"] == 2.0
    assert metrics["multi_segment/token_weight_mean"] == 1.0
    assert metrics["multi_segment/normalized_reward_mean"] == 0.0


def test_trajectory_weights_preserve_response_padded_seq_mean_loss():
    baseline_values = torch.tensor([[2.0, 2.0], [8.0, 8.0]])
    baseline_mask = torch.ones_like(baseline_values)
    baseline_sum, baseline_count = agg(
        baseline_values,
        baseline_mask,
        calculate_per_token_loss=False,
    )

    duplicated_values = torch.tensor(
        [[2.0, 2.0], [8.0, 8.0], [8.0, 8.0], [8.0, 8.0]]
    )
    duplicated_mask = torch.ones_like(duplicated_values)
    token_weights = torch.tensor([[2.0], [2.0 / 3.0], [2.0 / 3.0], [2.0 / 3.0]])
    duplicated_sum, duplicated_count = agg(
        duplicated_values,
        duplicated_mask,
        calculate_per_token_loss=False,
        token_weights=token_weights,
    )

    assert torch.allclose(baseline_sum / baseline_count, duplicated_sum / duplicated_count)


@patch(
    "gpatch_v4.training_backend.loss.ppo_loss.reduce_metrics_across_data_parallel_group"
)
def test_grpo_loss_accepts_response_padded_trajectory_weights(_mock_reduce_metrics):
    config = SimpleNamespace(
        ppo=PpoConfig(
            loss_func="grpo",
            grpo_kl_loss_beta=0.0,
        ),
        debug=DebugConfig(),
    )
    baseline_advantages = torch.tensor([[2.0, 2.0], [8.0, 8.0]])
    baseline_log_probs = torch.zeros_like(baseline_advantages)
    baseline_input = PolicyLossInput(
        advantages=baseline_advantages,
        prev_log_probs=baseline_log_probs,
        ref_log_probs=None,
        curr_log_probs=baseline_log_probs,
        response_mask=torch.ones_like(baseline_advantages),
        scaled_entropy=torch.tensor(0.0),
        per_token_entropy=torch.zeros_like(baseline_advantages),
        calculate_per_token_loss=False,
    )
    baseline_sum, baseline_count, _ = get_loss_fn("mcore", "grpo")(
        config,
        baseline_input,
    )

    duplicated_advantages = torch.tensor(
        [[2.0, 2.0], [8.0, 8.0], [8.0, 8.0], [8.0, 8.0]]
    )
    duplicated_log_probs = torch.zeros_like(duplicated_advantages)
    duplicated_input = PolicyLossInput(
        advantages=duplicated_advantages,
        prev_log_probs=duplicated_log_probs,
        ref_log_probs=None,
        curr_log_probs=duplicated_log_probs,
        response_mask=torch.ones_like(duplicated_advantages),
        scaled_entropy=torch.tensor(0.0),
        per_token_entropy=torch.zeros_like(duplicated_advantages),
        token_weights=torch.tensor(
            [[2.0], [2.0 / 3.0], [2.0 / 3.0], [2.0 / 3.0]]
        ),
        calculate_per_token_loss=False,
    )
    duplicated_sum, duplicated_count, _ = get_loss_fn("mcore", "grpo")(
        config,
        duplicated_input,
    )

    assert torch.allclose(
        baseline_sum / baseline_count,
        duplicated_sum / duplicated_count,
    )
