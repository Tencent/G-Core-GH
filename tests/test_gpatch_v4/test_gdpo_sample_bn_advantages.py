from types import SimpleNamespace

import torch

from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.core.advantage_helper import AdvantageContext, get_advantage_fn
from gpatch_v4.core.advantage_impl import (
    calculate_gdpo_advantages,
    calculate_gdpo_sample_bn_advantages,
    mask_single_valid_sample_groups,
)


def _scalar_rewards(values):
    return [torch.tensor(value, dtype=torch.float32) for value in values]


def test_gdpo_sample_bn_normalizes_sample_scalars_before_token_expansion():
    rewards_dict = {
        "correctness": _scalar_rewards([1.0, 3.0, 4.0, 8.0]),
        "format": _scalar_rewards([0.0, 0.0, 0.0, 0.0]),
    }
    masks = [
        torch.ones(1),
        torch.ones(3),
        torch.ones(1),
        torch.ones(1),
    ]

    sample_bn_advantages, _, _ = calculate_gdpo_sample_bn_advantages(
        rewards_dict=rewards_dict,
        mask=masks,
        grpo_sampling_times=2,
        grpo_advantage_epsilon=1e-8,
        gdpo_reward_weights={"correctness": 1.0, "format": 1.0},
    )
    token_bn_advantages, _ = calculate_gdpo_advantages(
        rewards_dict=rewards_dict,
        mask=masks,
        grpo_sampling_times=2,
        grpo_advantage_epsilon=1e-8,
        gdpo_reward_weights={"correctness": 1.0, "format": 1.0},
    )

    sample_scalars = torch.stack(
        [advantage[mask.bool()][0] for advantage, mask in zip(sample_bn_advantages, masks)]
    )
    sample_token_values = torch.cat(
        [advantage[mask.bool()] for advantage, mask in zip(sample_bn_advantages, masks)]
    )
    token_values = torch.cat(
        [advantage[mask.bool()] for advantage, mask in zip(token_bn_advantages, masks)]
    )

    assert torch.allclose(sample_scalars.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(sample_scalars.std(), torch.tensor(1.0), atol=1e-6)
    assert not torch.allclose(sample_token_values.mean(), torch.tensor(0.0), atol=1e-2)
    assert torch.allclose(token_values.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(token_values.std(), torch.tensor(1.0), atol=1e-6)
    assert not torch.allclose(sample_token_values, token_values)


def test_gdpo_sample_bn_registered_as_builtin_advantage_type():
    PpoConfig(advantage_type="group_gdpo_sample_bn")

    config = SimpleNamespace(
        training=SimpleNamespace(sampling_keep_n=2),
        ppo=SimpleNamespace(
            grpo_advantage_epsilon=1e-8,
            gdpo_reward_weights={"correctness": 1.0},
        ),
    )
    masks = [torch.ones(1), torch.ones(1)]
    ctx = AdvantageContext(
        rollout_batch={},
        config=config,
        mask=masks,
        logprobs=[],
        gdpo_rewards={"correctness": _scalar_rewards([1.0, 3.0])},
    )

    result = get_advantage_fn("group_gdpo_sample_bn")(ctx)

    assert len(result.advantages) == 2
    assert len(result.returns) == 2


def test_gdpo_sample_bn_zeros_invalid_samples_from_sample_mask():
    rewards_dict = {
        "correctness": _scalar_rewards([1.0, 3.0, 5.0, 7.0]),
    }
    masks = [
        torch.ones(1),
        torch.ones(1),
        torch.ones(2),
        torch.ones(2),
    ]
    sample_mask = [
        torch.tensor(1.0),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
    ]

    advantages, _, _ = calculate_gdpo_sample_bn_advantages(
        rewards_dict=rewards_dict,
        mask=masks,
        grpo_sampling_times=2,
        grpo_advantage_epsilon=1e-8,
        gdpo_reward_weights={"correctness": 1.0},
        sample_mask=sample_mask,
    )

    assert torch.count_nonzero(advantages[2]) == 0
    assert torch.count_nonzero(advantages[3]) == 0


def test_mask_single_valid_sample_groups_discards_only_affected_group():
    masks = [
        torch.ones(2),
        torch.zeros(2),
        torch.ones(2),
        torch.ones(2),
    ]
    sample_mask = [
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(1.0),
        torch.tensor(1.0),
    ]

    mask_single_valid_sample_groups(masks, sample_mask, sampling_keep_n=2)

    assert torch.count_nonzero(masks[0]) == 0
    assert sample_mask[0].item() == 0.0
    assert torch.count_nonzero(masks[2]) == 2
    assert torch.count_nonzero(masks[3]) == 2
    assert sample_mask[2].item() == 1.0
    assert sample_mask[3].item() == 1.0
