from types import SimpleNamespace

import pytest
import torch

from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.core.dynamic_batch_reward_normalize import (
    prepare_sample_masks,
    reward_normalize,
)
from gpatch_v4.utils.filter_samplings import (
    filter_rollout_samples,
)


def _config(advantage_type="grpo", strategies=None, reward_weights=None):
    return SimpleNamespace(
        ppo=SimpleNamespace(
            advantage_type=advantage_type,
            gdpo_reward_weights=reward_weights or {},
            grpo_advantage_epsilon=1e-8,
            custom_reward_normalize_py_path=None,
        ),
        training=SimpleNamespace(
            dynamic_batch_rollout_filter_strategy=list(strategies or []),
            sampling_keep_n=2,
        ),
    )


def _sample(group_id, traj_id, reward, *, response_tokens=2, sample_mask=None):
    sample = {
        "tokens": torch.arange(response_tokens + 2),
        "prompt_lengths": torch.tensor(2),
        "sequence_lengths": torch.tensor(response_tokens + 2),
        "group_id": group_id,
        "traj_id": traj_id,
        "rewards": torch.tensor(float(reward)),
    }
    if sample_mask is not None:
        sample["sample_mask"] = torch.tensor(sample_mask)
    return sample


def test_prepare_sample_masks_preserves_existing_mask_and_creates_missing_mask():
    existing = _sample(0, 0, 1.0, response_tokens=3)
    existing["mask"] = torch.tensor([0.0, 1.0])
    missing = _sample(0, 1, 2.0, response_tokens=3)

    prepare_sample_masks([existing, missing])

    assert torch.equal(existing["mask"], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(missing["mask"], torch.tensor([0.0, 1.0, 1.0, 1.0]))


def test_grpo_aggregates_segments_by_trajectory_before_normalizing():
    samples = [
        _sample(0, 0, 1.0),
        _sample(0, 0, 2.0),
        _sample(0, 1, 7.0),
    ]
    prepare_sample_masks(samples)

    reward_normalize(_config(), samples)

    expected = 1.0 / 2**0.5
    assert torch.allclose(samples[0]["normalized_rewards"], torch.tensor(-expected))
    assert torch.equal(samples[0]["normalized_rewards"], samples[1]["normalized_rewards"])
    assert torch.allclose(samples[2]["normalized_rewards"], torch.tensor(expected))
    assert [sample["rewards"].item() for sample in samples] == [1.0, 2.0, 7.0]


def test_grpo_soft_masks_group_with_one_valid_trajectory():
    samples = [
        _sample(0, 0, 1.0, sample_mask=True),
        _sample(0, 1, 2.0, sample_mask=False),
    ]
    prepare_sample_masks(samples)

    reward_normalize(_config(), samples)

    assert all(not sample["sample_mask"].item() for sample in samples)
    assert all(torch.count_nonzero(sample["mask"]) == 0 for sample in samples)
    assert all(sample["normalized_rewards"].item() == 0.0 for sample in samples)


def test_grpo_rejects_inconsistent_segment_sample_masks():
    samples = [
        _sample(0, 0, 1.0, sample_mask=True),
        _sample(0, 0, 2.0, sample_mask=False),
        _sample(0, 1, 3.0, sample_mask=True),
    ]
    prepare_sample_masks(samples)

    with pytest.raises(AssertionError, match="inconsistent sample_mask"):
        reward_normalize(_config(), samples)


def test_sample_mask_filter_drops_invalid_trajectories_and_underfilled_groups():
    samples = [
        _sample(0, 0, 1.0, sample_mask=True),
        _sample(0, 1, 2.0, sample_mask=False),
        _sample(0, 2, 3.0, sample_mask=True),
        _sample(1, 0, 1.0, sample_mask=True),
        _sample(1, 1, 2.0, sample_mask=False),
    ]

    filtered = filter_rollout_samples(
        _config(strategies=["sample-mask", "valid_group"]),
        samples,
    )

    assert [(sample["group_id"], sample["traj_id"]) for sample in filtered] == [(0, 0), (0, 2)]


def test_gdpo_token_bn_weights_response_tokens():
    samples = [
        _sample(0, 0, 1.0, response_tokens=1),
        _sample(0, 1, 3.0, response_tokens=3),
        _sample(1, 0, 4.0, response_tokens=1),
        _sample(1, 1, 8.0, response_tokens=1),
    ]
    for sample in samples:
        sample["correctness"] = sample["rewards"]
    prepare_sample_masks(samples)

    metrics = reward_normalize(
        _config("gdpo", reward_weights={"correctness": 1.0}),
        samples,
    )

    token_values = torch.cat([
        sample["normalized_rewards"].expand(int(sample["mask"].sum().item()))
        for sample in samples
    ])
    assert torch.allclose(token_values.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(token_values.std(), torch.tensor(1.0), atol=1e-6)
    assert "ppo-metrics/global_bn_mean" in metrics


@pytest.mark.parametrize("advantage_type", ["gdpo_sample_bn", "group_gdpo_sample_bn"])
def test_gdpo_sample_bn_normalizes_trajectory_scalars(advantage_type):
    samples = [
        _sample(0, 0, 1.0),
        _sample(0, 1, 3.0),
        _sample(1, 0, 1.0),
        _sample(1, 1, 2.0),
        _sample(1, 2, 6.0),
    ]
    for sample in samples:
        sample["correctness"] = sample["rewards"]
    prepare_sample_masks(samples)

    reward_normalize(
        _config(advantage_type, reward_weights={"correctness": 1.0}),
        samples,
    )

    if advantage_type.startswith("group_"):
        partitions = [
            [sample for sample in samples if sample["group_id"] == group_id]
            for group_id in (0, 1)
        ]
    else:
        partitions = [samples]
    for partition in partitions:
        values = torch.stack([sample["normalized_rewards"] for sample in partition])
        assert torch.allclose(values.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(values.std(), torch.tensor(1.0), atol=1e-6)


def test_group_gdpo_token_bn_uses_group_id_boundaries():
    samples = [
        _sample(0, 0, 1.0, response_tokens=1),
        _sample(0, 1, 3.0, response_tokens=3),
        _sample(1, 0, 1.0, response_tokens=1),
        _sample(1, 1, 2.0, response_tokens=1),
        _sample(1, 2, 6.0, response_tokens=2),
    ]
    for sample in samples:
        sample["correctness"] = sample["rewards"]
    prepare_sample_masks(samples)

    reward_normalize(
        _config("group_gdpo", reward_weights={"correctness": 1.0}),
        samples,
    )

    for group_id in (0, 1):
        partition = [
            sample for sample in samples if sample["group_id"] == group_id
        ]
        values = torch.cat([
            sample["normalized_rewards"].expand(int(sample["mask"].sum().item()))
            for sample in partition
        ])
        assert torch.allclose(values.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(values.std(), torch.tensor(1.0), atol=1e-6)


def test_custom_reward_normalize_config_accepts_module_path():
    PpoConfig(
        advantage_type="grpo",
        custom_reward_normalize_py_path="/tmp/custom.py",
        custom_reward_normalize_py_name="custom_reward_normalize",
    )
