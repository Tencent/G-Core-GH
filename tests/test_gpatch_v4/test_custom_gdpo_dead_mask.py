# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Unit tests for ``tasks/math_rl_v4/custom_gdpo_dead_mask_advantage.py``.

All tests run on CPU. The post-advantage all-reduce path is exercised through a
single-rank ``gloo`` process group initialised once per session.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Module loading: import the implementation file by its filesystem path so the
# test does not depend on ``tasks/`` being on ``PYTHONPATH``.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
# tests/test_gpatch_v4/test_custom_gdpo_dead_mask.py -> gcore-dev/
_GCORE_DEV = _HERE.parents[2]
_IMPL_PATH = (
    _GCORE_DEV / "tasks" / "math_rl_v4" / "custom_gdpo_dead_mask_advantage.py"
)
assert _IMPL_PATH.is_file(), f"impl file not found at {_IMPL_PATH}"

_spec = importlib.util.spec_from_file_location(
    "custom_gdpo_dead_mask_advantage", str(_IMPL_PATH)
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["custom_gdpo_dead_mask_advantage"] = _module
_spec.loader.exec_module(_module)  # type: ignore[union-attr]

# Bind for readability
_resolve_dead_threshold = _module.resolve_dead_threshold
_calc_grpo_advantages_func_with_dead_mask = (
    _module.calc_grpo_advantages_func_with_dead_mask
)
compute_gdpo_combined_advantages_with_dead_mask = (
    _module.compute_gdpo_combined_advantages_with_dead_mask
)
compute_gdpo_sample_bn_dead_mask_advantages = (
    _module.compute_gdpo_sample_bn_dead_mask_advantages
)
gdpo_sample_bn_dead_mask_post_advantage = (
    _module.gdpo_sample_bn_dead_mask_post_advantage
)
DEFAULT_DEAD_GROUP_THRESHOLD = _module.DEFAULT_DEAD_GROUP_THRESHOLD

from gpatch_v4.core import AdvantageContext, PostAdvantageContext  # noqa: E402


# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _single_rank_gloo_pg():
    """Initialise a single-rank gloo process group for distributed.all_reduce."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        # Use a PID-derived port so parallel CI workers do not collide on 29555.
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    yield dist.group.WORLD
    if dist.is_initialized():
        dist.destroy_process_group()


def _scalar_rewards(values: List[float]) -> List[torch.Tensor]:
    return [torch.tensor(v, dtype=torch.float32) for v in values]


def _make_advantage_context(
    *,
    config: SimpleNamespace,
    rewards_dict: dict,
    masks: List[torch.Tensor],
    sample_mask: Optional[List[torch.Tensor]] = None,
) -> AdvantageContext:
    rollout_batch: dict = {}
    rollout_batch.update(rewards_dict)
    return AdvantageContext(
        rollout_batch=rollout_batch,
        config=config,
        mask=masks,
        logprobs=[],
        sample_mask=sample_mask,
    )


def _make_ppo_config(
    *,
    loss_func: str = "grpo",
    grpo_advantage_epsilon: float = 1e-8,
    advantage_clip=None,
    advantage_clip_lower_bound=None,
    advantage_clip_upper_bound=None,
    gdpo_reward_weights: dict = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        loss_func=loss_func,
        grpo_advantage_epsilon=grpo_advantage_epsilon,
        advantage_clip=advantage_clip,
        advantage_clip_lower_bound=advantage_clip_lower_bound,
        advantage_clip_upper_bound=advantage_clip_upper_bound,
        gdpo_reward_weights=gdpo_reward_weights or {},
    )


def _make_top_config(
    *,
    sampling_keep_n: int,
    ppo_cfg: SimpleNamespace,
    task: Optional[object] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(sampling_keep_n=sampling_keep_n),
        ppo=ppo_cfg,
        task=task,
    )


# ===========================================================================
# 1. _calc_grpo_advantages_func_with_dead_mask
# ===========================================================================


def test_calc_with_dead_mask_all_same_rewards_marks_group_dead():
    """A group whose rewards are identical (std=0) is marked dead and adv=0."""
    rewards = _scalar_rewards([1.0, 1.0, 1.0, 1.0])
    masks = [torch.ones(3, dtype=torch.float32) for _ in range(4)]
    adv, dead = _calc_grpo_advantages_func_with_dead_mask(
        rewards=rewards,
        mask=masks,
        grpo_sampling_times=4,
        dead_threshold=0.01,
    )
    assert adv.shape == (4,)
    assert dead.shape == (4,)
    assert dead.all(), f"all samples in the only group should be dead; got {dead}"
    assert torch.equal(adv, torch.zeros(4))


def test_calc_with_dead_mask_normal_group_matches_trunk_behavior():
    """Two groups with non-trivial std behave like the trunk GRPO normalization."""
    # Group 0: [1, 3] -> mean=2 std=sqrt(2); Group 1: [2, 8] -> mean=5 std=sqrt(18)
    rewards = _scalar_rewards([1.0, 3.0, 2.0, 8.0])
    masks = [torch.ones(2, dtype=torch.float32) for _ in range(4)]
    adv, dead = _calc_grpo_advantages_func_with_dead_mask(
        rewards=rewards,
        mask=masks,
        grpo_sampling_times=2,
        grpo_advantage_epsilon=1e-8,
        dead_threshold=0.01,
    )
    assert not dead.any(), "no group should be dead with non-trivial std"
    # Group means are 0 within each group of size 2; opposite signs.
    assert adv[0].item() < 0 and adv[1].item() > 0
    assert adv[2].item() < 0 and adv[3].item() > 0
    # Sample-level mean within each group is 0.
    assert abs(adv[0].item() + adv[1].item()) < 1e-5
    assert abs(adv[2].item() + adv[3].item()) < 1e-5


def test_calc_with_dead_mask_single_sample_group_is_dead():
    """A group with only one valid sample (per sample_mask) is treated as dead."""
    rewards = _scalar_rewards([1.0, 3.0, 5.0, 7.0])
    masks = [torch.ones(1, dtype=torch.float32) for _ in range(4)]
    # Group 1 has only sample idx=3 valid.
    sample_mask = [
        torch.tensor(1.0),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(1.0),
    ]
    adv, dead = _calc_grpo_advantages_func_with_dead_mask(
        rewards=rewards,
        mask=masks,
        grpo_sampling_times=2,
        sample_mask=sample_mask,
        dead_threshold=0.01,
    )
    # Group 0: [1, 3] alive
    assert not dead[0].item() and not dead[1].item()
    # Group 1: only 1 valid sample -> dead for whole group (both indices)
    assert dead[2].item() and dead[3].item()
    # Dead-group advantages must be 0
    assert adv[2].item() == 0 and adv[3].item() == 0


# ===========================================================================
# 2. compute_gdpo_combined_advantages_with_dead_mask
# ===========================================================================


def test_combined_partial_dead_marks_sample_alive_if_any_dim_alive():
    """Sample alive in any reward dim -> all_dead=False; dead dims contribute 0."""
    # 4 samples, sampling_keep_n=4, so 1 group.
    # dim A: constant rewards -> dead. dim B: varying rewards -> alive.
    rewards_dict = {
        "dim_dead": _scalar_rewards([1.0, 1.0, 1.0, 1.0]),
        "dim_alive": _scalar_rewards([1.0, 2.0, 3.0, 4.0]),
    }
    masks = [torch.ones(1, dtype=torch.float32) for _ in range(4)]
    combined, all_dead = compute_gdpo_combined_advantages_with_dead_mask(
        rewards_dict=rewards_dict,
        mask=masks,
        grpo_sampling_times=4,
        grpo_advantage_epsilon=1e-8,
        gdpo_reward_weights={"dim_dead": 0.5, "dim_alive": 0.5},
        dead_threshold_cfg=0.01,
    )
    # any_dim_alive is True for every sample because dim_alive provides signal.
    assert not all_dead.any(), f"no sample should be all_dead; got {all_dead}"
    # combined comes purely from dim_alive's contribution (dim_dead -> 0).
    # Sum of combined within the single group should be ~0 (group-normalized).
    assert abs(combined.sum().item()) < 1e-4


def test_combined_all_dead_when_every_dim_is_dead():
    """When every reward dim has std=0, all_dead is True and combined is 0."""
    rewards_dict = {
        "dim_a": _scalar_rewards([0.0, 0.0, 0.0, 0.0]),
        "dim_b": _scalar_rewards([1.0, 1.0, 1.0, 1.0]),
    }
    masks = [torch.ones(1, dtype=torch.float32) for _ in range(4)]
    combined, all_dead = compute_gdpo_combined_advantages_with_dead_mask(
        rewards_dict=rewards_dict,
        mask=masks,
        grpo_sampling_times=4,
        gdpo_reward_weights={"dim_a": 1.0, "dim_b": 1.0},
        dead_threshold_cfg=0.01,
    )
    assert all_dead.all(), f"every sample should be all_dead; got {all_dead}"
    assert torch.equal(combined, torch.zeros(4))


# ===========================================================================
# 3. _resolve_dead_threshold (float / dict / fallback)
# ===========================================================================


def test_resolve_dead_threshold_float_and_dict_and_missing():
    # task_config is None -> default
    assert _resolve_dead_threshold(None, "any") == DEFAULT_DEAD_GROUP_THRESHOLD
    # task_config without the attribute -> default
    assert _resolve_dead_threshold(SimpleNamespace(), "any") == DEFAULT_DEAD_GROUP_THRESHOLD
    # explicit float
    task = SimpleNamespace(dead_group_threshold=0.05)
    assert _resolve_dead_threshold(task, "any") == 0.05
    # explicit dict with present key
    task = SimpleNamespace(dead_group_threshold={"r1": 0.1, "r2": 0.2})
    assert _resolve_dead_threshold(task, "r1") == 0.1
    assert _resolve_dead_threshold(task, "r2") == 0.2
    # dict with missing key falls back to default
    assert _resolve_dead_threshold(task, "missing") == DEFAULT_DEAD_GROUP_THRESHOLD
    # invalid type raises
    with pytest.raises(TypeError):
        _resolve_dead_threshold(SimpleNamespace(dead_group_threshold="bad"), "r1")


# ===========================================================================
# 4. compute_gdpo_sample_bn_dead_mask_advantages — rollout_batch mutation
# ===========================================================================


def test_advantage_entry_updates_rollout_batch_with_backups_and_zeroes_dead():
    """End-to-end check that the advantage entry mutates rollout_batch as designed."""
    # 4 samples, sampling_keep_n=4 -> 1 group, every reward dim has std=0
    # -> all_dead = True for every sample.
    rewards_dict = {
        "r_a": _scalar_rewards([1.0, 1.0, 1.0, 1.0]),
        "r_b": _scalar_rewards([2.0, 2.0, 2.0, 2.0]),
    }
    masks = [torch.ones(3, dtype=torch.float32) for _ in range(4)]
    sample_mask = [torch.tensor(1.0) for _ in range(4)]

    ppo_cfg = _make_ppo_config(
        gdpo_reward_weights={"r_a": 0.5, "r_b": 0.5},
    )
    cfg = _make_top_config(
        sampling_keep_n=4,
        ppo_cfg=ppo_cfg,
        task=SimpleNamespace(dead_group_threshold=0.01),
    )
    ctx = _make_advantage_context(
        config=cfg, rewards_dict=rewards_dict, masks=masks, sample_mask=sample_mask
    )

    # Capture references for in-place verification.
    rb = ctx.rollout_batch

    result = compute_gdpo_sample_bn_dead_mask_advantages(ctx)

    # pre_bn_advantages must be set; final advantages list left empty.
    assert result.advantages == []
    assert result.pre_bn_advantages is not None
    assert result.pre_bn_advantages.shape == (4,)
    assert torch.equal(result.pre_bn_advantages, torch.zeros(4))

    # rollout_batch must carry the new bookkeeping fields.
    # all_dead_mask is List[0-d bool tensor] to satisfy check_rollout_batch's
    # "every value is a list" contract (training_utils.py:97-131).
    assert "all_dead_mask" in rb and isinstance(rb["all_dead_mask"], list)
    assert all(t.dtype == torch.bool for t in rb["all_dead_mask"])
    assert all(bool(t.item()) for t in rb["all_dead_mask"])
    assert "original_sample_mask" in rb
    assert "original_mask_pre_dead" in rb
    # Backups preserve the *pre-zero* values.
    assert all(sm.item() == 1.0 for sm in rb["original_sample_mask"])
    assert all(m.sum().item() == 3.0 for m in rb["original_mask_pre_dead"])
    # In-place mutation: live sample_mask and mask are now all zero.
    assert all(sm.item() == 0.0 for sm in rb["sample_mask"])
    assert all(m.sum().item() == 0.0 for m in rb["mask"])


def test_advantage_entry_assert_rejects_non_grpo_loss_func():
    rewards_dict = {"r_a": _scalar_rewards([1.0, 2.0, 3.0, 4.0])}
    masks = [torch.ones(1, dtype=torch.float32) for _ in range(4)]
    ppo_cfg = _make_ppo_config(loss_func="gspo", gdpo_reward_weights={"r_a": 1.0})
    cfg = _make_top_config(sampling_keep_n=4, ppo_cfg=ppo_cfg)
    ctx = _make_advantage_context(config=cfg, rewards_dict=rewards_dict, masks=masks)
    with pytest.raises(AssertionError, match="loss_func=grpo"):
        compute_gdpo_sample_bn_dead_mask_advantages(ctx)


# ===========================================================================
# 5. gdpo_sample_bn_dead_mask_post_advantage — end-to-end BN excludes all_dead
# ===========================================================================


def test_post_advantage_excludes_all_dead_samples_from_bn(_single_rank_gloo_pg):
    """Global BN must skip all_dead samples (via sample_mask=0); their final
    advantages must be 0 and metrics must report dead_group_count/ratio."""
    # Construct two groups (sampling_keep_n=4):
    # - Group 0: rewards vary -> alive
    # - Group 1: rewards constant -> dead
    rewards_dict = {
        "r": _scalar_rewards([1.0, 2.0, 3.0, 4.0, 7.0, 7.0, 7.0, 7.0]),
    }
    masks = [torch.ones(2, dtype=torch.float32) for _ in range(8)]
    sample_mask = [torch.tensor(1.0) for _ in range(8)]

    ppo_cfg = _make_ppo_config(gdpo_reward_weights={"r": 1.0})
    cfg = _make_top_config(
        sampling_keep_n=4,
        ppo_cfg=ppo_cfg,
        task=SimpleNamespace(dead_group_threshold=0.01),
    )
    ctx = _make_advantage_context(
        config=cfg, rewards_dict=rewards_dict, masks=masks, sample_mask=sample_mask
    )

    adv_result = compute_gdpo_sample_bn_dead_mask_advantages(ctx)
    # Group 1 must be all_dead, Group 0 must not. all_dead_mask is now a
    # List[0-d bool tensor]; reduce to a 1-D tensor for ergonomic asserts.
    all_dead = torch.stack(ctx.rollout_batch["all_dead_mask"])
    assert not all_dead[:4].any()
    assert all_dead[4:].all()

    # Stash pre_bn_advantages onto rollout_batch (mimic mixin behaviour).
    ctx.rollout_batch["pre_bn_advantages"] = adv_result.pre_bn_advantages

    post_ctx = PostAdvantageContext(
        rollout_batches=[ctx.rollout_batch],
        config=cfg,
        dp_group=_single_rank_gloo_pg,
        num_samples=8,
    )
    post_result = gdpo_sample_bn_dead_mask_post_advantage(post_ctx)

    rb = ctx.rollout_batch
    assert "advantages" in rb and "returns" in rb
    # All-dead samples (indices 4..7) must yield 0 advantage everywhere.
    for i in range(4, 8):
        assert torch.equal(rb["advantages"][i], torch.zeros_like(rb["advantages"][i]))
    # Group 0 (alive samples) should have non-trivial advantages on the
    # subset of valid tokens (sample_mask still 1 for those samples).
    alive_token_vals = torch.cat([rb["advantages"][i] for i in range(4)])
    assert alive_token_vals.abs().sum().item() > 0, (
        "alive samples should have non-zero advantages after BN"
    )

    # Metrics: dead_group_count = 4, ratio metric is multiplied by num_samples=8
    # per the project's metrics-aggregator convention.
    assert post_result.metrics["ppo-metrics/dead_group_count"] == pytest.approx(4.0)
    assert post_result.metrics["ppo-metrics/dead_group_ratio"] == pytest.approx(
        (4.0 / 8.0) * 8
    )


def test_advantage_entry_synthesizes_sample_mask_when_none_provided():
    """When ctx.sample_mask is None, the entry synthesises ones-tensors and
    still flips them to 0 for all_dead samples; backups capture the synthetic
    ones for diagnostics."""
    rewards_dict = {"r": _scalar_rewards([1.0, 1.0, 1.0, 1.0])}
    masks = [torch.ones(2, dtype=torch.float32) for _ in range(4)]
    ppo_cfg = _make_ppo_config(gdpo_reward_weights={"r": 1.0})
    cfg = _make_top_config(
        sampling_keep_n=4,
        ppo_cfg=ppo_cfg,
        task=SimpleNamespace(dead_group_threshold=0.01),
    )
    ctx = _make_advantage_context(
        config=cfg, rewards_dict=rewards_dict, masks=masks, sample_mask=None
    )
    compute_gdpo_sample_bn_dead_mask_advantages(ctx)
    rb = ctx.rollout_batch
    # The synthesised sample_mask must now exist on the rollout batch and be all zero.
    assert "sample_mask" in rb and len(rb["sample_mask"]) == 4
    assert all(sm.item() == 0.0 for sm in rb["sample_mask"])
    # Backups capture the *pre*-zero synthetic ones.
    assert all(sm.item() == 1.0 for sm in rb["original_sample_mask"])


def test_advantage_entry_resolves_per_reward_dict_threshold():
    """End-to-end: dict-shaped task.dead_group_threshold drives per-dim decisions."""
    # 4 samples, sampling_keep_n=4. dim_a std~0.05; dim_b std~5.0.
    # With per-reward thresholds: r_a's threshold 0.1 declares it dead;
    # r_b's threshold 0.001 keeps it alive.
    rewards_dict = {
        "r_a": _scalar_rewards([1.00, 1.05, 1.00, 1.05]),
        "r_b": _scalar_rewards([0.0, 5.0, 10.0, 15.0]),
    }
    masks = [torch.ones(1, dtype=torch.float32) for _ in range(4)]
    ppo_cfg = _make_ppo_config(gdpo_reward_weights={"r_a": 1.0, "r_b": 1.0})
    cfg = _make_top_config(
        sampling_keep_n=4,
        ppo_cfg=ppo_cfg,
        task=SimpleNamespace(dead_group_threshold={"r_a": 0.1, "r_b": 0.001}),
    )
    ctx = _make_advantage_context(
        config=cfg, rewards_dict=rewards_dict, masks=masks,
        sample_mask=[torch.tensor(1.0) for _ in range(4)],
    )
    result = compute_gdpo_sample_bn_dead_mask_advantages(ctx)
    # r_b is alive -> any_dim_alive = True -> all_dead = False for every sample.
    all_dead_any = torch.stack(ctx.rollout_batch["all_dead_mask"]).any()
    assert not bool(all_dead_any.item())
    # combined advantages come purely from r_b's contribution; sum within the
    # single group should be ~0 (group-normalized).
    assert abs(result.pre_bn_advantages.sum().item()) < 1e-4


def test_post_advantage_applies_advantage_clip_bounds(_single_rank_gloo_pg):
    """Confirm the clip path stashes original_advantages and clamps the final ones."""
    # 4 samples, sampling_keep_n=4, alive group with large spread to drive
    # post-BN advantages outside the clip range.
    rewards_dict = {"r": _scalar_rewards([0.0, 1.0, 2.0, 3.0])}
    masks = [torch.ones(1, dtype=torch.float32) for _ in range(4)]
    ppo_cfg = _make_ppo_config(
        gdpo_reward_weights={"r": 1.0},
        advantage_clip=0.5,  # symmetric clip to [-0.5, 0.5]
    )
    cfg = _make_top_config(
        sampling_keep_n=4,
        ppo_cfg=ppo_cfg,
        task=SimpleNamespace(dead_group_threshold=0.01),
    )
    ctx = _make_advantage_context(
        config=cfg, rewards_dict=rewards_dict, masks=masks,
        sample_mask=[torch.tensor(1.0) for _ in range(4)],
    )
    adv_result = compute_gdpo_sample_bn_dead_mask_advantages(ctx)
    ctx.rollout_batch["pre_bn_advantages"] = adv_result.pre_bn_advantages

    post_ctx = PostAdvantageContext(
        rollout_batches=[ctx.rollout_batch],
        config=cfg,
        dp_group=_single_rank_gloo_pg,
        num_samples=4,
    )
    gdpo_sample_bn_dead_mask_post_advantage(post_ctx)
    rb = ctx.rollout_batch
    # Clip path must populate the original_advantages backup.
    assert "original_advantages" in rb
    # Final advantages are inside [-0.5, 0.5].
    for adv in rb["advantages"]:
        assert (adv >= -0.5).all() and (adv <= 0.5).all()
    # Pre-clip values had at least one entry outside the clip range
    # (group-normalized scaled by ~1.0 std fall outside ±0.5).
    flat_orig = torch.cat([a.reshape(-1) for a in rb["original_advantages"]])
    assert (flat_orig.abs() > 0.5).any()
