# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Custom GDPO sample-level batch-normalization variant with *dead group masking*.

This module is registered into the gcore advantage dispatch via the YAML triple::

    ppo:
      loss_func: "grpo"
      advantage_type: "custom_gdpo_sample_bn_dead_mask"
      custom_advantage_py_path: "tasks/math_rl_v4/custom_gdpo_dead_mask_advantage.py"
      custom_advantage_py_name: "compute_gdpo_sample_bn_dead_mask_advantages"
      custom_post_advantage_py_name: "gdpo_sample_bn_dead_mask_post_advantage"
    task:
      dead_group_threshold: 0.01          # float or dict[reward_name, float]
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed

from gpatch_v4.core import (
    AdvantageContext,
    AdvantageResult,
    PostAdvantageContext,
    PostAdvantageResult,
)
from gpatch_v4.utils import log
from gpatch_v4.utils.ppo_utils import get_advantage_clip_bounds


DEFAULT_DEAD_GROUP_THRESHOLD: float = 0.01


def get_current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def resolve_dead_threshold(
    task_config: Any,
    reward_name: str,
    default: float = DEFAULT_DEAD_GROUP_THRESHOLD,
) -> float:
    if task_config is None:
        return default
    assert "dead_group_threshold" in task_config
    cfg_val = getattr(task_config, "dead_group_threshold", None)
    if cfg_val is None:
        return default
    if isinstance(cfg_val, (int, float)):
        return float(cfg_val)
    if isinstance(cfg_val, dict):
        if reward_name in cfg_val:
            return float(cfg_val[reward_name])
        return default
    raise TypeError(
        f"task.dead_group_threshold must be float or dict[str, float]; "
        f"got {type(cfg_val).__name__}: {cfg_val!r}"
    )


def calc_grpo_advantages_func_with_dead_mask(
    rewards: List[torch.Tensor],
    mask: List[torch.Tensor],
    grpo_sampling_times: int = 1,
    grpo_advantage_epsilon: float = 1e-6,
    sample_mask: Optional[List[torch.Tensor]] = None,
    dead_threshold: float = DEFAULT_DEAD_GROUP_THRESHOLD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert grpo_sampling_times > 0
    for reward in rewards:
        assert reward.numel() == 1
    scores = torch.stack([reward.sum() for reward in rewards]).view(-1)
    assert scores.shape[0] == len(rewards)
    assert scores.shape[0] % grpo_sampling_times == 0, (
        f"num samples {scores.shape[0]} must be divisible by "
        f"grpo_sampling_times {grpo_sampling_times}"
    )

    grouped_scores = scores.view(-1, grpo_sampling_times)
    num_groups = grouped_scores.shape[0]

    if sample_mask is not None:
        assert len(sample_mask) == scores.shape[0]
        grouped_sample_mask = torch.stack(sample_mask).view(-1, grpo_sampling_times)
    else:
        grouped_sample_mask = torch.ones_like(grouped_scores, dtype=torch.bool)

    mean_grouped: List[float] = []
    std_grouped: List[float] = []
    group_dead: List[bool] = []
    for g_scores, g_sample_mask in zip(grouped_scores, grouped_sample_mask):
        masked_scores = g_scores[g_sample_mask.bool()]
        if masked_scores.numel() <= 1:
            # Empty or single-sample group: std is undefined; mark dead.
            mean_grouped.append(0.0)
            std_grouped.append(1.0)
            group_dead.append(True)
            continue
        g_mean = masked_scores.mean().item()
        g_std = masked_scores.std().item()
        mean_grouped.append(g_mean)
        std_grouped.append(g_std)
        group_dead.append(g_std < dead_threshold)

    mean_t = torch.tensor(mean_grouped, dtype=scores.dtype, device=scores.device)
    std_t = torch.tensor(std_grouped, dtype=scores.dtype, device=scores.device)
    dead_t = torch.tensor(group_dead, dtype=torch.bool, device=scores.device)
    assert mean_t.shape[0] == num_groups
    assert dead_t.shape[0] == num_groups

    mean_t = mean_t.repeat_interleave(grpo_sampling_times, dim=0)
    std_t = std_t.repeat_interleave(grpo_sampling_times, dim=0)
    dead_t = dead_t.repeat_interleave(grpo_sampling_times, dim=0)

    advantages = (scores - mean_t) / (std_t + grpo_advantage_epsilon)
    # Zero out dead-group contributions in this dimension.
    advantages = advantages.masked_fill(dead_t, 0.0)
    return advantages, dead_t


def compute_gdpo_combined_advantages_with_dead_mask(
    rewards_dict: Dict[str, List[torch.Tensor]],
    mask: List[torch.Tensor],
    grpo_sampling_times: int = 1,
    grpo_advantage_epsilon: float = 1e-6,
    gdpo_reward_weights: Optional[Dict[str, float]] = None,
    sample_mask: Optional[List[torch.Tensor]] = None,
    dead_threshold_cfg: Union[float, Dict[str, float], None] = None,
    task_config: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert gdpo_reward_weights is not None, "gdpo_reward_weights is required"
    assert isinstance(gdpo_reward_weights, dict), "gdpo_reward_weights must be a dict"
    assert len(rewards_dict) > 0, "rewards_dict must not be empty"

    combined: Optional[torch.Tensor] = None
    any_dim_alive: Optional[torch.Tensor] = None
    for reward_name, rewards in rewards_dict.items():
        weight = float(gdpo_reward_weights.get(reward_name, 1.0))

        # Resolve threshold: explicit cfg override wins, else read task_config.
        if dead_threshold_cfg is not None:
            if isinstance(dead_threshold_cfg, (int, float)):
                threshold = float(dead_threshold_cfg)
            elif isinstance(dead_threshold_cfg, dict):
                threshold = float(
                    dead_threshold_cfg.get(reward_name, DEFAULT_DEAD_GROUP_THRESHOLD)
                )
            else:
                raise TypeError(
                    f"dead_threshold_cfg must be float | dict | None; "
                    f"got {type(dead_threshold_cfg).__name__}"
                )
        else:
            threshold = resolve_dead_threshold(task_config, reward_name)

        adv, dead_mask = calc_grpo_advantages_func_with_dead_mask(
            rewards=rewards,
            mask=mask,
            grpo_sampling_times=grpo_sampling_times,
            grpo_advantage_epsilon=grpo_advantage_epsilon,
            sample_mask=sample_mask,
            dead_threshold=threshold,
        )

        weighted = adv * weight
        if combined is None:
            combined = weighted
            any_dim_alive = ~dead_mask
        else:
            combined = combined + weighted
            any_dim_alive = any_dim_alive | (~dead_mask)

    assert combined is not None and any_dim_alive is not None
    all_dead = ~any_dim_alive
    return combined, all_dead


def compute_gdpo_sample_bn_dead_mask_advantages(ctx: AdvantageContext) -> AdvantageResult:
    gdpo_reward_weights = ctx.config.ppo.gdpo_reward_weights
    assert gdpo_reward_weights, (
        "gdpo_reward_weights must be configured (non-empty dict) for "
        "custom_gdpo_sample_bn_dead_mask"
    )

    rollout_batch = ctx.rollout_batch
    rewards_dict: Dict[str, List[torch.Tensor]] = {}
    for reward_name in gdpo_reward_weights:
        assert reward_name in rollout_batch, (
            f"reward dimension {reward_name!r} (declared in gdpo_reward_weights) "
            f"not present in rollout_batch; available keys: {list(rollout_batch.keys())}"
        )
        rewards_dict[reward_name] = rollout_batch[reward_name]

    task_config = getattr(ctx.config, "task", None)
    combined, all_dead = compute_gdpo_combined_advantages_with_dead_mask(
        rewards_dict=rewards_dict,
        mask=ctx.mask,
        grpo_sampling_times=ctx.config.training.sampling_keep_n,
        grpo_advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
        gdpo_reward_weights=gdpo_reward_weights,
        sample_mask=ctx.sample_mask,
        task_config=task_config,
    )
    n = combined.shape[0]
    assert all_dead.shape[0] == n
    assert len(ctx.mask) == n

    sample_mask = ctx.sample_mask
    if sample_mask is None:
        ref = ctx.mask[0]
        sample_mask = [
            torch.ones((), dtype=ref.dtype, device=ref.device) for _ in range(n)
        ]

    original_sample_mask = [sm.clone() for sm in sample_mask]
    original_mask = [m.clone() for m in ctx.mask]

    dead_count = 0
    for i in range(n):
        if bool(all_dead[i].item()):
            sample_mask[i] = sample_mask[i] * 0
            ctx.mask[i] = ctx.mask[i] * 0
            dead_count += 1

    rollout_batch["mask"] = ctx.mask
    rollout_batch["sample_mask"] = sample_mask
    rollout_batch["original_sample_mask"] = original_sample_mask
    rollout_batch["original_mask_pre_dead"] = original_mask
    all_dead_bool = all_dead.to(dtype=torch.bool)
    rollout_batch["all_dead_mask"] = [
        all_dead_bool[i].clone() for i in range(n)
    ]

    if dead_count > 0:
        log(
            f"[gdpo_sample_bn_dead_mask] zeroed {dead_count}/{n} all-dead samples in "
            f"this micro-batch (advantage / mask / sample_mask all forced to 0)",
            rank=0,
        )

    return AdvantageResult(
        advantages=[],
        pre_bn_advantages=combined,
    )


def gdpo_sample_bn_dead_mask_post_advantage(
    ctx: PostAdvantageContext,
) -> PostAdvantageResult:
    rollout_batches = ctx.rollout_batches
    assert "pre_bn_advantages" in rollout_batches[0], (
        "gdpo_sample_bn_dead_mask_post_advantage requires pre_bn_advantages"
    )
    epsilon = ctx.config.ppo.grpo_advantage_epsilon
    device = get_current_device()

    all_pre_bn = torch.cat([rb["pre_bn_advantages"] for rb in rollout_batches])
    all_sample_masks: List[torch.Tensor] = []
    for rb in rollout_batches:
        sm = rb.get("sample_mask", None)
        if sm is not None:
            all_sample_masks.extend(sm)
    has_sample_mask = len(all_sample_masks) > 0

    if has_sample_mask:
        valid_mask = torch.stack(all_sample_masks).to(
            device=all_pre_bn.device, dtype=all_pre_bn.dtype
        )
    else:
        valid_mask = torch.ones_like(all_pre_bn)

    local_sum = (all_pre_bn * valid_mask).sum()
    local_count = valid_mask.sum()
    sum_and_count = torch.tensor(
        [local_sum.item(), local_count.item()],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(sum_and_count, group=ctx.dp_group)
    global_count = sum_and_count[1].item()
    if global_count > 0:
        global_mean = sum_and_count[0].item() / global_count
    else:
        global_mean = 0.0

    # ----- global std via all_reduce -----
    local_var_sum = (((all_pre_bn - global_mean) ** 2) * valid_mask).sum()
    var_sum_tensor = torch.tensor(
        [local_var_sum.item()],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(var_sum_tensor, group=ctx.dp_group)
    if global_count > 1:
        global_std = (var_sum_tensor[0].item() / (global_count - 1)) ** 0.5
    else:
        global_std = 1.0

    advantage_clip_bounds = get_advantage_clip_bounds(
        ctx.config.ppo.advantage_clip,
        ctx.config.ppo.advantage_clip_lower_bound,
        ctx.config.ppo.advantage_clip_upper_bound,
    )
    all_normalized: List[torch.Tensor] = []
    all_norm_masks: List[torch.Tensor] = []
    for rollout_batch in rollout_batches:
        mask = rollout_batch["mask"]
        n = len(mask)
        pre_bn = rollout_batch.pop("pre_bn_advantages")
        assert pre_bn.shape[0] == n, f"pre_bn.shape[0] {pre_bn.shape[0]} != n {n}"

        normalized = (pre_bn - global_mean) / (global_std + epsilon)

        sm = rollout_batch.get("sample_mask", None)
        if sm is not None:
            sm_tensor = torch.stack(sm).to(device=normalized.device, dtype=normalized.dtype)
            normalized = normalized * sm_tensor
            all_norm_masks.append(sm_tensor)
        else:
            all_norm_masks.append(
                torch.ones(n, device=normalized.device, dtype=normalized.dtype)
            )
        all_normalized.append(normalized)

        advantages = [adv.expand(m.shape[-1]) * m for adv, m in zip(normalized, mask)]
        assert advantages[0].dtype == torch.float32
        if advantage_clip_bounds is not None:
            clip_lo, clip_hi = advantage_clip_bounds
            rollout_batch["original_advantages"] = advantages
            advantages = [a.clamp(min=clip_lo, max=clip_hi) for a in advantages]
        rollout_batch["advantages"] = advantages
        rollout_batch["returns"] = advantages

    # ----- normalized stats (pre-expand) for observability -----
    all_norm = torch.cat(all_normalized)
    all_nm = torch.cat(all_norm_masks)
    local_norm_sum = (all_norm * all_nm).sum().item()
    local_norm_count = all_nm.sum().item()
    norm_sc = torch.tensor(
        [local_norm_sum, local_norm_count],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(norm_sc, group=ctx.dp_group)
    global_norm_count = norm_sc[1].item()
    norm_mean = norm_sc[0].item() / global_norm_count if global_norm_count > 0 else 0.0

    local_norm_var_sum = (((all_norm - norm_mean) ** 2) * all_nm).sum().item()
    norm_var_t = torch.tensor(
        [local_norm_var_sum],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(norm_var_t, group=ctx.dp_group)
    norm_std = (
        (norm_var_t[0].item() / (global_norm_count - 1)) ** 0.5
        if global_norm_count > 1
        else 0.0
    )

    # ----- dead-group metrics (local then all_reduce) -----
    local_total = 0.0
    local_dead = 0.0
    for rb in rollout_batches:
        all_dead = rb.get("all_dead_mask", None)
        if all_dead is None:
            continue
        if isinstance(all_dead, list):
            local_total += float(len(all_dead))
            local_dead += float(sum(bool(t.item()) for t in all_dead))
        else:
            local_total += float(all_dead.numel())
            local_dead += float(all_dead.sum().item())
    dead_sc = torch.tensor([local_dead, local_total], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(dead_sc, group=ctx.dp_group)
    global_dead = dead_sc[0].item()
    global_total = dead_sc[1].item()
    dead_ratio = (global_dead / global_total) if global_total > 0 else 0.0

    metrics: Dict[str, float] = {
        "ppo-metrics/global_bn_mean": global_mean * ctx.num_samples,
        "ppo-metrics/global_bn_std": global_std * ctx.num_samples,
        "ppo-metrics/normalized_advantages_mean": norm_mean * ctx.num_samples,
        "ppo-metrics/normalized_advantages_std": norm_std * ctx.num_samples,
        "ppo-metrics/dead_group_count": global_dead,
        "ppo-metrics/dead_group_ratio": dead_ratio * ctx.num_samples,
    }
    return PostAdvantageResult(rollout_batches=rollout_batches, metrics=metrics)
