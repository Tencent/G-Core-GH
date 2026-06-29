# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from typing import Dict, List, cast

import numpy as np
import torch

from gpatch_v4.utils import log

safezip = cast(type[zip], lambda *args, **kwargs: zip(*args, **kwargs, strict=True))


def mask_single_valid_sample_groups(
    mask: List[torch.Tensor],
    sample_mask: List[torch.Tensor],
    sampling_keep_n: int,
) -> None:
    """Discard GRPO groups with exactly one valid sample.

    ``torch.std`` is undefined for a single valid sample. Mutating the masks here
    keeps advantage implementations from seeing that invalid group.
    """
    assert sampling_keep_n > 0
    assert 0 == len(sample_mask) % sampling_keep_n
    for group_start in range(0, len(sample_mask), sampling_keep_n):
        group_end = group_start + sampling_keep_n
        group_valid_indices = [
            i for i in range(group_start, group_end) if sample_mask[i].bool().item()
        ]
        if 1 == len(group_valid_indices):
            mask[group_valid_indices[0]].zero_()
            sample_mask[group_valid_indices[0]].zero_()


def _calc_grpo_advantages_func(
    rewards: List[torch.Tensor],
    mask: List[torch.Tensor],
    grpo_sampling_times=1,
    grpo_advantage_epsilon=1e-6,
    sample_mask: List[torch.Tensor] = None,
    **kwargs,
):
    """Compute GRPO group-normalized advantages from scalar rewards.

    Parameters
    ----------
    rewards : list of torch.Tensor
    mask : list of torch.Tensor
    grpo_sampling_times : int, optional
    grpo_advantage_epsilon : float, optional
    sample_mask : list of torch.Tensor, optional
        Per-sample validity mask (0 or 1). 0 means invalid, 1 means valid.

    Returns
    """
    for reward in rewards:
        assert reward.numel() == 1
    scores = [reward.sum() for reward in rewards]
    scores = torch.stack(scores).view(-1)
    # Compute grouped-wise rewards
    if sample_mask is not None:
        mean_grouped_rewards = []
        std_grouped_rewards = []
        grouped_scores = scores.view(-1, grpo_sampling_times)
        grouped_sample_mask = torch.stack(sample_mask).view(-1, grpo_sampling_times)
        for g_scores, g_sample_mask in zip(grouped_scores, grouped_sample_mask):
            masked_scores = g_scores[g_sample_mask.bool()]
            assert masked_scores.shape[0
                                      ] != 1, "only 1 valid sample in the group, will get NAN loss"
            is_empty_group = 0 == masked_scores.shape[0]
            # For empty groups, use mean=0, std=1 as safe defaults;
            # downstream mask multiplication will zero out their contribution.
            mean_grouped_rewards.append(0 if is_empty_group else masked_scores.mean().item())
            std_grouped_rewards.append(1 if is_empty_group else masked_scores.std().item())
        mean_grouped_rewards = torch.tensor(
            mean_grouped_rewards, dtype=scores.dtype, device=scores.device
        )
        std_grouped_rewards = torch.tensor(
            std_grouped_rewards, dtype=scores.dtype, device=scores.device
        )
    else:
        mean_grouped_rewards = scores.view(-1, grpo_sampling_times).mean(dim=1)
        std_grouped_rewards = scores.view(-1, grpo_sampling_times).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(grpo_sampling_times, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(grpo_sampling_times, dim=0)
    advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + grpo_advantage_epsilon)
    return advantages


def calculate_grpo_advantages(
    rewards: List[torch.Tensor],
    mask: List[torch.Tensor],
    grpo_sampling_times=1,
    grpo_advantage_epsilon=1e-6,
    **kwargs,
):
    """Compute GRPO group-normalized advantages from scalar rewards.

    Parameters
    ----------
    rewards : list of torch.Tensor
    mask : list of torch.Tensor
    grpo_sampling_times : int, optional
    grpo_advantage_epsilon : float, optional

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor]]
        ``(advantages, returns)``.
    """
    advantages = _calc_grpo_advantages_func(
        rewards, mask, grpo_sampling_times, grpo_advantage_epsilon, **kwargs
    )
    advantages_mask = []
    for advantage, m in zip(advantages.chunk(len(rewards)), mask):
        assert m.ndim == 1
        advantages_mask.append(advantage.tile([m.shape[-1]]) * m)

    return advantages_mask, advantages_mask


def calculate_identity_advantages(
    rewards: List[torch.Tensor],
    mask: List[torch.Tensor],
):
    """Identity advantage — raw reward tiled to each token position, no normalization.

    Parameters
    ----------
    rewards : list of torch.Tensor
        Scalar rewards, one per sample.
    mask : list of torch.Tensor
        Response mask per sample.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor]]
        ``(advantages, returns)``, both are raw reward tiled to token positions.
    """
    advantages_mask = []
    for reward, m in zip(rewards, mask):
        assert reward.numel() == 1
        assert m.ndim == 1
        adv = reward.sum().tile([m.shape[-1]]) * m
        advantages_mask.append(adv)

    return advantages_mask, advantages_mask


def calculate_reinforce_advantages(
    rewards: List[torch.Tensor],
    mask: List[torch.Tensor],
    gamma: float = 1.0,
):
    """Compute REINFORCE-style discounted return-to-go per token.

    ``reward`` can be either:
    - scalar ``(numel==1)``: assigned to the last valid token in ``mask``;
    - per-token tensor with the same length as ``mask``.
    """
    gamma_v = float(gamma)
    advantages_mask = []

    for reward, m in zip(rewards, mask):
        assert m.ndim == 1
        token_rewards = torch.zeros_like(m, dtype=torch.float32)

        if reward.numel() == 1:
            # 如果 reward 是标量，则将 reward 分配给最后一个有效 token
            valid_idx = torch.nonzero(m.bool(), as_tuple=False).view(-1)
            if valid_idx.numel() > 0:
                token_rewards[valid_idx[-1]] = reward.sum().to(torch.float32)
        elif reward.numel() == m.numel():
            # 如果 进来的 reward 是 token level 的，则直接赋值
            token_rewards = reward.reshape_as(m).to(torch.float32)
        else:
            raise ValueError(
                f"reinforce advantages expect scalar reward or len(mask) reward, "
                f"got reward.numel={reward.numel()} and mask.numel={m.numel()}"
            )

        advantages = torch.zeros_like(token_rewards, dtype=torch.float32)
        cumulative_reward = torch.zeros((), dtype=torch.float32, device=token_rewards.device)
        for t in reversed(range(token_rewards.shape[-1])):
            local_reward = token_rewards[t] if m[t].bool() else torch.zeros_like(cumulative_reward)
            cumulative_reward = local_reward + gamma_v * cumulative_reward
            if m[t].bool():
                advantages[t] = cumulative_reward

        advantages_mask.append(advantages * m.to(torch.float32))

    return advantages_mask, advantages_mask


def compute_gdpo_combined_advantages(
    rewards_dict: Dict[str, List[torch.Tensor]],
    mask: List[torch.Tensor],
    grpo_sampling_times: int = 1,
    grpo_advantage_epsilon: float = 1e-6,
    gdpo_reward_weights: Dict[str, float] = None,
    sample_mask: List[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-group normalized, weighted-combined advantages WITHOUT batch normalization.

    Each reward dimension is independently group-normalized, then dimensions
    are combined with weights. Returns a 1-D tensor of per-sample scalars.

    Parameters
    ----------
    rewards_dict : dict[str, list[torch.Tensor]]
        Per-dimension scalar rewards keyed by reward name.
    mask : list of torch.Tensor
    grpo_sampling_times : int
    grpo_advantage_epsilon : float
    gdpo_reward_weights : dict[str, float]
    sample_mask : list of torch.Tensor or None

    Returns
    -------
    torch.Tensor
        1-D tensor of shape ``(num_samples,)`` with combined advantages (pre-BN).
    """
    assert gdpo_reward_weights is not None, "gdpo_reward_weights is required"
    assert isinstance(gdpo_reward_weights, dict), "gdpo_reward_weights must be a dict"
    combined = None
    for reward_name, rewards in rewards_dict.items():
        weight = gdpo_reward_weights.get(reward_name, 1.0)
        adv = _calc_grpo_advantages_func(
            rewards, mask, grpo_sampling_times, grpo_advantage_epsilon, sample_mask=sample_mask
        )
        if combined is None:
            combined = adv * weight
        else:
            combined = combined + adv * weight
    assert combined is not None, "rewards_dict must not be empty"
    return combined


def calculate_gdpo_advantages(
    rewards_dict: Dict[str, List[torch.Tensor]],
    mask: List[torch.Tensor],
    grpo_sampling_times=1,
    grpo_advantage_epsilon=1e-6,
    **kwargs,
):
    combined_advantages = compute_gdpo_combined_advantages(
        rewards_dict=rewards_dict,
        mask=mask,
        grpo_sampling_times=grpo_sampling_times,
        grpo_advantage_epsilon=grpo_advantage_epsilon,
        gdpo_reward_weights=kwargs["gdpo_reward_weights"],
        sample_mask=kwargs.get("sample_mask", None),
    )

    expanded_advantages = [adv.expand(m.shape[-1]) * m for adv, m in zip(combined_advantages, mask)]
    valid_values = torch.cat([adv[m.bool()] for adv, m in zip(expanded_advantages, mask)])

    if len(valid_values) > 0:
        bn_mean = valid_values.mean()
        bn_std = valid_values.std()
    else:
        bn_mean = 0.0
        bn_std = 1.0

    masked_advantages = [
        ((adv - bn_mean) / (bn_std + grpo_advantage_epsilon)) * m
        for adv, m in zip(expanded_advantages, mask)
    ]

    return masked_advantages, masked_advantages


def calculate_gdpo_sample_bn_advantages(
    rewards_dict: Dict[str, List[torch.Tensor]],
    mask: List[torch.Tensor],
    grpo_sampling_times=1,
    grpo_advantage_epsilon=1e-6,
    **kwargs,
):
    combined_advantages = compute_gdpo_combined_advantages(
        rewards_dict=rewards_dict,
        mask=mask,
        grpo_sampling_times=grpo_sampling_times,
        grpo_advantage_epsilon=grpo_advantage_epsilon,
        gdpo_reward_weights=kwargs["gdpo_reward_weights"],
        sample_mask=kwargs.get("sample_mask", None),
    )

    sample_mask = kwargs.get("sample_mask", None)
    if sample_mask is None:
        valid_values = combined_advantages
    else:
        sample_mask_tensor = torch.stack(sample_mask).to(
            device=combined_advantages.device,
            dtype=torch.bool,
        )
        valid_values = combined_advantages[sample_mask_tensor]

    if valid_values.numel() > 1:
        bn_mean = valid_values.mean()
        bn_std = valid_values.std()
    elif valid_values.numel() == 1:
        bn_mean = valid_values.mean()
        bn_std = torch.ones_like(bn_mean)
    else:
        bn_mean = torch.zeros(
            (), dtype=combined_advantages.dtype, device=combined_advantages.device
        )
        bn_std = torch.ones((), dtype=combined_advantages.dtype, device=combined_advantages.device)

    normalized_advantages = (combined_advantages - bn_mean) / (bn_std + grpo_advantage_epsilon)
    assert normalized_advantages.ndim == 1 and normalized_advantages.shape[0] == len(mask)
    if sample_mask is not None:
        normalized_advantages = normalized_advantages * sample_mask_tensor.to(
            dtype=normalized_advantages.dtype
        )

    n_samples = len(mask)
    num_zero_combined = (combined_advantages == 0).sum().item()
    valid_norm_adv = normalized_advantages[sample_mask_tensor
                                          ] if sample_mask is not None else normalized_advantages
    metrics = {
        "sample_normalized_advantages_mean":
            valid_norm_adv.mean().item() * n_samples,
        "sample_normalized_advantages_std":
            valid_norm_adv.std().item() * n_samples if valid_norm_adv.numel() > 1 else 0.0,
        "sample_bn_mean":
            bn_mean.item() * n_samples,
        "sample_bn_std":
            bn_std.item() * n_samples,
        "num_zero_combined_adv":
            num_zero_combined,
        "bn_std_is_zero":
            float(bn_std.item() == 0) * n_samples,
    }

    masked_advantages = [adv.expand(m.shape[-1]) * m for adv, m in zip(normalized_advantages, mask)]
    return masked_advantages, masked_advantages, metrics


def calculate_ppo_rewards(
    values, rewards, per_token_rewards, sequence_lengths, init_policy_kl, penalty_factor=0.0
):
    """Compute per-token PPO rewards (final-token reward + KL penalty).

    Parameters
    ----------
    values : torch.Tensor
    rewards : torch.Tensor
        Scalar reward.
    per_token_rewards : torch.Tensor or None
    sequence_lengths : torch.Tensor
    init_policy_kl : torch.Tensor
        KL penalty per token.
    penalty_factor : float, optional

    Returns
    -------
    torch.Tensor
        Per-token reward signal.
    """
    rewards_sequence = torch.zeros_like(values)

    idx = (sequence_lengths - 2).clamp(min=0, max=None)

    assert rewards_sequence.ndim == 1
    assert idx.numel() == 1
    assert rewards.numel() == 1

    rewards_sequence[idx] = rewards.flatten()

    if per_token_rewards is not None:
        rewards_sequence += per_token_rewards

    return rewards_sequence - penalty_factor * init_policy_kl


def calculate_ppo_advantages_and_returns(
    values,
    rewards,
    discount_factor,
    gae_lambda,
    mask=None,
    per_token_rewards=None,
    per_token_rewards_factor=1.0,
):
    """Compute GAE advantages and returns for the entire sequence.

    Parameters
    ----------
    values : torch.Tensor
        Value estimates of shape ``(S-1,)``.
    rewards : torch.Tensor
        Per-token rewards of shape ``(S-1,)``.
    discount_factor : float
    gae_lambda : float
    mask : torch.Tensor, optional
    per_token_rewards : torch.Tensor, optional
        Additional per-token rewards to add to advantages.
    per_token_rewards_factor : float, optional

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(advantages, returns)``.
    """
    assert mask is not None

    last_gae_lam = 0
    next_values = 0.0
    advantages = torch.zeros_like(rewards)
    max_seq_len = values.size(-1)

    for i in reversed(range(max_seq_len)):
        delta = rewards[i] + discount_factor * next_values - values[i]
        last_gae_lam_ = delta + discount_factor * gae_lambda * last_gae_lam

        # skip values and TD-error on observation tokens
        m = mask[i]
        next_values = values[i] * m + (1 - m) * next_values
        last_gae_lam = last_gae_lam_ * m + (1 - m) * last_gae_lam
        advantages[i] = last_gae_lam

    if per_token_rewards is not None:
        advantages += (per_token_rewards * per_token_rewards_factor)

    returns = advantages + values
    return advantages, returns


def discounted_future_sum_vectorized(x: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted sum of future values using scipy ``lfilter``.

    Parameters
    ----------
    x : np.ndarray
        1-D array of rewards.
    gamma : float

    Returns
    -------
    np.ndarray
        Discounted cumulative sums.
    """
    # Reverse x so lfilter processes from end to start
    import scipy.signal

    return scipy.signal.lfilter([1], [1, -gamma], x[::-1],
                                axis=0)[::-1].astype(x.dtype)  # type: ignore


def calculate_reverse_kl_advantages(
    rewards: List[torch.Tensor],
    mask_lst: List[torch.Tensor],
    logprobs: List[torch.Tensor],
    teacher_logprobs: List[torch.Tensor],
    sampling_repeat_n=1,
    advantage_epsilon=1e-6,
    kl_penalty_coef=1.0,
    kl_discount_factor=0.0,
):
    """Compute advantages using reverse KL (student vs teacher).

    Parameters
    ----------
    rewards : list of torch.Tensor or None
    mask_lst : list of torch.Tensor
    logprobs : list of torch.Tensor
        Student log probs.
    teacher_logprobs : list of torch.Tensor
    sampling_repeat_n : int, optional
    advantage_epsilon : float, optional
    kl_penalty_coef : float, optional
    kl_discount_factor : float, optional

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor], dict]
        ``(advantages, returns, metrics)``.
    """

    #TODO：和 thinker 有点不一样的事算reward 的 advantage 没有除以方差
    # https://github.com/thinking-machines-lab/tinker-cookbook/blob/c1b3b6fe48dd9105f063e8e1cf0eb1ae036a7ad8/tinker_cookbook/rl/data_processing.py#L27
    if rewards is not None:
        advantages, returns = calculate_grpo_advantages(
            rewards=rewards,
            mask=mask_lst,
            grpo_sampling_times=sampling_repeat_n,
            grpo_advantage_epsilon=advantage_epsilon,
        )
    else:
        advantages = [torch.zeros_like(logprobs[i]) for i in range(len(logprobs))]
        returns = [torch.zeros_like(logprobs[i]) for i in range(len(logprobs))]

    # pad teacher_logprobs to match student_logprobs
    student_len = logprobs[0].shape[-1]
    for i in range(len(teacher_logprobs)):
        assert teacher_logprobs[i].shape[
            -1] <= student_len, f"{teacher_logprobs[i].shape[-1]=} > {student_len=}"
        teacher_logprobs[i] = torch.nn.functional.pad(
            teacher_logprobs[i],
            (0, student_len - teacher_logprobs[i].shape[-1]),
            value=0,
        )

    reverse_kl = [
        (student_logps - teacher_logps) * mask
        for (teacher_logps, student_logps, mask) in safezip(teacher_logprobs, logprobs, mask_lst)
    ]

    assert len(advantages) == len(reverse_kl), f"{len(advantages)=} != {len(reverse_kl)=}"
    for i in range(len(advantages)):
        kl_advantages = -kl_penalty_coef * reverse_kl[i] * mask_lst[i]
        if kl_discount_factor > 0:
            #TODO: tinker 说 discount_factor > 0 效果并不好，所以这里暂时不测试，
            # https://thinkingmachines.ai/blog/on-policy-distillation/
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )
        advantages[i] = advantages[i] + kl_advantages

    # Compute average reverse KL over the batch for logging
    # diff 已经 mask 过了
    avg_per_log_diff = sum(diff.sum() / m.sum()
                           for diff, m in zip(reverse_kl, mask_lst)) / len(mask_lst)
    avg_logp_diff = sum([diff.sum()
                         for diff in reverse_kl]) / sum([mask.sum() for mask in mask_lst])
    metrics = {
        "avg_per_sample_teacher_kl": avg_per_log_diff.item(),
        "teacher_kl": avg_logp_diff.item(),
    }
    return advantages, returns, metrics


def _pad_logprobs_to_target_len(
    logprobs_lst: List[torch.Tensor], target_logprobs: List[torch.Tensor]
):
    """Pad each tensor in the list to ``target_len`` along the last dim."""
    for i in range(len(logprobs_lst)):
        if logprobs_lst[i].shape[-1] < target_logprobs[i].shape[-1]:
            logprobs_lst[i] = torch.nn.functional.pad(
                logprobs_lst[i],
                (0, target_logprobs[i].shape[-1] - logprobs_lst[i].shape[-1]),
                value=0,
            )
        elif logprobs_lst[i].shape[-1] > target_logprobs[i].shape[-1]:
            log(f"DEBUG {logprobs_lst[i].shape[-1]=} > {target_logprobs[i].shape[-1]=}")
            assert False, "logprobs_lst[i].shape[-1] > target_logprobs[i].shape[-1]"


def g_opd_reverse_kl_single(
    student_lp: torch.Tensor,
    teacher_lp: torch.Tensor,
    base_lp: torch.Tensor,
    mask: torch.Tensor,
    g_opd_lambda: float,
    has_base: bool,
) -> torch.Tensor:
    """Compute per-token reverse KL for a single sample."""
    if has_base and g_opd_lambda != 1.0:
        return ((student_lp - base_lp) - g_opd_lambda * (teacher_lp - base_lp)) * mask
    else:
        return (student_lp - teacher_lp) * mask


def g_opd_forward_kl_single(
    student_lp: torch.Tensor,
    teacher_lp: torch.Tensor,
    base_lp: torch.Tensor,
    mask: torch.Tensor,
    g_opd_lambda: float,
    has_base: bool,
) -> torch.Tensor:
    """Compute per-token forward KL for a single sample."""
    if has_base and g_opd_lambda != 1.0:
        return (g_opd_lambda * (teacher_lp - base_lp) - (student_lp - base_lp)) * mask
    else:
        return (teacher_lp - student_lp) * mask


def calculate_g_opd_advantages(
    mask_lst: List[torch.Tensor],
    logprobs: List[torch.Tensor],
    teacher_logprobs: List[torch.Tensor],
    base_logprobs: List[torch.Tensor] = None,
    g_opd_lambda: float = 1.0,
    rewards: List[torch.Tensor] = None,
    sampling_repeat_n: int = 1,
    advantage_epsilon: float = 1e-6,
    multi_teacher_logprobs: Dict[str, List[torch.Tensor]] = None,
    teacher_types: List[str] = None,
    default_teacher_name: str = "default",
):
    """Compute advantages using the G-OPD / ExOPD formula.

    Supports both single-teacher and multi-teacher distillation.

    **Single-teacher formula** (with base model, λ ≠ 1)::

        reverse_kl = log π_student - log π_base - λ * (log π_teacher - log π_base)
        advantage  = -reverse_kl

    **Multi-teacher**: each sample is routed to the appropriate teacher based
    on the ``teacher_types`` list. The ``teacher_logprobs`` argument provides
    the primary (default) teacher, and ``multi_teacher_logprobs`` provides
    additional named teachers keyed by name.

    Reference: `arXiv:2602.12125 <https://arxiv.org/abs/2602.12125>`_.

    Parameters
    ----------
    mask_lst : list of torch.Tensor
    logprobs : list of torch.Tensor
    teacher_logprobs : list of torch.Tensor
        Primary teacher per-token log probabilities.
    base_logprobs : list of torch.Tensor or None
        Base model (student's initial checkpoint) log probabilities.
    g_opd_lambda : float
    rewards : list of torch.Tensor or None
        Optional per-sample scalar rewards for mixing GRPO advantages.
    sampling_repeat_n : int
    advantage_epsilon : float
    multi_teacher_logprobs : dict[str, list[torch.Tensor]] or None
        Named teacher log probs for multi-teacher mode.
        Keys are teacher names, values are per-sample log prob lists.
    teacher_types : list of str or None
        Per-sample teacher name for routing. Length must match batch size.
        Required when ``multi_teacher_logprobs`` is not None.
    default_teacher_name : str

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor], dict]
        ``(advantages, returns, metrics)``.
    """
    batch_size = len(logprobs)
    student_len = logprobs[0].shape[-1]
    all_lens = [lp.shape[-1] for lp in logprobs]
    equal_lens = [l == student_len for l in all_lens]

    # Pad all teacher logprobs to student length
    _pad_logprobs_to_target_len(teacher_logprobs, logprobs)
    if base_logprobs is not None:
        _pad_logprobs_to_target_len(base_logprobs, logprobs)
    if multi_teacher_logprobs is not None:
        for _name, _lps in multi_teacher_logprobs.items():
            _pad_logprobs_to_target_len(_lps, logprobs)

    has_base = base_logprobs is not None
    is_multi_teacher = (
        multi_teacher_logprobs is not None and teacher_types is not None and
        len(multi_teacher_logprobs) > 0
    )

    # Build a unified teacher lookup: name -> per-sample logprobs list
    all_teachers: Dict[str, List[torch.Tensor]] = {default_teacher_name: teacher_logprobs}
    if is_multi_teacher:
        all_teachers.update(multi_teacher_logprobs)

    # Compute per-sample reverse KL
    reverse_kl = []
    per_teacher_count: Dict[str, int] = {}

    for i in range(batch_size):
        # Determine which teacher to use for this sample
        if is_multi_teacher:
            t_name = teacher_types[i]
        else:
            t_name = default_teacher_name

        assert t_name in all_teachers, f"{t_name=} not in {all_teachers.keys()=}"
        per_teacher_count[t_name] = per_teacher_count.get(t_name, 0) + 1

        teacher_lp = all_teachers[t_name][i]
        base_lp = base_logprobs[i] if has_base else None

        kl = g_opd_reverse_kl_single(
            logprobs[i], teacher_lp, base_lp, mask_lst[i], g_opd_lambda, has_base
        )
        reverse_kl.append(kl)

    # Advantages = -reverse_kl
    advantages = [(-kl) for kl in reverse_kl]

    # Optionally mix in GRPO reward-based advantages
    if rewards is not None:
        reward_advantages, _ = calculate_grpo_advantages(
            rewards=rewards,
            mask=mask_lst,
            grpo_sampling_times=sampling_repeat_n,
            grpo_advantage_epsilon=advantage_epsilon,
        )
        advantages = [adv + r_adv for adv, r_adv in safezip(advantages, reward_advantages)]

    returns = advantages

    # Metrics
    avg_per_log_diff = sum(kl.sum() / m.sum()
                           for kl, m in zip(reverse_kl, mask_lst)) / len(mask_lst)
    avg_logp_diff = sum(kl.sum() for kl in reverse_kl) / sum(m.sum() for m in mask_lst)
    metrics = {
        "avg_per_sample_teacher_kl": avg_per_log_diff.item(),
        "teacher_kl": avg_logp_diff.item(),
        "g_opd_lambda": g_opd_lambda,
    }
    if is_multi_teacher:
        for t_name, count in per_teacher_count.items():
            metrics[f"teacher_count/{t_name}"] = count

    return advantages, returns, metrics


def calculate_topk_advantages(
    mask_lst: List[torch.Tensor],
    stu_topk_logprobs: List[torch.Tensor],
    teacher_topk_logprobs: List[torch.Tensor],
    base_topk_logprobs: List[torch.Tensor] = None,
    g_opd_lambda: float = 1.0,
    rewards: List[torch.Tensor] = None,
    sampling_repeat_n: int = 1,
    advantage_epsilon: float = 1e-6,
    multi_teacher_topk_logprobs: Dict[str, List[torch.Tensor]] = None,
    teacher_types: List[str] = None,
    default_teacher_name: str = "default",
):
    """Compute per-sample 3D top-K advantages for the OPD / G-OPD top-K logits path.

    For each sample the per-token reverse KL is evaluated on the K candidates and
    weighted by the renormalized student probability mass over those candidates
    (``softmax(stu_topk_lp, dim=-1)``). When ``base_topk_logprobs`` is supplied
    (G-OPD with base model) and ``g_opd_lambda != 1.0`` the formula is
    ``rkl = (stu - base) - lambda * (teacher - base)``; otherwise it falls back to
    ``rkl = stu - teacher``. Optional outcome rewards are broadcast to ``[S-1, K]``
    and added on top via GRPO normalization.

    Parameters
    ----------
    mask_lst : list of torch.Tensor
        Response masks ``[S-1]`` (float / bool).
    stu_topk_logprobs : list of torch.Tensor
        Student top-K log-probs ``[S-1, K]`` per sample.
    teacher_topk_logprobs : list of torch.Tensor
        Primary teacher's log-probs gathered at the student top-K ids ``[S-1, K]``.
    base_topk_logprobs : list of torch.Tensor or None
        Base / ref log-probs gathered at the student top-K ids ``[S-1, K]``.
    g_opd_lambda, rewards, sampling_repeat_n, advantage_epsilon
        Same semantics as :func:`calculate_g_opd_advantages`.
    multi_teacher_topk_logprobs, teacher_types, default_teacher_name
        Multi-teacher routing identical to :func:`calculate_g_opd_advantages`.

    Returns
    -------
    tuple[list[torch.Tensor], dict]
        ``(topk_advantages [S-1, K] per sample, metrics)``.
    """
    batch_size = len(stu_topk_logprobs)
    assert batch_size == len(teacher_topk_logprobs)

    from gpatch_v4.utils.training_utils import pad_topk_logprobs_to_target_len
    pad_topk_logprobs_to_target_len(teacher_topk_logprobs, stu_topk_logprobs)
    if base_topk_logprobs is not None:
        pad_topk_logprobs_to_target_len(base_topk_logprobs, stu_topk_logprobs)
    if multi_teacher_topk_logprobs is not None:
        for _name, _lps in multi_teacher_topk_logprobs.items():
            pad_topk_logprobs_to_target_len(_lps, stu_topk_logprobs)

    has_base = base_topk_logprobs is not None
    is_multi_teacher = (
        multi_teacher_topk_logprobs is not None and teacher_types is not None and
        len(multi_teacher_topk_logprobs) > 0
    )

    all_teachers: Dict[str, List[torch.Tensor]] = {default_teacher_name: teacher_topk_logprobs}
    if is_multi_teacher:
        all_teachers.update(multi_teacher_topk_logprobs)

    advantages_3d: List[torch.Tensor] = []
    reverse_kl_token_sum = 0.0
    reverse_kl_per_sample_sum = 0.0
    mask_token_sum = 0.0
    per_teacher_count: Dict[str, int] = {}

    for i in range(batch_size):
        t_name = teacher_types[i] if is_multi_teacher else default_teacher_name
        assert t_name in all_teachers, f"{t_name=} not in {all_teachers.keys()=}"
        per_teacher_count[t_name] = per_teacher_count.get(t_name, 0) + 1

        stu_lp = stu_topk_logprobs[i].to(torch.float32)
        teacher_lp = all_teachers[t_name][i].to(torch.float32)
        mask_1d = mask_lst[i].to(torch.float32)
        mask_2d = mask_1d.unsqueeze(-1)  # [S-1, 1]

        if has_base and g_opd_lambda != 1.0:
            base_lp = base_topk_logprobs[i].to(torch.float32)
            rkl = (stu_lp - base_lp) - g_opd_lambda * (teacher_lp - base_lp)
        else:
            rkl = stu_lp - teacher_lp

        # softmax_K renormalizes the student's mass over the K candidates so that
        # tokens dominating the student's prediction contribute more weight.
        w = torch.softmax(stu_lp, dim=-1)
        adv = (-rkl) * w * mask_2d

        advantages_3d.append(adv)

        # weighted-sum KL on the K dim → 1D per-token KL for metric stability
        masked_rkl_per_token = (rkl * w).sum(dim=-1) * mask_1d  # [S-1]
        token_sum = masked_rkl_per_token.sum()
        mask_sum = mask_1d.sum()
        reverse_kl_token_sum += token_sum
        reverse_kl_per_sample_sum += token_sum / (mask_sum + 1e-12)
        mask_token_sum += mask_sum

    if rewards is not None:
        reward_advantages, _ = calculate_grpo_advantages(
            rewards=rewards,
            mask=mask_lst,
            grpo_sampling_times=sampling_repeat_n,
            grpo_advantage_epsilon=advantage_epsilon,
        )
        # (rionawang)TODO: grpo_outcome_weight是否需要参数化, 类似OPD
        topk = stu_topk_logprobs[0].shape[-1]
        grpo_outcome_weight = 1.0 / topk
        advantages_3d = [
            adv + grpo_outcome_weight * r_adv.to(torch.float32).unsqueeze(-1)
            for adv, r_adv in safezip(advantages_3d, reward_advantages)
        ]

    metrics = {
        "topk_avg_per_sample_teacher_kl": (reverse_kl_per_sample_sum / batch_size).item(),
        "topk_teacher_kl": (reverse_kl_token_sum / (mask_token_sum + 1e-12)).item(),
        "g_opd_lambda": g_opd_lambda,
    }
    if is_multi_teacher:
        for t_name, count in per_teacher_count.items():
            metrics[f"topk_teacher_count/{t_name}"] = count

    return advantages_3d, metrics
