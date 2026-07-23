# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from typing import List, Optional, Tuple, Union

import torch

from gpatch_v4.utils import log


def get_advantage_clip_bounds(
    advantage_clip: Optional[float],
    advantage_clip_lower_bound: Optional[float],
    advantage_clip_upper_bound: Optional[float],
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Resolve symmetric or explicit advantage clipping bounds.

    ``advantage_clip`` keeps the legacy symmetric clipping behavior. Explicit
    lower/upper bounds are mutually exclusive with the legacy setting.
    """
    has_explicit_bounds = (
        advantage_clip_lower_bound is not None or advantage_clip_upper_bound is not None
    )
    if has_explicit_bounds:
        assert advantage_clip is None, (
            "ppo.advantage_clip cannot be set together with "
            "ppo.advantage_clip_lower_bound or ppo.advantage_clip_upper_bound"
        )
        if advantage_clip_lower_bound is not None and advantage_clip_upper_bound is not None:
            assert advantage_clip_lower_bound < advantage_clip_upper_bound, (
                "ppo.advantage_clip_lower_bound must be less than "
                "ppo.advantage_clip_upper_bound, got "
                f"{advantage_clip_lower_bound} >= {advantage_clip_upper_bound}"
            )
        return advantage_clip_lower_bound, advantage_clip_upper_bound

    if advantage_clip is not None:
        assert advantage_clip > 0, f"ppo.advantage_clip must be positive, got {advantage_clip}"
        return -advantage_clip, advantage_clip

    # return None means no advantage clipping
    return None


def count_advantage_clip_samples(
    original_advantages: List[torch.Tensor],
    advantages: List[torch.Tensor],
    masks: List[torch.Tensor],
) -> Tuple[int, int, int]:
    """Count samples whose advantages were raised/lowered by clip bounds.

    A sample counts if any masked response token differs after clamp:
    lower clip raises values (``original < clipped``); upper clip lowers
    them (``original > clipped``).

    Parameters
    ----------
    original_advantages : list of torch.Tensor
        Pre-clamp advantages, one tensor per sample.
    advantages : list of torch.Tensor
        Post-clamp advantages, same layout as ``original_advantages``.
    masks : list of torch.Tensor
        Response masks; dead / padded tokens (0) are ignored.

    Returns
    -------
    tuple[int, int, int]
        ``(n_lower_clipped, n_upper_clipped, n_samples)``.
    """
    assert len(original_advantages) == len(advantages) == len(masks)
    n_lower = 0
    n_upper = 0
    for orig, clipped, mask in zip(original_advantages, advantages, masks, strict=True):
        valid = mask.bool()
        if not valid.any():
            continue
        n_lower += int((orig[valid] < clipped[valid]).any().item())
        n_upper += int((orig[valid] > clipped[valid]).any().item())
    return n_lower, n_upper, len(original_advantages)


def align_token_level_tensors_to_logprobs(
    tensors: List[torch.Tensor],
    logprobs: List[torch.Tensor],
    sequence_lengths: List[Union[int, torch.Tensor]],
    truncate_head: bool,
) -> List[torch.Tensor]:
    """Align full-token tensors to the log-probability axis.

    Full-token tensors have one entry per token (length ``S``), while
    next-token log probabilities have length ``S - 1``. Inputs are shifted
    according to ``truncate_head`` and then right-padded to the corresponding
    log-probability length.
    """
    assert len(tensors) == len(logprobs) == len(sequence_lengths)

    aligned = []
    for tensor, logps, sequence_length in zip(tensors, logprobs, sequence_lengths, strict=True):
        assert tensor.ndim == 1, f"expected a 1D token-level tensor, got {tensor.ndim}D"
        assert logps.ndim == 1, f"expected 1D logprobs, got {logps.ndim}D"

        sequence_length = int(sequence_length)
        tensor_length = tensor.size(-1)
        logprobs_length = logps.size(-1)

        assert tensor_length == sequence_length, (
            "token-level tensor must use the full, unpadded token axis: "
            f"{tensor_length=} != {sequence_length=}"
        )
        tensor = tensor[1:] if truncate_head else tensor[:-1]

        aligned.append(
            torch.nn.functional.pad(
                tensor,
                (0, logprobs_length - tensor.size(-1)),
                value=0,
            ).contiguous()
        )
    return aligned


def create_response_mask(
    values: List[torch.Tensor],
    prompt_lengths: List[torch.Tensor],
    sequence_lengths: List[torch.Tensor],
    dtype=None
):
    """Create a mask that keeps only the response (non-prompt, non-padding) tokens.

    Parameters
    ----------
    values : list of torch.Tensor
    prompt_lengths : list of torch.Tensor
    sequence_lengths : list of torch.Tensor
    dtype : torch.dtype, optional

    Returns
    -------
    list of torch.Tensor
    """
    mask = []
    for value, prompt_length, response_length in zip(values, prompt_lengths, sequence_lengths):
        # value shape is: [pad_seqlen - 1]
        tmp = torch.zeros_like(value, dtype=dtype)
        tmp[prompt_length - 1:response_length - 1] = 1.0
        mask.append(tmp)
    return mask


def calculate_kl_loss(
    cur_log_probs,
    ref_log_probs,
    use_absolute_kl=True,
    use_low_var_kl=False,
    clamp_kl_loss=False,
    clamp_kl_val=None
):
    """Compute per-token KL divergence between current and reference log probs.

    Parameters
    ----------
    cur_log_probs : torch.Tensor
    ref_log_probs : torch.Tensor
    use_absolute_kl : bool, optional
    use_low_var_kl : bool, optional
    clamp_kl_loss : bool, optional
        Clamp loss to [-10, 10].
    clamp_kl_val : float, optional
        Clamp intermediate KL values.

    Returns
    -------
    torch.Tensor
    """
    kl = ref_log_probs - cur_log_probs

    if use_low_var_kl:
        # For numerical stability
        if clamp_kl_val is not None:
            kl = torch.clamp(kl, min=-clamp_kl_val, max=clamp_kl_val)
        ratio = torch.exp(kl)
        kl_loss = (ratio - kl - 1).contiguous()
        if clamp_kl_loss:
            # 论文没看到有做这个 clamp, 但 verl 实现了，讨论看看有没有必要加上
            kl_loss = torch.clamp(kl_loss, min=-10, max=10)
    else:
        kl_loss = cur_log_probs - ref_log_probs

    if use_absolute_kl:
        kl_loss = kl_loss.abs()

    return kl_loss


def calculate_kl_penalty(
    log_probs_a: List[torch.Tensor], log_probs_b: List[torch.Tensor], use_absolute_kl=True
):
    """Compute per-token KL penalty between two sets of log probabilities.

    Parameters
    ----------
    log_probs_a : list of torch.Tensor
    log_probs_b : list of torch.Tensor
    use_absolute_kl : bool, optional

    Returns
    -------
    list of torch.Tensor
    """
    init_policy_kl = []
    for log_prob_a, log_prob_b in zip(log_probs_a, log_probs_b):
        tmp = log_prob_a - log_prob_b
        if use_absolute_kl:
            tmp = tmp.abs()
        init_policy_kl.append(tmp)

    return init_policy_kl


def get_ladder_reward(distinct_ratio):
    """Map a distinct-ngram ratio to a quadratic penalty reward.

    Parameters
    ----------
    distinct_ratio : float

    Returns
    -------
    float
    """
    penalty = 1 - distinct_ratio
    return -1 * penalty**2


def calculate_repetition_penalty_reward(
    tokens: List[Union[str, int]],
    ngram_size: int = 3,
    tail_len: int = 256,
    min_repeat_time: int = 3
) -> float:
    """Compute a repetition-penalty reward based on distinct N-gram ratio.

    Parameters
    ----------
    tokens : list of str or int
    ngram_size : int, optional
    tail_len : int, optional
    min_repeat_time : int, optional
        Tolerated number of repeated N-grams.

    Returns
    -------
    float
        Non-positive.
    """
    sequence = tokens[-tail_len:]

    if len(sequence) < ngram_size:
        return get_ladder_reward(1.0 if len(sequence) > 0 else 0.0)

    all_ngrams = []

    for i in range(len(sequence) - ngram_size + 1):
        ngram = tuple(sequence[i:i + ngram_size])
        all_ngrams.append(ngram)

    total_ngram_count = len(all_ngrams)
    distinct_ngram_count = len(set(all_ngrams))

    if total_ngram_count == 0:
        return get_ladder_reward(0.0)

    if (total_ngram_count - distinct_ngram_count) <= min_repeat_time:
        return get_ladder_reward(1.0)

    distinct_ratio = distinct_ngram_count / total_ngram_count

    return get_ladder_reward(distinct_ratio)
