"""SAPO experiment reward (accuracy + overlong penalty).

Standalone rule reward used by ``tasks/math_rl_v4/scripts/sapo.sh`` for the
GRPO / GSPO / SAPO comparison. Copied from ``bt_reward_dapo.py`` so the
experiment does not perturb the shared reward, with two deliberate changes:

1. Format correctness is returned as a monitoring metric (``fmt_reward``) but is
   NOT added to the reward, so a near-saturated format signal can no longer
   inflate / flatten the curve.
2. A DAPO overlong reward shaping term (ref: ``gpatch/training/arguments.py``
   ``--dapo-overlong-*`` / verl DAPO recipe ``overlong-reward-shaping``): a soft,
   length-aware penalty that ramps linearly from ``0`` at
   ``max_response_len - OVERLONG_BUFFER_LEN`` to ``-OVERLONG_PENALTY_FACTOR`` at
   ``max_response_len``. Disable by setting ``OVERLONG_PENALTY_FACTOR = 0.0``.

Answer grading uses ``mathruler.grader.grade_answer`` (same as DAPO), so it
supports arbitrary math expressions (fractions, radicals, etc.).

Exports ``acc_only_rule_reward`` as the ``parse_reward_fn_name`` entry point.
"""
import json
import re
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from gpatch_v4.utils.training_utils import list_of_tensor_to_list

try:
    from mathruler.grader import extract_boxed_content, grade_answer
except ImportError as e:
    raise ImportError(
        "mathruler is required for DAPO-Math reward grading. "
        "Install via: pip install mathruler"
    ) from e

# DAPO overlong reward shaping knobs (mirror of the v3 ``--dapo-overlong-*`` args
# consumed in ``gpatch/core/models/gpt/gpt_ppo_critic_model.py``). The cap
# ``max_response_len`` is taken from ``config`` (sampler ``generate_max_tokens``).
# Set ``OVERLONG_PENALTY_FACTOR = 0.0`` to disable the penalty.
OVERLONG_BUFFER_LEN = 2048
OVERLONG_PENALTY_FACTOR = 1.0


def math_format_reward(predict_str: str) -> float:
    """Check if the response contains ``\\boxed{...}``."""
    pattern = re.compile(r"\\boxed\{.+\}", re.DOTALL)
    return 1.0 if re.search(pattern, predict_str) else 0.0


def math_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """Grade the model's boxed answer against ground truth using mathruler.

    Parameters
    ----------
    predict_str : str
        Full model response string.
    ground_truth : str
        Ground-truth answer string (e.g. ``"12"``, ``"\\frac{1}{2}"``).

    Returns
    -------
    float
        1.0 if correct, 0.0 otherwise.
    """
    answer = extract_boxed_content(predict_str)
    if answer is None:
        return 0.0
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def cal_overlong_penalty(response_lengths: List[int], max_response_len: int) -> List[float]:
    """DAPO overlong reward shaping (always ``<= 0``).

    Penalty ramps linearly from ``0`` once a response exceeds
    ``max_response_len - OVERLONG_BUFFER_LEN`` down to ``-OVERLONG_PENALTY_FACTOR``
    at ``max_response_len``. Mirrors the v3 implementation in
    ``gpt_ppo_critic_model.py``.

    Parameters
    ----------
    response_lengths : list of int
        Valid response lengths (``sequence_length - prompt_length``).
    max_response_len : int
        Generation budget (sampler ``generate_max_tokens``).

    Returns
    -------
    list of float
        Per-sample penalty, each ``<= 0``.
    """
    if OVERLONG_PENALTY_FACTOR <= 0.0 or OVERLONG_BUFFER_LEN <= 0:
        return [0.0] * len(response_lengths)
    expected_len = max_response_len - OVERLONG_BUFFER_LEN
    penalties = []
    for resp_len in response_lengths:
        exceed_len = resp_len - expected_len
        penalty = -exceed_len / OVERLONG_BUFFER_LEN * OVERLONG_PENALTY_FACTOR
        penalties.append(min(penalty, 0.0))
    return penalties


def acc_only_rule_reward(
    batched_data: List[Dict[str, Union[int, List[Any]]]],
    tokenizer: AutoTokenizer = None,
    actor_tokenizer: AutoTokenizer = None,
    config=None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, torch.Tensor]]:
    """Rule reward for SAPO: accuracy + overlong penalty (format monitor-only).

    Parameters
    ----------
    batched_data : list of dict
        Each dict contains ``tokens``, ``sequence_lengths``,
        ``prompt_lengths``, and ``gt_label`` (ground-truth answer strings
        or float tensors).
    tokenizer : AutoTokenizer, optional
        RM tokenizer (unused here).
    actor_tokenizer : AutoTokenizer
        Actor tokenizer for decoding response tokens.
    config : RlConfig, optional
        Injected by the framework when present in the signature. Used to read the
        generation budget (``sampler.infer_engine_configs[0].generate_max_tokens``)
        for the overlong penalty.

    Returns
    -------
    rule_reward : Tensor, shape ``(N, 1)``
        ``acc_reward + overlong_penalty`` (``fmt_reward`` is NOT added).
    per_token_reward : None
    metrics : dict
        ``{"acc_reward": Tensor, "fmt_reward": Tensor, "overlong_penalty": Tensor}``
    """
    assert actor_tokenizer is not None
    assert config is not None, "config is required for the overlong penalty"

    inputs_list: List[torch.Tensor] = []
    sequence_lengths_list: List[torch.Tensor] = []
    prompt_lengths_list: List[torch.Tensor] = []
    gt_label_list: List[Any] = []

    for batch in batched_data:
        inputs_list.extend(batch["tokens"])
        sequence_lengths_list.extend(batch["sequence_lengths"])
        prompt_lengths_list.extend(batch["prompt_lengths"])
        if "gt_label" in batch:
            gt_label_list.extend(batch["gt_label"])
        elif "labels" in batch:
            gt_label_list.extend(batch["labels"])
        else:
            raise ValueError("neither 'gt_label' nor 'labels' found in batch")

    tokens_cpu: List[List[int]] = list_of_tensor_to_list(inputs_list, False)
    seq_len_cpu: List[int] = torch.stack(sequence_lengths_list).view(-1).tolist()
    prompt_len_cpu: List[int] = torch.stack(prompt_lengths_list).view(-1).tolist()

    assert len(tokens_cpu) == len(seq_len_cpu)
    assert len(tokens_cpu) == len(gt_label_list)

    # Decode only the response portion (after prompt).
    resp_token_ids = []
    for i in range(len(tokens_cpu)):
        resp_token_ids.append(tokens_cpu[i][prompt_len_cpu[i]:seq_len_cpu[i]])
    resp_strs = actor_tokenizer.batch_decode(resp_token_ids, skip_special_tokens=True)

    # Resolve gt_label to strings.
    gt_answers: List[str] = []
    for label in gt_label_list:
        if isinstance(label, str):
            # May be a raw string like "12" or a JSON-encoded dict.
            try:
                parsed = json.loads(label)
                if isinstance(parsed, dict) and "answer" in parsed:
                    gt_answers.append(str(parsed["answer"]))
                else:
                    gt_answers.append(label)
            except (json.JSONDecodeError, TypeError):
                gt_answers.append(label)
        elif isinstance(label, torch.Tensor):
            # Float tensor from dataset (integer answers stored as float).
            val = label.item()
            gt_answers.append(str(int(val)) if val == int(val) else str(val))
        else:
            gt_answers.append(str(label))

    # Compute rewards.
    acc_rewards: List[float] = []
    fmt_rewards: List[float] = []
    for resp, gt in zip(resp_strs, gt_answers):
        acc_rewards.append(math_accuracy_reward(resp, gt))
        fmt_rewards.append(math_format_reward(resp))

    acc_reward_tensor = torch.tensor(acc_rewards, dtype=torch.float32).view(-1, 1)
    fmt_reward_tensor = torch.tensor(fmt_rewards, dtype=torch.float32).view(-1, 1)

    max_response_len = config.sampler.infer_engine_configs[0].generate_max_tokens
    response_lengths = [s - p for s, p in zip(seq_len_cpu, prompt_len_cpu)]
    overlong_penalty = cal_overlong_penalty(response_lengths, max_response_len)
    overlong_penalty_tensor = torch.tensor(
        overlong_penalty, dtype=torch.float32
    ).view(-1, 1)

    # SAPO: fmt is monitor-only and does NOT enter the reward.
    rule_reward = acc_reward_tensor + overlong_penalty_tensor
    metrics = {
        "acc_reward": acc_reward_tensor,
        "fmt_reward": fmt_reward_tensor,
        "overlong_penalty": overlong_penalty_tensor,
    }

    return rule_reward, None, metrics
