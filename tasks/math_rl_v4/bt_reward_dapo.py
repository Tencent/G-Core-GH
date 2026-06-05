"""Rule-based reward for DAPO-Math-17K GRPO training.

Uses ``mathruler.grader.grade_answer`` to compare the model's boxed answer
against the ground-truth ``target`` string. This supports arbitrary math
expressions (fractions, radicals, etc.), not just integers.

Exports ``dapo_math_rule_reward`` as the ``parse_reward_fn_name`` entry point.
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


def dapo_math_rule_reward(
    batched_data: List[Dict[str, Union[int, List[Any]]]],
    tokenizer: AutoTokenizer = None,
    actor_tokenizer: AutoTokenizer = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, torch.Tensor]]:
    """Rule reward function for DAPO-Math-17K GRPO.

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

    Returns
    -------
    rule_reward : Tensor, shape ``(N, 1)``
        ``acc_reward + fmt_reward``.
    per_token_reward : None
    metrics : dict
        ``{"acc_reward": Tensor, "fmt_reward": Tensor}``
    """
    assert actor_tokenizer is not None

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

    rule_reward = acc_reward_tensor + fmt_reward_tensor
    metrics = {
        "acc_reward": acc_reward_tensor,
        "fmt_reward": fmt_reward_tensor,
    }

    return rule_reward, None, metrics
