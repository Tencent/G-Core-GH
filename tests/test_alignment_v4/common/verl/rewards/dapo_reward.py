"""DAPO-Math rule reward for verl, aligned with gcore
``tasks/math_rl_v4/bt_reward_dapo.py`` (``dapo_math_rule_reward``).

gcore grades with ``mathruler`` over ``\\boxed{...}`` (accuracy 1/0) and adds a
format reward (``\\boxed{...}`` present, 1/0), yielding ``rule_reward = acc + fmt``.
This scorer reproduces that judging exactly, and returns ``score = acc + fmt`` so
verl's naive reward manager assigns the gcore reward per sample.

Wire it in via:
    reward.reward_manager.name=naive
    reward.custom_reward_function.path=.../rewards/dapo_reward.py
    reward.custom_reward_function.name=compute_score
"""

import re
from typing import Any, Optional

from mathruler.grader import extract_boxed_content, grade_answer

_BOXED_RE = re.compile(r"\\boxed\{.+\}", re.DOTALL)


def math_format_reward(predict_str: str) -> float:
    """Return ``1.0`` if the response contains a ``\\boxed{...}`` span (gcore parity)."""
    return 1.0 if _BOXED_RE.search(predict_str) else 0.0


def math_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """Grade the boxed answer against ground truth via mathruler (``1.0`` / ``0.0``)."""
    answer = extract_boxed_content(predict_str)
    if answer is None:
        return 0.0
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl-compatible scorer mirroring gcore ``dapo_math_rule_reward``.

    Returns
    -------
    dict
        ``{"score": acc + fmt, "acc_reward": acc, "fmt_reward": fmt}``, where
        ``score`` is gcore's ``rule_reward = acc_reward + fmt_reward`` and the
        other two mirror gcore's ``metrics``.
    """
    acc = math_accuracy_reward(solution_str, str(ground_truth))
    fmt = math_format_reward(solution_str)
    return {"score": acc + fmt, "acc_reward": acc, "fmt_reward": fmt}
