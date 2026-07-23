"""GSM8K rule reward for verl, copied 1:1 from gcore
``tasks/math_rl_v4/bt_reward.py`` (the ``math_rl_rule_reward`` used by the gcore
dense alignment config).

The functions below (``is_number`` / ``extract_last_boxed_content`` /
``parse_boxed_content`` / ``parse_content`` / ``cal_accuracy_reward`` /
``cal_format_reward``) are verbatim copies of gcore's. Only the entry point
differs: gcore's ``math_rl_rule_reward`` decodes batched tokens itself, while
verl's ``compute_score`` receives the already-decoded ``solution_str`` and
``ground_truth`` per sample, so we just wrap the single sample into a
length-1 list and feed it to the original list-based functions.

Wire it in via:
    reward.reward_manager.name=naive
    reward.custom_reward_function.path=.../scripts/gsm8k_reward.py
    reward.custom_reward_function.name=compute_score
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional


# ----------------------------------------------------------------------------
# Below: verbatim copy of gcore tasks/math_rl_v4/bt_reward.py
# ----------------------------------------------------------------------------
def is_number(text):
    try:
        Decimal(text)
        return True
    except InvalidOperation:
        return False


def extract_last_boxed_content(text):
    boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
    matches = boxed_pattern.findall(text)

    if matches:
        return matches[-1]
    else:
        return None


def parse_boxed_content(content):
    try:
        content = content.strip()
        t1 = int(content)
        return t1
    except:
        return None


def parse_content(text):
    boxed_content = extract_last_boxed_content(text)
    boxed_value = parse_boxed_content(boxed_content)
    return boxed_value, boxed_content


def cal_accuracy_reward(resp_strs, gt_answer):
    acc_rewards = []
    boxed_content_tmp = []
    boxed_value_tmp = []
    for index, resp in enumerate(resp_strs):
        boxed_value, boxed_content = parse_content(resp)
        try:
            reward = float(boxed_value == gt_answer[index])
        except OverflowError as e:
            print(f"catch OverflowError {e} {index=} {boxed_value=}")
            reward = 0.0

        acc_rewards.append(reward)
        boxed_content_tmp.append(boxed_content)
        boxed_value_tmp.append(boxed_value)

    return acc_rewards, boxed_content_tmp, boxed_value_tmp


def cal_format_reward(resp_strs, **kwargs):
    """Reward function that checks if the completion has the format:
    'xxx final answer to this question is \boxed{...}' and extracts the content inside \boxed{}.
    """
    # pattern = r"final answer to this question is \\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    pattern = r'\\boxed\{[+-]?(\d+(\.\d*)?|\.\d+)\}'

    completion_contents = [resp for resp in resp_strs]
    matches = [
        re.search(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents
    ]

    fmt_reward = [1.0 if match else 0.0 for match in matches]

    return fmt_reward


# ----------------------------------------------------------------------------
# verl entry point: thin wrapper over the gcore functions above.
# gcore math_rl_rule_reward does (per response): rule_reward = acc + fmt, with
# gt_label cast to float (json {"answer": ...}) / int. Here ground_truth is the
# number after #### in the gsm8k answer, so we cast it the same way and feed a
# length-1 list to reuse the exact gcore logic.
# ----------------------------------------------------------------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl-compatible scorer mirroring gcore ``math_rl_rule_reward``.

    Returns
    -------
    dict
        ``{"score": acc + fmt, "acc_reward": acc, "fmt_reward": fmt}``, where
        ``score`` is gcore's ``rule_reward = acc_reward + fmt_reward`` and the
        other two mirror gcore's ``metrics``.
    """
    # gcore casts gt_label to float (str json {"answer": ..}) or int. gsm8k
    # ground_truth is a plain number string, so float() matches both paths.
    gt_answer = [float(ground_truth)]

    acc_reward, _, _ = cal_accuracy_reward([solution_str], gt_answer)
    fmt_reward = cal_format_reward([solution_str])

    acc = acc_reward[0]
    fmt = fmt_reward[0]
    return {"score": acc + fmt, "acc_reward": acc, "fmt_reward": fmt}
