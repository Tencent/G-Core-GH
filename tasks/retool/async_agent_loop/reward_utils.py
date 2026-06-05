# ReTool / VERL-aligned reward and boxed-answer helpers (ported from tasks/retool sampler).
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

CODE_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL)


def _get_int_env(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "")
        return int(v) if v != "" else default
    except Exception:
        return default


RETOOL_TOOL_OUTPUT_MAX_LINES = _get_int_env("RETOOL_TOOL_OUTPUT_MAX_LINES", 200)
RETOOL_TOOL_OUTPUT_MAX_CHARS = _get_int_env("RETOOL_TOOL_OUTPUT_MAX_CHARS", 8192)
RETOOL_TRAIN_TOOL_OUTPUT_MAX_LINES = _get_int_env("RETOOL_TRAIN_TOOL_OUTPUT_MAX_LINES", 50)
RETOOL_TRAIN_TOOL_OUTPUT_MAX_CHARS = _get_int_env("RETOOL_TRAIN_TOOL_OUTPUT_MAX_CHARS", 2048)


def truncate_tool_output(text: str, *, max_lines: int, max_chars: int) -> str:
    if not text:
        return text
    original_len = len(text)
    if max_lines > 0:
        lines = text.splitlines()
        if len(lines) > max_lines:
            text = "\n".join(
                lines[:max_lines]
            ) + f"\n...[TRUNCATED {len(lines) - max_lines} more lines]..."
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + f"\n...[TRUNCATED {original_len - max_chars} chars]..."
    return text


def _last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def _remove_boxed(s: str) -> str:
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left):-1]


def _normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    substitutions = [
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
    ]
    removed = ["\\ldots", "\\text{s}", "\\text{.}", "\\text{}^2", "\\text{}^3", "\\text{}", '"']
    for before, after in substitutions:
        final_answer = final_answer.replace(before, after)
    for expr in removed:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def _is_correct_strict_box(pred_text: str, gt_text: str) -> Tuple[bool, Optional[str]]:
    boxed_pred = _last_boxed_only_string(pred_text)
    if boxed_pred is None:
        return False, None
    extracted_pred = _remove_boxed(boxed_pred)
    pred_norm = _normalize_final_answer(extracted_pred)
    gt_norm = _normalize_final_answer(str(gt_text))
    return pred_norm == gt_norm, extracted_pred


def compute_retool_reward(
    response_str: str,
    ground_truth: str,
    num_turns: int,
) -> Dict[str, Any]:
    # 这个地方不是只取最近一个生成的，而是要取所有生成的轨迹
    tail = response_str[-300:] if len(response_str) > 300 else response_str
    correct, pred = _is_correct_strict_box(tail, ground_truth)
    base_score = 1.0 if correct else -1.0
    final_score = base_score
    tool_call_reward = 0.0
    if base_score < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        final_score = min(0.0, base_score + tool_call_reward)
    return {
        "score": final_score,
        "acc": 1 if correct else 0,
        "pred": pred if pred is not None else "",
        "base_score": base_score,
        "tool_call_reward": tool_call_reward,
        "num_turns": num_turns,
    }


def prepare_code_for_sandbox(code: str) -> str:
    matches = CODE_PATTERN.findall(code)
    if matches:
        code = matches[0].strip()
    lines = code.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "":
            continue
        if not lines[i].strip().startswith("print"):
            lines[i] = f"print({lines[i]})"
        break
    return "\n".join(lines)


if __name__ == "__main__":

    def test_compute_retool_reward(response_str, ground_truth, num_turns):
        # test compute_retool_reward
        rew_info = compute_retool_reward(response_str, ground_truth, num_turns)
        print(rew_info)

    test_compute_retool_reward(
        "Thus, the smallest value of \(n\) is \(10\).\nAnswer: \\boxed{10}<|im_end|>", "100", 6
    )
    test_compute_retool_reward("Answer: \\boxed{10}<|im_end|>", "10", 10)
