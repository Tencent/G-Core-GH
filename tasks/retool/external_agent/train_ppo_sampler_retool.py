# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# yeazhao@tencent.com
#
# ReTool GRPO Sampler - Multi-turn tool-calling rollout generation
# Based on Verl (Volcengine RL) ReTool implementation logic
# Upgraded to idio_new.py framework (2025-01)
#
# Key Features:
#   1. Multi-turn Generate -> Detect Code -> Execute -> Output -> Generate loop
#   2. Code detection via ```python ... ``` markdown blocks
#   3. Sandbox code execution via local_sandbox_server.py
#   4. Reward = Correctness (boxed answer) + Tool Usage Bonus (when wrong)
#   5. Robust failure handling with dummy data fallback (idio_new.py style)
#   6. log_probs/mask roll(-1) alignment (idio_new.py style)

import os
import re
import copy
import uuid
import json
import asyncio
import traceback
import logging
from typing import Optional, List, Dict, Any, Tuple

import httpx
import torch
import openai

from megatron.core import mpu
from megatron.training.global_vars import get_tokenizer, get_args

from gpatch.training.v3.grpo_sampler import run_grpo_sampler_v3, GrpoSamplerV3
from gpatch.core.parallel_state import is_mp_and_cp_head
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.v3.default_model_provider import default_sampler_model_provider
from gpatch.training.utils import print_with_rank_and_datetime

from tasks.retool.args import get_tasks_args

model_provider = default_sampler_model_provider

# =====================================================================================
# ======================== Debug Logging for gen_intervals Verification ==============
# =====================================================================================

# =====================================================================================
# ======================== Debug Logging for messages_traj (low-frequency) ============
# =====================================================================================

_MESSAGES_TRAJ_DEBUG_LOG_PATH = os.environ.get(
    "RETOOL_MESSAGES_TRAJ_DEBUG_LOG_PATH", "/tmp/messages_traj_debug.log"
)


def _should_debug_messages_traj(ppo_step: int, pending_idx: int, my_rank: int) -> bool:
    """
    Control when to log messages_traj debug info.
    Designed to be VERY low frequency to avoid log spam / I/O overhead.
    Enable with: export RETOOL_DEBUG_MESSAGES_TRAJ=1
    """
    if my_rank != 0:
        return False
    if not _get_bool_env("RETOOL_DEBUG_MESSAGES_TRAJ", False):
        return False
    interval = _get_int_env("RETOOL_DEBUG_MESSAGES_TRAJ_STEP_INTERVAL", 50)
    first_n = _get_int_env("RETOOL_DEBUG_MESSAGES_TRAJ_FIRST_N", 1)
    if interval > 0 and (ppo_step % interval != 0):
        return False
    if pending_idx >= max(first_n, 0):
        return False
    return True


def _log_messages_traj_debug(
    *,
    ppo_step: int,
    pending_idx: int,
    chat_id: str,
    messages_traj: list,
    final_messages: list,
):
    """
    Append a compact view of messages trajectory to a dedicated log file.
    Final source of truth is serving-side chat_state['messages'] (messages_traj),
    because it is aligned with chat_state['token_ids']/gen_intervals used for training.
    If serving-side messages are missing, fall back to local gen['messages'].
    """
    try:
        import datetime

        os.makedirs(os.path.dirname(_MESSAGES_TRAJ_DEBUG_LOG_PATH), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _compact(msgs: list) -> list:
            out = []
            if not isinstance(msgs, list):
                return out
            # NOTE: Do NOT truncate. Caller is responsible for keeping frequency low.
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "")
                name = m.get("name", "")
                tool_call_id = m.get("tool_call_id", "")
                content = m.get("content", "") or ""
                out.append(
                    {
                        "role": role,
                        "name": name,
                        "tool_call_id": tool_call_id,
                        "has_tool_calls": bool(m.get("tool_calls", None)),
                        "content": content,
                    }
                )
            return out

        with open(_MESSAGES_TRAJ_DEBUG_LOG_PATH, "a") as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write(
                f"[{timestamp}] PPO_STEP={ppo_step} PENDING_IDX={pending_idx} CHAT_ID={chat_id}\n"
            )
            f.write("=" * 100 + "\n")
            use_serving = isinstance(messages_traj, list) and len(messages_traj) > 0
            chosen = messages_traj if use_serving else (
                final_messages if isinstance(final_messages, list) else []
            )
            source = "serving chat_state['messages']" if use_serving else "local gen['messages'] (fallback)"
            f.write(f"source={source}\n")
            f.write(f"messages_len={len(chosen) if isinstance(chosen, list) else 'NA'}\n\n")

            f.write(">>> messages_traj (compact):\n")
            f.write(json.dumps(_compact(chosen), ensure_ascii=False, indent=2))
            f.write("\n")
            f.write("=" * 100 + "\n")
            f.flush()
    except Exception as ex:
        print_with_rank_and_datetime(f"[messages_traj debug] Failed to write log: {ex}", rank=0)


def _should_debug_gen_intervals(ppo_step: int, pending_idx: int) -> bool:
    """
    Control when to log gen_intervals debug info.
    Only log for first 3 samples of every 10th PPO step to avoid log spam.
    """
    if not _get_bool_env("RETOOL_DEBUG_GEN_INTERVALS", False):
        return False
    # Log every 10th PPO step
    if ppo_step % 10 != 0:
        return False
    # Only first 3 samples per step
    if pending_idx >= 3:
        return False
    return True


def _log_gen_intervals_debug(
    ppo_step: int,
    pending_idx: int,
    chat_id: str,
    gen_intervals: list,
    token_ids: list,
    mask_before_roll: torch.Tensor,
    mask_after_roll: torch.Tensor,
    final_messages: list,
    tokenizer,
):
    """
    Log detailed gen_intervals and mask debug info to a dedicated log file.
    This helps verify that:
    1. gen_intervals correctly marks model-generated tokens
    2. Tool response tokens are NOT included in gen_intervals
    3. The mask correctly excludes tool responses from training
    """
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_path = os.environ.get(
            "RETOOL_GEN_INTERVALS_DEBUG_LOG_PATH",
            f"/tmp/gen_intervals_debug_{uuid.uuid4().hex}.log",
        )
        with open(log_path, "a") as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write(
                f"[{timestamp}] PPO_STEP={ppo_step} PENDING_IDX={pending_idx} CHAT_ID={chat_id}\n"
            )
            f.write("=" * 100 + "\n\n")

            # 1. Log gen_intervals
            f.write(">>> GEN_INTERVALS (should only contain model-generated token ranges):\n")
            f.write(f"    Total intervals: {len(gen_intervals)}\n")
            for i, (s, e) in enumerate(gen_intervals):
                f.write(f"    [{i}] start={s}, end={e}, length={e-s}\n")
            f.write("\n")

            # 2. Log mask statistics
            mask_sum_before = int(mask_before_roll.sum().item())
            mask_sum_after = int(mask_after_roll.sum().item())
            total_tokens = len(token_ids)
            f.write(">>> MASK STATISTICS:\n")
            f.write(f"    Total tokens: {total_tokens}\n")
            f.write(
                f"    Masked tokens (before roll): {mask_sum_before} ({100*mask_sum_before/max(total_tokens,1):.1f}%)\n"
            )
            f.write(
                f"    Masked tokens (after roll): {mask_sum_after} ({100*mask_sum_after/max(total_tokens,1):.1f}%)\n"
            )
            f.write("\n")

            # 3. Log message structure (to compare with gen_intervals)
            f.write(">>> CONVERSATION MESSAGES:\n")
            for i, msg in enumerate(final_messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                has_tool_calls = "tool_calls" in msg
                content_preview = content[:100].replace("\n", "\\n"
                                                       ) + ("..." if len(content) > 100 else "")
                f.write(
                    f"    [{i}] role={role}, has_tool_calls={has_tool_calls}, content_len={len(content)}\n"
                )
                f.write(f"        preview: {content_preview}\n")
            f.write("\n")

            # 4. Decode and show token segments for each gen_interval
            f.write(">>> TOKEN SEGMENTS FOR EACH GEN_INTERVAL:\n")
            for i, (s, e) in enumerate(gen_intervals[:5]):  # Limit to first 5 intervals
                segment_ids = token_ids[s:min(e, s + 50)]  # First 50 tokens of segment
                try:
                    segment_text = tokenizer._tokenizer.decode(
                        segment_ids, skip_special_tokens=False
                    )
                    segment_text = segment_text[:200].replace("\n", "\\n")
                except:
                    segment_text = "[decode error]"
                f.write(f"    [interval {i}] tokens[{s}:{e}] (showing first 50 tokens):\n")
                f.write(f"        {segment_text}\n")
            f.write("\n")

            # 5. Show what's NOT in gen_intervals (tool responses)
            f.write(">>> TOKENS NOT IN GEN_INTERVALS (should be tool responses + prompts):\n")
            non_gen_ranges = []
            prev_end = 0
            for s, e in sorted(gen_intervals):
                if s > prev_end:
                    non_gen_ranges.append((prev_end, s))
                prev_end = max(prev_end, e)
            if prev_end < total_tokens:
                non_gen_ranges.append((prev_end, total_tokens))

            for i, (s, e) in enumerate(non_gen_ranges[:5]):  # Limit to first 5 gaps
                gap_ids = token_ids[s:min(e, s + 50)]
                try:
                    gap_text = tokenizer._tokenizer.decode(gap_ids, skip_special_tokens=False)
                    gap_text = gap_text[:200].replace("\n", "\\n")
                except:
                    gap_text = "[decode error]"
                f.write(f"    [gap {i}] tokens[{s}:{e}] (showing first 50 tokens):\n")
                f.write(f"        {gap_text}\n")
            f.write("\n")

            # 6. Check for potential issues
            f.write(">>> POTENTIAL ISSUES CHECK:\n")
            issues = []

            # Check if gen_intervals is empty
            if len(gen_intervals) == 0:
                issues.append("WARNING: gen_intervals is EMPTY - no tokens will be trained!")

            # Check if any gen_interval exceeds token length
            for s, e in gen_intervals:
                if e > total_tokens:
                    issues.append(
                        f"WARNING: gen_interval ({s}, {e}) exceeds total_tokens ({total_tokens})"
                    )

            # Check if mask has any True values
            if mask_sum_after == 0:
                issues.append("WARNING: mask is all zeros after roll - nothing will be trained!")

            # Check if tool responses might be included (heuristic: look for "tool" in decoded segments)
            for s, e in gen_intervals[:3]:
                segment_ids = token_ids[s:e]
                try:
                    segment_text = tokenizer._tokenizer.decode(
                        segment_ids, skip_special_tokens=False
                    ).lower()
                    if "<tool_response>" in segment_text or "tool_call_id" in segment_text:
                        issues.append(
                            f"WARNING: gen_interval ({s}, {e}) may contain tool response tokens!"
                        )
                except:
                    pass

            if issues:
                for issue in issues:
                    f.write(f"    ❌ {issue}\n")
            else:
                f.write("    ✅ No obvious issues detected\n")

            f.write("\n" + "=" * 100 + "\n\n")
            f.flush()

    except Exception as ex:
        print_with_rank_and_datetime(f"[gen_intervals debug] Failed to write log: {ex}", rank=0)


# =====================================================================================
# ======================== ReTool Configuration Constants =============================
# =====================================================================================

# Regex pattern to extract Python code from markdown code blocks (Verl-style)
CODE_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL)

# Regex pattern to extract boxed answer for correctness check (Verl-style)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

# Tool definition for Qwen-style function calling (aligned with VERL's sandbox_fusion_tool_config.yaml)
RETOOL_TOOLS = [
    {
        "type": "function",
        "function":
            {
                "name": "code_interpreter",
                "description": "A tool for executing code.",
                "parameters":
                    {
                        "type": "object",
                        "properties":
                            {
                                "code": {
                                    "type": "string",
                                    "description": "The code to execute."
                                }
                            },
                        "required": ["code"]
                    }
            }
    }
]

# =====================================================================================
# ======================== Sandbox Execution (Verl-style) =============================
# =====================================================================================


def _retool_debug_enabled() -> bool:
    return os.environ.get("RETOOL_DEBUG", "0").lower() in ("1", "true", "yes", "y")


def _retool_debug_first_n() -> int:
    # Only print debug for the first N rollouts to avoid log spam.
    try:
        return int(os.environ.get("RETOOL_DEBUG_FIRST_N", "2"))
    except Exception:
        return 2


def _get_int_env(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "")
        return int(v) if v != "" else default
    except Exception:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name, "")
    if v == "":
        return default
    return v.lower() in ("1", "true", "yes", "y")


# ==== Reward normalization helpers (aligned with VERL math_dapo strict_box_verify) ====
def _last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last \\boxed{...} block. Returns None if not found."""
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
    """Remove \\boxed{...} wrapper."""
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left):-1]


def _normalize_final_answer(final_answer: str) -> str:
    """Lightweight normalization copied from VERL math_dapo."""
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
        (",", ","),  # keep comma normalization simple
    ]
    removed = ["\\ldots", "\\text{s}", "\\text{.}", "\\text{}^2", "\\text{}^3", "\\text{}", '"']
    for before, after in substitutions:
        final_answer = final_answer.replace(before, after)
    for expr in removed:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def _is_correct_strict_box(pred_text: str, gt_text: str) -> tuple[bool, Optional[str]]:
    """
    Strict boxed verification: extract last boxed from pred_text tail, normalize both sides.
    Returns (is_correct, extracted_pred).
    """
    boxed_pred = _last_boxed_only_string(pred_text)
    if boxed_pred is None:
        return False, None
    extracted_pred = _remove_boxed(boxed_pred)
    pred_norm = _normalize_final_answer(extracted_pred)
    gt_norm = _normalize_final_answer(str(gt_text))
    return pred_norm == gt_norm, extracted_pred


def truncate_tool_output(text: str, *, max_lines: int, max_chars: int) -> str:
    """Truncate potentially huge tool/sandbox outputs to prevent runaway tokenization."""
    if not text:
        return text
    original_len = len(text)
    # Line cap first.
    if max_lines > 0:
        lines = text.splitlines()
        if len(lines) > max_lines:
            text = "\n".join(
                lines[:max_lines]
            ) + f"\n...[TRUNCATED {len(lines) - max_lines} more lines]..."
    # Char cap second (hard bound).
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + f"\n...[TRUNCATED {original_len - max_chars} chars]..."
    return text


# Safety bounds (overridable via env).
# These apply to the tool output fed back into the model as context.
# NOTE: Even if tool tokens are masked out for loss, they still participate in attention
# and can increase memory usage. Keep defaults conservative to avoid OOM from verbose stdout.
RETOOL_TOOL_OUTPUT_MAX_LINES = _get_int_env("RETOOL_TOOL_OUTPUT_MAX_LINES", 200)
RETOOL_TOOL_OUTPUT_MAX_CHARS = _get_int_env("RETOOL_TOOL_OUTPUT_MAX_CHARS", 8192)

# Training transcript policy:
# - If drop tool messages, training tokens will exclude sandbox stdout/stderr/traceback entirely.
# - Otherwise, keep tool messages but clamp their content.
RETOOL_TRAIN_DROP_TOOL_MESSAGES = _get_bool_env("RETOOL_TRAIN_DROP_TOOL_MESSAGES", False)
RETOOL_TRAIN_TOOL_OUTPUT_MAX_LINES = _get_int_env("RETOOL_TRAIN_TOOL_OUTPUT_MAX_LINES", 50)
RETOOL_TRAIN_TOOL_OUTPUT_MAX_CHARS = _get_int_env("RETOOL_TRAIN_TOOL_OUTPUT_MAX_CHARS", 2048)


async def execute_code_in_sandbox(
    code: str,
    sandbox_url: str,
    timeout: int = 20,
    memory_limit_mb: int = 1024,
    client: httpx.AsyncClient = None,
    debug: bool = False,
    debug_prefix: str = "",
) -> Tuple[str, bool]:
    """
    Execute Python code in sandbox and return output.
    
    Based on Verl's CustomSandboxFusionTool.execute() logic:
    1. Extract code from ```python ... ``` blocks if present
    2. Auto-add print() to last line if not present
    3. Send to sandbox and return stdout/stderr
    
    Args:
        code: Python code to execute (may contain markdown blocks)
        sandbox_url: URL to sandbox /run_code endpoint
        timeout: Execution timeout in seconds
        memory_limit_mb: Memory limit in MB
        client: httpx AsyncClient for connection reuse
        
    Returns:
        Tuple of (output_string, success_flag)
    """
    # 1. Extract code from markdown blocks (Verl-style)
    matches = CODE_PATTERN.findall(code)
    if matches:
        code = matches[0].strip()

    # 2. Auto-add print() to last non-empty line if not already printing (Verl-style)
    lines = code.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "":
            continue
        if not lines[i].strip().startswith("print"):
            # Wrap last expression in print()
            lines[i] = f"print({lines[i]})"
        break
    code = "\n".join(lines)

    # 3. Call sandbox API
    payload = {
        "compile_timeout": 10,
        "run_timeout": timeout,
        "code": code,
        "stdin": None,
        "memory_limit_MB": memory_limit_mb,
        "language": "python",
        "files": {},
        "fetch_files": []
    }

    start_t = None
    if debug:
        start_t = asyncio.get_event_loop().time()
    try:
        should_close = False
        if client is None:
            client = httpx.AsyncClient()
            should_close = True

        response = await client.post(
            sandbox_url,
            json=payload,
            timeout=timeout + 10  # Extra buffer for network
        )
        response.raise_for_status()
        result = response.json()

        if should_close:
            await client.aclose()

        # Parse sandbox response
        status = result.get("status", "Failed")
        run_result = result.get("run_result", {}) or {}
        compile_result = result.get("compile_result", {}) or {}

        if status == "Success":
            stdout = run_result.get("stdout", "").strip()
            stdout = truncate_tool_output(
                stdout,
                max_lines=RETOOL_TOOL_OUTPUT_MAX_LINES,
                max_chars=RETOOL_TOOL_OUTPUT_MAX_CHARS,
            )
            if debug and start_t is not None:
                dt = asyncio.get_event_loop().time() - start_t
                out_len = len(stdout) if stdout else 0
                print_with_rank_and_datetime(
                    f"{debug_prefix}[sandbox] ok=1 secs={dt:.2f} out_chars={out_len}",
                    rank=0,
                )
            return stdout if stdout else "(no output)", True
        else:
            # Compilation or runtime error
            stderr = run_result.get("stderr", "") if run_result else ""
            compile_err = compile_result.get("stderr", "") if compile_result else ""
            error_msg = stderr or compile_err or "Execution failed"
            error_msg = truncate_tool_output(
                error_msg,
                max_lines=RETOOL_TOOL_OUTPUT_MAX_LINES,
                max_chars=RETOOL_TOOL_OUTPUT_MAX_CHARS,
            )
            if debug and start_t is not None:
                dt = asyncio.get_event_loop().time() - start_t
                print_with_rank_and_datetime(
                    f"{debug_prefix}[sandbox] ok=0 secs={dt:.2f} err={error_msg[:120].replace(chr(10), ' ')}",
                    rank=0,
                )
            return f"Error: {error_msg[:500]}", False

    except asyncio.TimeoutError:
        if debug and start_t is not None:
            dt = asyncio.get_event_loop().time() - start_t
            print_with_rank_and_datetime(
                f"{debug_prefix}[sandbox] ok=0 secs={dt:.2f} err=timeout", rank=0
            )
        return "Error: Execution timed out", False
    except Exception as e:
        if debug and start_t is not None:
            dt = asyncio.get_event_loop().time() - start_t
            print_with_rank_and_datetime(
                f"{debug_prefix}[sandbox] ok=0 secs={dt:.2f} err={str(e)[:120].replace(chr(10), ' ')}",
                rank=0,
            )
        return f"Error: {str(e)[:200]}", False


# =====================================================================================
# ======================== Reward Calculation (Verl-style) ============================
# =====================================================================================


def extract_last_boxed(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} content from text.
    Based on Verl's math_dapo.last_boxed_only_string() logic.
    """
    # Find last occurrence of \boxed{
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None

    # Find matching closing brace
    i = idx
    num_open = 0
    right_idx = None

    while i < len(text):
        if text[i] == "{":
            num_open += 1
        elif text[i] == "}":
            num_open -= 1
            if num_open == 0:
                right_idx = i
                break
        i += 1

    if right_idx is None:
        return None

    # Extract content between \boxed{ and }
    boxed_str = text[idx:right_idx + 1]
    # Remove \boxed{ prefix and } suffix
    content = boxed_str[7:-1]  # len("\\boxed{") = 7
    return content


def compute_retool_reward(
    response_str: str,
    ground_truth: str,
    num_turns: int,
) -> Dict[str, Any]:
    """
    Compute ReTool reward based on Verl's retool.py compute_score() logic.
    
    Reward Formula:
    1. Base correctness: +1.0 if \\boxed{answer} matches ground_truth, else -1.0
    2. Tool usage bonus (only when wrong): (num_turns - 2) / 2 * 0.1
       - Clamped so final score <= 0 when wrong
    
    Args:
        response_str: Full model response text
        ground_truth: Expected answer (string)
        num_turns: Number of conversation turns (user + assistant + tool messages)
        
    Returns:
        Dict with keys: score, acc, pred, num_turns
    """
    # Extract boxed answer from tail of response (Verl uses last 300 chars)
    tail = response_str[-300:] if len(response_str) > 300 else response_str
    # Strict boxed verification with normalization (VERL-style)
    correct, pred = _is_correct_strict_box(tail, ground_truth)
    base_score = 1.0 if correct else -1.0

    # Tool usage bonus (Verl-style): only when incorrect
    # Encourages model to use tools even if answer is wrong
    final_score = base_score
    tool_call_reward = 0.0

    if base_score < 0:
        # num_turns=2 is baseline (user + assistant)
        # Each tool call adds 2 turns (tool_call + tool_result)
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        # Clamp so score never becomes positive when wrong
        final_score = min(0.0, base_score + tool_call_reward)

    return {
        "score": final_score,
        "acc": 1 if correct else 0,
        "pred": pred if pred is not None else "",
        "base_score": base_score,
        "tool_call_reward": tool_call_reward,
        "num_turns": num_turns,
    }


# =====================================================================================
# ======================== Multi-Turn ReAct Loop ======================================
# =====================================================================================


async def run_retool_multi_turn(
    messages: List[Dict[str, str]],
    model_base_url: str,
    sampling_params: Dict[str, Any],
    max_turns: int,
    max_response_per_turn: int,
    total_max_tokens: int,
    sandbox_url: str,
    sandbox_timeout: int,
    client: httpx.AsyncClient,
    chat_id: str,
    debug: bool = False,
    debug_prefix: str = "",
) -> Dict[str, Any]:
    """
    Run multi-turn ReAct loop: Generate -> Detect Tool Call -> Execute -> Append Result -> Repeat
    
    Based on Verl's multi-turn rollout logic with format=hermes.
    
    Returns dict with keys:
        - messages: final conversation messages
        - num_turns: number of turns
        - status: 1 for success, 0 for failure
        - chat_id: the chat identifier
        - train_data: additional training metadata
    """
    args = get_args()

    openai_client = openai.AsyncClient(
        api_key=chat_id, base_url=model_base_url, max_retries=0, timeout=3000
    )

    # Get model name
    try:
        resp = await openai_client.models.list()
        model = resp.data[0].id
    except Exception as e:
        print_with_rank_and_datetime(f"Failed to get model list: {e}", rank=0)
        return {
            "messages": messages,
            "num_turns": len(messages),
            "status": 0,
            "chat_id": chat_id,
            "train_data": {
                "error": str(e)
            },
        }

    current_messages = copy.deepcopy(messages)
    num_assistant_turns = 0
    num_user_turns = 0  # Verl-style: track user turns (tool responses) separately
    tool_calls_total = 0
    sandbox_ok = 0
    sandbox_fail = 0

    # Verl-style: max parallel tool calls per turn (prevents runaway tool execution)
    max_parallel_calls = 1  # Aligns with verl default

    if debug:
        print_with_rank_and_datetime(
            f"{debug_prefix}start max_turns={max_turns} sandbox_timeout={sandbox_timeout} sandbox_url={sandbox_url}",
            rank=0,
        )

    try:
        # Verl-style: while True loop with explicit break conditions
        while num_assistant_turns < max_turns:
            # NOTE:
            # - We intentionally do NOT enforce a client-side "remaining budget" across turns.
            #   sglang will handle response length (args.ppo_resp_seq_len) internally.
            # - Keep a per-turn cap only if explicitly configured (>0); otherwise fall back
            #   to total_max_tokens (aligned with --ppo-resp-seq-len).
            max_tokens_this_turn = total_max_tokens
            if isinstance(max_response_per_turn, int) and max_response_per_turn > 0:
                max_tokens_this_turn = min(max_tokens_this_turn, max_response_per_turn)

            max_connect_retry_times = 5
            while max_connect_retry_times > 0:
                try:
                    # Generate assistant response with tool calling
                    response = await openai_client.chat.completions.create(
                        model=model,
                        messages=current_messages,
                        tools=RETOOL_TOOLS,
                        tool_choice="auto",
                        temperature=sampling_params.get("temperature", 1.0),
                        top_p=sampling_params.get("top_p", 0.9),
                        max_tokens=max_tokens_this_turn,
                        extra_body={
                            "top_k": sampling_params.get("top_k", -1),
                            "task_id": chat_id  # Use consistent chat_id for all turns
                        }
                    )
                    break
                except openai.APIConnectionError as e:
                    print_with_rank_and_datetime(
                        f"{debug_prefix} turn={num_assistant_turns} retry={max_connect_retry_times}, connection error: {e}",
                        rank=0,
                    )
                finally:
                    max_connect_retry_times -= 1

            if max_connect_retry_times <= 0:
                break

            choice = response.choices[0]
            assistant_msg = choice.message

            num_assistant_turns += 1

            # VERL-style: Check if we've reached max assistant turns BEFORE processing tool calls
            # This ensures max num_turns = max_turns + (max_turns-1) + 1 = 16 (when max_turns=8)
            # instead of max_turns + max_turns + 1 = 17
            if num_assistant_turns >= max_turns:
                # Add assistant message and break without processing tool calls
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg.content or ""
                    }
                )
                break

            # Verl-style: Check for tool calls
            if not assistant_msg.tool_calls:
                # No tool call - add regular assistant message and break
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg.content or ""
                    }
                )
                break

            # Has tool calls - process them
            # Verl-style: limit tool calls per turn to max_parallel_calls
            tool_calls_to_process = assistant_msg.tool_calls[:max_parallel_calls]
            tool_calls_total += len(tool_calls_to_process)

            if debug:
                names = [tc.function.name for tc in tool_calls_to_process]
                names_str = ",".join(names[:4])
                if len(names) > 4:
                    names_str += f",+{len(names)-4}"
                print_with_rank_and_datetime(
                    f"{debug_prefix}turn={num_assistant_turns} tool_calls={len(tool_calls_to_process)} names={names_str}",
                    rank=0,
                )

            # Add assistant message with tool calls
            current_messages.append(
                {
                    "role":
                        "assistant",
                    "content":
                        assistant_msg.content or "",
                    "tool_calls":
                        [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function":
                                    {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    }
                            } for tc in tool_calls_to_process
                        ]
                }
            )

            # =====================================================================
            # VERL-aligned: Execute tool calls with VERL-style error handling
            # Key differences from original gcore:
            # 1. JSON parse failure → break WITHOUT adding tool response (like VERL Exception)
            # 2. Unknown tool → break WITHOUT adding tool response (like VERL KeyError)
            # 3. Sandbox timeout/error → add tool response and CONTINUE (VERL doesn't break)
            # =====================================================================
            any_fatal_failure = False  # Only for JSON parse / unknown tool (VERL-style Exception)
            tool_responses_to_add = []  # Collect tool responses first, add only if no fatal failure

            for tool_call in tool_calls_to_process:
                if tool_call.function.name == "code_interpreter":
                    # VERL-aligned: JSON parse failure = Exception = break without tool response
                    try:
                        args_dict = json.loads(
                            tool_call.function.arguments, strict=False
                        )  # strict=False, 对齐 verl
                        # args_dict 可能是个 str / array ... 单独用 AttributeError 处理下
                        code = args_dict.get("code", "")
                    except json.JSONDecodeError as e:
                        # VERL behavior: json.JSONDecoder().decode() failure → return Exception → break
                        # Do NOT add tool response, do NOT use raw string as code
                        if debug:
                            print_with_rank_and_datetime(
                                f"{debug_prefix}turn={num_assistant_turns} JSON parse failed for tool arguments: {e}",
                                rank=0,
                            )
                        any_fatal_failure = True
                        break  # Exit the for loop
                    except AttributeError as e:
                        if debug:
                            print_with_rank_and_datetime(
                                f"{debug_prefix}turn={num_assistant_turns} args_dict AttributeError: {e}",
                                rank=0,
                            )
                        any_fatal_failure = True
                        break  # Exit the for loop

                    # Execute code in sandbox
                    output, success = await execute_code_in_sandbox(
                        code=code,
                        sandbox_url=sandbox_url,
                        timeout=sandbox_timeout,
                        client=client,
                        debug=debug,
                        debug_prefix=f"{debug_prefix}turn={num_assistant_turns} ",
                    )
                    if success:
                        sandbox_ok += 1
                    else:
                        sandbox_fail += 1
                        # VERL-aligned: sandbox timeout/error → add tool response and CONTINUE
                        # VERL's sandbox returns error strings but does NOT cause Exception/break
                        # So we should NOT set any_fatal_failure here

                    # Collect tool result message (will add later if no fatal failure)
                    tool_responses_to_add.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "code_interpreter",
                            "content": output
                        }
                    )
                else:
                    # VERL-aligned: Unknown tool = KeyError = Exception = break without tool response
                    # VERL behavior: self.tools[tool_name] → KeyError → return Exception → break
                    # Do NOT add error message to messages
                    if debug:
                        print_with_rank_and_datetime(
                            f"{debug_prefix}turn={num_assistant_turns} Unknown tool: {tool_call.function.name}",
                            rank=0,
                        )
                    any_fatal_failure = True
                    break  # Exit the for loop

            # VERL-aligned: If fatal failure (JSON parse / unknown tool), break WITHOUT adding tool responses
            if any_fatal_failure:
                if debug:
                    print_with_rank_and_datetime(
                        f"{debug_prefix}turn={num_assistant_turns} breaking due to fatal tool failure (VERL-style: no tool response added)",
                        rank=0,
                    )
                break  # Exit the while loop - tool_responses_to_add is NOT added to current_messages

            # No fatal failure - add all collected tool responses to messages
            for tool_response in tool_responses_to_add:
                current_messages.append(tool_response)

            # Verl-style: increment user turns counter after processing tool responses
            num_user_turns += 1

            # Continue loop to next assistant turn
            # (loop condition will be checked at top of while)

    except Exception as e:
        if isinstance(e, openai.APIStatusError) and e.status_code == 406:
            # overlong conversation, just pass, it's a valid case in retool case
            # NOTE in this case the messages may not consist with the token_ids because of the last tool_response turn
            print_with_rank_and_datetime(f"message is overlong: {e}")
        else:
            print_with_rank_and_datetime(f"run_retool_multi_turn failed: {e}")
            traceback.print_exc()
            return {
                "messages": current_messages,
                "num_turns": len(current_messages),
                "status": 0,
                "chat_id": chat_id,
                "train_data": {
                    "error": str(e)
                },
            }

    # Verl-style: Calculate num_turns as user_turns + assistant_turns + 1 (initial user message)
    # This matches verl's tool_agent_loop.py line 131:
    #   num_turns=user_turns + assistant_turns + 1
    # Note: user_turns here means tool response turns, assistant_turns means LLM generation turns
    num_turns = num_user_turns + num_assistant_turns + 1

    if debug:
        print_with_rank_and_datetime(
            f"{debug_prefix}done num_turns={num_turns} (user={num_user_turns} assist={num_assistant_turns}) "
            f"tool_calls_total={tool_calls_total} sandbox_ok={sandbox_ok} sandbox_fail={sandbox_fail}",
            rank=0,
        )

    return {
        "messages": current_messages,
        "num_turns": num_turns,
        "status": 1,
        "chat_id": chat_id,
        "train_data":
            {
                "tool_calls_total": tool_calls_total,
                "sandbox_ok": sandbox_ok,
                "sandbox_fail": sandbox_fail,
                "num_assistant_turns": num_assistant_turns,
                "num_user_turns": num_user_turns,
            },
    }


# =====================================================================================
# ======================== Validation (idio_new.py style) =============================
# =====================================================================================


def is_valid_gen_data(data: Dict[str, Any]) -> bool:
    """
    Validate generation result (aligned with idio_new.py).
    Returns True if status == 1 (success).
    """
    try:
        status = data.get("status", 0)
        chat_id = data.get("chat_id", "")
        if status == 1:
            return True
        print(f"gen_data校验未通过，chat_id={chat_id}, status={status}")
        return False
    except Exception as e:
        print(f"判断gen_data失败: {e}")
        return False


# =====================================================================================
# ======================== Sampling Parameters ========================================
# =====================================================================================


def get_sampling_params(engine):
    """Get vLLM/sglang sampling parameters."""
    tokenizer = get_tokenizer()
    args = get_args()

    stop_at_token_id = tokenizer._tokenizer.eos_token_id

    return engine.get_sampling_params(
        n=1,
        temperature=args.ppo_rollout_temperature,
        top_k=args.ppo_rollout_top_k if args.ppo_rollout_top_k > 0 else -1,
        top_p=args.ppo_rollout_top_p,
        max_tokens=args.ppo_resp_seq_len,
        stop_token_ids=[stop_at_token_id],
        seed=args.seed,
    )


def get_model_conf(engine, sampling_params):
    """Convert sampling params to model config dict."""
    if engine.infer_engine_impl == "vllm":
        return sampling_params
    else:
        params = copy.deepcopy(sampling_params.__dict__)
        if 'seed' in params:
            params.pop('seed')  # seed not supported by sglang
        if "max_new_tokens" in params:
            params["max_new_tokens"] -= 1  # sglang position embedding limit
        return params


# =====================================================================================
# ======================== Main Rollout Generation (idio_new.py style) ================
# =====================================================================================


@torch.no_grad()
async def gen_rollouts(engine, batch, sampling_repeat, ppo_step):
    """
    Generate rollouts with multi-turn tool calling (ReTool style).
    Upgraded to idio_new.py framework with:
      - asyncio.sleep(10) staggered launch
      - Robust failure handling with dummy data
      - log_probs/mask roll(-1) alignment
      - messages_traj output
      - Enhanced sample_mask checks
    """
    args = get_args()
    tokenizer = get_tokenizer()
    my_rank = torch.distributed.get_rank()

    # Get sandbox URL from environment or args
    sandbox_url = os.environ.get(
        "SANDBOX_FUSION_URL",
        f"http://127.0.0.1:{os.environ.get('SANDBOX_LOCAL_PORT', '8008')}/run_code"
    )

    # Get ReTool-specific args
    max_turns = getattr(args, 'retool_max_turns', 8)
    sandbox_timeout = getattr(args, 'retool_sandbox_timeout', 20)
    max_response_per_turn = getattr(args, 'retool_max_response_per_turn', 4096)
    total_max_tokens = args.ppo_resp_seq_len

    # Engine endpoint
    ep_ip = args.ppo_sampler_ips[mpu.get_data_parallel_rank()]
    ep_port = args.ppo_sampler_ports[mpu.get_data_parallel_rank()]
    model_base_url = f"http://{ep_ip}:{ep_port}/v1"

    # Extract batch data
    prompt_token_ids = batch["prompt_token_ids"]
    lpad_lens = batch["lpad_lens"]
    gt_label = batch["gt_label"]
    messages = batch["messages"]

    num_prompts = len(prompt_token_ids)
    num_results = num_prompts * sampling_repeat

    # Prepare sampling params
    sampling_params = get_sampling_params(engine)
    model_conf = get_model_conf(engine, sampling_params)

    # Build sampling params dict for run_retool_multi_turn
    sampling_params_dict = {
        "temperature":
            model_conf.get("temperature", 1.0)
            if isinstance(model_conf, dict) else getattr(model_conf, "temperature", 1.0),
        "top_p":
            model_conf.get("top_p", 0.9)
            if isinstance(model_conf, dict) else getattr(model_conf, "top_p", 0.9),
        "top_k":
            model_conf.get("top_k", -1)
            if isinstance(model_conf, dict) else getattr(model_conf, "top_k", -1),
    }

    # =========================================================================
    # 1. Call WebAPI with staggered launch (idio_new.py style: asyncio.sleep(10))
    # =========================================================================
    pending_idx_list = list(range(num_results))
    gen_outputs = [None] * num_results

    async with httpx.AsyncClient(timeout=300) as client:
        tasks = []

        for pending_idx in pending_idx_list:
            prompt_idx = pending_idx // sampling_repeat

            # Deep copy messages for this rollout
            initial_messages = copy.deepcopy(messages[prompt_idx])
            chat_id = f"rank{my_rank}_step{ppo_step}_idx{pending_idx}_{uuid.uuid4().hex}"

            # Debug settings
            debug_enabled = _retool_debug_enabled() and (pending_idx < _retool_debug_first_n()
                                                        ) and (my_rank == 0)
            debug_prefix = f"[ReTool][dbg step={ppo_step} idx={pending_idx}] "

            # Create async task
            task = asyncio.create_task(
                run_retool_multi_turn(
                    messages=initial_messages,
                    model_base_url=model_base_url,
                    sampling_params=sampling_params_dict,
                    max_turns=max_turns,
                    max_response_per_turn=max_response_per_turn,
                    total_max_tokens=total_max_tokens,
                    sandbox_url=sandbox_url,
                    sandbox_timeout=sandbox_timeout,
                    client=client,
                    chat_id=chat_id,
                    debug=debug_enabled,
                    debug_prefix=debug_prefix,
                )
            )
            tasks.append((pending_idx, prompt_idx, task))

            # Staggered launch: wait 10 seconds between task starts (idio_new.py style)
            await asyncio.sleep(10)

        # Gather all results
        raw_results = await asyncio.gather(*[t for _, _, t in tasks], return_exceptions=True)

        # Map results back to pending indices
        for i, (pending_idx, prompt_idx, _) in enumerate(tasks):
            result = raw_results[i]
            if isinstance(result, Exception):
                print_with_rank_and_datetime(
                    f"[ReTool] pending_idx={pending_idx} failed with exception: {result}", rank=0
                )
                gen_outputs[pending_idx] = {
                    "messages": [],
                    "num_turns": 0,
                    "status": 0,
                    "chat_id": f"rank{my_rank}_step{ppo_step}_idx{pending_idx}_failed",
                    "train_data": {
                        "error": str(result)
                    },
                }
            else:
                gen_outputs[pending_idx] = result

    # =========================================================================
    # 2. Collect failed indices (idio_new.py style - no local fallback for ReTool)
    # =========================================================================
    local_indices = []
    for pending_idx, gen in enumerate(gen_outputs):
        if not is_valid_gen_data(gen):
            print_with_rank_and_datetime(
                f"train_ppo_sampler {pending_idx=} marked as failed (no local fallback for ReTool)",
                rank=0
            )
            local_indices.append(pending_idx)

    # =========================================================================
    # 3. Process results (idio_new.py style with enhanced failure handling)
    # =========================================================================
    tokens = [None] * num_results
    mask_list = [None] * num_results
    sequence_lengths = [None] * num_results
    prompt_lens = [None] * num_results
    gt_label_list = [None] * num_results
    num_webapi_fail = [None] * num_results
    sample_mask = [None] * num_results
    rollout_log_probs = [None] * num_results
    routed_experts_list = [None] * num_results
    messages_traj = [None] * num_results  # New: idio_new.py style
    response_lengths = [None] * num_results

    for pending_idx, gen in enumerate(gen_outputs):
        prompt_idx = pending_idx // sampling_repeat
        chat_id = gen.get('chat_id', '')

        # Get chat state from engine
        chat_state = engine.openai_serving_chat.get_chat_state(chat_id)

        # Prepare train_data with error flags (idio_new.py style)
        train_data = copy.deepcopy(gen.get("train_data", {}))
        train_data['status'] = gen.get('status', 0)
        train_data['remote_error'] = False
        train_data['remote_error_status'] = False
        train_data['remote_error_local_fallback'] = False

        # =====================================================================
        # Handle failures: fill with dummy data (idio_new.py style)
        # =====================================================================
        if chat_state.get('status') != 'success' or pending_idx in local_indices:
            print_with_rank_and_datetime(
                f"train_ppo_sampler get {chat_id=} fail, fill with dummy data", rank=0
            )

            # Dummy token data
            tokens[pending_idx] = torch.zeros(1, dtype=torch.long)
            rollout_log_probs[pending_idx] = torch.ones(1, dtype=torch.float32)
            sequence_lengths[pending_idx] = torch.tensor(1, dtype=torch.long)
            # Clamp prompt length to seq_len to avoid negative response lengths in metrics
            orig_prompt_len = lpad_lens[prompt_idx]
            orig_prompt_len = int(
                orig_prompt_len.item() if torch.is_tensor(orig_prompt_len) else orig_prompt_len
            )
            prompt_lens[pending_idx] = torch.tensor(min(orig_prompt_len, 1), dtype=torch.long)

            # Set error flags
            if chat_state.get('status') != 'success':
                train_data['remote_error'] = True
                train_data['remote_error_status'] = True
            if pending_idx in local_indices:
                train_data['remote_error'] = True
                train_data['remote_error_local_fallback'] = True

            # Copy gt_label with train_data
            gt_label_list[pending_idx] = copy.deepcopy(gt_label[prompt_idx])
            if not isinstance(gt_label_list[pending_idx], dict):
                gt_label_list[pending_idx] = {"ground_truth": str(gt_label_list[pending_idx])}
            gt_label_list[pending_idx]['train_data'] = train_data

            # Penalty reward for failed rollout
            gt_label_list[pending_idx]["retool_reward"] = -1.0
            gt_label_list[pending_idx]["retool_acc"] = 0
            gt_label_list[pending_idx]["retool_pred"] = ""
            gt_label_list[pending_idx]["retool_num_turns"] = 0

            mask_list[pending_idx] = torch.zeros(1, dtype=torch.bool)
            num_webapi_fail[pending_idx] = len(local_indices)
            sample_mask[pending_idx] = torch.tensor(False)
            messages_traj[pending_idx] = []
            response_lengths[pending_idx] = torch.tensor(0, dtype=torch.long)
            continue

        # =====================================================================
        # Successful rollout: process normally
        # =====================================================================
        token = torch.tensor(chat_state['token_ids'], dtype=torch.long)
        seq_len = len(token)

        # Absolute safety cap: never allow training token sequence to exceed model seq_length
        if getattr(args, "seq_length", None) is not None and seq_len > args.seq_length:
            token = token[:args.seq_length]
            seq_len = args.seq_length

        # Build mask from gen_intervals
        gen_intervals = chat_state.get('gen_intervals', [])
        mask = torch.zeros(seq_len, dtype=torch.bool)
        for s, e in gen_intervals:
            s_clamped = min(s, seq_len)
            e_clamped = min(e, seq_len)
            if s_clamped < e_clamped:
                mask[s_clamped:e_clamped] = True

        # DEBUG: Log gen_intervals and mask info for verification
        mask_before_roll = mask.clone()  # Save mask before roll for debug

        # Get log_probs and mask non-generated tokens
        token_logprobs = chat_state.get("token_logprobs", None)
        if token_logprobs is None:
            log_probs = torch.ones(seq_len, dtype=torch.float32)
        else:
            log_probs = torch.tensor(token_logprobs, dtype=torch.float32)
            if len(log_probs) > seq_len:
                log_probs = log_probs[:seq_len]
            elif len(log_probs) < seq_len:
                log_probs = torch.cat(
                    [log_probs,
                     torch.ones(seq_len - len(log_probs), dtype=torch.float32)]
                )

        log_probs.masked_fill_(~mask, 1)

        # =====================================================================
        # CRITICAL: roll(-1) alignment (idio_new.py style)
        # This aligns token's logprob with next token/action
        # =====================================================================
        log_probs = log_probs.roll(-1)
        mask = mask.roll(-1)

        # =====================================================================
        # DEBUG: Log gen_intervals and mask for verification
        # Enable with: export RETOOL_DEBUG_GEN_INTERVALS=1
        # Logs to: retool_output/log/gen_intervals_debug.log
        # =====================================================================
        if _should_debug_gen_intervals(ppo_step, pending_idx):
            final_messages_for_debug = gen.get('messages', [])
            _log_gen_intervals_debug(
                ppo_step=ppo_step,
                pending_idx=pending_idx,
                chat_id=chat_id,
                gen_intervals=gen_intervals,
                token_ids=token.tolist(),
                mask_before_roll=mask_before_roll,
                mask_after_roll=mask,
                final_messages=final_messages_for_debug,
                tokenizer=tokenizer,
            )

        # MoE router replay support (idio_new.py style)
        if getattr(args, 'moe_router_replay', False):
            routed_experts = chat_state.get('token_routed_experts', None)
            if routed_experts is not None:
                assert len(routed_experts) == len(token), \
                    f"routed_experts length mismatch: {len(routed_experts)} vs {len(token)}"
                routed_experts_list[pending_idx] = torch.tensor(routed_experts, dtype=torch.int32)

        # Store token data
        tokens[pending_idx] = token
        mask_list[pending_idx] = mask
        rollout_log_probs[pending_idx] = log_probs
        sequence_lengths[pending_idx] = torch.tensor(seq_len, dtype=torch.long)
        # Clamp prompt length so it never exceeds seq_len (prevents negative response lengths)
        orig_prompt_len = lpad_lens[prompt_idx]
        orig_prompt_len = int(
            orig_prompt_len.item() if torch.is_tensor(orig_prompt_len) else orig_prompt_len
        )
        prompt_lens[pending_idx] = torch.tensor(min(orig_prompt_len, seq_len), dtype=torch.long)
        response_lengths[pending_idx] = mask.sum().long()

        # =====================================================================
        # Compute ReTool reward
        # =====================================================================
        final_messages = gen.get('messages', [])
        num_turns = gen.get('num_turns', len(final_messages))

        # Extract response text for reward calculation
        response_text = ""
        for msg in final_messages[2:]:  # Skip system and user messages
            if msg.get("role") == "assistant":
                response_text += msg.get("content", "") + "\n"

        # Get ground truth
        gt = gt_label[prompt_idx]
        if isinstance(gt, dict):
            ground_truth = gt.get("ground_truth", gt.get("answer", ""))
        else:
            ground_truth = str(gt)

        # Compute reward
        reward_result = compute_retool_reward(
            response_str=response_text,
            ground_truth=ground_truth,
            num_turns=num_turns,
        )

        # Pack into gt_label
        gt_label_copy = copy.deepcopy(gt_label[prompt_idx])
        if not isinstance(gt_label_copy, dict):
            gt_label_copy = {"ground_truth": str(gt_label_copy)}

        gt_label_copy['train_data'] = train_data
        gt_label_copy["retool_reward"] = reward_result["score"]
        gt_label_copy["retool_acc"] = reward_result["acc"]
        gt_label_copy["retool_pred"] = reward_result["pred"]
        gt_label_copy["retool_num_turns"] = reward_result["num_turns"]
        gt_label_copy["retool_base_score"] = reward_result["base_score"]
        gt_label_copy["retool_tool_bonus"] = reward_result["tool_call_reward"]
        gt_label_list[pending_idx] = gt_label_copy

        # Failure count
        num_webapi_fail[pending_idx] = len(local_indices)

        # =====================================================================
        # Enhanced sample_mask check (idio_new.py style)
        # =====================================================================
        # overlong = chat_state.get('reach_context_len_thr', False) \
        #         or chat_state.get('input_messages_overlong', False)
        sample_mask[pending_idx] = torch.tensor(True)

        # Store messages trajectory (idio_new.py style)
        messages_traj[pending_idx] = chat_state.get('messages', final_messages)

        # DEBUG: Low-frequency dump of messages_traj to file (rank0 only)
        # Enable with: export RETOOL_DEBUG_MESSAGES_TRAJ=1
        # Optional controls:
        #   - RETOOL_DEBUG_MESSAGES_TRAJ_STEP_INTERVAL (default 50)
        #   - RETOOL_DEBUG_MESSAGES_TRAJ_FIRST_N (default 1)
        # Logs to: tasks/retool/retool_output/log/messages_traj_debug.log
        if _should_debug_messages_traj(ppo_step, pending_idx, my_rank):
            _log_messages_traj_debug(
                ppo_step=ppo_step,
                pending_idx=pending_idx,
                chat_id=chat_id,
                messages_traj=messages_traj[pending_idx] or [],
                final_messages=final_messages or [],
            )

        # Periodic logging
        if pending_idx % 50 == 0:
            gen_token_count = int(mask.sum().item())
            print_with_rank_and_datetime(
                f"[ReTool] idx={pending_idx} turns={num_turns} "
                f"seq_len={seq_len} gen_tokens={gen_token_count} "
                f"score={reward_result['score']:.3f} acc={reward_result['acc']} "
                f"pred={reward_result['pred'][:50] if reward_result['pred'] else 'None'}",
                rank=0
            )

    # =========================================================================
    # 4. Build rollout batch
    # =========================================================================
    rollout_batch = {
        'tokens': tokens,
        'rollout_log_probs': rollout_log_probs,
        'sequence_lengths': sequence_lengths,
        'prompt_lengths': prompt_lens,
        'gt_label': gt_label_list,
        'mask': mask_list,
        'num_webapi_fail': num_webapi_fail,
        'sample_mask': sample_mask,
        'messages_traj': messages_traj,  # New: idio_new.py style
        'response_lengths': response_lengths,
    }

    # MoE router replay support (idio_new.py style)
    if getattr(args, 'moe_router_replay', False):
        rollout_batch["routed_experts"] = routed_experts_list

    return rollout_batch


# =====================================================================================
# ======================== Entry Point ================================================
# =====================================================================================

if __name__ == "__main__":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    init_gpatch_for_mcore()
    grpo_sampler = GrpoSamplerV3()
    run_grpo_sampler_v3(
        grpo_sampler,
        model_provider,
        gen_rollouts,
        extra_args_provider=get_tasks_args,
    )
