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
from tasks.retool.internal_agent.tool_agent_loop import ToolAgentLoop

model_provider = default_sampler_model_provider

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
    engine,
    tokenizer,
    messages: List[Dict[str, str]],
    model_base_url: str,
    sampling_params: Dict[str, Any],
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
    _tool_agent_loop = ToolAgentLoop(engine, tokenizer)
    output = await _tool_agent_loop.run(messages, sampling_params)
    return output.model_dump()


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

            tmp_sampling_params = copy.deepcopy(sampling_params)
            tmp_sampling_params.seed += ppo_step * num_results + pending_idx
            # Create async task
            task = asyncio.create_task(
                run_retool_multi_turn(
                    engine,
                    tokenizer._tokenizer,
                    messages=initial_messages,
                    model_base_url=model_base_url,
                    sampling_params=tmp_sampling_params,
                )
            )
            tasks.append((pending_idx, prompt_idx, task))

            # Staggered launch: wait 10 seconds between task starts (idio_new.py style)
            await asyncio.sleep(10)

        # Gather all results
        raw_results = await asyncio.gather(*[t for _, _, t in tasks], return_exceptions=False)

        # Map results back to pending indices
        for i, (pending_idx, prompt_idx, _) in enumerate(tasks):
            result = raw_results[i]
            if isinstance(result, Exception):
                print_with_rank_and_datetime(
                    f"[ReTool] pending_idx={pending_idx} failed with exception: {result}", rank=0
                )
                gen_outputs[pending_idx] = result
            else:
                gen_outputs[pending_idx] = result

    # =========================================================================
    # 3. Process results (idio_new.py style with enhanced failure handling)
    # =========================================================================
    tokens = [None] * num_results
    mask_list = [None] * num_results
    sequence_lengths = [None] * num_results
    prompt_lens = [None] * num_results
    gt_label_list = [None] * num_results
    routed_experts_list = [None] * num_results
    messages_traj = [None] * num_results  # New: idio_new.py style
    response_lengths = [None] * num_results

    for pending_idx, gen in enumerate(gen_outputs):
        prompt_idx = pending_idx // sampling_repeat
        # =====================================================================
        # Handle failures: fill with dummy data (idio_new.py style)
        # =====================================================================
        token = gen["prompt_ids"] + gen["response_ids"]
        mask = torch.tensor([0] * len(gen["prompt_ids"]) + gen["response_mask"], dtype=torch.bool)
        mask = mask.roll(-1)
        seq_len = len(mask)

        assert len(token) == len(mask), "token and mask length mismatch"

        tokens[pending_idx] = torch.tensor(token, dtype=torch.long)
        mask_list[pending_idx] = mask
        sequence_lengths[pending_idx] = torch.tensor(seq_len, dtype=torch.long)
        prompt_lens[pending_idx] = torch.tensor(len(gen["prompt_ids"]), dtype=torch.long)
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

        gt_label_copy["retool_reward"] = reward_result["score"]
        gt_label_copy["retool_acc"] = reward_result["acc"]
        gt_label_copy["retool_pred"] = reward_result["pred"]
        gt_label_copy["retool_num_turns"] = reward_result["num_turns"]
        gt_label_copy["retool_base_score"] = reward_result["base_score"]
        gt_label_copy["retool_tool_bonus"] = reward_result["tool_call_reward"]
        gt_label_list[pending_idx] = gt_label_copy

        # Store messages trajectory (idio_new.py style)
        messages_traj[pending_idx] = gen.get('messages', final_messages)

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
        'sequence_lengths': sequence_lengths,
        'prompt_lengths': prompt_lens,
        'gt_label': gt_label_list,
        'mask': mask_list,
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
