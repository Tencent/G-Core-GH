"""RetoolEnv – async-rollout-friendly env for math ReTool.

Task data (question + ground truth) is injected via ``data``
in ``reset()`` by ``EnvAgentLoopActor``, instead of being loaded
from a metadata file internally.

Expected ``data`` keys (from DataSource / cleaned_data)::

    {
        "question": ["What is 2+2?"],   # list[str], mbs=1 → len 1
        "target":   ["4"],               # list[str], ground truth
    }
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from gem import Env

from gpatch_v4.agentic.tools.tool_parsers.core_types import ParseResult
from gpatch_v4.utils import log

from .reward_utils import (
    RETOOL_TRAIN_TOOL_OUTPUT_MAX_CHARS,
    RETOOL_TRAIN_TOOL_OUTPUT_MAX_LINES,
    compute_retool_reward,
    truncate_tool_output,
)
from .sandbox_sync import execute_code_in_sandbox_sync


def _default_sandbox_url() -> str:
    return os.environ.get(
        "SANDBOX_FUSION_URL",
        f"http://127.0.0.1:{os.environ.get('SANDBOX_LOCAL_PORT', '8008')}/run_code",
    )


class RetoolEnv(Env):
    """One episode = one math problem, task data from external DataSource.

    Each step is one assistant generation.
    Tool calls → sandbox stdout as next observation (use_tool=True).
    No tool calls → terminal step with ReTool reward.
    """
    def __init__(
        self,
        answer_format_suffix: str = (
            "\nThe answer format must be: \\boxed{'The final answer goes here.'}"
        ),
        sandbox_timeout: int = 20,
        memory_limit_mb: int = 1024,
        env_instruction: Optional[str] = None,
        max_steps: int = 8,
        **kwargs: Any,
    ):
        self.answer_format_suffix = answer_format_suffix or ""
        self.sandbox_timeout = int(sandbox_timeout)
        self.memory_limit_mb = int(memory_limit_mb)
        self.max_steps = int(max_steps)
        self._env_instruction = env_instruction or (
            "Solve the problem. You may call code_interpreter to run Python when helpful."
        )
        self._ground_truth: str = ""
        self._observation: str = ""
        self._msg_turn_count: int = 0
        self._last_terminal_metrics: Dict[str, Any] = {}
        self._all_actions: List[str] = []
        self._ppo_step: Optional[int] = None

    def reset(
        self,
        seed: int = 42,
        ppo_step: int = 0,
        data: Dict[str, Any] = None,
    ):
        Env.reset(self, seed)
        assert data is not None, (
            "RetoolEnv requires data; set data.py_path to a dataset "
            "that returns {question, target} fields"
        )
        self._ppo_step = ppo_step
        question_list = data["question"]
        target_list = data["target"]
        problem = question_list[0] if isinstance(question_list, list) else question_list
        self._ground_truth = (target_list[0] if isinstance(target_list, list) else str(target_list))

        self._observation = str(problem) + self.answer_format_suffix
        self._msg_turn_count = 0
        self._last_terminal_metrics = {
            "success": False,
            "format_penalty": 0.0,
            "action_is_valid": True,
            "action_is_effective": True,
        }
        self._all_actions = []
        info = {
            "env_instruction": self._env_instruction,
            "ground_truth": self._ground_truth,
            "metrics": dict(self._last_terminal_metrics),
            "metrics_agg_mode":
                {
                    "success": "last",
                    "format_penalty": "mean",
                    "action_is_valid": "mean",
                    "action_is_effective": "mean",
                },
        }
        log(f"[RetoolEnv] reset: {self._observation}", rank=0)
        return self._observation, info, False

    def _parse_tool_code(self, tool_call_parse_result: ParseResult) -> Tuple[Optional[str], bool]:
        if not tool_call_parse_result.calls:
            return None, False
        tc = tool_call_parse_result.calls[0]
        name = (tc.function.name or "").strip()
        if name != "code_interpreter":
            return None, True
        raw_args = tc.function.arguments
        try:
            if isinstance(raw_args, dict):
                args_dict = raw_args
            else:
                args_dict = json.loads(raw_args or "{}", strict=False)
            code = args_dict.get("code", "") if isinstance(args_dict, dict) else ""
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None, True
        return str(code), False

    def step(
        self,
        action: str,
        tool_call_parse_result: Optional[ParseResult] = None,
    ) -> Tuple[Any, float, bool, bool, Dict]:
        self._msg_turn_count += 1
        metrics_agg_mode = {
            "success": "last",
            "format_penalty": "mean",
            "action_is_valid": "mean",
            "action_is_effective": "mean",
        }
        self._all_actions.append(action)

        if tool_call_parse_result is None or len(tool_call_parse_result.calls) == 0:
            rew_info = compute_retool_reward(
                "\n".join(self._all_actions),
                self._ground_truth,
                2 * self._msg_turn_count,
            )
            won = rew_info["acc"] == 1
            metrics = {
                "success": won,
                "format_penalty": 0.0,
                "action_is_valid": True,
                "action_is_effective": True,
                "retool_acc": float(rew_info["acc"]),
            }
            self._last_terminal_metrics = metrics
            info = {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
                "won": won,
                "valid": True,
                "use_tool": False,
                "retool_detail": rew_info,
            }
            return "", float(rew_info["score"]), True, False, info

        code, fatal = self._parse_tool_code(tool_call_parse_result)
        if fatal or code is None or not str(code).strip():
            rew_info = compute_retool_reward(
                "\n".join(self._all_actions),
                self._ground_truth,
                2 * self._msg_turn_count,
            )
            metrics = {
                "success": False,
                "format_penalty": -0.5,
                "action_is_valid": False,
                "action_is_effective": False,
            }
            self._last_terminal_metrics = metrics
            info = {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
                "won": False,
                "valid": False,
                "use_tool": False,
                "err_flag": ["tool_parse_fatal"],
            }
            return "", float(rew_info["score"]), True, False, info

        out, ok = execute_code_in_sandbox_sync(
            code,
            _default_sandbox_url(),
            timeout=self.sandbox_timeout,
            memory_limit_mb=self.memory_limit_mb,
        )
        out = truncate_tool_output(
            out,
            max_lines=RETOOL_TRAIN_TOOL_OUTPUT_MAX_LINES,
            max_chars=RETOOL_TRAIN_TOOL_OUTPUT_MAX_CHARS,
        )
        self._msg_turn_count += 1
        metrics = {
            "success": False,
            "format_penalty": 0.0,
            "action_is_valid": True,
            "action_is_effective": ok,
            "sandbox_ok": ok,
        }
        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "use_tool": True,
            "valid": True,
        }
        return out, 0.0, False, False, info

    def report_traj(self, payload: Dict) -> Dict:
        return {}

    def sample_random_action(self) -> str:
        return "\\boxed{0}"
