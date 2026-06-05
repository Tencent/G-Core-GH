"""Minimal env for agentic RL tests (TrajEnvManager + ToolReasonEnvWrapper). Importable from training workers."""
from __future__ import annotations

from contextlib import suppress
from typing import Any, Dict, Optional, Tuple

with suppress(ImportError):
    from gem import Env
if 'Env' not in globals():
    Env = object

from gpatch_v4.agentic.reasoning_parsers.core_types import ReasoningParseResult
from gpatch_v4.agentic.tools.tool_parsers.core_types import ParseResult


class SimpleAgenticEnv(Env):
    """
    Hardcoded problem: "What is 2+2?", ground_truth = "4".
    - Tool call  -> mock output "4", reward 0, not terminated (unless max_steps).
    - No tool call -> check \\boxed{4}, reward, terminated.
    """
    def __init__(self, max_steps: int = 3, **kwargs: Any):
        self.max_steps = int(max_steps)
        self._step_count = 0
        self._ground_truth = "4"

    def reset(
        self,
        seed: int = 42,
        ppo_step: int = 0,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Match ``RetoolEnv`` / ``TrajEnvManager`` call signature; task stays hardcoded."""
        Env.reset(self, seed)
        self._step_count = 0
        observation = (
            "What is 2+2?"
            "\nThe answer format must be: \\boxed{'The final answer goes here.'}"
        )
        info = {
            "env_instruction":
                "Solve the math problem. You may call code_interpreter to run Python.",
            "ground_truth":
                self._ground_truth,
            "metrics": {
                "success": False
            },
            "metrics_agg_mode": {
                "success": "last"
            },
        }
        return observation, info, False

    def step(
        self,
        action: str,
        tool_call_parse_result: Optional[ParseResult] = None,
        reasoning_parse_result: Optional[ReasoningParseResult] = None,
    ) -> Tuple[Any, float, bool, bool, Dict]:
        self._step_count += 1
        metrics_agg_mode = {"success": "last"}

        has_tool_call = (
            tool_call_parse_result is not None and len(tool_call_parse_result.calls) > 0
        )

        if not has_tool_call:
            correct = "\\boxed{4}" in action
            reward = 1.0 if correct else -1.0
            info = {
                "metrics": {
                    "success": correct
                },
                "metrics_agg_mode": metrics_agg_mode,
                "use_tool": False,
                "valid": True,
            }
            return "", reward, True, False, info

        terminated = self._step_count >= self.max_steps
        info = {
            "metrics": {
                "success": False
            },
            "metrics_agg_mode": metrics_agg_mode,
            "use_tool": True,
            "valid": True,
        }
        return "4", 0.0, terminated, False, info

    def report_traj(self, payload: Dict) -> Dict:
        return {}

    def sample_random_action(self) -> str:
        return "\\boxed{0}"
