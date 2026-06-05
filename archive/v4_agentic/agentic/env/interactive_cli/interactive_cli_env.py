"""
CLI interactive environment for TrajEnvManager testing.

Displays each LLM response in the terminal, then reads user input as the next observation.
Only for debugging.
Special inputs: ``\\term`` (terminated), ``\\abort`` (truncated).
"""

from __future__ import annotations

from contextlib import suppress

with suppress(ImportError):
    from gem import Env
if 'Env' not in globals():
    Env = object
import json

from gpatch_v4.agentic.tools.tool_parsers.core_types import ParseResult


class InteractiveCliEnv(Env):
    """Terminal-based env: step(action) prints the model response and blocks on stdin."""

    _TERM = "\\term"
    _ABORT = "\\abort"

    def __init__(self, welcome_message: str | None = None, **kwargs):
        # gem.make may pass extra keys; ignore unknown kwargs like SokobanEnv.
        self._welcome = welcome_message or (
            "Interactive CLI env ready. After each model reply, type the next observation. "
            f"Send '{self._TERM}' to end episode (terminated), '{self._ABORT}' for truncated."
        )
        self._step_idx = 0

    def reset(self, seed=None):
        Env.reset(self, seed)
        self._step_idx = 0
        info = {
            "env_instruction":
                (
                    "You are in an interactive CLI test: the user will type the next observation after each reply."
                ),
        }
        return self._welcome, info, False

    def step(self, action: str, tool_call_parse_result: ParseResult = None):
        self._step_idx += 1
        print("\n" + "=" * 60)
        print(f"[assistant / step {self._step_idx}]")
        print(f"Action: {action}")
        if tool_call_parse_result is not None and \
            len(tool_call_parse_result.calls) > 0:
            print(f"Tool calls: {json.dumps(tool_call_parse_result.model_dump(), indent=4)}")
        else:
            print(f"Normal text: {tool_call_parse_result.normal_text}")
        print("=" * 60 + "\n")

        user_line = input("Observation (your reply as env state): ").rstrip("\n")

        if user_line == self._TERM:
            return "", 0.0, True, False, {}
        if user_line == self._ABORT:
            return "", 0.0, False, True, {}

        return user_line, 0.0, False, False, {}

    def report_traj(self, payload):
        return {}

    def sample_random_action(self) -> str:
        """Used by RandomProxy smoke tests."""
        return "<random_test_reply>ok</random_test_reply>"
