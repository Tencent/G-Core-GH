from __future__ import annotations

import inspect
import json
from contextlib import suppress
from typing import Any, Dict, Optional, Tuple

with suppress(ImportError):
    from gem.core import Env, EnvWrapper
if 'EnvWrapper' not in globals():
    EnvWrapper = object

from gpatch_v4.agentic.reasoning_parsers import BaseReasoningParser, get_reasoning_parser
from gpatch_v4.agentic.tools.tool_parsers import BaseToolParser, get_tool_parser
from gpatch_v4.utils import log


class ToolReasonEnvWrapper(EnvWrapper):
    """Env wrapper that parses tool calls and/or reasoning before ``env.step``.

    Both parsers are optional and independently enabled:

    * Tool parsing is enabled when ``env_tool_config.use_tools`` is True.
      Parsed tool calls are forwarded to ``env.step`` via the
      ``tool_call_parse_result`` keyword argument; the inner env's ``step``
      MUST declare this parameter.
    * Reasoning parsing is enabled when both ``enable_thinking`` is True
      and ``reasoning_parser_name`` is a non-empty string. Parsed reasoning
      is forwarded via the ``reasoning_parse_result`` keyword argument; the
      inner env's ``step`` MUST declare this parameter.

    When reasoning parsing is enabled, tool parsing runs on the
    ``normal_text`` (post-think) portion to avoid spurious detection of
    function-call markup inside the chain-of-thought.
    """
    def __init__(
        self,
        env: Env,
        env_tool_config: Optional[Any] = None,
        reasoning_parser_name: str = "",
        enable_thinking: bool = False,
    ):
        super().__init__(env)
        self.env_tool_config = env_tool_config
        self.reasoning_parser_name = reasoning_parser_name
        self.enable_thinking = bool(enable_thinking)
        self.tools: Optional[list] = None
        self.tool_parser: Optional[BaseToolParser] = None
        self.reasoning_parser: Optional[BaseReasoningParser] = None

        self._setup_tool_parser()
        self._setup_reasoning_parser()

    def _step_has_param(self, name: str) -> bool:
        return inspect.signature(self.env.step).parameters.get(name) is not None

    def _setup_tool_parser(self) -> None:
        if self.env_tool_config is None or not self.env_tool_config.use_tools:
            return
        if not self._step_has_param("tool_call_parse_result"):
            log(
                "Env.step must declare a 'tool_call_parse_result' parameter when "
                "env_tool_config.use_tools is enabled."
            )
            raise ValueError("Env.step missing 'tool_call_parse_result' parameter for tool parsing")

        tools_json_path = self.env_tool_config.tools_json_path
        try:
            with open(tools_json_path, "r") as f:
                self.tools = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load tools from {tools_json_path}: {e}")

        tool_parser_cls = get_tool_parser(self.env_tool_config.tool_call_parser)
        try:
            self.tool_parser = tool_parser_cls(self.tools)
        except Exception as e:
            raise ValueError(f"Failed to initialize tool parser: {e}")

    def _setup_reasoning_parser(self) -> None:
        if not (self.enable_thinking and self.reasoning_parser_name):
            return
        if not self._step_has_param("reasoning_parse_result"):
            log(
                "Env.step must declare a 'reasoning_parse_result' parameter when "
                "training.enable_thinking is True and reasoning_parser is configured."
            )
            raise ValueError(
                "Env.step missing 'reasoning_parse_result' parameter for reasoning parsing"
            )

        reasoning_parser_cls = get_reasoning_parser(self.reasoning_parser_name)
        try:
            self.reasoning_parser = reasoning_parser_cls()
        except Exception as e:
            raise ValueError(f"Failed to initialize reasoning parser: {e}")

    def reset(self, **kwargs) -> Tuple[Any, Dict, bool]:
        observation, info, terminated = self.env.reset(**kwargs)
        if self.tools is not None:
            info.update({"tools": self.tools})
        return observation, info, terminated

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        extra: Dict[str, Any] = {}
        text_for_tool = action

        if self.reasoning_parser is not None:
            reasoning_result = self.reasoning_parser.extract_reasoning(action)
            extra["reasoning_parse_result"] = reasoning_result
            if reasoning_result.normal_text:
                text_for_tool = reasoning_result.normal_text

        if self.tool_parser is not None:
            tool_call_parse_result = self.tool_parser.extract_tool_calls(text_for_tool)
            extra["tool_call_parse_result"] = tool_call_parse_result

        obs, reward, terminated, truncated, info = self.env.step(action, **extra)
        if info is None:
            info = {}
        if "tool_call_parse_result" in extra:
            info["use_tool"] = len(extra["tool_call_parse_result"].calls) > 0
        return obs, reward, terminated, truncated, info
