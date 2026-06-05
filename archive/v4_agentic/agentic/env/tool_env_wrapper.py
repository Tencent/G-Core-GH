from __future__ import annotations

import inspect
import json
from contextlib import suppress
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

with suppress(ImportError):
    from gem.core import Env, EnvWrapper
if 'EnvWrapper' not in globals():
    EnvWrapper = object
with suppress(ImportError):
    from gem.tools.base_tool import BaseTool

from gpatch_v4.agentic.tools.tool_parsers import BaseToolParser, get_tool_parser
from gpatch_v4.agentic.utils import LoggerAdaptor


class ToolEnvWrapper(EnvWrapper):
    def __init__(
        self,
        env: Env,
        env_tool_config: Dict,
    ):
        super().__init__(env)
        self.logger = LoggerAdaptor()
        if inspect.signature(self.env.step).parameters.get("tool_call_parse_result") is None:
            self.logger.error(f"Env.step has a parameter called 'tool_call_parse_result'," \
                                "which is not supported by ToolEnvWrapper. " \
                                "Please use a different environment or update the ToolEnvWrapper to support it.")
            raise ValueError("Env.step has a parameter called 'tool_call_parse_result'")
        self.env_tool_config = env_tool_config
        self.setup_tools()

    def setup_tools(self):
        tools_json_path = self.env_tool_config.tools_json_path
        tool_call_parser_cls: type[BaseToolParser] = get_tool_parser(
            self.env_tool_config.tool_call_parser
        )
        try:
            with open(tools_json_path, "r") as f:
                self.tools = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load tools from {tools_json_path}: {e}")
        try:
            self.tool_parser = tool_call_parser_cls(self.tools)
        except Exception as e:
            raise ValueError(f"Failed to initialize tool parser: {e}")
        return self.tool_parser

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        observation, info, terminated = self.env.reset(**kwargs)
        info.update({"tools": self.tools})
        return observation, info, terminated

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        tool_call_parse_result = self.tool_parser.extract_tool_calls(action)
        obs, reward, terminated, truncated, info = self.env.step(action, tool_call_parse_result)
        if info is None:
            info = {}
        info["use_tool"] = len(tool_call_parse_result.calls) > 0
        return obs, reward, terminated, truncated, info
