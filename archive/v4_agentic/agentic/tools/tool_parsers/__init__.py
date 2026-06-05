import json
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Type

from gpatch_v4.agentic.tools.tool_parsers.core_types import ParseResult, Tool
from gpatch_v4.agentic.utils import LoggerAdaptor


class BaseToolParser(ABC):
    """
    LLMProxy defines a unified interface for generating responses based on messages or lm_input DataProto.
    Subclasses will implement specific inference apis.
    """
    def __init__(self, tools: Union[List[Dict], str], **kwargs):
        self.logger = LoggerAdaptor()
        if isinstance(tools, str):
            tools = json.loads(tools)
            if not isinstance(tools, List):
                tools = [tools]
        self.tools = [Tool(**tool) for tool in tools]

    @abstractmethod
    def extract_tool_calls(self, text: str) -> ParseResult:
        """Extract tool calls from the text.

        Args:
            text (str): The text to extract tool calls from.

        Returns:
            ParseResult: The result of the tool calls extraction.
        """
        raise NotImplementedError


TOOL_PARSER_REGISTRY = {}


def _sglang_parser_with_default(base: Type["BaseToolParser"], default_tool_call_parser: str) -> Type["BaseToolParser"]:
    """Wrap SGLangToolParser so ``cls(tools)`` uses *default_tool_call_parser* (e.g. from ``sglang/qwen``)."""
    class SGLangToolParserWithDefault(base):  # type: ignore[valid-type,misc]
        def __init__(
            self,
            tools: Union[List[Tool], str, Dict],
            tool_call_parser: Optional[str] = None,
        ):
            tp = tool_call_parser if tool_call_parser is not None else default_tool_call_parser
            super().__init__(tools, tp)

    SGLangToolParserWithDefault.__name__ = f"{base.__name__}_{default_tool_call_parser.replace('/', '_')}"
    SGLangToolParserWithDefault.__qualname__ = f"{base.__qualname__}_{default_tool_call_parser.replace('/', '_')}"
    return SGLangToolParserWithDefault


def register_tool_parser(name: str):
    def register_class(cls):
        TOOL_PARSER_REGISTRY[name] = cls
        return cls

    return register_class


def get_tool_parser(name: str) -> type[BaseToolParser]:
    if name.startswith("sglang"):
        register_name = "sglang"
        if register_name not in TOOL_PARSER_REGISTRY:
            raise ValueError(f"Unknown tool parser: {register_name}")
        base_cls = TOOL_PARSER_REGISTRY[register_name]
        # ``sglang/<tool_call_parser>``：与 SGLang FunctionCallParser 的 tool_call_parser 名一致，如 qwen
        suffix = name[len("sglang"):].lstrip("/")
        if suffix:
            return _sglang_parser_with_default(base_cls, suffix)
        return base_cls
    register_name = name
    if register_name not in TOOL_PARSER_REGISTRY:
        raise ValueError(f"Unknown tool parser: {register_name}")
    return TOOL_PARSER_REGISTRY[register_name]


from gpatch_v4.agentic.tools.tool_parsers.sglang_tool_parsers import SGLangToolParser

__all__ = ["BaseToolParser", "register_tool_parser", "get_tool_parser"]
