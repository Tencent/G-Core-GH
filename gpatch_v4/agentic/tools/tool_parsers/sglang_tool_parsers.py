import uuid
from typing import Dict, List, Union

from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from gpatch_v4.agentic.tools.tool_parsers import BaseToolParser, register_tool_parser
from gpatch_v4.agentic.tools.tool_parsers.core_types import (
    FunctionResponse,
    ParseResult,
    Tool,
    ToolCall,
)
from gpatch_v4.utils import log


@register_tool_parser("sglang")
class SGLangToolParser(BaseToolParser):
    """Tool parser bridge to SGLang's ``FunctionCallParser``."""
    def __init__(self, tools: Union[List[Dict], str], tool_call_parser: str):
        super().__init__(tools)
        self.function_call_parser = FunctionCallParser(
            tools=self.tools, tool_call_parser=tool_call_parser
        )

    def extract_tool_calls(self, text: str) -> ParseResult:
        """Extract tool calls from ``text``.

        Args:
            text (str): Input text.

        Returns:
            ParseResult.
        """
        try:
            # 这个地方有时候 text 能被 json.loads 成功，但是 json load 之后不是 dict 类型
            # 而是 str 类型导致 crash，所以这里需要 try-except
            normal_text, tool_call_list = self.function_call_parser.parse_non_stream(text)
        except Exception as e:
            log(f"Failed to extract tool calls from text: {e}")
            normal_text = text
            tool_call_list = []
        return ParseResult(
            normal_text=normal_text,
            calls=[
                ToolCall(
                    id=f"tool_call_{str(uuid.uuid4())}",
                    index=i,
                    function=FunctionResponse(name=call.name, arguments=call.parameters)
                ) for i, call in enumerate[ToolCallItem](tool_call_list)
            ]
        )
