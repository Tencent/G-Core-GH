from sglang.srt.function_call.core_types import ToolCallItem


from typing import List, Union, Dict
import uuid
from gpatch_v4.agentic.tools.tool_parsers import BaseToolParser
from gpatch_v4.agentic.tools.tool_parsers.core_types import ParseResult, ToolCall, Tool, FunctionResponse
from gpatch_v4.agentic.tools.tool_parsers import register_tool_parser
from sglang.srt.function_call.function_call_parser import FunctionCallParser

@register_tool_parser("sglang")
class SGLangToolParser(BaseToolParser):
    """
    SGLangToolParser is a tool parser for SGLang.
    """
    def __init__(self, tools: Union[List[Dict], str], tool_call_parser: str):
        super().__init__(tools)
        self.function_call_parser = FunctionCallParser(tools=self.tools, tool_call_parser=tool_call_parser)  
          
    def extract_tool_calls(self, text: str) -> ParseResult:
        """Extract tool calls from the text.

        Args:
            text (str): The text to extract tool calls from.

        Returns:
            ParseResult: The result of the tool calls extraction.
        """
        try:
            # 这个地方有时候 text 能被 json.loads 成功，但是 json load 之后不是 dict 类型
            # 而是 str 类型导致 crash，所以这里需要 try-except
            normal_text, tool_call_list = self.function_call_parser.parse_non_stream(text)
        except Exception as e:
            self.logger.warning(f"Failed to extract tool calls from text: {e}")
            normal_text = text
            tool_call_list = []
        return ParseResult(
            normal_text=normal_text,
            calls=[ToolCall(
                id=f"tool_call_{str(uuid.uuid4())}",
                index=i,
                function=FunctionResponse(
                    name=call.name,
                    arguments=call.parameters
                )
            ) for i, call in enumerate[ToolCallItem](tool_call_list)]
        )