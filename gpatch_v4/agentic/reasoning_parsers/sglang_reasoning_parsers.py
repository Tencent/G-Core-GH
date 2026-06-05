from sglang.srt.parser.reasoning_parser import ReasoningParser as _SGLangReasoningParser

from gpatch_v4.agentic.reasoning_parsers import BaseReasoningParser, register_reasoning_parser
from gpatch_v4.agentic.reasoning_parsers.core_types import ReasoningParseResult
from gpatch_v4.utils import log


@register_reasoning_parser("sglang")
class SGLangReasoningParser(BaseReasoningParser):
    """Bridge to SGLang's reasoning parser.

    Parameters
    ----------
    model_type : str
        ``sglang.srt.parser.reasoning_parser.ReasoningParser`` model type
        (``"qwen3"``, ``"deepseek-r1"``, ``"gpt-oss"``, ...).
    force_reasoning : bool, optional
        Assume parsed text starts inside a reasoning block even when the
        ``<think>`` start token is absent. Defaults to True.

        Rationale: when ``enable_thinking=True``, the chat template (e.g.
        Qwen3) appends ``<think>`` to the *prompt*, so sampler-produced
        ``response_ids`` start directly inside the reasoning body and only
        contain the closing ``</think>``. Without ``force_reasoning=True``,
        SGLang's default ``Qwen3Detector`` would miss the block and return
        the entire response as ``normal_text``.
    """
    def __init__(self, model_type: str, force_reasoning: bool = True):
        self.reasoning_parser = _SGLangReasoningParser(
            model_type=model_type,
            force_reasoning=force_reasoning,
        )

    def extract_reasoning(self, text: str) -> ReasoningParseResult:
        try:
            reasoning_text, normal_text = self.reasoning_parser.parse_non_stream(text)
        except Exception as e:
            log(f"Failed to extract reasoning from text: {e}")
            reasoning_text, normal_text = "", text
        return ReasoningParseResult(
            reasoning_text=reasoning_text or "",
            normal_text=normal_text or "",
        )
