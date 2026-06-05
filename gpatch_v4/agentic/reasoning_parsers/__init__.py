from abc import ABC, abstractmethod
from typing import Optional, Type

from gpatch_v4.agentic.reasoning_parsers.core_types import ReasoningParseResult


class BaseReasoningParser(ABC):
    """Extract chain-of-thought from a raw text generation.

    Returns both the reasoning and the remaining normal text.
    """
    @abstractmethod
    def extract_reasoning(self, text: str) -> ReasoningParseResult:
        """Extract reasoning content from ``text``.

        Parameters
        ----------
        text : str

        Returns
        -------
        ReasoningParseResult
            Parsed reasoning and remaining normal text.
        """
        raise NotImplementedError


REASONING_PARSER_REGISTRY: dict = {}


def _sglang_reasoning_parser_with_default(
    base: Type["BaseReasoningParser"], default_model_type: str
) -> Type["BaseReasoningParser"]:
    """Wrap SGLangReasoningParser so ``cls()`` uses *default_model_type* (e.g. from ``sglang/qwen3``)."""
    class SGLangReasoningParserWithDefault(base):  # type: ignore[valid-type,misc]
        def __init__(self, model_type: Optional[str] = None):
            mt = model_type if model_type is not None else default_model_type
            super().__init__(mt)

    SGLangReasoningParserWithDefault.__name__ = (
        f"{base.__name__}_{default_model_type.replace('/', '_')}"
    )
    SGLangReasoningParserWithDefault.__qualname__ = (
        f"{base.__qualname__}_{default_model_type.replace('/', '_')}"
    )
    return SGLangReasoningParserWithDefault


def register_reasoning_parser(name: str):
    def register_class(cls):
        REASONING_PARSER_REGISTRY[name] = cls
        return cls

    return register_class


def get_reasoning_parser(name: str) -> Type[BaseReasoningParser]:
    """Resolve a reasoning parser class by ``name``.

    The ``sglang/<model_type>`` convention forwards ``model_type`` to SGLang's
    ReasoningParser (e.g. ``sglang/qwen3``, ``sglang/deepseek-r1``).
    """
    if name.startswith("sglang"):
        register_name = "sglang"
        if register_name not in REASONING_PARSER_REGISTRY:
            raise ValueError(f"Unknown reasoning parser: {register_name}")
        base_cls = REASONING_PARSER_REGISTRY[register_name]
        suffix = name[len("sglang"):].lstrip("/")
        if suffix:
            return _sglang_reasoning_parser_with_default(base_cls, suffix)
        return base_cls
    register_name = name
    if register_name not in REASONING_PARSER_REGISTRY:
        raise ValueError(f"Unknown reasoning parser: {register_name}")
    return REASONING_PARSER_REGISTRY[register_name]


from gpatch_v4.agentic.reasoning_parsers.sglang_reasoning_parsers import (  # noqa: E402,F401
    SGLangReasoningParser,
)

__all__ = [
    "BaseReasoningParser",
    "register_reasoning_parser",
    "get_reasoning_parser",
    "ReasoningParseResult",
]
