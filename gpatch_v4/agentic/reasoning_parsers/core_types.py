from typing import Optional

from pydantic import BaseModel


class ReasoningParseResult(BaseModel):
    """Result of reasoning extraction.

    Parameters
    ----------
    reasoning_text : str
        Text inside the reasoning/think section; empty if absent.
    normal_text : str
        Text outside the reasoning section (the "real" response); empty if
        the entire output is still inside an unclosed reasoning block.
    """

    reasoning_text: str = ""
    normal_text: str = ""
