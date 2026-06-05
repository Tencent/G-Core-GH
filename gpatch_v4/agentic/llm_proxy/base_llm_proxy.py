from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizer

with suppress(ImportError):
    import gem
# from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from gpatch_v4.client import SamplerClient
from gpatch_v4.configs.agentic_config import LLMProxyConfig


class BaseLLMProxy(ABC):
    """Unified interface for generating responses from messages or lm_input."""
    def __init__(
        self, sampler_client: SamplerClient, llm_proxy_config: LLMProxyConfig,
        tokenizer: PreTrainedTokenizer, env: gem.Env
    ):
        self.sampler_client = sampler_client
        self.llm_proxy_config = llm_proxy_config
        self.tokenizer = tokenizer
        self.env = env

    @abstractmethod
    def generate(self, data: List[Dict[str, str]], engine_index: Optional[int] = None):
        """
        Generate a response from conversation messages or model input.

        Args:
            messages (List[Dict[str, str]]): Conversation history,
                e.g. ``[{"role": "user", "content": "Hello!"}, ...]``.
            lm_input (DataProto): Tokenized prompts and tensor inputs.
            generation_config (Dict[str, Any]): Override default generation parameters.
            engine_index (Optional[int]): Hint to pin the request to a specific backend
                engine (e.g. sglang cluster index). Multi-engine subclasses (notably
                ``EngineProxy``) should honour this; others may ignore.

        Returns:
            DataProto: Generated sequences and metadata; the batch contains a
                ``'responses'`` key with returned token_ids.
        """
        pass


LLM_PROXY_REGISTRY = {}


def register_llm_proxy(name):
    def register_class(cls):
        LLM_PROXY_REGISTRY[name] = cls
        return cls

    return register_class
