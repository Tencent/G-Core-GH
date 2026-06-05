from __future__ import annotations

from contextlib import suppress
from typing import Optional

from transformers import PreTrainedTokenizer

with suppress(ImportError):
    import gem

import gpatch_v4.agentic.llm_proxy.sglang_proxy  # noqa: F401 — registers "sglang" proxy
from gpatch_v4.agentic.llm_proxy.base_llm_proxy import (
    LLM_PROXY_REGISTRY,
    BaseLLMProxy,
    register_llm_proxy,
)
from gpatch_v4.agentic.llm_proxy.engine_proxy import EngineProxy
from gpatch_v4.agentic.llm_proxy.random_proxy import RandomProxy
from gpatch_v4.client import SamplerClient

#from gpatch_v4.distributed.scheduler.generate_scheduler import RequestScheduler
from gpatch_v4.configs.agentic_config import LLMProxyConfig


def create_llm_proxy(
    sampler_client: Optional[SamplerClient],
    llm_proxy_config: LLMProxyConfig,
    tokenizer: PreTrainedTokenizer,
    env: gem.Env,
) -> BaseLLMProxy:
    proxy_type = llm_proxy_config.proxy_type
    if proxy_type in LLM_PROXY_REGISTRY:
        cls = LLM_PROXY_REGISTRY[proxy_type]
        return cls(sampler_client, llm_proxy_config, tokenizer, env)
    else:
        raise ValueError(f"Unknown proxy type: {proxy_type}")
