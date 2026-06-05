from typing import Any, Dict, Optional

from gpatch_v4.agentic.llm_proxy.base_llm_proxy import BaseLLMProxy, register_llm_proxy


@register_llm_proxy("random")
class RandomProxy(BaseLLMProxy):
    """Cheap proxy for dry-runs: encodes ``env.sample_random_action()`` as token ids."""
    def generate(self, data: Dict[str, Any], engine_index: Optional[int] = None) -> Dict[str, Any]:
        del engine_index  # No backend fan-out; routing hint is ignored.
        response_text = str(self.env.sample_random_action())
        enc = self.tokenizer(response_text, add_special_tokens=False, return_attention_mask=False)
        response_ids = enc["input_ids"]
        if len(response_ids) == 0:
            response_ids = [self.tokenizer.eos_token_id or 0]
        return {"response_ids": response_ids}
