from typing import List, Dict, Any

import ray
import asyncio

from gpatch_v4.agentic.llm_proxy import BaseLLMProxy, register_llm_proxy
from gpatch_v4.agentic.proto import DataProto
import numpy as np


@register_llm_proxy("engine")
class EngineProxy(BaseLLMProxy):
    """
    A proxy for engine that invokes the engine (e.g. vllm/sglang) to perform generation.
    """
    def generate(self, data: Dict[str, Any]) -> DataProto:
        # one request a time
        num_engine = self.sampler_client.svr_cluster_num_per_sampler[0]
        engine_index = np.random.randint(low=0, high=num_engine)
        # sampler expect a dict of list
        data = {k: [v] for (k, v) in data.items()}
        task = self.sampler_client.generate(0, 0, engine_index, data, 1)
        result = asyncio.run(task)
        return result
