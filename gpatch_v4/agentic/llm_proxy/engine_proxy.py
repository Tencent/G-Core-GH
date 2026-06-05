import asyncio
import time
from typing import Any, Dict, Optional

import numpy as np
from typing_extensions import override

from gpatch_v4.agentic.llm_proxy import BaseLLMProxy, register_llm_proxy
from gpatch_v4.utils import log


@register_llm_proxy("engine")
class EngineProxy(BaseLLMProxy):
    """Proxy that invokes a backend engine (e.g. vllm/sglang) for generation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ``np.random.RandomState`` only accepts a uint32 seed, while
        # ``time.time()`` is a float-second timestamp that loses sub-second
        # entropy when cast to ``int``.  Use ``time.time_ns()`` (mod 2**32) to
        # keep the spirit of seeding from current time while avoiding seed
        # collisions for proxies constructed within the same second.
        self._rng = np.random.RandomState(seed=int(time.time_ns()) % (2**32))

    @override
    def generate(self, data: Dict[str, Any], engine_index: Optional[int] = None):
        num_engine = self.sampler_client.svr_cluster_num_per_sampler[0]
        engine_index_in = engine_index
        if engine_index is None:
            engine_index = int(self._rng.randint(low=0, high=num_engine))
        else:
            engine_index = int(engine_index) % num_engine
        log(
            f"[EngineProxy] route generate: engine_index_in={engine_index_in} "
            f"-> engine_index={engine_index} (num_engine={num_engine})"
        )
        data = {k: [v] for (k, v) in data.items()}
        task = self.sampler_client.generate(0, 0, engine_index, data, 1, load_aware=True)
        return asyncio.run(task)
