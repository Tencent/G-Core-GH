import random
from typing import Any, Dict, List

import torch
from typing_extensions import override

from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.configs.reward_config import T2iBtRewardModelInfo
from gpatch_v4.reward import BaseT2iRewardModel


class StubRewardModel(BaseT2iRewardModel):
    def __init__(
        self,
        config: T2iRlConfig = None,
        rm_idx: int = None,
        reward_model_info: T2iBtRewardModelInfo = None
    ):
        super().__init__(reward_model_info)
        assert config is not None
        assert rm_idx is not None
        assert reward_model_info is not None
        self.model = 1  # a stub

    @override
    def compute_rewards(self, batched_data: Dict[str, List[Any]] = None):
        assert batched_data is not None
        images = batched_data["images"]
        captions = batched_data["prompt"]
        assert len(images) == len(captions), f"{len(images)=} != {len(captions)}"
        n = len(images)

        py_rng = random.Random(42)

        reward_lst = []
        for i, caption in enumerate(captions):
            reward_lst.extend([py_rng.random()])

        return torch.tensor(reward_lst, dtype=torch.float32)

    @override
    def sleep(self):
        pass

    @override
    def wake_up(self):
        pass
