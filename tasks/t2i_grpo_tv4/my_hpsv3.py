import os
import shutil
import uuid
from typing import Any, Dict, List

import torch
from typing_extensions import override

from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.configs.reward_config import T2iBtRewardModelInfo
from gpatch_v4.reward import BaseT2iRewardModel
from gpatch_v4.utils import clear_memory, log


class Hpsv3RewardModel(BaseT2iRewardModel):
    # HPSv3 的 code 不成熟，目前能跑，而且与 sglang 不冲突的版本是:
    # ```
    # pip3 install hpsv3==1.0.0
    # pip3 install transformers==4.51.3 peft==0.10.0
    # export _CHECK_PEFT=0
    # ```
    # 而且 inference 的构造函数会创建 transformers TrainingArg，会影响进程内 transformers
    # model 的 gpu mem 分配，不过我们 model 在进程内所以还 ok。

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

        from hpsv3 import HPSv3RewardInferencer

        self.model = HPSv3RewardInferencer(
            device=self.device,
            config_path='tasks/t2i_grpo_tv4/yaml/HPSv3_7B.yaml',
            checkpoint_path="hf-hub/MizzenAI/HPSv3/HPSv3.safetensors"
        )

    def save_images_to_tmp_pathes(self, images):
        pathes = []
        for im in images:
            p = f'/tmp/gcore-t2i-grpo-hpsv3-{torch.distributed.get_rank()}-{uuid.uuid4()}.png'
            im.save(p)
            pathes.append(p)
        return pathes

    def remove_tmp_pathes(self, pathes):
        for p in pathes:
            os.remove(p)

    @override
    def compute_rewards(self, batched_data: Dict[str, List[Any]] = None):
        assert batched_data is not None
        images = batched_data["images"]
        captions = batched_data["prompt"]
        assert len(images) == len(captions), f"{len(images)=} != {len(captions)}"
        n = len(images)

        tmp_pathes = self.save_images_to_tmp_pathes(images)
        try:
            reward_lst = []
            for i, (tmp_path, caption) in enumerate(zip(tmp_pathes, captions)):
                rewards = self.model.reward([tmp_path], [caption])
                reward_lst.extend([reward[0].item() for reward in rewards])
        finally:
            self.remove_tmp_pathes(tmp_pathes)

        return torch.tensor(reward_lst, dtype=torch.float32)

    @override
    def sleep(self):
        assert self.model is not None
        self.model.model.cpu()
        clear_memory()

    @override
    def wake_up(self):
        assert self.model is not None
        self.model.model.to(self.device)
        clear_memory()
