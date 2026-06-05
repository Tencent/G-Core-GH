from abc import ABC
from typing import Any, Dict, List

import torch

from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.configs.reward_config import T2iBtRewardModelInfo
from gpatch_v4.utils import clear_memory, log
'''
继承 BaseT2iRewardModel， 根据 reward model 类型重写 __init__ 和 compute_rewards 函数
比如下面 Hpsv2RewardModel

class Hpsv2RewardModel(BaseT2iRewardModel)
    def __init__(self, reward_model_info):
        # 函数功能: 创建 model 且赋值给 self.model
        # Input: a T2iBtRewardModelInfo object, 包含模型路径，权重路径等

    def compute_rewards(self, batch_data: Dict[str, List[Any]]):
        # 函数功能: 对生成的图片进行评测打分
        # Input: dict of list, 包含图片和 caption 等信息
        # Return: a tensor of rewards

'''


class BaseT2iRewardModel:
    """Base reward model template for T2I tasks.

    Subclasses override ``__init__`` to create the model and
    ``compute_rewards`` to score generated images.

    Parameters
    ----------
    config : T2iRlConfig, optional
    rm_idx : int, optional
    reward_model_info : T2iBtRewardModelInfo, optional
        Model path and weight information.
    """
    def __init__(
        self,
        config: T2iRlConfig = None,
        rm_idx: int = None,
        reward_model_info: T2iBtRewardModelInfo = None
    ):
        self.config = config
        self.rm_idx = rm_idx
        self.reward_model_info = reward_model_info
        self.device = torch.device(torch.cuda.current_device())
        self.model = None

    def compute_rewards(self, batched_data: List[Dict[str, Any]] = None):
        """Score a batch of generated images.

        Parameters
        ----------
        batched_data : list of dict, optional
            Batch with images and captions.

        Raises
        ------
        NotImplementedError
            Always; must be overridden.
        """
        raise NotImplementedError(f"BaseReardModel infer_reward not implemented")

    def sleep(self):
        """Move the model to CPU to free GPU memory."""
        assert self.config.placement_type != "disaggregated", (
            f"{self.__class__.__name__}.sleep should not be called in disaggregated placement"
        )
        assert self.model is not None
        self.model.cpu()
        clear_memory()

    def wake_up(self):
        """Move the model back to GPU."""
        assert self.config.placement_type != "disaggregated", (
            f"{self.__class__.__name__}.wake_up should not be called in disaggregated placement"
        )
        assert self.model is not None
        self.model.to(self.device)
        clear_memory()
