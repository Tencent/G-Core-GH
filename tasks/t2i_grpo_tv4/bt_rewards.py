from typing import Any, Dict, List
import torch

from hpsv2.src import open_clip
from typing_extensions import override

from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.configs.reward_config import T2iBtRewardModelInfo
from gpatch_v4.models.weclip_v2 import WeCLIPv2Large
from gpatch_v4.models.weclip_v3 import WeCLIPv3Large
from gpatch_v4.reward import BaseT2iRewardModel
from gpatch_v4.utils import log


class Hpsv2RewardModel(BaseT2iRewardModel):
    '''
    继承 BaseT2iRewardModel， 根据 reward model 类型重写 __init__ 和 compute_rewards 函数
    比如下面 Hpsv2RewardModel

    class Hpsv2RewardModel(BaseT2iRewardModel)
        def __init__(self, reward_model_info):
            # 函数功能: 创建 model 且赋值给 self.model
            # Input: a T2iBtRewardModelInfo object, 包含模型路径，权重路径等

        def compute_rewards(self, batched_data: Dict[str, List[Any]]):
            # 函数功能: 对生成的图片进行评测打分
            # Input: dict of list, 包含图片和 caption 等信息
            # Return: a tensor of rewards
    '''
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

        model_name = "ViT-H-14"
        model_weight_path = reward_model_info.model_weight_path

        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=reward_model_info.hf_model_path,
            precision='amp',
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        checkpoint = torch.load(model_weight_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        tokenizer = open_clip.get_tokenizer(model_name)

        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_val = preprocess_val

    @override
    def compute_rewards(self, batched_data: Dict[str, List[Any]] = None):
        assert batched_data is not None
        images = batched_data["images"]
        captions = batched_data["prompt"]
        assert len(images) == len(captions), f"{len(images)=} != {len(captions)}"
        n = len(images)

        reward_lst = []
        for i, (image, caption) in enumerate(zip(images, captions)):
            image = self.preprocess_val(image).unsqueeze(0).to(
                device=self.device, non_blocking=True
            )
            # Process the prompt
            text = self.tokenizer([caption]).to(device=self.device, non_blocking=True)

            # Calculate the HPS
            # 强行对齐
            with torch.amp.autocast('cuda'):
                outputs = self.model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image)
            reward_lst.append(hps_score)
        return torch.tensor(reward_lst, dtype=torch.float32)


class WeClipV3RewardModel(BaseT2iRewardModel):
    def __init__(self, reward_model_info: T2iBtRewardModelInfo = None):
        pass

    @override
    def compute_rewards(self, batched_data: Dict[str, List[Any]] = None):
        pass
