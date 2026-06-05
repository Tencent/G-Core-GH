"""
less code is more code, clever man is lazy man.
reuse transformers for rope index calculation.
"""
from typing import Optional

import torch


class BaseRopeIndexHelper:
    def __init__(self, config, hf_class):
        self.config = config
        self.hf_class = hf_class

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        return self.hf_class.get_rope_index(
            self,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            **kwargs
        )
