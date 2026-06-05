from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
from llamafactory.extras.constants import IGNORE_INDEX
from transformers import DataCollatorForSeq2Seq, ProcessorMixin


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images, videos and audios.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None
    post_collate_hook: Optional["Callable"] = None

    def __post_init__(self):
        assert self.label_pad_token_id == IGNORE_INDEX
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")
        image_token = self.template.mm_plugin.image_token
        if image_token is not None:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        else:
            self.image_token_id = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        batch_images, batch_videos, batch_audios = [], [], []
        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
        origin_json_data_list = []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            audios = feature.pop("audios", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_input_ids.append(feature["input_ids"])
            origin_json_data_list.append(feature.pop("json_data", None))

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images,
            batch_videos,
            batch_audios,
            batch_imglens,
            batch_vidlens,
            batch_audlens,
            batch_input_ids,
            self.processor,
        )
        features: dict[str, torch.Tensor] = super().__call__(features)
        if self.post_collate_hook is not None:
            # custom process logic
            custom_data = self.post_collate_hook(features, mm_inputs, origin_json_data_list)
            features.update(custom_data)

        #
        has_image = ("image_grid_thw" in mm_inputs) and (mm_inputs["image_grid_thw"] is not None)
        mm_inputs["has_image"] = torch.tensor([has_image], dtype=torch.bool)
        features.update(mm_inputs)

        if self.image_token_id is not None:
            features["image_input_mask"] = features["input_ids"] == self.image_token_id
        else:
            assert not has_image
            features["image_input_mask"] = torch.zero_like()
        features["image_padded"] = torch.tensor([False] * len(batch_input_ids), dtype=torch.bool)
        features["loss_mask"] = (features["labels"] != IGNORE_INDEX).to(torch.float32)

        return features
