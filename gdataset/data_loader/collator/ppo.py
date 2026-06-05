from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch

from gdataset.data_loader.collator.base import MultiModalDataCollatorForSeq2Seq


@dataclass
class PPODataCollator(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for 4d attention mask."""

    compute_dtype: "torch.dtype" = torch.float32

    @contextmanager
    def keep_regularized_images(self):
        """
        hajack the output of mm_plugin _regularize_images
        """
        assert not self._keep_regularized_images
        try:
            self._keep_regularized_images = True
            self.template.mm_plugin._regularize_images = self._regularize_images_hijack
            yield
        finally:
            self.template.mm_plugin._regularize_images = self.template.mm_plugin._regularize_images_backup
            self._keep_regularized_images = False

    def __post_init__(self):
        super().__post_init__()
        # hijack template method for keep image
        hijack_mm_plugin(self)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:

        input_ids_for_gen = [e.pop("input_ids_for_gen", []) for e in features]
        num_imgs = [len(e.get("images", [])) for e in features]
        json_data_list = [e.get("json_data", None) for e in features]
        with self.keep_regularized_images():
            features = super().__call__(features)

        for key, value in features.items():  # cast data dtype for paligemma
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)

        # get regularized_images
        images, self._tmp_images = self._tmp_images, None
        # process image and input_ids_for_gen for sglang or vllm generation
        raw_images_list = []
        index = 0
        for e in num_imgs:
            tmp = []
            if e:
                tmp = images[index:index + e]
                index = index + e
            raw_images_list.append(tmp)

        if not input_ids_for_gen[0]:
            input_ids_for_gen = []
        else:
            input_ids_for_gen = [torch.LongTensor(e) for e in input_ids_for_gen]

        features["json_data_list"] = json_data_list
        features["imgs_np_array_list"] = [
            [np.array(img) for img in imgs] for imgs in raw_images_list
        ]
        features["input_ids_for_gen"] = input_ids_for_gen
        features["images_padded"] = torch.tensor([0], dtype=torch.int64)

        return features


def hijack_mm_plugin(collator):
    """
    hijack _regularize_images of mm_plugin
    """
    template = collator.template
    mm_plugin = template.mm_plugin
    # method hijack
    _regularize_images = mm_plugin._regularize_images_backup if hasattr(
        mm_plugin, "_regularize_images_backup"
    ) else mm_plugin._regularize_images
    collator._keep_regularized_images = False

    def regularize_images(self, *args, **kwargs):
        res = _regularize_images(*args, **kwargs)
        # save for later process
        if collator._keep_regularized_images:
            collator._tmp_images = res["images"]
            # print(f"_tmp_images {collator._tmp_images}", flush=True)
        return res

    mm_plugin._regularize_images_backup = _regularize_images
    collator._regularize_images_hijack = MethodType(regularize_images, mm_plugin)
