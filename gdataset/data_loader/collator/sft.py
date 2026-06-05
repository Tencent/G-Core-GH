from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch

from gdataset.data_loader.collator.base import MultiModalDataCollatorForSeq2Seq


@dataclass
class SFTDataCollator(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for 4d attention mask."""

    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        for key, value in features.items():  # cast data dtype for paligemma
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)

        return features
