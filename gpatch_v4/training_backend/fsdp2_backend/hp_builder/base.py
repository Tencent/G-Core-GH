# coding=utf-8
"""Shared HP model construction pipeline."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module


def meta_construct_fp32(model_cls: type, cfg: Any) -> Module:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            return model_cls(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)


class HpModelBuilder:
    """Template: validate → prepare_config → meta construct → apply_hp → load."""
    def validate(self, engine: Any) -> None:
        return

    def prepare_config(self, model_cls: type, hf_model_path: str, engine: Any) -> Any:
        raise NotImplementedError

    def apply_hp(self, model: Module, engine: Any) -> Module:
        raise NotImplementedError

    def build(
        self,
        engine: Any,
        model_cls: type,
        hf_model_path: str,
        model_only_inference: bool,
    ) -> Module:
        self.validate(engine)
        cfg = self.prepare_config(model_cls, hf_model_path, engine)
        model = meta_construct_fp32(model_cls, cfg)
        model = self.apply_hp(model, engine)
        model.load_checkpoint_hp(hf_model_path)
        if not model_only_inference:
            model.train()
        else:
            model.eval()
        return model
