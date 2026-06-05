import copy
from dataclasses import fields

import torch

from megatron.core.transformer.transformer_config import TransformerConfig


def merge_config(mbridge_config, mg_config):
    assert mbridge_config.num_layers == mg_config.num_layers
    config_merged = copy.deepcopy(mg_config)
    hf_fields = {e.name for e in fields(mbridge_config)}
    mg_fields = {e.name for e in fields(TransformerConfig)}
    # config fields TransformerConfig - TransformerConfig
    diff = hf_fields - mg_fields

    # generally, use parallel config from megatron config, use model config from hf config
    for f in diff:
        setattr(config_merged, f, getattr(mbridge_config, f))

    return config_merged
