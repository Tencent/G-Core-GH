# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

import pprint

import torch
from PIL.Image import Image as PilImage
from PIL.ImageFile import ImageFile as PilImageFile
from torch.utils.data import default_collate as torch_default_collate


def default_collate(batches):
    # Unfortunately `default_collate_fn_map` is not
    # supported in torch 2.6. So we have to implement it in such a stupid way.
    #
    # See:
    # 1. https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.default_collate
    # 2. https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/data/_utils/collate.py
    #
    # TODO: use `default_collate_fn_map` instead

    assert isinstance(batches, list) and isinstance(batches[0], dict)
    fname_to_fvs = {}
    fnames = [k for k in batches[0].keys()]
    for fname in fnames:
        fv = batches[0][fname]
        if isinstance(fv, PilImage):
            fvs = [batch.pop(fname) for batch in batches]
            fname_to_fvs[fname] = fvs

    batch = torch_default_collate(batches)

    for fname, fvs in fname_to_fvs.items():
        batch[fname] = fvs
    return batch
