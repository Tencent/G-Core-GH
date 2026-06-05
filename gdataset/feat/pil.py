# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com, guanyouhe@tencent.com

import base64
import io
import json
from abc import ABC

from PIL import Image
from typing_extensions import override

from gdataset.feat.base import Feat
from gdataset.store import store_cli_provider


class PilImageFeat(Feat):
    '''
    parse a string value / dict to PIL image.
    '''
    def __init__(
        self,
        cos=False,
        new_name='image',
        field_value_is_json_str=False,
        **kwargs,
    ):
        self.cos = cos
        self.new_name = new_name
        self.field_value_is_json_str = field_value_is_json_str

    @override
    def post_init(self, metadata):
        self.metadata = metadata
        self.store_cli = store_cli_provider(metadata, cos=self.cos)

    @override
    def encode_example(self, fk, fv):
        # 多方协作是这样子的...
        if isinstance(fv, dict):
            body = self.store_cli.get(**fv)
        else:
            assert isinstance(fv, str)
            if self.field_value_is_json_str:
                fv = json.loads(fv)
                body = self.store_cli.get(**fv)
            else:
                # NOTE 目前只有这么一种 case，先 hardcode，但应该尽可能避免这种用法。
                body = self.store_cli.get(url=fv, cos_bucket_name=self.metadata['cos_bucket_name'])

        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
        img = Image.open(io.BytesIO(body))
        return {
            self.new_name: img,
        }


"""
{
  "images": [
    {
      "image_path": "train-00000-of-00003.parquet/8.png"
    }
  ]
}
"""


class PilImageListFeat(Feat):
    '''
    parse a string value to python obj with ``json.loads``.

    ``fv`` 是一个数组，它的 json 格式如下：

    .. highlight:: json
    .. code-block:: json

        {
            "images": [
                {
                "image_path": "train-00000-of-00003.parquet/8.png"
                }
            ]
        }
    '''
    def __init__(
        self,
        cos=False,
        lmdb=False,
        new_name='__images_feat__',
        field_value_is_json_str=False,
        return_src_data=False,
        convert_to_rgb=False,
        **kwargs,
    ):
        self.cos = cos
        self.lmdb = lmdb
        self.new_name = new_name
        self.field_value_is_json_str = field_value_is_json_str
        self.return_src_data = return_src_data
        self.convert_to_rgb = convert_to_rgb

    @override
    def post_init(self, metadata):
        self.metadata = metadata
        self.store_cli = store_cli_provider(metadata, cos=self.cos, lmdb=self.lmdb)

    @override
    def encode_example(self, fk, fv):
        # 多方协作是这样子的...
        body_list = []
        for ele in fv:
            value = ele['image_path']
            if self.field_value_is_json_str:
                body = self.store_cli.get(**json.loads(value))
            else:
                # NOTE 目前只有这么一种 case，先 hardcode，但应该尽可能避免这种用法。
                body = self.store_cli.get(
                    url=value,
                    cos_bucket_name=self.metadata.get('cos_bucket_name', None),
                )
            body_list.append(body)

        imgs = [Image.open(io.BytesIO(body)) for body in body_list]
        if self.convert_to_rgb:
            imgs = [img.convert("RGB") for img in imgs]

        res = {self.new_name: imgs}
        if self.return_src_data:
            res.update({fk: fv})
        return res
