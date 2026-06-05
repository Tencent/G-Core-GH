# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

from abc import ABC
from typing import Any


class Feat(ABC):
    '''
    Feature 的名字来自于 huggingface ``datasets``.

    The name ``feature`` comes from huggingface ``datasets``.

    .. highlight:: python
    .. code-block:: python

        dataset = load_dataset("AI-Lab-Makerere/beans", split="train").cast_column("image", Image(decode=False))

    ``gdataset`` 的 ``Feat`` 提供了类似的功能，同时还封装了 remote storage 读取的能力。具体可参考 ``gdataset.PilImageFeat``。

    .. highlight:: python
    .. code-block:: python

        class PilImageFeat(Feat):

            def __init__(
                self,
                cos=False, # 是否存储在 cos，否则存储在 LFS
                field_value_is_json_str=False, # 如果输入数据是 string，默认是 cos 路径，也可能 parse 之后是 json
            ):


    将 uniondb 存的 string 转换成 json obj。

    .. highlight:: python
    .. code-block:: python

        class JsonFeat(Feat):

            def __init__(
                self,
                nest=True,
            ):
    '''
    def post_init(self, metadata):
        pass

    def encode_example(self, fk, fv) -> Any:
        pass
