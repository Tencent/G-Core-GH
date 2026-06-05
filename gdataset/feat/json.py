# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

import io
import json
from abc import ABC

from typing_extensions import override

from gdataset.feat.base import Feat
from gdataset.store import store_cli_provider


class JsonFeat(Feat):
    '''
    parse a string value to python obj with ``json.loads``.
    '''
    def __init__(
        self,
        nest=True,
        **kwargs,
    ):
        self.nest = nest

    @override
    def post_init(self, metadata):
        self.metadata = metadata

    @override
    def encode_example(self, fk, fv):
        assert isinstance(fv, str)
        try:
            jobj = json.loads(fv)
        except json.decoder.JSONDecodeError as ex:
            print(f'unexpected {fk} {fv}')
            raise ex
        if not self.nest:
            return jobj
        else:
            return {fk: jobj}
