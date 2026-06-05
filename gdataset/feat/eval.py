# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

import io
import json
from abc import ABC

from typing_extensions import override

from gdataset.feat.base import Feat
from gdataset.store import store_cli_provider


class EvalFeat(Feat):
    '''
    parse a string value to python obj with ``eval``.

    dangerous! for debugging purpose only.
    dangerous! for debugging purpose only.
    dangerous! for debugging purpose only.

    危险，只用于 debug！
    危险，只用于 debug！
    危险，只用于 debug！
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
        jobj = eval(fv)
        if not self.nest:
            return jobj
        else:
            return {fk: jobj}
