# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

import io
from abc import ABC

from typing_extensions import override

from gdataset.feat.base import Feat


class FfmpegVideoFeat(Feat):
    @override
    def post_init(self, metadata):
        self.metadata = metadata

    @override
    def encode_example(self, fk, fv):
        raise NotImplementedError()
