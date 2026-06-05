# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

import io

from typing_extensions import override

from gdataset.store.base import CliBase


class LfsClient(CliBase):
    def __init__(self, metadata, **kwargs):
        pass

    def get(self, **kwargs):
        url = kwargs['url']
        with open(url, 'rb') as inf:
            body = inf.read()
        return body
