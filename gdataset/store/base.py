# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

from abc import ABC


class CliBase(ABC):
    def get(self, **kwargs):
        raise NotImplementedError()
