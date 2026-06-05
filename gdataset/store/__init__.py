# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

from gdataset.store.base import CliBase
from gdataset.store.cos import CosClient
from gdataset.store.fkv import FkvClient
from gdataset.store.lfs import LfsClient
from gdataset.store.lmdb import LmdbClient
from gdataset.store.uniondb import UniondbClient


def store_cli_provider(metadata, **kwargs):
    if kwargs.get('cos', False):
        return CosClient(metadata, **kwargs)
    elif kwargs.get('lmdb', False):
        return LmdbClient(metadata, **kwargs)
    elif kwargs.get('uniondb', False):
        return UniondbClient(metadata, **kwargs)
    elif kwargs.get('fkv', False):
        return FkvClient(metadata, **kwargs)
    else:
        return LfsClient(metadata, **kwargs)
