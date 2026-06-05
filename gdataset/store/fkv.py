# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

import asyncio
import io

from typing_extensions import override

try:
    from svrkit.client.kv import FKVOLClient, build_mix_module_key_fkv_only
    from svrkit.config import Config
except:
    pass

from gdataset.store.base import CliBase


class FkvClient(CliBase):
    def __init__(self, metadata, **kwargs):
        assert "fkv_config" in metadata
        assert "fkv_fk" in metadata
        assert "fkv_tid" in metadata

        self.fkv_rm = 1
        self.fkv_gm = 2
        self.config = Config()
        self.config.load_from_file(metadata["fkv_config"])
        self.mix_module_key = build_mix_module_key_fkv_only(
            self.fkv_rm,
            metadata["fkv_fk"],
            metadata["fkv_tid"],
        )
        self.client = FKVOLClient(self.config, self.mix_module_key)

    def get(self, **kwargs):
        url = kwargs['url']
        is_input_list = True
        if isinstance(url, str):
            is_input_list = False
            url = [url]
        assert isinstance(url, list)

        loop = asyncio.get_event_loop()
        resp = loop.run_until_complete(self.client.batch_get(keys=url, strategy=self.fkv_gm))
        res = []
        for kv in resp:
            res.append(kv.value)
        assert len(res) == len(url)

        if not is_input_list:
            res = res[0]
        return res
