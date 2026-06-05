# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

import io

import requests
from typing_extensions import override

from gdataset.store.base import CliBase


class LmdbClient(CliBase):
    def __init__(self, metadata, **kwargs):
        assert "lmdb_http_url" in metadata
        self.lmdb_http_url = metadata['lmdb_http_url']

    @override
    def get(self, url, **kwargs):

        response = requests.post(self.lmdb_http_url, json=[url])
        assert response.status_code == 200
        json_data_list = response.json()['data']
        res_bytes = {}
        assert len(json_data_list) == 1
        for json_data in json_data_list:
            assert json_data["res"], f"http read from lmdb serve fail"
            res_bytes[json_data["key"]] = json_data["value"].encode('latin1')

        body = res_bytes[url]
        return body
