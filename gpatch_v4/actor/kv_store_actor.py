import asyncio
import os
import threading
from typing import Any, Dict, List

import lmdb
import torch
import uvicorn
from fastapi import FastAPI, Query, Response

from gpatch_v4.core.parallel_state import init_pg, initlize_parallel_state
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.utils import log


def lmdb_put(env, key: bytes, value: bytes):
    with env.begin(write=True) as txn:
        txn.put(key, value)


def lmdb_get(env, key: bytes) -> bytes | None:
    with env.begin() as txn:
        return txn.get(key)


class KvStoreActor(BaseActor):
    async def init(self, config):
        super().init(config, pg_backend='gloo')
        initlize_parallel_state(config, config.kv.dist_config)
        init_pg(config.kv.dist_config)
        self.setup_lmdb(config)

    def setup_lmdb(self, config):
        # Define the path for the LMDB environment (it's a directory)
        lmdb_path = os.path.join(config.kv.root, f'{torch.distributed.get_rank()}')
        if not os.path.exists(lmdb_path):
            os.makedirs(lmdb_path)

        # Open an LMDB environment. The map_size must be set large enough to hold all data.
        # It can be dynamically resized, but setting it large initially is simpler.
        map_size = 1 * (2**40)  # 1TB
        self.kv = lmdb.open(lmdb_path, map_size=map_size)

    def start_http_server(self, host: str, port: int):
        """Start a FastAPI HTTP server in a background thread."""
        app = FastAPI()

        @app.get("/get")
        async def http_get(key: str):
            result = await self.get_co(key)
            if result is None:
                return Response(status_code=404)
            return Response(content=result, media_type="application/octet-stream")

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve())

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        log(f"KvStoreActor HTTP server started on {host}:{port}")

    async def set_co(self, k: str | bytes, v: bytes):
        if isinstance(k, str):
            k = k.encode("utf-8")
        await asyncio.to_thread(lmdb_put, self.kv, k, v)

    def set(self, k: str | bytes, v: bytes):
        if isinstance(k, str):
            k = k.encode("utf-8")
        lmdb_put(self.kv, k, v)

    async def get_co(self, k: str | bytes) -> bytes | None:
        if isinstance(k, str):
            k = k.encode("utf-8")
        return await asyncio.to_thread(lmdb_get, self.kv, k)

    def get(self, k: str | bytes) -> bytes | None:
        if isinstance(k, str):
            k = k.encode("utf-8")
        return lmdb_get(self.kv, k)
