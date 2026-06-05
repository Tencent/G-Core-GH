# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import asyncio
import pickle
from contextlib import asynccontextmanager
from typing import Any, Dict

import fastapi
import uvicorn

from gpatch.rpc import once_rpc
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils.common_utils import find_process_using_port, logging_with_rank_and_datetime


# HttpServer
class BaseServer:
    def __init__(self, endpoint_ip: str, endpoint_port: int, config: RlConfig, legacy_hook=None):
        self.endpoint_ip = endpoint_ip
        self.endpoint_port = endpoint_port
        self.ctx = None
        self.router = None
        self.is_running = False
        self.lock = asyncio.Lock()

        self.config = config
        self.infer_engine = None
        self.legacy_hook = legacy_hook
        self.monitor_kwargs = {
            "do_monitor": config.monitor_config.do_monitor,
            "monitor_server_ip": config.monitor_config.monitor_server_ip,
            "monitor_port": config.monitor_config.monitor_port,
        }

        self.app = self._create_app()
        self.info_msg = f"Server {self.__class__.__name__} {self.endpoint_ip=} {self.endpoint_port=}"

    def _create_app(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            yield

        app = fastapi.FastAPI(lifespan=lifespan)
        self._register_routes(app)
        return app

    def serve_forever(self, server_timeout_keep_alive):
        find_process_using_port(self.endpoint_ip, self.endpoint_port)
        logging_with_rank_and_datetime(f'RUN {self.__class__.__name__} http://{self.endpoint_ip}:{self.endpoint_port}')
        uvicorn.run(
            self.app,
            host=self.endpoint_ip,
            port=self.endpoint_port,
            log_level='error',
            use_colors=False,
            timeout_keep_alive=server_timeout_keep_alive,
            ssl_keyfile=None,
            ssl_certfile=None,
            ssl_ca_certs=None,
            ssl_cert_reqs=None
        )

    def _register_routes(self, app):
        # register api routes
        @app.post("/heartbeat")
        @once_rpc(**self.monitor_kwargs)
        async def heartbeat(req_dict):
            return {"ret": "ok"}

        @app.post("/exit")
        @once_rpc(**self.monitor_kwargs)
        async def exit(req_dict):
            return {'ret': 'ok'}

        # you can override this function and add more routes here
