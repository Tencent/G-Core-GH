import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
import ray
import torch
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.client.base_client import BaseClientAbc
from gpatch_v4.configs.utils import MappingProtocol
from gpatch_v4.utils.common_utils import log, logging_rank0

# Default timeout for HTTP requests (seconds)
_DEFAULT_HTTP_TIMEOUT = 60.0

# Max consecutive failures before emitting a warning
_WARN_AFTER_FAILURES = 5


class KvStoreClient(BaseClientAbc):
    """KV store client that supports both ``ray`` and ``http`` RPC backends.

    The ``http`` backend uses ``httpx`` so it can work from non-ray processes
    (e.g. dataset / dataloader child processes).  Every call retries until
    success to simulate RPC at-least-once semantics.
    """
    def __init__(self, config: MappingProtocol):
        self.config = config
        self.rpc_type = config.policy.kv_store_client.rpc_type

        self.sync_http_client = httpx.Client(timeout=_DEFAULT_HTTP_TIMEOUT)

        if self.rpc_type == "ray":
            dist_config = self.config.kv.dist_config
            self.num_clusters, _, _ = self.get_engine_info(dist_config)
        else:
            raise ValueError(f"invalid rpc type: {self.rpc_type}")
        logging_rank0(f"{self.__class__.__name__} init with {self.rpc_type} client")

    # ------------------------------------------------------------------
    # BaseClientAbc overrides (no-ops for KV client)
    # ------------------------------------------------------------------

    @override
    async def mark_ppo_step_begin(self, idx, ppo_step):
        pass

    @override
    async def mark_ppo_step_end(self, idx, ppo_step):
        pass

    @override
    async def wake_up(self, idx, tag_names=None):
        pass

    @override
    async def sleep(self, idx):
        pass

    # ------------------------------------------------------------------
    # Key hashing
    # ------------------------------------------------------------------

    def str_to_int(self, s: str) -> int:
        md5 = hashlib.md5(s.encode('utf-8')).hexdigest()
        return int(md5, 16)

    # ------------------------------------------------------------------
    # Async set / get
    # ------------------------------------------------------------------

    async def set_co(self, k: str, v: bytes):
        ep_idx = self.str_to_int(k) % self.num_clusters
        target_ep = ray.get_actor(f'kv_{ep_idx}')
        await target_ep.set_co.remote(k, v)

    async def get_co(self, k: str) -> bytes | None:
        ep_idx = self.str_to_int(k) % self.num_clusters
        target_ep = ray.get_actor(f'kv_{ep_idx}')
        return await target_ep.get_co.remote(k)

    # ------------------------------------------------------------------
    # Sync set / get
    # ------------------------------------------------------------------

    def set(self, k: str, v: bytes):
        ep_idx = self.str_to_int(k) % self.num_clusters
        target_ep = ray.get_actor(f'kv_{ep_idx}')
        ray.get(target_ep.set.remote(k, v))

    def get(self, k: str) -> bytes | None:
        ep_idx = self.str_to_int(k) % self.num_clusters
        target_ep = ray.get_actor(f'kv_{ep_idx}')
        return ray.get(target_ep.get.remote(k))

    # ------------------------------------------------------------------
    # Sync get with HTTP (in dataloader)
    # ------------------------------------------------------------------

    def get_http(self, k: str) -> bytes | None:
        ep_idx = self.str_to_int(k) % self.num_clusters
        http_endpoints = self.config.kv.http_endpoints
        url = f"http://{http_endpoints[ep_idx]}/get?{urlencode({'key': k})}"
        return self._http_get_bytes_with_retry_sync(url)

    # ------------------------------------------------------------------
    # Retry helpers – retry until success, warn on repeated failure
    # ------------------------------------------------------------------

    def _http_get_bytes_with_retry_sync(self, url: str) -> bytes | None:
        """Synchronous version of :meth:`_http_get_bytes_with_retry`."""
        client = self.sync_http_client
        failures = 0
        while True:
            try:
                resp = client.get(url)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                failures = 0
                return resp.content
            except Exception as e:
                failures += 1
                if failures % _WARN_AFTER_FAILURES == 0:
                    log(
                        f"[WARN] KvStoreClient HTTP sync GET {url} failed {failures} times "
                        f"(latest: {e}). Retrying …"
                    )
                time.sleep(min(1.0 * failures, 5.0))
