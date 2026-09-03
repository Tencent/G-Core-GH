"""Direct payload adapter around TransferQueue's native KV batch API."""
from __future__ import annotations

import logging
import os
import subprocess
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Sequence
from urllib.parse import urlparse

from tensordict import TensorDict

try:
    import transfer_queue as tq
    from transfer_queue.metadata import KVBatchMeta
except ImportError:
    tq = None
    KVBatchMeta = None

from gpatch_v4.configs.tq_config import TqConfig
from gpatch_v4.orches.utils import get_current_node_ip
from gpatch_v4.utils import log

__all__ = ["TqConnector"]


class TqConnector:
    """Offload and restore payload data through TransferQueue."""
    def __init__(self, tq_config: TqConfig) -> None:
        self.tq_config = tq_config
        if not self.tq_config.partition_id:
            raise ValueError("tq.partition_id must be set before initializing TqConnector")

        self._apply_transfer_queue_log_level()
        if self.tq_config.backend == "MooncakeStore":
            os.environ["MC_TCP_BIND_ADDRESS"] = get_current_node_ip()
        tq.init(self.tq_config.to_tq_config())
        log(
            f"[TQ] initialized backend={self.tq_config.backend} "
            f"partition_id={self.tq_config.partition_id}"
        )

    def _apply_transfer_queue_log_level(self) -> None:
        """Set only ``transfer_queue*`` loggers; leave gcore / root loggers alone."""
        level_name = self.tq_config.log_level
        os.environ["TQ_LOGGING_LEVEL"] = level_name
        logging.getLogger("transfer_queue").setLevel(level_name)
        for name in logging.root.manager.loggerDict:
            if name == "transfer_queue" or name.startswith("transfer_queue."):
                logging.getLogger(name).setLevel(level_name)

    def close(self) -> None:
        """Close this process's TransferQueue runtime and its managed subprocesses."""
        tq.close()
        if self.tq_config.backend == "MooncakeStore":
            self._terminate_mooncake_processes()

    def _terminate_mooncake_processes(self) -> None:
        """Terminate mooncake_master and metadata_server matching this session's ports."""
        mc = self.tq_config.mooncake
        master_port = urlparse(
            mc.master_server_address if "://" in mc.master_server_address else "//" +
            mc.master_server_address
        ).port
        metadata_port = urlparse(
            mc.metadata_server if "://" in mc.metadata_server else "//" + mc.metadata_server
        ).port

        patterns = []
        if master_port:
            patterns.append(f"[m]ooncake_master.*--rpc_port={master_port}")
        if metadata_port:
            patterns.append(f"[m]ooncake_http_metadata_server.*--port.*{metadata_port}")

        for pattern in patterns:
            subprocess.run(
                ["pkill", "-TERM", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        time.sleep(2)
        for pattern in patterns:
            subprocess.run(
                ["pkill", "-KILL", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    @staticmethod
    def _key_prefix(ppo_step: int) -> str:
        return f"step{ppo_step}:uid:"

    @staticmethod
    def make_keys(count: int, *, ppo_step: int) -> List[str]:
        """Generate *count* unique user keys for a given step."""
        prefix = TqConnector._key_prefix(ppo_step)
        return [f"{prefix}{uuid.uuid4().hex}" for _ in range(count)]

    async def async_set(
        self,
        tensor_dict: TensorDict,
        key: str | Sequence[str],
        ppo_step: int,
    ) -> KVBatchMeta:
        """Store one or more TensorDict rows under caller-provided keys."""
        keys = [key] if isinstance(key, str) else list(key)
        assert len(keys) == int(tensor_dict.batch_size[0])
        field_names = sorted(str(name) for name in tensor_dict.keys())
        return await tq.async_kv_batch_put(
            keys=keys,
            partition_id=self.tq_config.partition_id,
            fields=tensor_dict,
            tags=[{
                "ppo_step": ppo_step,
                "fields": field_names,
            }] * len(keys),
        )

    async def async_get(self, key: str | Sequence[str]) -> TensorDict:
        """Restore TensorDict rows by their user-provided keys."""
        keys = [key] if isinstance(key, str) else list(key)
        return await tq.async_kv_batch_get(
            keys=keys,
            partition_id=self.tq_config.partition_id,
        )

    def get(self, key: str | Sequence[str]) -> TensorDict:
        """Synchronously restore TensorDict rows by user-provided keys."""
        keys = [key] if isinstance(key, str) else list(key)
        return tq.kv_batch_get(
            keys=keys,
            partition_id=self.tq_config.partition_id,
        )

    @staticmethod
    def _group_keys_by_fields(
        key_meta: Dict[str, Any],
        prefix: str,
    ) -> List[List[str]]:
        """Group listed keys that share the same TransferQueue field schema.

        Mooncake/KV ``async_kv_clear`` only keeps fields present on every key
        in one call. Prompt and hidden payloads have disjoint fields, so a
        mixed clear produces empty ``field_names`` and fails.
        """
        groups: dict[tuple[str, ...], List[str]] = defaultdict(list)
        ungrouped: List[str] = []
        for key, meta in key_meta.items():
            if not key.startswith(prefix):
                continue
            fields = meta.get("fields") if isinstance(meta, dict) else None
            if fields:
                groups[tuple(fields)].append(key)
            else:
                ungrouped.append(key)
        grouped_keys = list(groups.values())
        grouped_keys.extend([[key] for key in ungrouped])
        return grouped_keys

    async def async_clear_step(self, ppo_step: int) -> None:
        """Clear all keys generated for one training step."""
        partition_id = self.tq_config.partition_id
        assert partition_id is not None
        partition_info = await tq.async_kv_list(partition_id=partition_id)
        prefix = self._key_prefix(ppo_step)
        for keys in self._group_keys_by_fields(
            partition_info.get(partition_id, {}),
            prefix,
        ):
            await tq.async_kv_clear(
                keys=keys,
                partition_id=partition_id,
            )
