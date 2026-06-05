import collections
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class KvConfig(MappingProtocol):
    dist_config: DistConfig = field(default_factory=DistConfig)
    root: str = field(default="/root/kv")
    # Populated at runtime by RayKvStoreGroup after port allocation.
    # Each entry is "addr:port" for the corresponding KV actor's HTTP server.
    http_endpoints: List[str] = field(default_factory=list)
