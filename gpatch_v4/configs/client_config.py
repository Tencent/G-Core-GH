from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class ClientConfig(MappingProtocol):
    """Configuration for RPC clients connecting to remote services.

    Attributes
    ----------
    endpoint_ips : list of list of str or None
        Nested list (one per replica group).
    endpoint_ports : list of list of int or None
    rpc_type : str
        ``"http"`` or ``"ray"``.
    rpc_timeout : int
        Seconds.
    update_weight_max_size_mb : int
        Max weight tensor size per RPC call.
    update_weight_use_bucketed_ipc : bool
        When *True* and the backend is vLLM, use flat-IPC bucketed transfer
        (one CUDA IPC handle per (rank, dtype, bucket); finalize once).
        Required for MoE models whose experts straddle multiple buckets.
        Requires colocated placement.
    split_generate_repeat_n : bool
        Split generation across ``repeat_n`` into separate RPC calls.
    """
    endpoint_ips: Optional[List[List[str]]
                          ] = field(default=None, metadata={"help": "List of endpoint ips"})
    endpoint_ports: Optional[List[List[int]]
                            ] = field(default=None, metadata={"help": "List of endpoint ports"})
    rpc_type: str = field(default='http', metadata={"help": "RPC type must in [http, ray"})
    rpc_timeout: int = field(default=60, metadata={"help": "RPC client timeout"})
    update_weight_max_size_mb: int = field(
        default=512, metadata={"help": "Max size of the weight to update in MB"}
    )
    update_weight_use_bucketed_ipc: bool = field(
        default=True,
        metadata={
            "help":
                "vLLM only: use flat-IPC bucketed weight transfer "
                "(one CUDA IPC handle per dtype per bucket + finalize-once). "
                "Default True; required for MoE models."
        },
    )
    split_generate_repeat_n: bool = field(
        default=False, metadata={"help": "Whether to split generate repeat n"}
    )

    def __post_init__(self):
        #TODO: support 'zmq' rpc_type
        assert self.rpc_type in [
            "http",
            "ray",
        ]
