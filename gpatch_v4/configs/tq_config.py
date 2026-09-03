"""Minimal G-Core configuration for the TransferQueue connector."""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class TqSimpleStorageConfig(MappingProtocol):
    """Configuration for TransferQueue's in-memory SimpleStorage backend.

    Attributes
    ----------
    num_storage_units : int
        Number of distributed SimpleStorage units. TransferQueue splits the
        total sample capacity across these units.
    total_storage_size : int
        Maximum number of samples retained across all SimpleStorage units.
    """
    num_storage_units: int = field(default=2)
    total_storage_size: int = field(default=100000)

    def __post_init__(self) -> None:
        assert self.num_storage_units >= 1
        assert self.total_storage_size >= 1


@dataclass
class TqMooncakeConfig(MappingProtocol):
    """Configuration for TransferQueue's MooncakeStore backend.

    Attributes
    ----------
    auto_init : bool
        Whether TransferQueue starts the Mooncake metadata and master services.
        The driver starts them once and worker processes then reuse them.
    metadata_server : str
        Metadata coordination server address in ``host:port`` form. When empty,
        :class:`TqConfig` assigns an available address on the head node.
    master_server_address : str
        Mooncake master RPC address in ``host:port`` form. When empty,
        :class:`TqConfig` assigns an available address on the head node.
    local_hostname : str
        Address advertised by the local Mooncake client. An empty value lets
        TransferQueue detect the current Ray node IP.
    protocol : str
        Data transport protocol, either ``"tcp"`` or ``"rdma"``.
    global_segment_size_gb : int
        Mounted global memory segment size in GiB for each Mooncake client.
        Converted to bytes when mapped onto TransferQueue.
    local_buffer_size_gb : int
        Local buffer size in GiB for each Mooncake client.
        Converted to bytes when mapped onto TransferQueue.
    device_name : str
        Network device passed to Mooncake. For RDMA, an empty value is replaced
        with the first active ``mlx5_bond_*`` or ``mlx5_*`` device found locally.
    """
    auto_init: bool = field(default=True)
    metadata_server: str = field(default="")
    master_server_address: str = field(default="")
    local_hostname: str = field(default="")
    protocol: str = field(default="tcp")
    global_segment_size_gb: int = field(default=4)
    local_buffer_size_gb: int = field(default=1)
    device_name: str = field(default="")

    def __post_init__(self) -> None:
        assert self.protocol in ("tcp", "rdma")
        assert self.global_segment_size_gb >= 1
        assert self.local_buffer_size_gb >= 1


@dataclass
class TqConfig(MappingProtocol):
    """Top-level TransferQueue configuration.

    Attributes
    ----------
    enable : bool
        Whether to initialize TransferQueue and use its off-policy data path.
    backend : str
        Storage backend, either ``"SimpleStorage"`` or ``"MooncakeStore"``.
    partition_id : str or None
        Namespace shared by all TransferQueue keys in one run. When TransferQueue
        is enabled and this value is empty, a 12-character hexadecimal ID is
        generated automatically.
    log_level : str
        Logging level applied only to TransferQueue loggers. It is normalized to
        uppercase during initialization.
    simple_storage : TqSimpleStorageConfig
        Settings used when ``backend`` is ``"SimpleStorage"``.
    mooncake : TqMooncakeConfig
        Settings used when ``backend`` is ``"MooncakeStore"``.
    """
    enable: bool = field(default=False)
    backend: str = field(default="SimpleStorage")
    partition_id: Optional[str] = field(default=None)
    log_level: str = field(default="WARNING")
    simple_storage: TqSimpleStorageConfig = field(default_factory=TqSimpleStorageConfig)
    mooncake: TqMooncakeConfig = field(default_factory=TqMooncakeConfig)

    def __post_init__(self) -> None:
        self.log_level = self.log_level.upper()
        assert self.backend in ("SimpleStorage", "MooncakeStore")
        if self.enable and not self.partition_id:
            self.partition_id = uuid.uuid4().hex[:12]
        if not self.enable or self.backend != "MooncakeStore":
            return

        # import here to avoid circular import
        from gpatch_v4.orches.utils import get_current_node_ip
        head_ip = get_current_node_ip()
        if (not self.mooncake.metadata_server or not self.mooncake.master_server_address):
            metadata_port, master_port = (self.find_available_mooncake_port_pair())
            self.mooncake.metadata_server = f"{head_ip}:{metadata_port}"
            self.mooncake.master_server_address = f"{head_ip}:{master_port}"
        if self.mooncake.protocol == "rdma" and not self.mooncake.device_name:
            self.mooncake.device_name = self.get_active_rdma_device()
            assert self.mooncake.device_name, ("No active RDMA device found for MooncakeStore")

    def find_available_mooncake_port_pair(self) -> tuple[int, int]:
        """Find two consecutive ports for Mooncake's managed services."""
        # import here to avoid circular import
        from gpatch_v4.orches.utils import is_port_available

        port_min = 55000
        port_max = 56000
        default_ports = (55050, 55051)

        if all(is_port_available(port) for port in default_ports):
            return default_ports

        for metadata_port in range(port_min, port_max):
            master_port = metadata_port + 1
            if (is_port_available(metadata_port) and is_port_available(master_port)):
                return metadata_port, master_port

        raise RuntimeError(
            "No two consecutive Mooncake ports are available in "
            f"[{port_min}, {port_max}]"
        )

    def get_active_rdma_device(self) -> str:
        infiniband_dir = Path("/sys/class/infiniband")
        for pattern in ("mlx5_bond_*", "mlx5_*"):
            for device_path in sorted(infiniband_dir.glob(pattern)):
                try:
                    state = (device_path / "ports/1/state").read_text().strip()
                except OSError:
                    continue
                if state.startswith("4:"):
                    return device_path.name
        return ""

    def to_tq_config(self) -> DictConfig:
        """Map the G-Core schema onto TransferQueue's native schema."""
        return OmegaConf.create(
            {
                "backend":
                    {
                        "storage_backend": self.backend,
                        "SimpleStorage":
                            {
                                "total_storage_size": self.simple_storage.total_storage_size,
                                "num_data_storage_units": self.simple_storage.num_storage_units,
                            },
                        "MooncakeStore":
                            {
                                "auto_init": self.mooncake.auto_init,
                                "metadata_server": self.mooncake.metadata_server,
                                "master_server_address": self.mooncake.master_server_address,
                                "local_hostname": self.mooncake.local_hostname,
                                "protocol": self.mooncake.protocol,
                                "global_segment_size": self.mooncake.global_segment_size_gb * (1024**3),
                                "local_buffer_size": self.mooncake.local_buffer_size_gb * (1024**3),
                                "device_name": self.mooncake.device_name,
                            },
                    }
            }
        )
