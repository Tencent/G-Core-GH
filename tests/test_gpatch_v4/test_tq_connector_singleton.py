import asyncio
from types import SimpleNamespace

import pytest

import gpatch_v4.transfer as transfer
import gpatch_v4.transfer.tq_connector as tq_connector
import gpatch_v4.orches.utils as orches_utils
from gpatch_v4.configs.tq_config import TqConfig, TqMooncakeConfig


class _FakeConnector:
    def __init__(self, tq_config):
        self.tq_config = tq_config
        self.closed = False

    def close(self):
        self.closed = True


def test_tq_config_generates_partition_id_when_enabled():
    config = TqConfig(enable=True)

    assert config.partition_id is not None
    assert len(config.partition_id) == 12
    int(config.partition_id, 16)
    assert TqConfig().partition_id is None
    configured = TqConfig(enable=True, partition_id="configured")
    assert configured.partition_id == "configured"


def test_tq_config_resolves_mooncake_runtime_defaults(monkeypatch):
    monkeypatch.setattr(
        orches_utils, "get_current_node_ip", lambda: "10.0.0.1"
    )
    monkeypatch.setattr(
        orches_utils, "is_port_available", lambda port: True
    )
    monkeypatch.setattr(
        TqConfig, "get_active_rdma_device", lambda self: "mlx5_bond_0"
    )

    config = TqConfig(
        enable=True,
        backend="MooncakeStore",
        mooncake=TqMooncakeConfig(protocol="rdma"),
    )

    assert config.mooncake.metadata_server == "10.0.0.1:55050"
    assert config.mooncake.master_server_address == "10.0.0.1:55051"
    assert config.mooncake.device_name == "mlx5_bond_0"


def test_tq_config_replaces_occupied_mooncake_port_pair(monkeypatch):
    monkeypatch.setattr(
        orches_utils, "get_current_node_ip", lambda: "10.0.0.1"
    )
    monkeypatch.setattr(
        orches_utils,
        "is_port_available",
        lambda port: port != 55050,
    )

    config = TqConfig(enable=True, backend="MooncakeStore")

    assert config.mooncake.metadata_server == "10.0.0.1:55000"
    assert config.mooncake.master_server_address == "10.0.0.1:55001"


def test_tq_config_fails_when_mooncake_port_range_is_exhausted(monkeypatch):
    monkeypatch.setattr(
        orches_utils, "get_current_node_ip", lambda: "10.0.0.1"
    )
    monkeypatch.setattr(
        orches_utils, "is_port_available", lambda port: False
    )

    with pytest.raises(RuntimeError, match="No two consecutive Mooncake ports"):
        TqConfig(enable=True, backend="MooncakeStore")


def test_tq_config_preserves_explicit_mooncake_endpoints(monkeypatch):
    monkeypatch.setattr(
        TqConfig,
        "find_available_mooncake_port_pair",
        lambda self: pytest.fail(
            "explicit endpoints must not trigger port probing"
        ),
    )

    config = TqConfig(
        enable=True,
        backend="MooncakeStore",
        mooncake=TqMooncakeConfig(
            metadata_server="metadata.example:1234",
            master_server_address="master.example:5678",
        ),
    )

    assert config.mooncake.metadata_server == "metadata.example:1234"
    assert config.mooncake.master_server_address == "master.example:5678"


def test_tq_config_converts_mooncake_sizes_from_gb_to_bytes():
    config = TqConfig(
        mooncake=TqMooncakeConfig(global_segment_size_gb=8, local_buffer_size_gb=2),
    )
    mooncake = config.to_tq_config().backend.MooncakeStore

    assert config.mooncake.global_segment_size_gb == 8
    assert config.mooncake.local_buffer_size_gb == 2
    assert mooncake.global_segment_size == 8 * 1024**3
    assert mooncake.local_buffer_size == 2 * 1024**3

    defaults = TqConfig().to_tq_config().backend.MooncakeStore
    assert defaults.global_segment_size == 4 * 1024**3
    assert defaults.local_buffer_size == 1 * 1024**3


def test_tq_connector_clears_only_keys_for_exact_step(monkeypatch):
    cleared = []

    async def async_kv_list(partition_id):
        assert partition_id == "test-partition"
        return {
            partition_id: {
                "step1:uid:prompt": {
                    "ppo_step": 1,
                    "fields": ["tokens", "images"],
                },
                "step1:uid:hidden": {
                    "ppo_step": 1,
                    "fields": ["teacher_hidden_states"],
                },
                "step10:uid:other": {
                    "ppo_step": 10,
                    "fields": ["tokens", "images"],
                },
                "custom-key": {},
            }
        }

    async def async_kv_clear(keys, partition_id):
        cleared.append((list(keys), partition_id))

    monkeypatch.setattr(
        tq_connector,
        "tq",
        SimpleNamespace(
            async_kv_list=async_kv_list,
            async_kv_clear=async_kv_clear,
        ),
    )
    connector = object.__new__(transfer.TqConnector)
    connector.tq_config = TqConfig(
        enable=True,
        partition_id="test-partition",
    )

    asyncio.run(connector.async_clear_step(1))

    assert cleared == [
        (["step1:uid:prompt"], "test-partition"),
        (["step1:uid:hidden"], "test-partition"),
    ]


def test_tq_connector_clears_legacy_keys_individually(monkeypatch):
    cleared = []

    async def async_kv_list(partition_id):
        return {
            partition_id: {
                "step2:uid:a": {"ppo_step": 2},
                "step2:uid:b": {"ppo_step": 2},
            }
        }

    async def async_kv_clear(keys, partition_id):
        cleared.append((list(keys), partition_id))

    monkeypatch.setattr(
        tq_connector,
        "tq",
        SimpleNamespace(
            async_kv_list=async_kv_list,
            async_kv_clear=async_kv_clear,
        ),
    )
    connector = object.__new__(transfer.TqConnector)
    connector.tq_config = TqConfig(
        enable=True,
        partition_id="test-partition",
    )

    asyncio.run(connector.async_clear_step(2))

    assert cleared == [
        (["step2:uid:a"], "test-partition"),
        (["step2:uid:b"], "test-partition"),
    ]


def test_tq_connector_process_singleton_lifecycle(monkeypatch):
    monkeypatch.setattr(transfer, "TqConnector", _FakeConnector)
    monkeypatch.setattr(transfer, "_TQ_CONNECTOR", None)

    config = TqConfig(enable=True, partition_id="first")
    connector = transfer.init_tq_connector(config)

    assert transfer.init_tq_connector(config) is connector
    assert transfer.get_tq_connector() is connector

    with pytest.raises(AssertionError, match="different config"):
        transfer.init_tq_connector(
            TqConfig(enable=True, partition_id="second")
        )

    transfer.close_tq_connector()
    assert connector.closed
    with pytest.raises(AssertionError, match="not initialized"):
        transfer.get_tq_connector()

    replacement = transfer.init_tq_connector(
        TqConfig(enable=True, partition_id="second")
    )
    assert replacement is not connector
    transfer.close_tq_connector()
