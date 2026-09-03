"""Public TransferQueue adapter API and process-local connector lifecycle."""

from gpatch_v4.configs.tq_config import TqConfig
from gpatch_v4.transfer.tq_connector import TqConnector

_TQ_CONNECTOR: TqConnector | None = None


def init_tq_connector(tq_config: TqConfig) -> TqConnector:
    """Initialize and return this process's connector."""
    global _TQ_CONNECTOR
    if _TQ_CONNECTOR is None:
        _TQ_CONNECTOR = TqConnector(tq_config)
    else:
        assert _TQ_CONNECTOR.tq_config == tq_config, (
            "TqConnector is already initialized with a different config"
        )
    return _TQ_CONNECTOR


def get_tq_connector() -> TqConnector:
    """Return this process's initialized connector."""
    assert _TQ_CONNECTOR is not None, "TqConnector is not initialized"
    return _TQ_CONNECTOR


def close_tq_connector() -> None:
    """Close and clear this process's connector."""
    global _TQ_CONNECTOR
    if _TQ_CONNECTOR is None:
        return
    try:
        _TQ_CONNECTOR.close()
    finally:
        _TQ_CONNECTOR = None


__all__ = [
    "TqConnector",
    "close_tq_connector",
    "get_tq_connector",
    "init_tq_connector",
]
