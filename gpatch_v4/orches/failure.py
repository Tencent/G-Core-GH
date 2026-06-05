"""Failure event definitions for distributed training fault tolerance."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class FailureType(Enum):
    """Types of failure that can occur during distributed training."""

    HANG = "hang"
    CRASH = "crash"


@dataclass
class FailureEvent:
    """Describes a detected failure during distributed training.

    Attributes
    ----------
    failure_type : FailureType
        Type of failure: ``FailureType.HANG`` (actor lost liveness / no progress) or
        ``FailureType.CRASH`` (actor process died, RayActorError propagated).
    failed_node_ips : list[str]
        IP addresses of the nodes where the failure was detected.
    failed_actor_names : list[str]
        Names of the Ray actors that failed.
    timestamp : float
        Unix timestamp when the failure was detected.
    details : str
        Additional information about the failure (e.g. exception message).
    """

    failure_type: FailureType
    failed_node_ips: List[str]
    failed_actor_names: List[str]
    timestamp: float = field(default_factory=time.time)
    details: str = ""

    def __bool__(self) -> bool:
        """FailureEvent is always truthy (for backward compat with old bool return)."""
        return True

    def __repr__(self) -> str:
        return (
            f"FailureEvent(type={self.failure_type.value!r}, "
            f"nodes={self.failed_node_ips}, "
            f"actors={self.failed_actor_names}, "
            f"details={self.details!r})"
        )
