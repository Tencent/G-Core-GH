"""Abstract interface for node eviction and provisioning.

Resource platforms should implement :class:`NodeReplacer` to enable
automatic node replacement on failure. The trainer calls these methods
during the restart path of ``launch_then_run_with_recovery``.
"""

import time
from abc import ABC, abstractmethod
from typing import List

import ray
from typing_extensions import override

from gpatch_v4.utils import log


class NodeReplacer(ABC):
    """Abstract interface for node eviction and provisioning.

    A resource platform must implement this to enable automatic
    node replacement when a failure is detected during training.
    """
    @abstractmethod
    def evict_nodes(self, node_ips: List[str]) -> None:
        """Remove faulty nodes from the cluster.

        Parameters
        ----------
        node_ips : list[str]
        """
        ...

    @abstractmethod
    def provision_nodes(self, count: int) -> List[str]:
        """Request *count* new nodes and add them to the cluster.

        Blocks until the new nodes are ready (Ray workers connected to GCS).

        Parameters
        ----------
        count : int

        Returns
        -------
        list[str]
            IP addresses of newly provisioned nodes.
        """
        ...

    @abstractmethod
    def wait_cluster_ready(self, expected_nnodes: int, timeout: float = 300) -> bool:
        """Wait until the Ray cluster has *expected_nnodes* alive nodes.

        Parameters
        ----------
        expected_nnodes : int
        timeout : float

        Returns
        -------
        bool
            *False* on timeout.
        """
        ...


class MockNodeReplacer(NodeReplacer):
    """Mock implementation for testing and manual-replacement scenarios.

    - ``evict_nodes``: only logs the eviction (does not actually remove nodes).
    - ``provision_nodes``: does nothing (expects the user to manually add nodes).
    - ``wait_cluster_ready``: polls ``ray.nodes()`` until the target count is met.
    """
    def __init__(self, config=None):
        pass

    @override
    def evict_nodes(self, node_ips: List[str]) -> None:
        log(f"[MockNodeReplacer] evict_nodes called: {node_ips}")

    @override
    def provision_nodes(self, count: int) -> List[str]:
        log(f"[MockNodeReplacer] provision_nodes({count}): waiting for external provisioning...")
        return []

    @override
    def wait_cluster_ready(self, expected_nnodes: int, timeout: float = 300) -> bool:
        """Poll ray.nodes() until *expected_nnodes* alive nodes are found."""
        log(
            f"[MockNodeReplacer] waiting for cluster to reach {expected_nnodes} nodes "
            f"(timeout={timeout}s)..."
        )
        start = time.time()
        while time.time() - start < timeout:
            alive = len([n for n in ray.nodes() if n["Alive"]])
            if alive >= expected_nnodes:
                log(f"[MockNodeReplacer] cluster ready: {alive} nodes alive")
                return True
            time.sleep(5)
        alive = len([n for n in ray.nodes() if n["Alive"]])
        log(
            f"[MockNodeReplacer] timeout: only {alive}/{expected_nnodes} nodes alive "
            f"after {timeout}s"
        )
        return False
