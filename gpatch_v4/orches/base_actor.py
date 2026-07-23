import os

from gpatch_v4.orches.utils import find_free_port, get_current_node_ip


class RayBaseActor:
    """Base Ray actor providing IP and port discovery utilities."""
    def _get_current_node_ip(self) -> str:
        """Return the IP address of the current node.

        Resolution order:

        1. ``__HOST_IP__`` environment variable
        2. IP of the ``bond1`` / ``bond0`` network interface
        3. IP of the ``eth0`` network interface
        4. ``127.0.0.1`` as last resort

        Returns
        -------
        str
        """
        return get_current_node_ip()

    def _get_current_node_ip_and_free_port(self, start_port=10000, consecutive=1):
        """Find the node IP and a block of consecutive free ports.

        Parameters
        ----------
        start_port : int, optional
        consecutive : int, optional

        Returns
        -------
        tuple[str, int]
            ``(ip_address, first_free_port)``.
        """
        master_addr = get_current_node_ip()
        master_port = find_free_port(start_port, consecutive)
        return master_addr, master_port

    def get_master_addr_and_port(self):
        """Return ``(master_addr, master_port)`` for distributed init.

        Returns
        -------
        tuple[str, int]
        """
        return self.master_addr, self.master_port
