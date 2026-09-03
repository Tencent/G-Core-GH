"""CUDA kernel-device registrations.

Importing this package registers all CUDA kernels for
``kernel_device_name="cuda"``.
"""

from gpatch_v4.kernel.registry import register_kernel
from gpatch_v4.kernel.triton.linear_cross_entropy import (
    linear_cross_entropy,
    set_linear_ce_backend,
)

register_kernel("linear_cross_entropy", linear_cross_entropy, devices="cuda")
register_kernel("set_linear_ce_backend", set_linear_ce_backend, devices="cuda")
