"""XPU kernel-device registrations.

Importing this package registers all XPU kernels for
``kernel_device_name="xpu_priv"``.
这里的 name 是 cuda，具可见：gpatch_v4/core/device/device_xpu_priv.py
"""

from gpatch_v4.kernel.registry import register_kernel
from gpatch_v4.kernel.triton.linear_cross_entropy import (
    linear_cross_entropy,
    set_linear_ce_backend,
)

register_kernel("linear_cross_entropy", linear_cross_entropy, devices="cuda")
register_kernel("set_linear_ce_backend", set_linear_ce_backend, devices="cuda")
