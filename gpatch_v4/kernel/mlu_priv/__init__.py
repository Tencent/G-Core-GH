"""mlu kernel-device registrations.

Importing this package registers all mlu kernels for
``kernel_device_name="mlu_priv"``. Lookup still uses backend name ``mlu``.
"""

from gpatch_v4.kernel.registry import register_kernel
from gpatch_v4.kernel.mlu_priv.linear_cross_entropy import (
    linear_cross_entropy,
    set_linear_ce_backend,
)

register_kernel("linear_cross_entropy", linear_cross_entropy, devices="mlu")
register_kernel("set_linear_ce_backend", set_linear_ce_backend, devices="mlu")
