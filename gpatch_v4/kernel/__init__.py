"""Hardware-dispatched kernels.

Public functions look up the implementation registered for
``get_kernel_device_name()``. CUDA registrations live in
:mod:`gpatch_v4.kernel.cuda`. Add another kernel-device with a
``<kernel_device_name>/`` package whose ``__init__`` registers functions::

    # gpatch_v4/kernel/npu/__init__.py
    from gpatch_v4.kernel.registry import register_kernel
    from gpatch_v4.kernel.npu_impl.linear_cross_entropy import linear_cross_entropy

    register_kernel("linear_cross_entropy", linear_cross_entropy, devices="npu")
"""

from __future__ import annotations

import typing

import torch
import torch.distributed as dist

from .registry import (
    get_cur_kernel_device_name,
    get_kernel,
    is_kernel_registered,
    list_kernels,
    register_kernel,
)


def set_linear_ce_backend(backend: str):
    fn = get_kernel("set_linear_ce_backend", required=False)
    if fn is None:
        return
    fn(backend)


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: typing.Optional[float] = 1.0,
    reduction: typing.Optional[str] = "none",
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    return_entropy: bool = False,
):
    return get_kernel("linear_cross_entropy")(
        hidden,
        weight,
        labels,
        temperature,
        reduction,
        dist_process_group,
        return_entropy,
    )


__all__ = [
    "get_cur_kernel_device_name",
    "get_kernel",
    "is_kernel_registered",
    "linear_cross_entropy",
    "list_kernels",
    "register_kernel",
    "set_linear_ce_backend",
]
