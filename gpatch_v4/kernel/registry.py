"""Device-dispatched kernel registry.

Kernels are plain functions. Each kernel-device has a package
``gpatch_v4.kernel.<kernel_device_name>`` (e.g. ``cuda/``) whose
``__init__`` registers implementations. ``kernel_device_name`` can be
finer than the torch backend (``cuda`` / ``npu`` / ``mlu``) when the
same backend needs different ops.

Register (in ``gpatch_v4/kernel/cuda/__init__.py``)::

    from gpatch_v4.kernel.registry import register_kernel
    from gpatch_v4.kernel.triton.linear_cross_entropy import linear_cross_entropy

    register_kernel("linear_cross_entropy", linear_cross_entropy, devices="cuda")

Dispatch (public API in ``gpatch_v4.kernel``)::

    def linear_cross_entropy(*args, **kwargs):
        return get_kernel("linear_cross_entropy")(*args, **kwargs)

Adding a new device
-------------------
1. Put the implementation under ``gpatch_v4/kernel/<impl>/``.
2. Create ``gpatch_v4/kernel/<kernel_device_name>/__init__.py`` and call
   ``register_kernel(name, fn, devices="<kernel_device_name>")``.
3. ``<kernel_device_name>`` must match
   :func:`gpatch_v4.core.device.get_kernel_device_name`. Override
   ``DeviceProtocol.kernel_device_name`` when it should differ from
   :func:`gpatch_v4.core.device.get_device_backend_name`.

``get_kernel`` imports ``gpatch_v4.kernel.<kernel_device_name>`` once so
Triton kernels are not loaded on NPU / MLU.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from functools import lru_cache

# op_name -> device_name -> fn
_KERNEL_REGISTRY: dict[str, dict[str, Callable]] = {}

_KERNEL_PACKAGE = "gpatch_v4.kernel"


@lru_cache(maxsize=1)
def _current_device() -> str:
    from gpatch_v4.core.device import get_device_backend_name
    return get_device_backend_name()


@lru_cache(maxsize=1)
def get_cur_kernel_device_name() -> str:
    """Return the kernel-device name for the current process."""
    from gpatch_v4.core.device import get_kernel_device_name
    return get_kernel_device_name()


@lru_cache(maxsize=None)
def _import_device(device: str) -> None:
    """Import ``gpatch_v4.kernel.<device>`` once so it can register kernels."""
    module_name = f"{_KERNEL_PACKAGE}.{device}"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Missing device module is expected until that hardware lands.
        # Nested import failures inside an existing module must still raise.
        if exc.name != module_name:
            raise


def register_kernel(
    name: str,
    fn: Callable,
    devices: str | Sequence[str],
) -> None:
    """Register ``fn`` as the implementation of ``name`` for ``devices``.

    Parameters
    ----------
    name : str
        Logical kernel name, e.g. ``"linear_cross_entropy"``.
    fn : Callable
        Kernel implementation.
    devices : str or sequence of str, optional
        Device backend names matching ``get_device_backend_name()``.
        If omitted, registers for the current device.

    Raises
    ------
    ValueError
        If ``name`` is already registered for one of ``devices``.
    """
    assert devices is not None
    if isinstance(devices, str):
        device_list = (devices, )
    else:
        device_list = tuple(devices)
    if not device_list:
        raise ValueError("At least one device must be specified")
    if len(device_list) != len(set(device_list)):
        raise ValueError(f"Duplicate devices: {device_list}")

    registry = _KERNEL_REGISTRY.setdefault(name, {})
    for device in device_list:
        if device in registry:
            registered = registry[device]
            raise ValueError(
                f"Kernel '{name}' for device '{device}' already registered by "
                f"{registered.__module__}.{registered.__qualname__}"
            )
        registry[device] = fn


def get_kernel(
    name: str,
    device: str | None = None,
    *,
    required: bool = True,
) -> Callable | None:
    """Return the registered implementation of ``name`` for ``device``.

    Parameters
    ----------
    name : str
        Logical kernel name.
    device : str, optional
        Device backend name. Defaults to the current device backend.
    required : bool
        If True (default), raise when no implementation is registered.
        If False, return None.

    Returns
    -------
    Callable or None
        Registered function, or None when ``required`` is False and
        nothing is registered.

    Raises
    ------
    RuntimeError
        If ``required`` is True and no implementation is registered.
    """
    if device is None:
        device = _current_device()
    _import_device(get_cur_kernel_device_name())
    fn = _KERNEL_REGISTRY.get(name, {}).get(device)
    if fn is not None:
        return fn
    if not required:
        return None
    available = sorted(_KERNEL_REGISTRY.get(name, {}))
    available_str = ", ".join(available) if available else "(none)"
    raise RuntimeError(
        f"No '{name}' kernel registered for device '{device}'. "
        f"Registered devices: [{available_str}]. "
        f"Add gpatch_v4/kernel/{device}/ and "
        f"register_kernel('{name}', fn, devices='{device}')."
    )


def is_kernel_registered(name: str, device: str | None = None) -> bool:
    """Return True if ``name`` is registered for ``device``."""
    return get_kernel(name, device, required=False) is not None


def list_kernels() -> dict[str, list[str]]:
    """Return ``{op_name: [device, ...]}`` for currently registered kernels.

    Does not trigger device-module imports.
    """
    return {name: sorted(devices) for name, devices in _KERNEL_REGISTRY.items()}
