from typing import Any, Callable, Dict

from gpatch_v4.utils import log
from gpatch_v4.utils.common_utils import import_fn_from_path

BACKEND_LOSS_FUNC_REGISTRIES: Dict[str, Dict[str, Callable]] = {}
LOSS_HISTOGRAM_FINALIZERS: Dict[str, Callable] = {}


def register_loss(
    backends: tuple[str, ...],
    loss_name: str,
    log_registration: bool = False,
):
    def decorator(fn: Callable) -> Callable:
        assert backends, "At least one backend must be registered"
        assert len(backends) == len(set(backends)), f"Duplicate backends: {backends}"
        registries = {
            backend: BACKEND_LOSS_FUNC_REGISTRIES.setdefault(backend, {})
            for backend in backends
        }
        for backend, registry in registries.items():
            if loss_name in registry:
                registered = registry[loss_name]
                raise ValueError(
                    f"{backend} loss type '{loss_name}' already registered by "
                    f"{registered.__module__}.{registered.__qualname__}"
                )
        for registry in registries.values():
            registry[loss_name] = fn
        if log_registration:
            for backend in backends:
                log(f"Registered {backend.upper()} loss function: {loss_name}")
        return fn

    return decorator


def get_loss_fn(backend: str, loss_name: str) -> Callable:
    backend_registry = BACKEND_LOSS_FUNC_REGISTRIES.get(backend, {})
    if loss_name in backend_registry:
        return backend_registry[loss_name]
    available_backend = ", ".join(sorted(backend_registry)) or "(none)"
    raise ValueError(
        f"Unknown {backend} loss type: '{loss_name}'. Available: [{available_backend}]"
    )


def is_loss_registered(backend: str, loss_name: str) -> bool:
    return loss_name in BACKEND_LOSS_FUNC_REGISTRIES.get(backend, {})


def register_histogram_finalizer(loss_name: str):
    def decorator(fn: Callable) -> Callable:
        if loss_name in LOSS_HISTOGRAM_FINALIZERS:
            registered = LOSS_HISTOGRAM_FINALIZERS[loss_name]
            raise ValueError(
                f"Histogram finalizer for loss type '{loss_name}' is already registered by "
                f"{registered.__module__}.{registered.__qualname__}"
            )
        LOSS_HISTOGRAM_FINALIZERS[loss_name] = fn
        return fn

    return decorator


def finalize_histogram_metrics(
    loss_name: str,
    config: Any,
    histograms: Dict[str, Any],
) -> Dict[str, Any]:
    if not histograms:
        return {}
    if loss_name not in LOSS_HISTOGRAM_FINALIZERS:
        raise ValueError(
            f"Loss type '{loss_name}' produced histogram metrics but has no registered finalizer"
        )
    return LOSS_HISTOGRAM_FINALIZERS[loss_name](config, histograms)


def register_custom_loss_fn(
    backend: str,
    loss_name: str,
    py_path: str,
    fn_name: str,
) -> Callable:
    """Load and register a custom loss for one backend.

    Parameters
    ----------
    backend : str
        Backend key used for subsequent ``get_loss_fn`` lookups.
    loss_name : str
        Value configured as ``loss_func``.
    py_path : str
        Absolute path to the Python module that defines the loss.
    fn_name : str
        Callable exported by ``py_path``.

    Returns
    -------
    Callable
        Registered loss function.
    """
    fn = import_fn_from_path(py_path, fn_name)
    return register_loss((backend, ), loss_name)(fn)
