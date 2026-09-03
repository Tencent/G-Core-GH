import importlib
import pathlib

from gpatch_v4.core.device.protocol import CudaDeviceBackend

_DEVICE_BACKEND = CudaDeviceBackend()

# Auto-discover *_priv backend modules in this package.
# The *_priv modules are stripped during open-source release, so failure
# here simply means we fall back to the default CudaDeviceBackend.
_pkg_dir = pathlib.Path(__file__).parent
_priv_files = sorted(_pkg_dir.glob("*_priv.py"))

for _priv_file in _priv_files:
    _module_name = f"gpatch_v4.core.device.{_priv_file.stem}"
    try:
        _mod = importlib.import_module(_module_name)
        _backend = _mod.initialize_device_backend()
        if _backend is not None:
            _DEVICE_BACKEND = _backend
            del _backend
            break
    except (ImportError, ModuleNotFoundError):
        pass


def get_device_perf_activity():
    return _DEVICE_BACKEND.perf_activity


def get_profiler_module():
    return _DEVICE_BACKEND.profiler_module


def is_cuda() -> bool:
    return _DEVICE_BACKEND.is_cuda


def get_device_module():
    return _DEVICE_BACKEND.device_module


def get_device_backend_name() -> str:
    """Return the device backend name, e.g. ``"cuda"``."""
    return _DEVICE_BACKEND.name


def get_kernel_device_name() -> str:
    # 调用 kernel 的 device name
    # 同是 cuda，也可以有不同的 name
    return _DEVICE_BACKEND.kernel_device_name


def get_dist_backend() -> str:
    return _DEVICE_BACKEND.dist_backend


def get_visible_devices_env_var() -> str:
    """Return the environment variable name for visible devices.

    For example ``"CUDA_VISIBLE_DEVICES"`` for CUDA backend.
    """
    return _DEVICE_BACKEND.visible_devices_env_var


def get_propagate_env_keys() -> list[str]:
    return _DEVICE_BACKEND.propagate_env_keys


def sync_param_offload() -> bool:
    return _DEVICE_BACKEND.sync_param_offload


def preprocess_before_build_sgl_engine() -> None:
    return _DEVICE_BACKEND.preprocess_before_build_sgl_engine()


def get_flash_attn_varlen_func(max_segment_len=4096):
    return _DEVICE_BACKEND.get_flash_attn_varlen_func(max_segment_len=max_segment_len)
