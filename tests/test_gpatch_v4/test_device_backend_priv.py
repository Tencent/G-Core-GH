"""Tests for ``gpatch_v4.core.device`` backend abstraction.

These tests verify:
- All backends implement ``DeviceProtocol`` correctly.
- The auto-detect fallback logic (open-source scenario without ``_priv`` modules).
- The public API returns correct types.
"""

import importlib
import sys
from unittest import mock

import pytest

from gpatch_v4.core.device.protocol import CudaDeviceBackend, DeviceProtocol


class TestDeviceProtocolCompliance:
    """All backends implement DeviceProtocol."""
    def test_cuda_is_subclass(self):
        assert issubclass(CudaDeviceBackend, DeviceProtocol)

    def test_cuda_instantiation(self):
        backend = CudaDeviceBackend()
        assert backend.is_cuda is True
        assert backend.name == "cuda"
        assert backend.dist_backend == "nccl"
        assert backend.visible_devices_env_var == "CUDA_VISIBLE_DEVICES"
        assert isinstance(backend.name, str)
        assert isinstance(backend.visible_devices_env_var, str)


class TestAutoDetect:
    """When _priv modules are absent (open-source scenario), fallback to cuda."""
    def test_fallback_to_cuda_when_no_priv(self):
        # Remove the core.device module from cache so it can be re-imported
        modules_to_remove = [key for key in sys.modules if key.startswith("gpatch_v4.core.device")]
        saved = {}
        for key in modules_to_remove:
            saved[key] = sys.modules.pop(key)

        # Block _priv imports via importlib.import_module which is what __init__.py uses
        real_import_module = importlib.import_module

        def mock_import_module(name, *args, **kwargs):
            if "device_ascend_priv" in name or "device_mlu_priv" in name:
                raise ImportError(f"mocked: {name}")
            return real_import_module(name, *args, **kwargs)

        try:
            with mock.patch("importlib.import_module", side_effect=mock_import_module):
                mod = importlib.reload(real_import_module("gpatch_v4.core.device"))
                assert mod.is_cuda() is True
                assert mod.get_device_backend_name() == "cuda"
        finally:
            # Restore
            for key, val in saved.items():
                sys.modules[key] = val


class TestPublicAPI:
    """Public functions return correct types."""
    def test_is_cuda_returns_bool(self):
        from gpatch_v4.core.device import is_cuda
        assert isinstance(is_cuda(), bool)

    def test_get_visible_devices_env_var_returns_str(self):
        from gpatch_v4.core.device import get_visible_devices_env_var
        result = get_visible_devices_env_var()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_device_backend_name_returns_str(self):
        from gpatch_v4.core.device import get_device_backend_name
        result = get_device_backend_name()
        assert isinstance(result, str)
        assert result in ("cuda", "mlu", "npu")

    def test_get_dist_backend_returns_str(self):
        from gpatch_v4.core.device import get_dist_backend
        result = get_dist_backend()
        assert isinstance(result, str)

    def test_old_module_removed(self):
        # Ensure the old top-level gpatch_v4.device is gone
        # Remove from sys.modules cache first so we get a fresh import attempt
        sys.modules.pop("gpatch_v4.device", None)
        with pytest.raises((ImportError, ModuleNotFoundError)):
            importlib.import_module("gpatch_v4.device")
