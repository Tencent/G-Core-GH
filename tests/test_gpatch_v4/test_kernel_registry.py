"""Kernel registry: per-device function registration and dispatch."""

from unittest.mock import patch

import pytest

from gpatch_v4.kernel.registry import (
    get_kernel,
    is_kernel_registered,
    list_kernels,
    register_kernel,
)

def _no_real_import():
    # get_kernel always calls get_cur_kernel_device_name() to import the
    # impl package. Stub it so fake-device tests do not load core.device.
    return patch(
        "gpatch_v4.kernel.registry.get_cur_kernel_device_name",
        return_value="no_such_kernel_impl",
    )


def test_register_and_get_kernel():
    def _add(x, y):
        return x + y

    register_kernel("_test_add", _add, devices="fake_dev")
    with _no_real_import():
        assert is_kernel_registered("_test_add", "fake_dev")
        assert get_kernel("_test_add", device="fake_dev")(1, 2) == 3
    assert "fake_dev" in list_kernels()["_test_add"]


def test_register_kernel_multiple_devices():
    def _shared():
        return 7

    register_kernel("_test_shared", _shared, devices=("dev_a", "dev_b"))
    with _no_real_import():
        assert get_kernel("_test_shared", device="dev_a") is _shared
        assert get_kernel("_test_shared", device="dev_b") is _shared


def test_duplicate_registration_raises():
    def _first():
        return 1

    def _second():
        return 2

    register_kernel("_test_dup", _first, devices="fake_dev")
    with pytest.raises(ValueError, match="already registered"):
        register_kernel("_test_dup", _second, devices="fake_dev")


def test_missing_kernel_raises_with_hint():
    with _no_real_import():
        with pytest.raises(RuntimeError, match="No '_test_missing' kernel"):
            get_kernel("_test_missing", device="npu")


def test_get_kernel_required_false():
    with _no_real_import():
        assert get_kernel("_test_missing", device="npu", required=False) is None


def test_empty_devices_raises():
    with pytest.raises(ValueError, match="At least one device"):
        register_kernel("_test_empty", lambda: None, devices=())


def test_default_lookup_uses_backend_name_import_uses_kernel_name():
    def _fn():
        return 1

    register_kernel("_test_split_names", _fn, devices="backend_name")
    with patch(
        "gpatch_v4.kernel.registry._current_device",
        return_value="backend_name",
    ), patch(
        "gpatch_v4.kernel.registry.get_cur_kernel_device_name",
        return_value="impl_name",
    ):
        assert get_kernel("_test_split_names") is _fn


def test_cuda_linear_cross_entropy_registers():
    pytest.importorskip("gpatch_v4.kernel.cuda")
    assert is_kernel_registered("linear_cross_entropy", "cuda")
    assert is_kernel_registered("set_linear_ce_backend", "cuda")

    from gpatch_v4.kernel.triton.linear_cross_entropy import (
        linear_cross_entropy as cuda_fn,
        set_linear_ce_backend as cuda_set,
    )

    assert get_kernel("linear_cross_entropy", device="cuda") is cuda_fn
    assert get_kernel("set_linear_ce_backend", device="cuda") is cuda_set


def test_public_api_dispatches_to_cuda_when_device_is_cuda():
    pytest.importorskip("gpatch_v4.kernel.cuda")
    from gpatch_v4.kernel.registry import _current_device
    from gpatch_v4.kernel.triton.linear_cross_entropy import (
        linear_cross_entropy as cuda_fn,
    )

    if _current_device() != "cuda":
        pytest.skip("current device backend is not cuda")
    assert get_kernel("linear_cross_entropy") is cuda_fn


def test_set_linear_ce_backend_noop_when_unregistered():
    from gpatch_v4.kernel import set_linear_ce_backend

    with patch(
        "gpatch_v4.kernel.get_kernel",
        return_value=None,
    ):
        set_linear_ce_backend("split_n")
