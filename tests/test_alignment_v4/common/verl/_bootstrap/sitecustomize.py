"""Auto-imported at interpreter startup (incl. Ray workers).

Two unrelated environment fixes that must be active in *every* process (driver +
Ray actor/rollout workers) before the offending libraries are imported. verl
imports torch / TransformerEngine / sglang internally, so we cannot wrap those
imports the way gcore does -- instead we hook them at import time from here.

1. flashinfer libcudart stub redirect
   tilelang ships ``libcudart_stub.so`` -- a compile-time linker stub that lacks
   runtime symbols like ``cudaDeviceReset``. When sglang imports
   ``flashinfer.comm`` *after* tilelang has been imported, flashinfer's
   ``CudaRTLibrary`` discovers the already-loaded stub via ``/proc/self/maps``
   (its ``find_loaded_library`` matches the ``libcudart`` substring) and binds to
   it, raising ``AttributeError: undefined symbol: cudaDeviceReset``. Mirrors
   gcore ``gpatch_v4/orches/flashinfer_cudart_fix.py``.

2. deterministic compute (gcore ``apply_deterministic_mode`` parity)
   gcore enables the torch-level deterministic flags and unloads FA3 inside
   ``enable_deterministic_mode()`` (which only runs in non-infer training
   processes). verl colocates actor training + sglang rollout in ONE process
   (``ActorRolloutRefWorker`` role ``actor_rollout_ref``), so we cannot scope it
   per-process. We therefore:
     - call ``torch.use_deterministic_algorithms(True, warn_only=True)`` right
       after torch imports: ops with a deterministic impl run deterministically,
       the rest (e.g. the sglang sampler's cumsum, pytorch#89492) fall back with a
       warning instead of raising -- the strictest setting safe under colocation.
     - set cuDNN deterministic / disable benchmark.
     - unload FA3 from TransformerEngine (FA3's deterministic backward is
       unreliable on Hopper) so the ``attention_backend=flash`` override falls
       back to FA2; mirrors gcore ``disable_flash_attn_3``. Gated on
       ``GCORE_VERL_DETERMINISTIC`` so it only activates for the deterministic run.
   The env vars (NCCL_DETERMINISTIC / NCCL_ALGO / FLASH_ATTENTION_DETERMINISTIC /
   NVTE_ALLOW_NONDETERMINISTIC_ALGO / CUBLAS_WORKSPACE_CONFIG) are exported by the
   launch script -- they must be set before the process starts, not here.

Put this directory on PYTHONPATH (the launch script prepends it) so Python's
``site`` machinery auto-imports this module at startup.
"""

import ctypes
import glob
import importlib.abc
import importlib.util
import logging
import os
import sys


def _find_real_cudart():
    for candidate in [
        os.environ.get("CUDART_LIBRARY_PATH"),
        "/usr/local/cuda/lib64/libcudart.so",
        *sorted(glob.glob("/usr/local/cuda-*/targets/x86_64-linux/lib/libcudart.so")),
        *sorted(glob.glob("/usr/local/cuda-*/lib64/libcudart.so")),
    ]:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _install_cudart_stub_redirect():
    real_cudart = _find_real_cudart()
    if real_cudart is None:
        return

    orig_init = ctypes.CDLL.__init__

    # guard against double-patching if sitecustomize is imported twice
    if getattr(orig_init, "_cudart_stub_redirect", False):
        return

    def patched_init(self, name, *args, **kwargs):
        if name and "libcudart_stub" in str(name):
            name = real_cudart
        orig_init(self, name, *args, **kwargs)

    patched_init._cudart_stub_redirect = True
    ctypes.CDLL.__init__ = patched_init


class _PostImportHook(importlib.abc.MetaPathFinder):
    """Run ``callback(module)`` once, right after ``fullname`` is imported.

    Inserted at the front of ``sys.meta_path``; it claims ``find_spec`` only for
    its target module, wraps the loader's ``exec_module`` to fire the callback
    after execution, then becomes inert. Lets us patch torch / TransformerEngine
    the moment they load without forcing an eager import of them at startup.
    """

    def __init__(self, fullname, callback):
        self._fullname = fullname
        self._callback = callback
        self._fired = False

    def find_spec(self, fullname, path=None, target=None):
        if self._fired or fullname != self._fullname:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        orig_exec = spec.loader.exec_module
        callback = self._callback

        def exec_module(module, _orig=orig_exec, _cb=callback):
            _orig(module)
            self._fired = True
            try:
                _cb(module)
            except Exception:  # noqa: BLE001 - never let a hook break the import
                logging.exception("post-import hook for %s failed", self._fullname)

        spec.loader.exec_module = exec_module
        return spec


def _register_post_import_hook(fullname, callback):
    # already imported (unusual at sitecustomize time): fire immediately.
    if fullname in sys.modules:
        callback(sys.modules[fullname])
        return
    sys.meta_path.insert(0, _PostImportHook(fullname, callback))


def _enable_torch_determinism(torch_module):
    # warn_only=True: deterministic where available, fall back (with a warning)
    # for ops lacking a deterministic impl (e.g. sglang sampler cumsum), which is
    # required because training and rollout share this process under colocation.
    torch_module.use_deterministic_algorithms(True, warn_only=True)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    # use_deterministic_algorithms(True) flips fill_uninitialized_memory on, which
    # poison-fills torch.empty() buffers. flashinfer's paged-KV attention
    # (BatchPrefillWithPagedKVCache) reads padding slots of its pre-allocated
    # index buffers during cuda graph capture; the poison ints become page
    # indices -> illegal memory access (surfaces async at the next deepgemm
    # launch). Disabling the poison fill does NOT weaken determinism.
    torch_module.utils.deterministic.fill_uninitialized_memory = False
    logging.info("Deterministic torch enabled (use_deterministic_algorithms warn_only, cuDNN deterministic)")


def _disable_flash_attn_3(_te_utils_module):
    # mirror gcore gpatch_v4/core/parallel_state.py::disable_flash_attn_3
    from transformer_engine.pytorch.attention.dot_product_attention.utils import (
        FlashAttentionUtils,
    )

    was_installed = getattr(FlashAttentionUtils, "v3_is_installed", False)
    FlashAttentionUtils.v3_is_installed = False
    FlashAttentionUtils.set_flash_attention_3_params = staticmethod(lambda: None)
    if was_installed:
        logging.info("Disabled FA3 for deterministic mode, falling back to FA2")


def _install_determinism():
    if os.environ.get("GCORE_VERL_DETERMINISTIC") != "1":
        return
    _register_post_import_hook("torch", _enable_torch_determinism)
    _register_post_import_hook(
        "transformer_engine.pytorch.attention.dot_product_attention.utils",
        _disable_flash_attn_3,
    )


_install_cudart_stub_redirect()
_install_determinism()
