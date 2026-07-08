"""Monkey patch for FSDP2 to backport upstream PyTorch PR #164009.

Upstream fix: when ``fully_shard`` is wrapped inside an Activation Checkpointing
(AC) region, the AC re-runs the FSDP module's ``_pre_forward`` during the
backward pass, which would also trigger ``post_forward`` and incorrectly
``reshard()`` / mutate ``post_forward_order``. This patch makes
``FSDPParamGroup.post_forward`` skip ``reshard()`` and ``_record_post_forward``
when it is being executed inside the autograd backward pass.

References:
- https://github.com/pytorch/pytorch/pull/164009
- ``torch/distributed/fsdp/_fully_shard/_fsdp_common.py``
- ``torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py``

The patch is idempotent: multiple calls to :func:`apply_fsdp2_post_forward_patch`
will only install the override once.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from packaging.version import Version as _V
from torch.distributed.fsdp import fully_shard as _torch_fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
from torch.profiler import record_function

from ...core.device import get_device_module

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_wegen_pr164009_patched"


def _is_bw() -> bool:
    """Return True iff we are currently inside an autograd backward pass.

    Mirrors ``is_bw`` introduced by PR #164009 in ``_fsdp_common.py``.
    Falls back to ``False`` if the underlying C++ API is unavailable.
    """
    fn = getattr(torch._C, "_current_graph_task_id", None)
    if fn is None:
        return False
    try:
        return fn() != -1
    except Exception:  # pragma: no cover - defensive
        return False


debug = False


@contextlib.contextmanager
def debug_context():
    global debug
    debug = True
    try:
        yield
    finally:
        debug = False


def apply_fsdp2_post_forward_patch() -> None:
    """Install the ``FSDPParamGroup.post_forward`` backport patch.

    Safe to call multiple times. No-ops if the running PyTorch version already
    contains the upstream fix or if FSDP2 is unavailable.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_common import (
            compiled_autograd_enabled,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
            FSDPParamGroup,
            TrainingState,
        )
    except Exception as exc:  # pragma: no cover - older torch w/o FSDP2
        logger.warning("Skip FSDP2 post_forward patch: %s", exc)
        return

    if getattr(FSDPParamGroup, _PATCH_FLAG, False):
        return

    # If upstream already shipped the fix (function ``is_bw`` exists in
    # ``_fsdp_common``), avoid double-patching.
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_common as _common_mod

        if hasattr(_common_mod, "is_bw"):
            setattr(FSDPParamGroup, _PATCH_FLAG, True)
            return
    except Exception:
        pass

    def patched_post_forward(
        self: FSDPParamGroup,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> Any:
        if not compiled_autograd_enabled():
            logger.debug("%s", self._with_fqn("FSDP::post_forward"))

        with record_function(self._with_fqn("FSDP::post_forward")):
            if not compiled_autograd_enabled():
                # For AC(fully_shard(model)), AC re-runs FSDP's ``_pre_forward``
                # in backward; it should not change ``post_forward_order``
                # nor reshard parameters again.
                if not _is_bw():
                    self.reshard()
                    self._record_post_forward()
            else:
                self.reshard()
                self._record_post_forward()
            self._training_state = TrainingState.IDLE

        # clear post_forward_order and _post_forward_indices if no grad
        state = _get_module_fsdp_state(module)
        is_root = state._state_ctx.iter_forward_root is state
        if is_root and not torch.is_grad_enabled():
            self.comm_ctx.post_forward_order.clear()
            for state in state._state_ctx.all_states:
                if state._fsdp_param_group:
                    state._fsdp_param_group._post_forward_indices.clear()

        return output

    @contextlib.contextmanager
    def _patched_wait_all_gather_streams_on_event():
        """Intercept the point where FSDP2 is about to drop the previous layer's
        ``all_gather_state``.  At that moment the data has already been copied
        out to ``fsdp_param.all_gather_outputs``, so the raw ``all_gather_output``
        buffer is safe to release.
        """

        original_func = FSDPParamGroup._wait_all_gather_streams_on_event

        def wrapper(self: FSDPParamGroup, event: Optional[torch.Event]) -> None:
            original_func(self, event)
            # in wait for unshard, gather in buffer can be released if copyout is done
            if self._training_state == TrainingState.FORWARD:
                storage = None
                if self.comm_ctx.all_gather_state is not None:
                    # storage of the previous layer
                    storage = (
                        self.comm_ctx.all_gather_state.all_gather_result.all_gather_output.
                        untyped_storage()
                    )
                elif self._all_gather_result is not None:
                    # storage of the current layer, with stream properlly syncnized
                    storage = self._all_gather_result.all_gather_output.untyped_storage()
                if storage is not None and storage.size() > 0:
                    global debug
                    if debug and torch.distributed.get_rank() == 0:
                        print("FSDPParamGroup release storage")
                    storage.resize_(0)
                    # get_device_module().synchronize()

        try:
            FSDPParamGroup._wait_all_gather_streams_on_event = wrapper
            yield
        finally:
            FSDPParamGroup._wait_all_gather_streams_on_event = original_func

    origin_wait_for_unshard = FSDPParamGroup.wait_for_unshard

    def patched_wait_for_unshard(self: FSDPParamGroup) -> None:
        with _patched_wait_all_gather_streams_on_event():
            return origin_wait_for_unshard(self)

    FSDPParamGroup.post_forward = patched_post_forward
    FSDPParamGroup.wait_for_unshard = patched_wait_for_unshard

    setattr(FSDPParamGroup, _PATCH_FLAG, True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info("Applied FSDP2 post_forward patch (PR #164009 backport).")
    else:
        logger.info("Applied FSDP2 post_forward patch (PR #164009 backport).")


# ---------------------------------------------------------------------------
# FSDP2 reduce-scatter input ring buffer (monkey patch, torch >= 2.10.0)
# ---------------------------------------------------------------------------
#
# Problem (upstream ``_fsdp_param_group.post_backward``):
#   Before each layer's RS copy-in, FSDP2 waits on the *previous* layer's
#   ``reduce_scatter_state.event`` on the compute stream. That serialises
#   adjacent layers because the caching allocator may reuse the same storage
#   for consecutive ``reduce_scatter_comm.allocate`` calls.
#
# Fix:
#   1. Keep dedicated RS *input* buffers in a depth-N ring per (size, dtype, device).
#   2. Wait only on the slot about to be overwritten (2 layers back for depth=2).
#   3. Skip the global inter-layer ``reduce_scatter_state`` wait in post_backward.
#
# Safety:
#   - RS *output* still uses ``orig_allocate``; grads live in ``param.grad`` refs.
#   - Per-layer ``_post_reduce_event`` chain and root ``finalize_backward`` still
#     run before ``optimizer.step()`` — only RS-input overlap is widened.
#   - Scope: modules wrapped via :func:`enable_rs_ring` + :func:`fully_shard`, or
#     explicitly tagged via :func:`mark_rs_ring_eligible`.
#
# Baseline: ``work/pytorch/torch/distributed/fsdp/_fully_shard/``
#   ``_fsdp_collectives.foreach_reduce`` / ``_fsdp_param_group.post_backward``
#
_RS_RING_PATCH_FLAG = "_light_rs_ring_buffer_patched"
_RS_RING_PATCH_VERSION = "1.5"
_RS_RING_PATCH_MIN_TORCH = _V("2.10.0")  # needs reduce_scatter_comm.allocate API
_RS_RING_ELIGIBLE_ATTR = "_light_rs_ring_eligible"

# Set by :func:`enable_rs_ring` — :func:`fully_shard` auto-marks modules when True.
_rs_ring_mark_on_shard: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_rs_ring_mark_on_shard", default=False
)
# Set in post_backward wrapper — foreach_reduce uses ring buffer when True.
_rs_ring_use_ring_buffer: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_rs_ring_use_ring_buffer", default=False
)


def mark_rs_ring_eligible(module: nn.Module) -> nn.Module:
    """Mark a module so its FSDP group uses the RS ring-buffer patch."""
    setattr(module, _RS_RING_ELIGIBLE_ATTR, True)
    return module


def fully_shard(module: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
    """FSDP2 ``fully_shard``; auto-marks for RS ring buffer inside :func:`enable_rs_ring`."""
    if _rs_ring_mark_on_shard.get():
        mark_rs_ring_eligible(module)
    return _torch_fully_shard(module, *args, **kwargs)


@contextlib.contextmanager
def enable_rs_ring():
    """Context manager: :func:`fully_shard` calls inside auto-apply ring-buffer marking."""
    token = _rs_ring_mark_on_shard.set(True)
    try:
        yield
    finally:
        _rs_ring_mark_on_shard.reset(token)


def _param_group_uses_rs_ring_buffer(param_group: Any) -> bool:
    """True iff the param group's wrapped root module was marked in :func:`enable_rs_ring`."""
    modules = getattr(param_group, "modules", None) or ()
    if not modules:
        return False
    return any(getattr(mod, _RS_RING_ELIGIBLE_ATTR, False) for mod in modules)


class _RingSlot:
    """One ring slot: dedicated buffer + completion event of the last RS that read it."""

    __slots__ = ("buffer", "event")

    def __init__(self, buffer: torch.Tensor) -> None:
        self.buffer = buffer
        self.event: Optional[torch.Event] = None  # set after RS on reduce_scatter_stream


class _ReduceScatterInputPool:
    """Ring of RS input buffers keyed by ``(size, dtype, device)``.

    Same FSDP param groups (e.g. all decoder blocks) share one bucket when their
    RS input shape matches.  Depth 2 lets layer N copy-in while layer N-1's RS
    still reads the other slot.
    """

    _capacity: int = 2
    _rings: dict[tuple, tuple[list[_RingSlot], int]] = {}

    @classmethod
    def configure(cls, capacity: int) -> None:
        cls._capacity = max(2, capacity)

    @classmethod
    def acquire(
        cls,
        size: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        wait_stream: torch.Stream,
    ) -> tuple[torch.Tensor, _RingSlot]:
        key = (size, dtype, device)
        if key not in cls._rings:
            slots = [
                _RingSlot(torch.empty(size, dtype=dtype, device=device))
                for _ in range(cls._capacity)
            ]
            cls._rings[key] = (slots, 0)
        slots, idx = cls._rings[key]
        slot = slots[idx]
        cls._rings[key] = (slots, (idx + 1) % cls._capacity)
        # Per-slot wait replaces the global reduce_scatter_state wait for this buffer.
        if slot.event is not None:
            wait_stream.wait_event(slot.event)
        return slot.buffer, slot


def _patch_foreach_reduce_with_ring_buffer(orig_foreach_reduce):
    """Wrap upstream ``foreach_reduce``.

    ``foreach_reduce`` calls ``reduce_scatter_comm.allocate`` twice:
      1st → RS input  (patched: ring buffer)
      2nd → RS output (unchanged: orig_allocate; held via param.grad)
    """
    @torch.no_grad()
    def _wrapped(*args, **kwargs):
        reduce_scatter_comm = args[4] if len(args) > 4 else kwargs["reduce_scatter_comm"]
        device = args[7] if len(args) > 7 else kwargs["device"]

        from torch.distributed.device_mesh import _get_device_handle

        device_handle = _get_device_handle(
            device.type if isinstance(device, torch.device) else torch.device(device).type
        )
        # copy-in runs on the compute stream; wait here before overwriting a slot.
        wait_stream = device_handle.current_stream()
        slot_holder: list[_RingSlot] = []
        alloc_calls = [0]
        orig_allocate = reduce_scatter_comm.allocate

        def _ring_allocate(size, *, dtype=None, device=None, **kw):
            alloc_calls[0] += 1
            if alloc_calls[0] == 1 and _rs_ring_use_ring_buffer.get():
                dev = device if isinstance(device, torch.device) else torch.device(device)
                size_tuple = tuple(int(s) for s in size)
                buf, slot = _ReduceScatterInputPool.acquire(size_tuple, dtype, dev, wait_stream)
                slot_holder.append(slot)
                return buf
            return orig_allocate(size, dtype=dtype, device=device, **kw)

        reduce_scatter_comm.allocate = _ring_allocate
        try:
            result = orig_foreach_reduce(*args, **kwargs)
        finally:
            reduce_scatter_comm.allocate = orig_allocate
        # result[1] is reduce_scatter_event on reduce_scatter_stream.
        if slot_holder:
            slot_holder[0].event = result[1]
        return result

    return _wrapped


def _patch_post_backward_skip_global_rs_wait(orig_post_backward):
    """Wrap upstream ``post_backward``.

    Marked modules only: clear ``comm_ctx.reduce_scatter_state`` so upstream does
    not ``compute_stream.wait_event(prev_rs_event)`` before the next copy-in.
    """
    def _wrapped(self, *unused: Any):
        use_ring = _param_group_uses_rs_ring_buffer(self)
        token = _rs_ring_use_ring_buffer.set(use_ring)
        try:
            if use_ring:
                self.comm_ctx.reduce_scatter_state = None
            return orig_post_backward(self, *unused)
        finally:
            _rs_ring_use_ring_buffer.reset(token)

    return _wrapped


def apply_fsdp2_reduce_scatter_ring_buffer_patch(ring_size: int = 2) -> None:
    """Monkey-patch FSDP2 reduce-scatter with (size, dtype, device) ring buffers.

    Only FSDP groups wrapped inside :func:`enable_rs_ring` (or tagged via
    :func:`mark_rs_ring_eligible`) use the ring buffer.
    Requires PyTorch >= 2.10.0 (``reduce_scatter_comm.allocate`` API).
    Safe to call multiple times (idempotent).
    """
    if ring_size <= 1:
        return

    torch_ver = _V(torch.__version__.split("+")[0])
    if torch_ver < _RS_RING_PATCH_MIN_TORCH:
        logger.warning(
            "Skip FSDP2 RS ring-buffer patch v%s: torch %s < %s",
            _RS_RING_PATCH_VERSION,
            torch_ver,
            _RS_RING_PATCH_MIN_TORCH,
        )
        return

    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives as coll_mod
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg_mod
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    except Exception as exc:  # pragma: no cover
        logger.warning("Skip FSDP2 RS ring-buffer patch: %s", exc)
        return

    if getattr(coll_mod, _RS_RING_PATCH_FLAG, False):
        _ReduceScatterInputPool.configure(ring_size)
        return

    _ReduceScatterInputPool.configure(ring_size)
    orig_foreach = coll_mod.foreach_reduce
    orig_post_backward = FSDPParamGroup.post_backward
    patched_foreach = _patch_foreach_reduce_with_ring_buffer(orig_foreach)
    patched_post_backward = _patch_post_backward_skip_global_rs_wait(orig_post_backward)

    # Patch both the module export and the import bound in _fsdp_param_group;
    # post_backward calls the latter, not coll_mod.foreach_reduce.
    coll_mod.foreach_reduce = patched_foreach
    pg_mod.foreach_reduce = patched_foreach
    FSDPParamGroup.post_backward = patched_post_backward
    setattr(coll_mod, _RS_RING_PATCH_FLAG, True)
    setattr(coll_mod, "_light_rs_ring_buffer_patch_version", _RS_RING_PATCH_VERSION)

    msg = (
        f"Applied FSDP2 reduce_scatter ring-buffer monkey patch "
        f"v{_RS_RING_PATCH_VERSION} (depth={ring_size}, torch={torch_ver})."
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(msg)
    else:
        logger.info(msg)
