# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com
"""Flat-IPC bucketed weight transfer for the vLLM backend.

Design (inspired by G-Core v3, see ``gpatch/core/models/gpt/
gpt_ppo_sampler_client_v3.py::batch_update_func`` and
``gpatch/training/v3/grpo_sampler.py::update_engine_weight``)
-------------------------------------------------------------------------
Instead of creating one CUDA IPC handle per tensor (which was the 30B-MoE
bottleneck -- thousands of ``cudaIpcOpenMemHandle`` round-trips plus a
giant per-tensor handle dict over Ray/ZMQ), we:

1. Group a bucket's tensors by ``dtype``.
2. Flatten each ``dtype`` sub-bucket into a single contiguous flat GPU
   tensor via :func:`gpatch.core.aligner_helper.flatten_weights`.
3. Produce exactly one :func:`torch.multiprocessing.reductions.reduce_tensor`
   IPC handle per (rank, dtype, bucket), keyed by the trainer rank's
   physical GPU UUID.
4. Ship ``(names, key_size, key_numel, {uuid: handle}, flat_shape,
   flat_dtype_name)`` to the vLLM worker via the existing ``collective_rpc``
   path. The worker rebuilds the flat tensor once, slices it into the
   original weights with :func:`torch.Tensor.narrow`, and calls
   :meth:`model.load_weights`.
5. After the final bucket, a dedicated ``finalize`` RPC runs
   :func:`process_weights_after_loading` exactly once.

This keeps the transport (Ray RPC + vLLM ``collective_rpc`` fan-out)
unchanged -- we only shrink the IPC metadata size and eliminate per-tensor
handle setup, which is the cost that actually dominated the 30B-MoE update.

The :class:`FlatIpcBucketBuilder` below is a small stateful helper used by
the trainer-side mixin (``_update_weights_by_bucketed_ipc_vllm``). It
owns the per-dtype accumulation and the flatten+handle step, emitting
:class:`FlatIpcBucket` objects on :meth:`drain`; the actual ``all_gather``
+ Ray RPC lives in the mixin so it stays colocated with the rest of the
trainer plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.multiprocessing.reductions import reduce_tensor

from gpatch.core.aligner_helper import flatten_weights


@dataclass
class FlatIpcBucket:
    """One dtype-homogeneous flattened bucket ready to ship over RPC.

    Attributes
    ----------
    names : list of str
        Weight names in the order they appear in :attr:`flat`.
    key_size : dict
        ``{name: [d0, d1, ...]}`` original tensor shapes (preserves the
        pickle-friendly dict format used by the G-Core v3 protocol).
    key_numel : dict
        ``{name: int}`` original tensor numel.
    local_ipc_handle : tuple
        ``reduce_tensor(flat)`` for THIS rank. The mixin wraps it into
        ``{gpu_uuid: handle}`` and all-gathers across trainer ranks
        before emitting the RPC.
    flat_shape : list of int
        ``[total_numel]``.
    flat_dtype_name : str
        Short dtype name (e.g. ``"bfloat16"``) — matches what vLLM's
        existing handlers consume.
    flat : torch.Tensor
        MUST be kept alive until the RPC has been acknowledged, otherwise
        the CUDA IPC backing goes away underneath the receiver.
    """

    names: list
    key_size: dict
    key_numel: dict
    local_ipc_handle: tuple
    flat_shape: list
    flat_dtype_name: str
    flat: torch.Tensor


class FlatIpcBucketBuilder:
    """Per-rank builder that accumulates weights and emits :class:`FlatIpcBucket`.

    Weights are added one at a time via :meth:`add`. The builder tracks a
    running byte count across all dtypes; when it exceeds
    ``max_bucket_bytes`` the caller should invoke :meth:`drain` to get
    back one :class:`FlatIpcBucket` per dtype currently buffered. At the
    end of the update the caller drains whatever remains.

    The builder is GPU-affinity agnostic: it lives on whichever rank the
    trainer is running on. The physical GPU UUID of the current CUDA
    device is attached to each :class:`FlatIpcBucket` by the caller (kept
    external so tests don't require a CUDA context to instantiate the
    class).

    Parameters
    ----------
    max_bucket_bytes : int
        Soft cap on total bytes before :meth:`is_full` returns ``True``.
        One :class:`FlatIpcBucket` per dtype is emitted on :meth:`drain`;
        each individual bucket may be smaller than this cap.
    """
    def __init__(self, max_bucket_bytes: int):
        self.max_bucket_bytes = int(max_bucket_bytes)
        # dtype -> {"names": [...], "tensors": [...], "bytes": int}
        self._by_dtype: dict[torch.dtype, dict] = {}
        self._total_bytes: int = 0

    def add(self, name: str, weight: torch.Tensor) -> None:
        buf = self._by_dtype.setdefault(weight.dtype, {"names": [], "tensors": [], "bytes": 0})
        nbytes = weight.element_size() * weight.numel()
        buf["names"].append(name)
        buf["tensors"].append(weight)
        buf["bytes"] += nbytes
        self._total_bytes += nbytes

    def is_full(self) -> bool:
        return self._total_bytes >= self.max_bucket_bytes

    def has_content(self) -> bool:
        return self._total_bytes > 0

    def drain(self) -> list[FlatIpcBucket]:
        """Flush all buffered tensors as one :class:`FlatIpcBucket` per dtype.

        The returned list's ``FlatIpcBucket.flat`` tensors hold GPU
        memory that the receiver will alias via CUDA IPC -- callers
        MUST keep them alive at least until the matching RPC is
        acknowledged. Typical pattern::

            alive: list[FlatIpcBucket] = []
            ...
            buckets = builder.drain()
            alive.extend(buckets)
            # issue RPCs using buckets
            ray.get(obj_refs)
            alive.clear()  # safe to release flat tensors now
        """
        out: list[FlatIpcBucket] = []
        for dtype, buf in self._by_dtype.items():
            if not buf["names"]:
                continue
            flat, key_size, key_numel, _total_numel = flatten_weights(buf["names"], buf["tensors"])
            handle = reduce_tensor(flat)
            out.append(
                FlatIpcBucket(
                    names=list(buf["names"]),
                    key_size=key_size,
                    key_numel=key_numel,
                    local_ipc_handle=handle,
                    flat_shape=list(flat.shape),
                    flat_dtype_name=str(dtype).split(".")[-1],
                    flat=flat,
                )
            )
        self._by_dtype.clear()
        self._total_bytes = 0
        return out


def rebuild_flat_cuda_tensor(ipc_handle: tuple, device_id: int) -> torch.Tensor:
    """Rebuild a CUDA IPC flat tensor on the vLLM-worker side.

    Mirrors the device-id patching logic from
    ``gpatch/training/v3/grpo_sampler.py::update_engine_weight`` so the
    tensor comes back on the worker's local CUDA ordinal even if the
    trainer process used a different ``CUDA_VISIBLE_DEVICES`` mapping.
    """
    func, args = ipc_handle
    list_args = list(args)
    # ``reduce_tensor`` packs device_id at position 6 in the args tuple
    # (see torch.multiprocessing.reductions.reduce_tensor).
    list_args[6] = device_id
    return func(*list_args)


def gcore_gpu_uuid(device: torch.device) -> str:
    """Physical GPU UUID string for ``device``.

    Used on both sides of the flat-IPC protocol: the trainer tags each
    :class:`FlatIpcBucket` with its current rank's GPU UUID, and the
    worker looks up its own UUID to pick the matching handle out of the
    merged ``ipc_handles`` dict.
    """
    return str(torch.cuda.get_device_properties(device.index).uuid)


def open_flat_ipc_bucket(
    update_info: dict,
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
    """Worker-side inverse of :meth:`FlatIpcBucketBuilder.drain`.

    Looks up this GPU's IPC handle from the merged dict, rebuilds the
    flat CUDA tensor with device-id retargeting, validates its declared
    shape/dtype, then slices it into zero-copy ``(name, view)`` pairs
    ready for ``model.load_weights``.

    Parameters
    ----------
    update_info : dict
        Per-bucket payload shipped over ``collective_rpc``. Expected
        keys (produced by :class:`FlatIpcBucketBuilder` + the trainer
        all_gather in ``UpdateWeightIpcMixin._update_weights_by_bucketed_ipc_vllm``)::

            {
                "names":       [<str>, ...],           # weight order in flat
                "key_size":    {name: [d0, d1, ...]},  # original shapes
                "key_numel":   {name: int},            # original numel
                "ipc_handles": {gpu_uuid: reduce_tensor(flat)},
                "flat_shape":  [total_numel],
                "flat_dtype":  "bfloat16",             # short dtype name
            }
    device : torch.device
        Caller's local CUDA device. Used to look up the UUID in
        ``ipc_handles`` and to retarget the rebuilt tensor.

    Returns
    -------
    flat : torch.Tensor
        Caller MUST keep it alive for any downstream
        ``model.load_weights`` call — the views in ``pairs`` are
        ``narrow`` windows into it.
    pairs : list of (str, torch.Tensor)
        ``(name, view)`` pairs in the same order as ``update_info['names']``.
    """
    names = update_info["names"]
    key_size = update_info["key_size"]
    key_numel = update_info["key_numel"]
    ipc_handles = update_info["ipc_handles"]
    flat_shape = update_info["flat_shape"]
    flat_dtype = getattr(torch, update_info["flat_dtype"])

    my_uuid = gcore_gpu_uuid(device)
    if my_uuid not in ipc_handles:
        raise RuntimeError(
            f"open_flat_ipc_bucket: no IPC handle for GPU {my_uuid} "
            f"in {list(ipc_handles.keys())}; trainer and vLLM worker "
            f"are not colocated on the same physical device."
        )

    flat = rebuild_flat_cuda_tensor(ipc_handles[my_uuid], device.index)
    assert flat.dtype == flat_dtype, (
        f"flat dtype mismatch: got {flat.dtype}, expected {flat_dtype}"
    )
    assert list(
        flat.shape
    ) == list(flat_shape
             ), (f"flat shape mismatch: got {tuple(flat.shape)}, expected {tuple(flat_shape)}")

    pairs: list[tuple[str, torch.Tensor]] = []
    offset = 0
    for name in names:
        numel = key_numel[name]
        size = key_size[name]
        pairs.append((name, flat.narrow(0, offset, numel).view(*size)))
        offset += numel
    return flat, pairs
