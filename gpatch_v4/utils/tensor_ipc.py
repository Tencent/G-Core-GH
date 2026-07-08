"""CUDA IPC tensor serialization utilities.

Provides :class:`FlattenedTensorBucket` (flatten multiple tensors into a
single contiguous buffer) and IPC serialize/deserialize helpers based on
``ForkingPickler``. CUDA tensors are converted to IPC handles (~64 bytes
each) so they can be transferred between processes on the same host
without copying tensor data.

No sglang / vllm dependency — only ``torch`` and stdlib.
"""

import io
import os
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import List, Tuple

import pybase64
import torch


@dataclass
class FlattenedTensorMetadata:
    """Metadata for a tensor in a flattened bucket."""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    start_idx: int
    end_idx: int
    numel: int


class FlattenedTensorBucket:
    """Flatten multiple named tensors into one contiguous ``uint8`` buffer.

    Two construction modes:

    * **From named tensors** — pass ``named_tensors``.
    * **From pre-flattened data** — pass ``flattened_tensor`` + ``metadata``
      (used for reconstruction on the receiving end).
    """

    supports_multi_dtypes = True

    def __init__(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]] = None,
        flattened_tensor: torch.Tensor = None,
        metadata: List[FlattenedTensorMetadata] = None,
    ):
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")
            self.metadata: List[FlattenedTensorMetadata] = [None] * len(named_tensors)
            current_idx = 0
            parts: List[torch.Tensor] = [None] * len(named_tensors)
            for i, (name, tensor) in enumerate(named_tensors):
                flat = tensor.flatten().view(torch.uint8)
                parts[i] = flat
                numel = flat.numel()
                self.metadata[i] = FlattenedTensorMetadata(
                    name=name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    start_idx=current_idx,
                    end_idx=current_idx + numel,
                    numel=numel,
                )
                current_idx += numel
            self.flattened_tensor = torch.cat(parts, dim=0)
        else:
            if flattened_tensor is None or metadata is None:
                raise ValueError(
                    "Must provide either named_tensors or both flattened_tensor and metadata"
                )
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata

    def get_flattened_tensor(self) -> torch.Tensor:
        return self.flattened_tensor

    def get_metadata(self) -> List[FlattenedTensorMetadata]:
        return self.metadata

    def reconstruct_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        """Reconstruct original ``(name, tensor)`` pairs from the flat buffer."""
        result = [None] * len(self.metadata)
        for i, meta in enumerate(self.metadata):
            tensor = (
                self.flattened_tensor[meta.start_idx:meta.end_idx].view(meta.dtype).reshape(
                    meta.shape
                )
            )
            result[i] = (meta.name, tensor)
        return result


def serialize_ipc(obj) -> bytes:
    """Serialize ``obj`` using ``ForkingPickler`` (CUDA tensors → IPC handles).

    Parameters
    ----------
    obj
        Any picklable object. CUDA tensors are converted to IPC handles
        (~64 bytes each) by ``ForkingPickler``.

    Returns
    -------
    bytes
    """
    buf = io.BytesIO()
    ForkingPickler(buf).dump(obj)
    return buf.getvalue()


def serialize_ipc_base64(obj) -> str:
    """Like :func:`serialize_ipc` but returns a base64-encoded string."""
    return pybase64.b64encode(serialize_ipc(obj)).decode("utf-8")


def serialize_pickle_base64(obj) -> str:
    """Serialize with standard ``pickle`` (no fd sharing) + base64.

    Does **not** use ``ForkingPickler``, so CPU tensors are serialized via
    in-memory copy rather than fd-based shared memory. Safe for
    cross-process transfer where authkeys may differ (e.g. Ray actors).
    """
    return pybase64.b64encode(pickle.dumps(obj)).decode("utf-8")


def deserialize_ipc(data):
    """Deserialize a payload produced by :func:`serialize_ipc`.

    Parameters
    ----------
    data : bytes or str
        Raw bytes or base64-encoded string.

    Returns
    -------
    object
        CUDA tensors are recovered from IPC handles.
    """
    if isinstance(data, str):
        data = pybase64.b64decode(data, validate=True)
    return pickle.loads(data)


@contextmanager
def disable_expandable_segments():
    """Temporarily disable ``expandable_segments`` for CUDA IPC compatibility.

    On kernels that lack ``pidfd_getfd``, CUDA IPC handles created from
    tensors allocated with ``expandable_segments:True`` cannot be rebuilt
    in another process.  This context manager switches the allocator to
    ``expandable_segments:False`` for the duration of the block and
    restores the original setting on exit.

    If ``PYTORCH_CUDA_ALLOC_CONF`` does not contain ``expandable_segments:True``,
    this is a no-op.
    """
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    need_disable = "expandable_segments:True" in alloc_conf
    if need_disable:
        torch._C._accelerator_setAllocatorSettings("expandable_segments:False")
    try:
        yield
    finally:
        if need_disable:
            torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
