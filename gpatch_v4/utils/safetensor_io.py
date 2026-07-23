# coding=utf-8
"""Safetensors I/O helpers (WFS-safe save)."""

from __future__ import annotations

import os
import shutil
import tempfile
from importlib.metadata import version

from safetensors.torch import save_file as _save_file

_SAFETENSORS_VERSION = version("safetensors")


def _copy_file_large_buffer(src: str, dst: str, bufsize: int = 64 * 1024 * 1024) -> None:
    """Sequential copy with a large buffer (better for remote FS than shutil.move)."""
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=bufsize)
        fdst.flush()
        os.fsync(fdst.fileno())


def save_file(tensors, filename, metadata=None):
    """Save safetensors; for /mnt/wfs/ paths, write to /tmp first then copy."""
    if ("/mnt/wfs" not in os.path.abspath(filename) or _SAFETENSORS_VERSION == "0.7.0"):
        _save_file(tensors, filename, metadata=metadata)
        return

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".safetensors", dir="/tmp")
    os.close(fd)
    try:
        _save_file(tensors, tmp_path, metadata=metadata)
        _copy_file_large_buffer(tmp_path, filename)
        os.remove(tmp_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
