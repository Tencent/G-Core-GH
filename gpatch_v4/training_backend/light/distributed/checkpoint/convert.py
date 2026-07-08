"""DCP → plain PyTorch / safetensors checkpoint conversion.

Single-process, no ``dist.init_process_group`` required.  Useful for:

* Loading a trained WeGen model for inference without a distributed setup.
* Changing parallelism topology (different EP / FSDP sizes).
* Exporting to HuggingFace / safetensors format.

DCP key layout
--------------
WeGen saves via::

    dcp.save({"model_state": ModelState(model)}, checkpoint_id=model_dir)

``ModelState.state_dict()`` returns ``{"model": {fqn: tensor}}``, so the DCP
metadata keys have the form ``model_state.model.{fqn}``.  The default
``key_prefix="model_state.model."`` strips this wrapper so the output dict
keys match ``model.named_parameters()`` FQNs directly.

Entry points
------------
``dcp_to_state_dict``
    Load a DCP checkpoint directory into a ``{fqn: Tensor}`` dict.
``convert_dcp_checkpoint``
    Convenience wrapper: load → optional dtype cast → save as ``.pt`` or
    ``.safetensors``.
"""

from __future__ import annotations

import gc
import logging
import os
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.checkpoint.metadata import Metadata

logger = logging.getLogger(__name__)

# Default DCP key prefix for WeGen model state.
# WeGen saves: dcp.save({"model_state": ModelState(model)}, ...)
# ModelState.state_dict() → {"model": {fqn: tensor}}
# → DCP flat key = "model_state.model.{fqn}"
_DEFAULT_MODEL_PREFIX = "model_state.model."

_DTYPE_BYTES: Dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float64: 8,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _tensor_byte_size(tensor_meta, dtype: Optional[torch.dtype] = None) -> int:
    """Byte size of a tensor from DCP ``ChunkStorageMetadata``."""
    if dtype is None:
        dtype = getattr(tensor_meta.properties, "dtype", torch.float32)
    numel = 1
    for d in tensor_meta.size:
        numel *= d
    return numel * _DTYPE_BYTES.get(dtype, 4)


def _collect_model_keys(
    metadata: Metadata,
    prefix: str,
    save_dtype: Optional[torch.dtype],
) -> List[Dict]:
    """Return sorted list of {dcp_key, fqn, byte_size} for model tensors."""
    infos = []
    for key, meta in metadata.state_dict_metadata.items():
        if not key.startswith(prefix):
            continue
        fqn = key[len(prefix):]
        if not hasattr(meta, "properties"):
            continue  # non-tensor metadata entry
        dtype = save_dtype if save_dtype is not None else getattr(meta.properties, "dtype", None)
        if dtype is None:
            logger.warning("Cannot determine dtype for '%s'; skipping.", key)
            continue
        infos.append(
            {
                "dcp_key": key,
                "fqn": fqn,
                "byte_size": _tensor_byte_size(meta, dtype),
                "orig_dtype": getattr(meta.properties, "dtype", dtype),
                "size": meta.size,
            }
        )
    infos.sort(key=lambda x: x["fqn"])
    return infos


def _build_shards(
    infos: List[Dict],
    shard_size_bytes: Optional[int],
) -> List[List[Dict]]:
    """Partition ``infos`` into memory-bounded shards."""
    if shard_size_bytes is None:
        return [infos]

    shards: List[List[Dict]] = []
    current: List[Dict] = []
    current_bytes = 0
    for info in infos:
        if current and current_bytes + info["byte_size"] > shard_size_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(info)
        current_bytes += info["byte_size"]
    if current:
        shards.append(current)
    return shards


def _load_shard(
    shard: List[Dict],
    checkpoint_dir: str,
    save_dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Load one shard of tensors from DCP, gather shards, return plain tensors."""
    reader = FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()

    # Allocate placeholder tensors (use original dtype for allocation).
    raw: Dict[str, torch.Tensor] = {}
    for info in shard:
        orig_dtype = info["orig_dtype"]
        raw[info["dcp_key"]] = torch.empty(info["size"], dtype=orig_dtype)

    # Load all shards from disk into raw (no distributed collective needed).
    load(
        raw,
        checkpoint_id=checkpoint_dir,
        storage_reader=FileSystemReader(checkpoint_dir),
        no_dist=True,
    )

    result: Dict[str, torch.Tensor] = OrderedDict()
    for info in shard:
        tensor = raw[info["dcp_key"]]

        # EP / FSDP parameters may come back as DTensors; gather to full tensor.
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()

        if save_dtype is not None:
            tensor = tensor.to(dtype=save_dtype)

        result[info["fqn"]] = tensor.cpu().detach().clone()
        del tensor

    del raw
    del metadata
    del reader
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def dcp_to_state_dict(
    checkpoint_dir: Union[str, os.PathLike],
    *,
    key_prefix: str = _DEFAULT_MODEL_PREFIX,
    save_dtype: Optional[torch.dtype] = None,
    shard_size_gb: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Load a WeGen DCP checkpoint into a plain ``{fqn: Tensor}`` state dict.

    Runs entirely on a single process (no ``dist.init_process_group``).
    EP / FSDP-sharded parameters are gathered from all shard files in the
    checkpoint directory and reconstructed into full tensors.

    Args:
        checkpoint_dir: Path to the DCP model checkpoint directory
            (the directory that contains ``*.distcp`` / ``*.bin`` shard files
            and ``metadata.json``).
        key_prefix: DCP flat-key prefix to strip.  The default
            ``"model_state.model."`` matches WeGen's
            ``dcp.save({"model_state": ModelState(model)}, ...)`` layout.
        save_dtype: Cast all tensors to this dtype before returning.
            ``None`` keeps the original checkpoint dtype.
        shard_size_gb: Process tensors in batches of at most this many GiB to
            limit peak memory.  ``None`` loads everything in one pass.

    Returns:
        ``{fqn: Tensor}`` dict whose keys match ``model.named_parameters()``.

    Note:
        Run this function on a single rank only to avoid redundant I/O.
    """
    checkpoint_dir = str(checkpoint_dir)
    reader = FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()

    infos = _collect_model_keys(metadata, key_prefix, save_dtype)
    if not infos:
        available = sorted({k.split(".")[0] for k in metadata.state_dict_metadata})
        raise ValueError(
            f"No model state keys found with prefix {key_prefix!r} in "
            f"{checkpoint_dir!r}.  Top-level DCP keys: {available}.  "
            "Pass a matching key_prefix= to override."
        )

    shard_bytes = int(shard_size_gb * (1024**3)) if shard_size_gb else None
    shards = _build_shards(infos, shard_bytes)

    total_gb = sum(i["byte_size"] for i in infos) / (1024**3)
    logger.info(
        "Converting DCP checkpoint: %d tensors, %.2f GiB total, %d shard(s)",
        len(infos),
        total_gb,
        len(shards),
    )

    state_dict: Dict[str, torch.Tensor] = OrderedDict()
    for idx, shard in enumerate(shards):
        logger.info("  Loading shard %d / %d …", idx + 1, len(shards))
        state_dict.update(_load_shard(shard, checkpoint_dir, save_dtype))

    return state_dict


def convert_dcp_checkpoint(
    checkpoint_dir: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    *,
    key_prefix: str = _DEFAULT_MODEL_PREFIX,
    save_dtype: Optional[torch.dtype] = None,
    shard_size_gb: Optional[float] = None,
    use_safetensors: bool = False,
) -> None:
    """Convert a WeGen DCP checkpoint to a single ``.pt`` or ``.safetensors`` file.

    Args:
        checkpoint_dir: Source DCP checkpoint directory.
        output_path: Destination file path.  Use ``.safetensors`` extension
            (or pass ``use_safetensors=True``) for safetensors format.
        key_prefix: DCP key prefix to strip (see :func:`dcp_to_state_dict`).
        save_dtype: Cast tensors to this dtype.  ``None`` keeps original.
        shard_size_gb: Memory-bounded shard size for loading (see above).
        use_safetensors: Force safetensors output regardless of extension.
    """
    output_path = str(output_path)
    if use_safetensors or output_path.endswith(".safetensors"):
        try:
            from safetensors.torch import save_file as _sf_save
        except ImportError as exc:
            raise ImportError(
                "safetensors is not installed.  Run: pip install safetensors"
            ) from exc
        _save = _sf_save
        fmt = "safetensors"
    else:
        _save = torch.save
        fmt = "pt"

    state_dict = dcp_to_state_dict(
        checkpoint_dir,
        key_prefix=key_prefix,
        save_dtype=save_dtype,
        shard_size_gb=shard_size_gb,
    )

    logger.info("Saving %d tensors to %s (%s format) …", len(state_dict), output_path, fmt)
    _save(state_dict, output_path)
    logger.info("Done: %s", output_path)
