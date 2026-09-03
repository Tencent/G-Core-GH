# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Streaming checkpoint I/O for Qwen3.8-Flash-Next.

Bound onto the model by :func:`.hp.apply_hp` as ``load_checkpoint_hp`` /
``save_checkpoint_hp``.

The released checkpoint is 1658 tensors / 360 GB across 131 safetensors shards. Loading it
through ``from_pretrained`` would need the whole thing on one rank, so each rank streams
the shards itself and keeps only what it owns.

Name mapping (verified exhaustively against the real index: 0 unmapped keys in either
direction, and every shape matches with **no** transpose, expert merge or QKV fusion):

===================================================== =========================
disk                                                  model
===================================================== =========================
``model.language_model.X``                            ``model.X``
``lm_head.weight``                                    unchanged
``mtp.*`` (31 tensors)                                skipped — head not trained
``model.visual.*`` (333 tensors)                      skipped — language-only
``...ngram_embedding.shard_{0..127}.weight``          rows of ``...ngram_embedding.weight``
===================================================== =========================

That is why this module does **not** use HF's ``conversion_mapping``: nothing needs
converting, and the ``qwen4_exp_text`` mapping would otherwise pull in a
``Concatenate`` over all 128 Engram shards, which upstream itself annotates as
"~95 GiB, cannot afford this on device".
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from gpatch_v4.utils import log

from .engram import OwnerShardedNGramEmbedding, materialize_engram_tables

__all__ = ["load_checkpoint_hp", "save_checkpoint_hp"]

_SKIP_PATTERNS = (re.compile(r"^mtp\."), re.compile(r"^model\.visual\."))
_ENGRAM_SHARD_RE = re.compile(r"^(?P<prefix>.*ngram_embedding)\.shard_(?P<index>\d+)\.weight$")
_EXPERT_SUFFIXES = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
_INDEX_FILE = "model.safetensors.index.json"


def _disk_to_model_key(disk_key: str) -> str:
    """``model.language_model.X`` -> ``model.X``; everything else unchanged."""
    return disk_key.replace("model.language_model.", "model.", 1)


def _read_weight_map(hf_path: Path) -> dict:
    """Load the safetensors index, failing loudly if the checkpoint looks wrong.

    ``PretrainedConfig.from_pretrained`` silently returns *library defaults* for a
    directory with no ``config.json`` — a typo'd path would otherwise train a
    40-layer/2048-hidden model instead of raising. Check both files explicitly.
    """
    if not (hf_path / "config.json").is_file():
        raise FileNotFoundError(
            f"{hf_path}/config.json is missing. Note that transformers would silently "
            "fall back to default config values here rather than raising, so this is "
            "checked explicitly."
        )
    index_path = hf_path / _INDEX_FILE
    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} is missing; expected a sharded checkpoint.")
    return json.loads(index_path.read_text())["weight_map"]


def load_checkpoint_hp(
    self: nn.Module,
    hf_path: str,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Stream the HF checkpoint into this rank's shard of an EP + FSDP2 model.

    Parameters
    ----------
    hf_path : str
        Directory holding ``config.json`` and ``model.safetensors.index.json``.
    dtype : torch.dtype, default ``torch.float32``
        Target dtype for floating-point parameters. Disk is BF16; ``apply_hp`` requires
        an fp32 master, hence the default.

    Notes
    -----
    Three routing cases, all driven by the target's own type/shape rather than by name
    heuristics beyond the expert/Engram suffixes:

    * **Engram table** — FSDP-ignored, holding rows no other rank owns. Only overlaps
      with this rank's row range are read, through ``local_weight`` because ``apply_hp``
      wraps the parameter in a global ``DTensor(Shard(0))``.
    * **Expert weights** — read only this rank's ``[ep_rank * n : (ep_rank+1) * n)``
      slice off disk via ``get_slice``, so the full 512-expert tensor (3.35 GB per key
      per layer) is never materialized.
    * **Everything else** — read whole, then ``distribute_tensor`` onto the parameter's
      own mesh/placements. Peak transient is the largest single tensor
      (``embed_tokens``: 248320x2560, 2.5 GB at fp32).
    """
    hf_dir = Path(hf_path).expanduser()
    weight_map = _read_weight_map(hf_dir)
    rank = dist.get_rank() if dist.is_initialized() else 0

    param_device = _load_device()
    materialize_engram_tables(self, param_device, dtype)

    target_state = self.state_dict()
    engram_plan = _plan_engram(self)

    # Group by shard file so each of the 131 files is opened at most once.
    keys_by_shard: dict[str, list[str]] = {}
    skipped = 0
    for disk_key, shard in weight_map.items():
        if any(pattern.match(disk_key) for pattern in _SKIP_PATTERNS):
            skipped += 1
            continue
        keys_by_shard.setdefault(shard, []).append(disk_key)

    loaded: dict[str, torch.Tensor] = {}
    engram_rows_filled = 0
    for shard, disk_keys in sorted(keys_by_shard.items()):
        shard_path = hf_dir / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"{shard_path} referenced by {_INDEX_FILE} is missing")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for disk_key in disk_keys:
                model_key = _disk_to_model_key(disk_key)
                engram_match = _ENGRAM_SHARD_RE.match(model_key)
                if engram_match is not None:
                    if engram_match.group("prefix") not in engram_plan["tables"]:
                        # A debug-truncated model can drop the only Engram-owning layer.
                        skipped += 1
                        continue
                    engram_rows_filled += _load_engram_shard(
                        handle, disk_key, engram_match, engram_plan, dtype
                    )
                    continue
                if model_key not in target_state:
                    # debug_truncate_num_hidden_layers leaves extra disk layers.
                    skipped += 1
                    continue
                loaded[model_key] = _load_regular_tensor(
                    handle,
                    disk_key,
                    model_key,
                    target_state[model_key],
                    self,
                    dtype,
                    param_device,
                )

    missing = set(target_state) - set(loaded) - engram_plan["target_keys"]
    if missing:
        raise RuntimeError(
            f"{len(missing)} parameters were not present in the checkpoint, e.g. "
            f"{sorted(missing)[:5]}"
        )
    if engram_plan["target_keys"] and engram_rows_filled != engram_plan["total_local_rows"]:
        raise RuntimeError(
            f"Engram load filled {engram_rows_filled} rows but this rank owns "
            f"{engram_plan['total_local_rows']}"
        )

    self.load_state_dict(loaded, strict=False, assign=True)
    _materialize_meta_buffers(self)
    self._engram_param_ids = {
        id(module.weight)
        for module in self.modules() if isinstance(module, OwnerShardedNGramEmbedding)
    }
    log(
        f"[rank {rank}] loaded {len(loaded)} tensors + "
        f"{engram_rows_filled} Engram rows from {hf_dir} "
        f"(skipped {skipped} tensors not in the model)",
        rank=0,
    )


def _load_device() -> torch.device:
    # fully_shard leaves parameters on meta until assign; do not read target.device.
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _load_regular_tensor(
    handle,
    disk_key: str,
    model_key: str,
    target: torch.Tensor,
    model: nn.Module,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Read one non-Engram tensor, EP-slicing experts, and match the target's layout."""
    is_expert = model_key.endswith(_EXPERT_SUFFIXES)
    if is_expert:
        ep_size = model._ep_size
        ep_rank = model._ep_rank
        full_experts = handle.get_slice(disk_key).get_shape()[0]
        if full_experts % ep_size != 0:
            raise ValueError(
                f"{disk_key} has {full_experts} experts, not divisible by ep_size={ep_size}"
            )
        num_local = full_experts // ep_size
        start = ep_rank * num_local
        # Slice on disk: the full tensor is 3.35 GB per key per layer.
        value = handle.get_slice(disk_key)[start:start + num_local]
    else:
        value = handle.get_tensor(disk_key)

    if value.dtype.is_floating_point:
        value = value.to(dtype)

    if isinstance(target, DTensor):
        return distribute_tensor(value, target.device_mesh, target.placements)
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f"{model_key}: checkpoint shape {tuple(value.shape)} != model shape "
            f"{tuple(target.shape)}"
        )
    # Hash buffers are not DTensors; safetensors + assign=True would leave them
    # on CPU. target.device is also CPU if they were born there.
    return value.to(device)


def _plan_engram(model: nn.Module) -> dict:
    """Locate the owner-sharded tables and how many rows this rank must fill.

    No alignment constraint between owner ranges and checkpoint shards is needed:
    :func:`_load_engram_shard` intersects the two in **global row space**, so any
    combination works — a rank may read part of one shard, or several whole shards.
    (NVIDIA's implementation does require whole-shard alignment, because it narrows
    write-through views into its own DTensor shard instead of copying overlaps. Don't
    port that restriction; it doesn't apply here.)
    """
    tables = {
        name: module
        for name, module in model.named_modules() if isinstance(module, OwnerShardedNGramEmbedding)
    }
    plan = {"tables": tables, "target_keys": set(), "total_local_rows": 0}
    for name, table in tables.items():
        plan["target_keys"].add(f"{name}.weight")
        plan["total_local_rows"] += table.local_rows
    return plan


def _load_engram_shard(
    handle,
    disk_key: str,
    match: re.Match,
    plan: dict,
    dtype: torch.dtype,
) -> int:
    """Copy the overlap between one checkpoint shard and this rank's owned rows.

    Returns the number of rows written (0 when the shard belongs to another rank).
    """
    module_name = match.group("prefix")
    shard_index = int(match.group("index"))
    table = plan["tables"].get(module_name)
    if table is None:
        raise KeyError(f"{disk_key} has no owner-sharded table at {module_name!r}")

    rows_per_shard = handle.get_slice(disk_key).get_shape()[0]
    global_start = shard_index * rows_per_shard
    global_end = global_start + rows_per_shard
    owned_start = table.global_row_start
    owned_end = owned_start + table.local_rows

    overlap_start = max(global_start, owned_start)
    overlap_end = min(global_end, owned_end)
    if overlap_start >= overlap_end:
        return 0

    value = handle.get_slice(disk_key)[overlap_start - global_start:overlap_end - global_start]
    with torch.no_grad():
        # `local_weight` unwraps the DTensor `apply_hp` puts around the shard.
        table.local_weight[overlap_start - owned_start:overlap_end - owned_start].copy_(
            value.to(dtype)
        )
    return overlap_end - overlap_start


def _materialize_meta_buffers(model: nn.Module) -> None:
    """Restore the non-persistent buffers, which the checkpoint does not contain.

    Only ``Qwen4ExpTextRotaryEmbedding`` has any: ``inv_freq`` / ``original_inv_freq``
    (``persistent=False``). The Engram metadata buffers (``layer_multipliers``,
    ``ngram_heads_vocab_sizes``, ``ngram_heads_offsets``) are persistent and arrive from
    the checkpoint.

    These are recomputed **unconditionally**, not only when still on meta. They are pure
    functions of the config and are absent from the checkpoint, so recomputing is always
    correct — whereas keying off ``is_meta`` silently leaves uninitialized garbage behind
    if anything called ``to_empty()`` on the way here.

    Any *other* meta buffer raises: a silent zero-init would only show up later as a bad
    loss curve.
    """
    device: Optional[torch.device] = None
    for param in model.parameters():
        if not param.is_meta:
            device = param.to_local().device if isinstance(param, DTensor) else param.device
            break
    if device is None:
        raise RuntimeError("_materialize_meta_buffers must run after parameters are loaded")

    for module in model.modules():
        if type(module).__name__ == "Qwen4ExpTextRotaryEmbedding":
            rope_type = module.config.rope_parameters["rope_type"]
            rope_init_fn = (
                type(module).compute_default_rope_parameters
                if rope_type == "default" else ROPE_INIT_FUNCTIONS[rope_type]
            )
            inv_freq, module.attention_scaling = rope_init_fn(module.config)
            inv_freq = inv_freq.to(device)
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        remaining = sorted(
            name for name, buffer in module.named_buffers(recurse=False) if buffer.is_meta
        )
        if remaining:
            raise RuntimeError(
                f"unhandled meta buffers in {type(module).__name__}: {remaining}. "
                "Add a branch here — silent zero-init is a correctness bug."
            )


_SAVE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


class _ShardWriter:
    """Accumulate tensors and flush a safetensors shard once past ``max_bytes``.

    Necessary rather than convenient: this model is ~352 GB in bf16, so neither a single
    output file nor a rank-0 dict holding every tensor is viable. Files are named with a
    placeholder count and renamed once the total is known, matching HF's layout.
    """
    def __init__(self, out_dir: Path, max_bytes: int = 4 * 1024**3) -> None:
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0
        self.shard_index = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        self.pending[key] = tensor
        nbytes = tensor.numel() * tensor.element_size()
        self.pending_bytes += nbytes
        self.total_bytes += nbytes
        if self.pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        self.shard_index += 1
        name = f"model-{self.shard_index:05d}-of-PENDING.safetensors"
        save_file(self.pending, str(self.out_dir / name), metadata={"format": "pt"})
        for key in self.pending:
            self.weight_map[key] = name
        self.pending.clear()
        self.pending_bytes = 0

    def finalize(self) -> None:
        self.flush()
        total = self.shard_index
        renames = {}
        for index in range(1, total + 1):
            old = f"model-{index:05d}-of-PENDING.safetensors"
            new = f"model-{index:05d}-of-{total:05d}.safetensors"
            (self.out_dir / old).rename(self.out_dir / new)
            renames[old] = new
        weight_map = {key: renames[shard] for key, shard in self.weight_map.items()}
        (self.out_dir / _INDEX_FILE).write_text(
            json.dumps(
                {"metadata": {"total_size": self.total_bytes}, "weight_map": weight_map},
                indent=2,
            )
        )


def _gather_expert_for_save(model: nn.Module, param: torch.Tensor) -> torch.Tensor:
    """Reassemble an expert tensor in model expert order on every EP rank."""
    if isinstance(param, DTensor):
        if model._ep_size == 1:
            return param.full_tensor()
        local = param.to_local().contiguous()
        ep_local = DTensor.from_local(
            local,
            device_mesh=model._ep_fsdp_mesh,
            placements=[Shard(0)],
            shape=param.shape,
            stride=param.stride(),
        ).full_tensor()
    else:
        ep_local = param.detach()

    if model._ep_size == 1:
        return ep_local

    ep_chunks = [torch.empty_like(ep_local) for _ in range(model._ep_size)]
    dist.all_gather(ep_chunks, ep_local, group=model._ep_group)
    value = torch.cat(ep_chunks, dim=0)
    del ep_chunks, ep_local
    return value


def save_checkpoint_hp(
    self: nn.Module,
    save_path: str,
    orig_ckpt_dir: Optional[str] = None,
    preserve_mtp: bool = False,
) -> None:
    """Write a ``from_pretrained``-ready bf16 HF directory.

    Rank 0 writes, but **every** rank must call this: gathering DTensors and the Engram
    table is collective. Names are converted back to the disk convention
    (``model.X`` -> ``model.language_model.X``) and the Engram table is re-emitted as
    ``shard_{i}`` keys so the output reloads through :func:`load_checkpoint_hp`.

    Parameters
    ----------
    save_path : str
        Output directory.
    orig_ckpt_dir : str, optional
        Source checkpoint whose config / tokenizer files are copied alongside. Required:
        without them the output cannot be reloaded.
    preserve_mtp : bool, default False
        Accepted for signature compatibility with the DSV4 saver. The MTP head is never
        loaded here, so ``True`` raises rather than silently writing nothing.

    """
    if preserve_mtp:
        raise NotImplementedError(
            "preserve_mtp=True is meaningless for qwen4_exp: the 4 B MTP head is never "
            "loaded, so it cannot be written back."
        )
    if orig_ckpt_dir is None:
        raise ValueError(
            "orig_ckpt_dir is required so config/tokenizer files can be copied; without "
            "them the output directory is not loadable."
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    out_dir = Path(save_path).expanduser()
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    engram_tables = {
        name: module
        for name, module in self.named_modules() if isinstance(module, OwnerShardedNGramEmbedding)
    }
    engram_keys = {f"{name}.weight" for name in engram_tables}
    writer = _ShardWriter(out_dir) if rank == 0 else None

    for name, param in self.state_dict().items():
        if name in engram_keys:
            continue
        if name.endswith(_EXPERT_SUFFIXES):
            # Expert DTensors are only global within the ep_fsdp sub-mesh. Recover that
            # local EP block first, then concatenate EP blocks in model expert order.
            value = _gather_expert_for_save(self, param)
        else:
            # Collective for DTensors, so it runs on every rank; only rank 0 keeps it.
            value = param.full_tensor() if isinstance(param, DTensor) else param
        if rank == 0:
            gathered = value.detach().to("cpu")
            if gathered.dtype.is_floating_point:
                gathered = gathered.to(torch.bfloat16)
            writer.add(_model_to_disk_key(name), gathered)
        del value

    # Owner-sharded and FSDP-ignored, so gather explicitly — one owner shard at a time,
    # because the full table is 102 GB in bf16 and must never be resident.
    for name, table in engram_tables.items():
        for disk_key, shard in _iter_engram_shards(table, f"{name}.weight", rank):
            if rank == 0:
                writer.add(_model_to_disk_key(disk_key), shard)

    if rank == 0:
        writer.finalize()
        source = Path(orig_ckpt_dir).expanduser()
        for filename in _SAVE_FILES:
            candidate = source / filename
            if candidate.is_file():
                shutil.copy2(candidate, out_dir / filename)
        log(
            f"save_checkpoint_hp wrote {len(writer.weight_map)} tensors in "
            f"{writer.shard_index} shards ({writer.total_bytes / 1e9:.1f} GB) to {out_dir}",
            rank=0,
        )

    if dist.is_initialized():
        dist.barrier()


def _model_to_disk_key(model_key: str) -> str:
    """Inverse of :func:`_disk_to_model_key`."""
    if model_key.startswith("model.") and not model_key.startswith("model.language_model."):
        return model_key.replace("model.", "model.language_model.", 1)
    return model_key


def _iter_engram_shards(table: OwnerShardedNGramEmbedding, target_key: str, rank: int):
    """Yield ``(shard_key, cpu_tensor)`` for each owner's slice of the Engram table."""
    prefix = target_key[:-len(".weight")]
    local = table.local_weight.detach().to(torch.bfloat16)
    for owner in range(table.world_size):
        if dist.is_initialized() and table.world_size > 1:
            buffer = local.clone() if owner == rank else torch.empty_like(local)
            dist.broadcast(buffer, src=owner, group=table.process_group)
        else:
            buffer = local
        if rank == 0:
            yield f"{prefix}.shard_{owner}.weight", buffer.to("cpu")
        del buffer
