# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Rewrite a DeepSeek-V4-Flash native checkpoint with per-router expert permutations.

DSV4-Flash stores routed experts *per-expert* and quantized: for each layer/MTP
depth, ``{scope}.ffn.experts.{i}.w{1,2,3}.weight`` (+ ``.scale``) where
``scope`` is ``layers.{L}`` or ``mtp.{d}``. The router is
``{scope}.ffn.gate.weight`` and ``{scope}.ffn.gate.bias``.

Rewrite rules (per routing-map row ``r`` with position-major ``perm[r]``):

* Experts — **relocate the (weight, scale) key group, bytes unchanged**: an old
  key at expert id ``i`` moves to new position ``inv_perm[r][i]`` (i.e. new
  ``experts.{p}`` holds old ``experts.{perm[r][p]}``).
* Router gate — **gather**: ``new = old.index_select(0, perm[r])`` for both
  ``gate.weight`` and ``gate.bias``.
* Everything else (hash layers, ``shared_experts``, attention, norms,
  embed/head, and — when the dump run had no MTP — all ``mtp.*``) is copied
  verbatim.

Output is written to a NEW destination directory; the source is never modified.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..perm import assert_valid_perm
from ..repermute_hf_ckpt import (
    _compute_total_tensor_bytes,
    _copy_aux_files,
    _copy_routing_map,
)
from gpatch_v4.models.deepseek_v4.checkpoint import is_wo_a_bf16_on_disk, _WO_A_SCALE_RE

_EXPERT_RE = re.compile(
    r"^(?P<scope>layers\.\d+|mtp\.\d+)\.ffn\.experts\."
    r"(?P<eid>\d+)\.(?P<wk>w[123])\.(?P<kind>weight|scale)$"
)
_GATE_RE = re.compile(
    r"^(?P<scope>layers\.\d+|mtp\.\d+)"
    r"\.ffn\.gate\.(?P<kind>weight|bias)$"
)


def _scope_to_row(routing_map: Dict) -> Dict[Tuple[str, int], int]:
    """Map on-disk scope (``layers.{L}`` / ``mtp.{d}``) -> permutation row index."""
    out: Dict[Tuple[str, int], int] = {}
    for row, (kind, idx) in enumerate(routing_map["row_scope"]):
        disk_kind = "layers" if kind == "backbone" else "mtp"
        out[(disk_kind, int(idx))] = row
    return out


def _parse_scope(scope: str) -> Tuple[str, int]:
    kind, idx = scope.split(".")
    return kind, int(idx)


def repermute_hf_ckpt(
    *,
    src_dir: str,
    dst_dir: str,
    routing_map_path: str,
    dry_run: bool = False,
    force_refresh_aux: bool = False,
) -> Optional[List[str]]:
    """Rewrite a DSV4-Flash checkpoint according to a DSV4 routing map."""
    with open(routing_map_path) as fp:
        routing_map = json.load(fp)
    perm = np.asarray(routing_map["perm"], dtype=np.int64)
    inv_perm = np.asarray(routing_map["inv_perm"], dtype=np.int64)
    num_rows = int(routing_map["num_layers"])
    num_experts = int(routing_map["num_experts"])
    assert perm.shape == (num_rows, num_experts)
    assert inv_perm.shape == (num_rows, num_experts)
    assert_valid_perm(perm, num_experts)
    assert_valid_perm(inv_perm, num_experts)
    scope_to_row = _scope_to_row(routing_map)
    perm_torch = torch.from_numpy(perm)

    src_dir = os.path.realpath(src_dir)
    dst_dir = os.path.realpath(dst_dir)
    assert src_dir != dst_dir, "src and dst must differ"
    os.makedirs(dst_dir, exist_ok=True)

    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    with open(index_path) as fp:
        index = json.load(fp)
    weight_map: Dict[str, str] = index["weight_map"]

    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for k, shard in weight_map.items():
        shard_to_keys[shard].append(k)

    print(
        f"[repermute_dsv4] src={src_dir}\n[repermute_dsv4] dst={dst_dir}\n"
        f"[repermute_dsv4] {len(weight_map)} tensors, {len(shard_to_keys)} shards, "
        f"permuted rows={len(scope_to_row)} num_experts={num_experts}"
    )

    if dry_run:
        _report_plan(weight_map, scope_to_row, num_rows, num_experts)
        return

    new_weight_map: Dict[str, str] = {}
    # (row, wk, kind) -> set of old expert ids seen, to fail-fast on partial coverage.
    seen_expert_ids: Dict[Tuple[int, str, str], set] = defaultdict(set)
    seen_gate: Dict[str, set] = defaultdict(set)
    n_expert_tensors = n_gate_tensors = 0
    # sgl ckpt has no wo_a.scale tensors
    # and its total_size in model.safetensors.index.json is wrong
    is_sgl_ckpt_fmt, _wo_a_dtype = is_wo_a_bf16_on_disk(src_dir, weight_map)
    phantom_keys: List[str] = []

    for shard, keys in sorted(shard_to_keys.items()):
        src_shard = os.path.join(src_dir, shard)
        dst_shard = os.path.join(dst_dir, shard)
        tmp_shard = dst_shard + ".tmp"
        new_state: Dict[str, torch.Tensor] = {}
        shard_meta: Dict[str, str] = {"format": "pt"}
        n_expert_tensors_shard = n_gate_tensors_shard = 0
        with safe_open(src_shard, framework="pt", device="cpu") as f:
            src_meta = f.metadata() or {}
            shard_meta.update({k: v for k, v in src_meta.items() if k != "format"})
            for k in keys:
                if is_sgl_ckpt_fmt and _WO_A_SCALE_RE.match(k):
                    phantom_keys.append(k)
                    continue
                t = f.get_tensor(k)
                new_key, new_t, tag = _rewrite_one(
                    k, t, scope_to_row, perm_torch, inv_perm, num_experts
                )
                if tag is not None:
                    row, kind_tag = tag
                    if kind_tag == "expert":
                        m = _EXPERT_RE.match(k)
                        seen_expert_ids[(row, m.group("wk"), m.group("kind"))].add(
                            int(m.group("eid"))
                        )
                        n_expert_tensors_shard += 1
                    else:
                        seen_gate[m.group("kind")].add(row)
                        n_gate_tensors_shard += 1
                assert new_key not in new_state, f"duplicate key after rename: {new_key}"
                new_state[new_key] = new_t
                new_weight_map[new_key] = shard
        save_file(new_state, tmp_shard, metadata=shard_meta)
        os.replace(tmp_shard, dst_shard)
        print(
            f"[repermute_dsv4] {shard}: {len(keys)} tensors "
            f"(repermuted {n_expert_tensors_shard} expert tensors, "
            f"{n_gate_tensors_shard} gate tensors)"
        )
        n_expert_tensors += n_expert_tensors_shard
        n_gate_tensors += n_gate_tensors_shard

    if phantom_keys:
        print(
            f"[repermute_dsv4] dropped {len(phantom_keys)} phantom "
            f"wo_a.scale index entries (dequantized ckpt): "
            f"{', '.join(sorted(phantom_keys))}",
            flush=True,
        )

    print(
        f"[repermute_dsv4] total: repermuted {n_expert_tensors} expert tensors, "
        f"{n_gate_tensors} gate tensors, rest copied verbatim"
    )
    validate_num_tensors(n_expert_tensors, n_gate_tensors, num_rows, num_experts)

    _assert_full_coverage(seen_expert_ids, seen_gate, num_rows, num_experts)
    assert len(new_weight_map) == len(weight_map) - len(phantom_keys), (
        f"key count changed: {len(weight_map) - len(phantom_keys)} -> {len(new_weight_map)}"
    )

    _write_index(
        index,
        new_weight_map,
        dst_dir,
        routing_map_path,
        shard_to_keys,
        is_sgl_ckpt_fmt,
        phantom_keys,
    )
    _copy_aux_files(src_dir, dst_dir, force=force_refresh_aux)
    _copy_routing_map(routing_map_path, dst_dir)
    total_disk = sum(os.path.getsize(os.path.join(dst_dir, s)) for s in shard_to_keys)
    print(f"[repermute_dsv4] done. {dst_dir} ({total_disk / (1024 ** 3):.2f} GiB on disk)")
    return phantom_keys


def _rewrite_one(
    key: str,
    tensor: torch.Tensor,
    scope_to_row: Dict[Tuple[str, int], int],
    perm_torch: torch.Tensor,
    inv_perm: np.ndarray,
    num_experts: int,
) -> Tuple[str, torch.Tensor, Tuple[int, str] | None]:
    """Return ``(new_key, new_tensor, tag)`` for one checkpoint tensor.

    ``tag`` is ``(row, "expert")`` / ``(row, "gate")`` when the tensor was
    remapped, else ``None`` (verbatim copy).
    """
    m = _EXPERT_RE.match(key)
    if m is not None:
        scope = _parse_scope(m.group("scope"))
        row = scope_to_row.get(scope)
        if row is None:
            return key, tensor, None
        eid = int(m.group("eid"))
        assert 0 <= eid < num_experts, f"expert id {eid} out of range for {key}"
        new_eid = int(inv_perm[row][eid])
        new_key = (
            f"{m.group('scope')}.ffn.experts.{new_eid}"
            f".{m.group('wk')}.{m.group('kind')}"
        )
        return new_key, tensor, (row, "expert")

    m = _GATE_RE.match(key)
    if m is not None:
        scope = _parse_scope(m.group("scope"))
        row = scope_to_row.get(scope)
        if row is None:
            return key, tensor, None
        assert tensor.shape[0] == num_experts, (
            f"gate {m.group('kind')} dim0={tensor.shape[0]} != num_experts={num_experts} "
            f"for {key}"
        )
        new_t = tensor.index_select(0, perm_torch[row]).contiguous()
        assert new_t.dtype == tensor.dtype, (
            f"index_select dtype drift for {key}: {tensor.dtype}->{new_t.dtype}"
        )
        return key, new_t, (row, "gate")

    return key, tensor, None


def validate_num_tensors(
    n_expert_tensors: int,
    n_gate_tensors: int,
    num_rows: int,
    num_experts: int,
) -> None:
    # w 1/2/3 weight/scale, 6 tensors per expert in total
    _NUM_TENSOR_PER_EXPERT = 6
    assert n_expert_tensors == num_rows * num_experts * _NUM_TENSOR_PER_EXPERT, (
        f"expected {num_rows * num_experts} experts, got {n_expert_tensors}"
    )
    # weight/bias, 2 tensors per gate in total
    _NUM_TENSOR_PER_GATE = 2
    assert n_gate_tensors == num_rows * _NUM_TENSOR_PER_GATE, (
        f"expected {num_rows} gate tensors, got {n_gate_tensors}"
    )

def _assert_full_coverage(
    seen_expert_ids: Dict[Tuple[int, str, str], set],
    seen_gate: Dict[str, set],
    num_rows: int,
    num_experts: int,
) -> None:
    expected_expert_ids = set(range(num_experts))
    expected_gate_rows = set(range(num_rows))
    for (row, wk, kind), ids in seen_expert_ids.items():
        assert ids == expected_expert_ids, (
            f"row {row} {wk}.{kind}: saw {len(ids)}/{num_experts} experts; "
            f"missing {sorted(expected_expert_ids - ids)[:8]}...; "
            f"checkpoint expert set is incomplete for this scope"
        )
    for kind, rows in seen_gate.items():
        assert rows == expected_gate_rows, (
            f"saw {len(rows)}/{num_rows} gate {kind}; "
            f"missing {sorted(expected_gate_rows - rows)[:8]}...; "
            f"checkpoint gate set is incomplete for this scope"
        )


def _report_plan(
    weight_map: Dict[str, str],
    scope_to_row: Dict[Tuple[str, int], int],
    num_rows: int,
    num_experts: int,
) -> None:
    n_expert_tensors = n_gate_tensors = 0
    for k in weight_map:
        m = _EXPERT_RE.match(k)
        if m is not None and _parse_scope(m.group("scope")) in scope_to_row:
            n_expert_tensors += 1
            continue
        m = _GATE_RE.match(k)
        if m is not None and _parse_scope(m.group("scope")) in scope_to_row:
            n_gate_tensors += 1
    print(
        f"[repermute_dsv4] dry-run: repermute {n_expert_tensors} expert tensors, "
        f"{n_gate_tensors} gate tensors, rest copied verbatim"
    )
    validate_num_tensors(n_expert_tensors, n_gate_tensors, num_rows, num_experts)

def _write_index(
    src_index: Dict,
    new_weight_map: Dict[str, str],
    dst_dir: str,
    routing_map_path: str,
    shard_to_keys: Dict[str, List[str]],
    is_sgl_ckpt_fmt: bool,
    phantom_keys: List[str],
) -> None:
    new_index = {
        "metadata": dict(src_index.get("metadata", {})),
        "weight_map": new_weight_map,
    }
    new_index["metadata"]["repermuted"] = "true"
    new_index["metadata"]["routing_map_path"] = os.path.realpath(routing_map_path)
    old_total_size = int(src_index.get("metadata", {}).get("total_size", -1))
    new_total_size = _compute_total_tensor_bytes(dst_dir, shard_to_keys, phantom_keys)
    assert is_sgl_ckpt_fmt or old_total_size == -1 or new_total_size == old_total_size, (
        f"total size changed: {old_total_size} -> {new_total_size}"
    )
    new_index["metadata"]["total_size"] = str(new_total_size)
    index_tmp = os.path.join(dst_dir, "model.safetensors.index.json.tmp")
    index_dst = os.path.join(dst_dir, "model.safetensors.index.json")
    with open(index_tmp, "w") as fp:
        json.dump(new_index, fp, indent=2)
    os.replace(index_tmp, index_dst)
