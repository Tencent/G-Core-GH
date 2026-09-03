# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Rewrite a HF safetensors checkpoint with per-layer expert permutations.

Reads ``--src-dir/{shard.safetensors,model.safetensors.index.json,...}`` and
writes a ``--dst-dir/`` with the same shard layout. For each MoE-routed
tensor in scope, the dim-0 slice (expert id) is gathered by ``perm[L]``;
all other tensors are copied as-is. Non-tensor files (config, tokenizer,
generation_config, ...) are hardlinked when possible, otherwise copied.

Convention: ``perm[L, p] = old_expert_id``; ``new_tensor = old_tensor[perm[L]]``
along dim 0. The MTP block (``mtp.layers.*``) is left untouched -- we have
no routing statistic for it.

Usage::

    python -m tools.moe_offline_repermute.repermute_hf_ckpt \\
        --src-dir       /work/wepsdl/.../Qwen3.6-35B-A3B \\
        --dst-dir       /work/wepsdl/.../Qwen3.6-35B-A3B-permuted \\
        --routing-map   /work/wepsdl/.../routing_map.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .perm import assert_valid_perm

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.")
_PERMUTABLE_SUFFIXES = (
    "mlp.gate.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src-dir", required=True)
    p.add_argument("--dst-dir", required=True)
    p.add_argument("--routing-map", required=True)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="don't write tensors, just enumerate keys and report the plan",
    )
    p.add_argument(
        "--force-refresh-aux",
        action="store_true",
        help=(
            "remove and re-link/copy auxiliary files (config, tokenizer, ...) "
            "even if a previous run already populated them. Recommended when "
            "rerunning with a different routing-map."
        ),
    )
    return p.parse_args()


def repermute_hf_ckpt(
    *,
    src_dir: str,
    dst_dir: str,
    routing_map_path: str,
    dry_run: bool = False,
    force_refresh_aux: bool = False,
) -> None:
    """Rewrite a HF checkpoint according to a routing-map permutation."""
    routing_map = _load_routing_map(routing_map_path)
    perm: np.ndarray = np.asarray(routing_map["perm"], dtype=np.int64)
    num_layers = routing_map["num_layers"]
    num_experts = routing_map["num_experts"]
    assert perm.shape == (num_layers, num_experts
                         ), (f"perm shape {perm.shape} != ({num_layers}, {num_experts})")
    assert_valid_perm(perm, num_experts)

    src_dir = os.path.realpath(src_dir)
    dst_dir = os.path.realpath(dst_dir)
    assert src_dir != dst_dir, "src and dst must differ"
    os.makedirs(dst_dir, exist_ok=True)

    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    with open(index_path) as fp:
        index = json.load(fp)
    weight_map: Dict[str, str] = index["weight_map"]

    permutable_keys = _classify_keys(weight_map.keys(), num_layers)
    n_permutable = sum(len(v) for v in permutable_keys.values())
    print(
        f"[repermute] src={src_dir}\n[repermute] dst={dst_dir}\n"
        f"[repermute] {len(weight_map)} total tensors, "
        f"{n_permutable} permutable ({3 * num_layers} expected)"
    )
    assert n_permutable == 3 * num_layers, (
        f"expected {3 * num_layers} permutable tensors (3 per layer x "
        f"{num_layers}), got {n_permutable}"
    )

    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for k, shard in weight_map.items():
        shard_to_keys[shard].append(k)

    if dry_run:
        for L in sorted(permutable_keys):
            print(f"  layer {L}: {permutable_keys[L]}")
        print(f"[repermute] dry-run: would write {len(shard_to_keys)} shards")
        return

    perm_torch = torch.from_numpy(perm)

    permuted_count = 0
    for shard, keys in sorted(shard_to_keys.items()):
        src_shard = os.path.join(src_dir, shard)
        dst_shard = os.path.join(dst_dir, shard)
        tmp_shard = dst_shard + ".tmp"
        new_state: Dict[str, torch.Tensor] = {}
        shard_meta: Dict[str, str] = {"format": "pt"}
        shard_perm_count = 0
        with safe_open(src_shard, framework="pt", device="cpu") as f:
            src_meta = f.metadata() or {}
            shard_meta.update({k: v for k, v in src_meta.items() if k != "format"})
            for k in keys:
                t = f.get_tensor(k)
                if _is_permutable(k):
                    L = _layer_id(k)
                    assert 0 <= L < num_layers
                    assert t.shape[0] == num_experts, (
                        f"expected dim0={num_experts} for {k}, got {tuple(t.shape)}"
                    )
                    idx = perm_torch[L]
                    orig_dtype = t.dtype
                    t = t.index_select(0, idx).contiguous()
                    assert t.dtype == orig_dtype, (
                        f"index_select dtype drift for {k}: {orig_dtype}->{t.dtype}"
                    )
                    shard_perm_count += 1
                new_state[k] = t
        save_file(new_state, tmp_shard, metadata=shard_meta)
        os.replace(tmp_shard, dst_shard)
        permuted_count += shard_perm_count
        print(f"[repermute] {shard}: {len(keys)} tensors "
              f"({shard_perm_count} permuted)")
    assert permuted_count == 3 * num_layers, (
        f"permuted {permuted_count} tensors but expected {3 * num_layers}"
    )

    new_index = {
        "metadata": dict(index.get("metadata", {})),
        "weight_map": dict(weight_map),
    }
    new_index["metadata"]["repermuted"] = "true"
    new_index["metadata"]["routing_map_path"] = os.path.realpath(routing_map_path)
    new_index["metadata"]["total_size"] = str(_compute_total_tensor_bytes(dst_dir, shard_to_keys))
    index_tmp = os.path.join(dst_dir, "model.safetensors.index.json.tmp")
    index_dst = os.path.join(dst_dir, "model.safetensors.index.json")
    with open(index_tmp, "w") as fp:
        json.dump(new_index, fp, indent=2)
    os.replace(index_tmp, index_dst)

    _copy_aux_files(src_dir, dst_dir, force=force_refresh_aux)
    _copy_routing_map(routing_map_path, dst_dir)
    total_disk = sum(os.path.getsize(os.path.join(dst_dir, s)) for s in shard_to_keys)
    print(f"[repermute] done. {dst_dir} ({total_disk / (1024 ** 3):.2f} GiB on disk)")


def main() -> None:
    args = parse_args()
    repermute_hf_ckpt(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        routing_map_path=args.routing_map,
        dry_run=args.dry_run,
        force_refresh_aux=args.force_refresh_aux,
    )


def _is_permutable(key: str) -> bool:
    if not _LAYER_RE.match(key):
        return False
    return any(key.endswith(s) for s in _PERMUTABLE_SUFFIXES)


def _layer_id(key: str) -> int:
    m = _LAYER_RE.match(key)
    assert m is not None, f"key without layer id: {key}"
    return int(m.group(1))


def _compute_total_tensor_bytes(
    dst_dir: str,
    shard_to_keys: Dict[str, List[str]],
    phantom_keys: Optional[List[str]] = None,
) -> int:
    """HF convention: ``total_size`` is the sum of tensor data bytes (not file size)."""
    total = 0
    for shard, keys in shard_to_keys.items():
        with safe_open(os.path.join(dst_dir, shard), framework="pt", device="cpu") as f:
            for k in keys:
                if phantom_keys is not None and k in phantom_keys:
                    continue
                sl = f.get_slice(k)
                shape = sl.get_shape()
                dtype = str(sl.get_dtype()).split("_")[0]
                total += _shape_bytes(shape, dtype)
    return total


_DTYPE_BYTES = {
    "F8": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "BOOL": 1,
}


def _shape_bytes(shape, dtype: str) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    bytes_per = _DTYPE_BYTES.get(dtype.upper())
    assert bytes_per is not None, f"unknown dtype {dtype}"
    return n * bytes_per


def _load_routing_map(path: str) -> Dict:
    with open(path) as fp:
        return json.load(fp)


def _classify_keys(keys, num_layers: int) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = defaultdict(list)
    for k in keys:
        m = _LAYER_RE.match(k)
        if not m:
            continue
        if not any(k.endswith(s) for s in _PERMUTABLE_SUFFIXES):
            continue
        L = int(m.group(1))
        assert 0 <= L < num_layers
        out[L].append(k)
    for L, ks in out.items():
        assert len(ks) == 3, f"layer {L} expected 3 permutable keys, got {ks}"
    return out


def _copy_aux_files(src_dir: str, dst_dir: str, force: bool = False) -> None:
    """Hardlink (or copy) every non-shard, non-index file into dst_dir."""
    skip_suffixes = (".safetensors", ".tmp")
    skip_names = {"model.safetensors.index.json", "routing_map.json"}
    for name in sorted(os.listdir(src_dir)):
        if name in skip_names:
            continue
        if any(name.endswith(s) for s in skip_suffixes):
            continue
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.isdir(s):
            if os.path.exists(d):
                if not force:
                    continue
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            if os.path.exists(d) or os.path.islink(d):
                if not force:
                    continue
                os.remove(d)
            try:
                os.link(s, d)
            except OSError:
                try:
                    shutil.copy2(s, d)
                except shutil.SameFileError:
                    pass


def _copy_routing_map(routing_map_path: str, dst_dir: str) -> None:
    out = os.path.join(dst_dir, "routing_map.json")
    if os.path.realpath(routing_map_path) == os.path.realpath(out):
        return
    shutil.copy2(routing_map_path, out)


if __name__ == "__main__":
    main()
