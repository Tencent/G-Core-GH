# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Equivalence check for a DSV4-Flash checkpoint after expert re-permutation.

Per permuted routing-map row:

**Byte-exact relocation** — the permuted checkpoint's ``experts.{p}.wK.*``
bytes equal the source's ``experts.{perm[r][p]}.wK.*`` bytes (weight AND
scale), and ``gate.{weight,bias}`` equal ``old.index_select(0, perm[r])``.
A sample of untouched tensors is confirmed byte-identical too.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from transformers import DeepseekV4Config

from ..verify_equivalence import load_index


def verify_equivalence_dsv4(
    *,
    src_dir: str,
    dst_dir: str,
    routing_map_path: str,
    hf_config: DeepseekV4Config,
    phantom_keys: List[str],
    seed: int = 0,
) -> None:
    """Verify byte-exact relocation after repermute."""
    with open(routing_map_path) as fp:
        rmap = json.load(fp)
    perm = torch.tensor(rmap["perm"], dtype=torch.long)
    num_experts = int(rmap["num_experts"])
    num_rows = int(rmap["num_layers"])
    row_scope = [(kind, int(idx)) for kind, idx in rmap["row_scope"]]
    assert len(row_scope) == num_rows, "row_scope must have exactly num_rows entries"

    top_k = int(hf_config.num_experts_per_tok)
    scoring_func = hf_config.scoring_func
    routed_scaling_factor = float(hf_config.routed_scaling_factor)
    swiglu_limit = float(hf_config.swiglu_limit)

    src_index = load_index(src_dir)
    dst_index = load_index(dst_dir)

    print(
        f"[verify_dsv4] num_experts={num_experts} num_rows={num_rows} top_k={top_k} "
        f"scoring_func={scoring_func} rsf={routed_scaling_factor} swiglu_limit={swiglu_limit}"
    )
    torch.manual_seed(seed)

    failures: List[str] = []
    for r in range(num_rows):
        kind, idx = row_scope[r]
        scope = f"layers.{idx}" if kind == "backbone" else f"mtp.{idx}"
        try:
            _verify_bytes_one_row(
                scope=scope,
                perm_row=perm[r],
                num_experts=num_experts,
                src_dir=src_dir,
                src_index=src_index,
                dst_dir=dst_dir,
                dst_index=dst_index,
            )
            print(f"[verify_dsv4] row {r:>3d} ({scope}): OK")
        except AssertionError as e:
            print(f"[verify_dsv4] row {r:>3d} ({scope}): FAIL  {e}")
            failures.append(f"row {r} ({scope}): {e}")

    # verify bytes of untouched tensors that must be copied verbatim
    _verify_untouched_sample(
        src_dir=src_dir,
        src_index=src_index,
        dst_dir=dst_dir,
        dst_index=dst_index,
        num_rows=num_rows,
        row_scope=row_scope,
        failures=failures,
        phantom_keys=phantom_keys,
    )

    if failures:
        print("\n".join(["[verify_dsv4] FAILURES:"] + failures))
        raise SystemExit(1)
    print(f"[verify_dsv4] all {num_rows} rows passed.")


# ======================================================================
# byte-exact relocation
# ======================================================================


def _verify_bytes_one_row(
    *,
    scope: str,
    perm_row: torch.Tensor,
    num_experts: int,
    src_dir: str,
    src_index: Dict,
    dst_dir: str,
    dst_index: Dict,
) -> None:
    # verify bytes of experts weights and scales
    for p in range(num_experts):
        old_eid = int(perm_row[p])
        for wk in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                new_t = _load(
                    dst_dir, dst_index, f"{scope}.ffn.experts.{p}.{wk}.{kind}"
                )
                old_t = _load(
                    src_dir, src_index, f"{scope}.ffn.experts.{old_eid}.{wk}.{kind}"
                )
                assert _raw_equal(new_t, old_t), (
                    f"expert byte mismatch {scope}.experts.{p}.{wk}.{kind} "
                    f"(should equal src experts.{old_eid})"
                )

    # verify bytes of gate weights and biases
    for kind in ("weight", "bias"):
        key = f"{scope}.ffn.gate.{kind}"
        assert key in dst_index["weight_map"], f"gate.{kind} not in dst index: {key}"
        assert key in src_index["weight_map"], f"gate.{kind} not in src index: {key}"
        new_t = _load(dst_dir, dst_index, key)
        old_t = _load(src_dir, src_index, key)
        assert _raw_equal(new_t, old_t.index_select(0, perm_row)), (
            f"gate.{kind} mismatch at {scope}"
        )


# ======================================================================
# untouched sample
# ======================================================================


def _verify_untouched_sample(
    *,
    src_dir: str,
    src_index: Dict,
    dst_dir: str,
    dst_index: Dict,
    num_rows: int,
    row_scope: List[Tuple[str, int]],
    failures: List[str],
    phantom_keys: List[str],
    limit: int = 20,
) -> None:
    """Byte-check a sample of tensors that must be copied verbatim."""
    permuted_scopes = set()
    for r in range(num_rows):
        kind, idx = row_scope[r]
        permuted_scopes.add(f"layers.{idx}" if kind == "backbone" else f"mtp.{idx}")

    checked = 0
    for key in src_index["weight_map"]:
        if key in phantom_keys or any(
            f"{s}.ffn.experts." in key or f"{s}.ffn.gate." in key
            for s in permuted_scopes
        ):
            continue
        if key not in dst_index["weight_map"]:
            failures.append(f"untouched key missing in dst: {key}")
            continue
        try:
            if not _raw_equal(_load(src_dir, src_index, key), _load(dst_dir, dst_index, key)):
                failures.append(f"untouched key bytes changed: {key}")
        except AssertionError as e:
            failures.append(f"untouched key load error {key}: {e}")
        checked += 1
        if checked >= limit:
            break
    print(f"[verify_dsv4] untouched sample: checked {checked} tensors byte-identical")


# ======================================================================
# io helpers
# ======================================================================


def _load(base_dir: str, index: Dict, key: str) -> torch.Tensor:
    wm = index["weight_map"]
    assert key in wm, f"key not in index: {key}"
    with safe_open(os.path.join(base_dir, wm[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _raw_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    # 1-byte quantized dtypes (FP8 / packed FP4) may lack a torch.equal kernel,
    # and float NaN != NaN would break a byte-exact check; compare raw bytes instead.
    if a.element_size() == 1:
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    return torch.equal(a, b)
