# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Build a DeepSeek-V4 routing map from a Stage-1 ``counts.pt`` dump.

Reads the ``[num_topk_layers, num_experts]`` global-batch counts saved by the
counts-dump run and the model ``config.json``, reconstructs which backbone /
MTP router each counts row belongs to, computes the EP-agnostic recursive
balanced-bisection permutation (reusing ``tools.moe_offline_repermute.perm``),
and writes ``routing_map.json``.

Row ordering (must match ``freeze_update_router._iter_topk_routers`` /
``model.modules()``): backbone TopK layers ascending (hash-MoE layers skipped),
then — only when the dump run had MTP enabled — MTP depths ``0..D-1``.

Convention (position-major): ``perm[r, p] = old_expert_id`` — new position ``p``
of router row ``r`` holds the expert originally at id ``perm[r, p]``.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import DeepseekV4Config

from ..perm import (
    assert_valid_perm,
    compute_perm,
    imbalance_ratio,
    inverse_perm,
    multi_ep_load,
)
from ..aggregate import _sha1_of_file
from ..build_routing_map import save_imbalance_curve

_EP_SIZES = (2, 4, 8, 16, 32, 64, 128)

RowScope = Tuple[str, int]  # ("backbone", layer_idx) | ("mtp", depth)


def build_row_scope(
    hf_config: DeepseekV4Config,
    enable_mtp: bool,
) -> List[RowScope]:
    """Reconstruct the counts-row -> (scope, index) mapping from ``config``.

    Parameters
    ----------
    config : dict
        Parsed ``config.json``; must contain ``mlp_layer_types`` and (when
        ``enable_mtp``) ``num_nextn_predict_layers``.
    enable_mtp : bool

    Returns
    -------
    list[RowScope]
        One entry per counts row, in row order.
    """
    scope: List[RowScope] = [
        ("backbone", i) for i, t in enumerate(hf_config.mlp_layer_types) if t != "hash_moe"
    ]
    if enable_mtp:
        num_mtp = int(hf_config.num_nextn_predict_layers)
        scope.extend(("mtp", d) for d in range(num_mtp))
    return scope


def build_routing_map(
    *,
    counts_path: str,
    hf_config: DeepseekV4Config,
    output: str,
    imbalance_curve: str,
    enable_mtp: bool = False,
    ep_sizes: Tuple[int, ...] = _EP_SIZES,
) -> Dict[str, Any]:
    """Load counts, compute the permutation, and write ``routing_map.json``."""
    assert os.path.exists(
        counts_path
    ), f"counts_path {counts_path} does not exist, check stage 1 of the repermute pipeline"
    counts_t = torch.load(counts_path, map_location="cpu")
    assert counts_t.ndim == 2, f"counts.pt must be 2-D [R, E], got {tuple(counts_t.shape)}"
    weights = counts_t.to(torch.float64).numpy()
    num_rows, num_experts = weights.shape
    assert (num_experts & (num_experts - 1)) == 0, (
        f"num_experts must be a power of two for recursive bisection, got {num_experts}"
    )

    row_scope = build_row_scope(hf_config, enable_mtp)
    assert len(row_scope) == num_rows, (
        f"counts rows ({num_rows}) != reconstructed TopK routers ({len(row_scope)}); "
        f"enable_mtp={enable_mtp}. A mismatch means the config / enable-mtp flag "
        f"does not match the dump run."
    )

    print(
        f"[build_routing_map] counts={counts_path} rows={num_rows} "
        f"num_experts={num_experts} enable_mtp={enable_mtp}"
    )
    perm = compute_perm(weights)
    assert_valid_perm(perm, num_experts)
    inv = inverse_perm(perm)
    assert_valid_perm(inv, num_experts)

    identity = np.broadcast_to(np.arange(num_experts, dtype=np.int64), weights.shape).copy()
    multi_ep_load_before: Dict[str, list] = {}
    multi_ep_load_after: Dict[str, list] = {}
    multi_ep_imbalance: Dict[str, Dict[str, float]] = {}
    for ep in ep_sizes:
        if ep > num_experts:
            continue
        load_b = multi_ep_load(weights, identity, ep)
        load_a = multi_ep_load(weights, perm, ep)
        ratio_b = imbalance_ratio(load_b)
        ratio_a = imbalance_ratio(load_a)
        multi_ep_load_before[str(ep)] = load_b.tolist()
        multi_ep_load_after[str(ep)] = load_a.tolist()
        multi_ep_imbalance[str(ep)] = {
            "before_mean_max_over_min": float(ratio_b.mean()),
            "before_max_layer_max_over_min": float(ratio_b.max()),
            "after_mean_max_over_min": float(ratio_a.mean()),
            "after_max_layer_max_over_min": float(ratio_a.max()),
            "improvement_factor_mean": float(ratio_b.mean() / ratio_a.mean()),
        }
        print(
            f"  EP={ep:>3d}  before mean(max/min)={ratio_b.mean():.3f} "
            f"max={ratio_b.max():.3f}  after mean={ratio_a.mean():.3f} "
            f"max={ratio_a.max():.3f}  gain={ratio_b.mean() / ratio_a.mean():.2f}x"
        )

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "convention":
            (
                "perm[r, p] = old_expert_id; new_tensor = old_tensor[perm[r]] "
                "along dim 0; inv_perm[r, q] = new_position_of_old_expert_q. "
                "Row r maps to row_scope[r]."
            ),
        "algorithm": "recursive_balanced_bisection",
        "num_layers": num_rows,
        "num_experts": num_experts,
        "enable_mtp": enable_mtp,
        "row_scope": [[kind, idx] for kind, idx in row_scope],
        "source_dir": os.path.dirname(os.path.realpath(counts_path)),
        "counts_path": os.path.realpath(counts_path),
        "counts_sha1": _sha1_of_file(counts_path),
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "perm": perm.tolist(),
        "inv_perm": inv.tolist(),
        "weights": weights.tolist(),
        "multi_ep_load_before": multi_ep_load_before,
        "multi_ep_load_after": multi_ep_load_after,
        "multi_ep_imbalance": multi_ep_imbalance,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"[build_routing_map] wrote {output} ({os.path.getsize(output) / 1e6:.1f} MB)")

    if imbalance_curve:
        save_imbalance_curve(multi_ep_imbalance, imbalance_curve)
        print(f"[build_routing_map] wrote {imbalance_curve}")

    return payload
