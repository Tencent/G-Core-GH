# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Aggregate MoE routing-dump jsonl files and compute an EP-agnostic permutation.

Usage (run from ``gcore-dev/``)::

    python -m tools.moe_offline_repermute.build_routing_map \\
        --source-dir /mnt/ceph-hz1-csp/.../moe_dist_64K \\
        --output      /work/wepsdl/.../routing_map.json \\
        --imbalance-curve /work/wepsdl/.../imbalance_curve.png

Outputs ``routing_map.json`` with the position-major permutation, the inverse
permutation, the input weight matrix, and a multi-EP imbalance comparison
(identity vs recursive-bisect). Also saves a summary PNG comparing per-layer
``max / min`` segment-load ratios across EP sizes.

Convention reminder:
    ``perm[L, p] = old_expert_id`` -- gather along dim 0 with this array.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Any, Dict

import numpy as np

from .aggregate import aggregate_jsonl_to_weights
from .perm import (
    assert_valid_perm,
    compute_perm,
    imbalance_ratio,
    inverse_perm,
    multi_ep_load,
)

_EP_SIZES = (2, 4, 8, 16, 32, 64, 128)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source-dir", required=True, help="dir with step_1_pp*.jsonl")
    p.add_argument("--output", required=True, help="output routing_map.json path")
    p.add_argument(
        "--imbalance-curve",
        default=None,
        help="optional PNG path summarising imbalance vs EP size",
    )
    p.add_argument("--num-layers", type=int, default=40)
    p.add_argument("--num-experts", type=int, default=256)
    return p.parse_args()


def build_routing_map(
    *,
    source_dir: str,
    output: str,
    imbalance_curve: str | None = None,
    num_layers: int = 40,
    num_experts: int = 256,
) -> Dict[str, Any]:
    """Aggregate routing stats, compute permutation, and write routing map."""
    print(f"[build_routing_map] reading jsonl from {source_dir}")
    weights, records, file_sha = aggregate_jsonl_to_weights(
        source_dir=source_dir,
        num_layers=num_layers,
        num_experts=num_experts,
        keep_recompute_factor=1,
    )
    print(
        f"[build_routing_map] weights shape={weights.shape} "
        f"records_per_layer min={min(records.values())} "
        f"max={max(records.values())}"
    )

    print("[build_routing_map] computing recursive-bisection permutation ...")
    perm = compute_perm(weights)
    assert_valid_perm(perm, num_experts)
    inv = inverse_perm(perm)
    assert_valid_perm(inv, num_experts)

    identity = np.broadcast_to(np.arange(num_experts, dtype=np.int64), weights.shape).copy()

    multi_ep_load_before: Dict[str, list] = {}
    multi_ep_load_after: Dict[str, list] = {}
    multi_ep_imbalance: Dict[str, Dict[str, float]] = {}
    for ep in _EP_SIZES:
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
                "perm[L, p] = old_expert_id; new_tensor = old_tensor[perm[L]] "
                "along dim 0; inv_perm[L, q] = new_position_of_old_expert_q."
            ),
        "algorithm": "recursive_balanced_bisection",
        "num_layers": num_layers,
        "num_experts": num_experts,
        "source_dir": os.path.realpath(source_dir),
        "source_sha1": file_sha,
        "records_per_layer": {
            str(k): v
            for k, v in sorted(records.items())
        },
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "perm": perm.tolist(),
        "inv_perm": inv.tolist(),
        "weights": weights.tolist(),
        "multi_ep_load_before": multi_ep_load_before,
        "multi_ep_load_after": multi_ep_load_after,
        "multi_ep_imbalance": multi_ep_imbalance,
    }

    out_path = output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"[build_routing_map] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    if imbalance_curve:
        save_imbalance_curve(multi_ep_imbalance, imbalance_curve)
        print(f"[build_routing_map] wrote {imbalance_curve}")

    return payload


def main() -> None:
    args = parse_args()
    build_routing_map(
        source_dir=args.source_dir,
        output=args.output,
        imbalance_curve=args.imbalance_curve,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
    )


def save_imbalance_curve(multi_ep_imbalance: Dict[str, Dict[str, float]], out_png: str) -> None:
    """Plot mean per-layer max/min vs EP size for before / after the permute."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = sorted(multi_ep_imbalance.keys(), key=int)
    eps_int = [int(e) for e in eps]
    before = [multi_ep_imbalance[e]["before_mean_max_over_min"] for e in eps]
    after = [multi_ep_imbalance[e]["after_mean_max_over_min"] for e in eps]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eps_int, before, marker="o", label="identity (before)", color="#cc5050")
    ax.plot(eps_int, after, marker="s", label="recursive-bisect (after)", color="#3070c0")
    ax.set_xscale("log", base=2)
    ax.set_xticks(eps_int)
    ax.set_xticklabels([str(e) for e in eps_int])
    ax.set_xlabel("EP size (contiguous segments)")
    ax.set_ylabel("mean per-layer max / min  (lower is better)")
    ax.set_title("Per-layer EP load imbalance: before vs after permute")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
