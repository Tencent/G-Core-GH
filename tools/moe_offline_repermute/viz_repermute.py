# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Visualisation + report for the offline routing-map permutation.

For each ``EP_size`` of interest produces:

* ``before_load_ep{K}.png``    -- ``[L, K]`` per-segment load heatmap (identity)
* ``after_load_ep{K}.png``     -- same with the permutation applied
* ``placement_ep{K}.png``      -- new-layout expert ids placed on a single
  fictional node with ``K`` GPUs (using EPLB_visualization's grid).
* ``repermute_report.md``      -- table of mean / max ``max/min`` ratios per
  EP size, plus per-layer worst residual imbalance after the permute.

Usage (run from any cwd; no GPU required)::

    python -m tools.moe_offline_repermute.viz_repermute \\
        --routing-map /work/wepsdl/.../routing_map.json \\
        --out-dir     /home/.../memory/moe_repermute/figs
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

_DEFAULT_EP_SIZES = (4, 8, 16)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--routing-map", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--ep-sizes",
        type=int,
        nargs="+",
        default=list(_DEFAULT_EP_SIZES),
    )
    return p.parse_args()


def visualize_repermute(
    *,
    routing_map_path: str,
    out_dir: str,
    ep_sizes=_DEFAULT_EP_SIZES,
) -> None:
    """Generate heatmaps, placement grids, and markdown report."""
    os.makedirs(out_dir, exist_ok=True)

    with open(routing_map_path) as fp:
        rmap = json.load(fp)

    weights = np.asarray(rmap["weights"], dtype=np.float64)
    perm = np.asarray(rmap["perm"], dtype=np.int64)
    num_layers, num_experts = weights.shape
    assert perm.shape == weights.shape, (
        f"perm shape {perm.shape} != weights shape {weights.shape}; "
        "routing_map.json appears inconsistent."
    )
    for ep in ep_sizes:
        if num_experts % ep != 0:
            raise SystemExit(
                f"--ep-sizes contains {ep} which does not divide num_experts="
                f"{num_experts}; choose a power of 2 dividing {num_experts}."
            )

    for ep in ep_sizes:
        load_b = _ep_load(weights, identity_perm(num_layers, num_experts), ep)
        load_a = _ep_load(weights, perm, ep)
        _save_load_heatmap(
            load_b,
            os.path.join(out_dir, f"before_load_ep{ep}.png"),
            title=f"Per-EP-rank load (identity, EP={ep})"
        )
        _save_load_heatmap(
            load_a,
            os.path.join(out_dir, f"after_load_ep{ep}.png"),
            title=f"Per-EP-rank load (recursive-bisect, EP={ep})"
        )
        _save_placement_grid(
            perm,
            ep,
            os.path.join(out_dir, f"placement_ep{ep}.png"),
        )
        print(f"[viz] EP={ep}: heatmaps + placement saved")

    report_path = os.path.join(out_dir, "repermute_report.md")
    _write_report(rmap, report_path, ep_sizes)
    print(f"[viz] wrote {report_path}")


def main() -> None:
    args = parse_args()
    visualize_repermute(
        routing_map_path=args.routing_map,
        out_dir=args.out_dir,
        ep_sizes=tuple(args.ep_sizes),
    )


def identity_perm(num_layers: int, num_experts: int) -> np.ndarray:
    return np.broadcast_to(np.arange(num_experts, dtype=np.int64), (num_layers, num_experts)).copy()


def _ep_load(weights: np.ndarray, perm: np.ndarray, ep: int) -> np.ndarray:
    L, N = weights.shape
    assert N % ep == 0
    seg = N // ep
    return np.take_along_axis(weights, perm, axis=1).reshape(L, ep, seg).sum(axis=-1)


def _save_load_heatmap(load: np.ndarray, out_png: str, title: str) -> None:
    """Heatmap of ``[num_layers, ep_size]`` aggregate load (viridis)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, K = load.shape
    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), max(5, L * 0.18)))
    im = ax.imshow(load, aspect="auto", cmap="viridis")
    ax.set_xlabel("EP rank (contiguous segment)")
    ax.set_ylabel("layer")
    ax.set_xticks(range(K))
    ax.set_yticks(range(0, L, max(1, L // 20)))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="total tokens")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _save_placement_grid(
    perm: np.ndarray, ep_size: int, out_png: str, max_layers: int = 12
) -> None:
    """Show the permuted expert ids per (layer, GPU). Capped to ``max_layers``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    L, N = perm.shape
    seg = N // ep_size
    sel = list(range(min(L, max_layers)))
    arr = perm[sel].reshape(len(sel), ep_size, seg)

    fig, ax = plt.subplots(figsize=(min(20, ep_size * 1.1), max(6, len(sel) * 0.7)))
    cell_w, cell_h = 1.0, 1.0
    for i, layer_idx in enumerate(sel):
        for k in range(ep_size):
            x = k * cell_w
            y = (len(sel) - 1 - i) * cell_h
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    cell_w * 0.95,
                    cell_h * 0.95,
                    facecolor="#f5f5f5",
                    edgecolor="black",
                )
            )
            ids = arr[i, k]
            label = ", ".join(str(v) for v in ids[:min(8, seg)])
            if seg > 8:
                label += " ..."
            ax.text(
                x + cell_w * 0.5,
                y + cell_h * 0.5,
                label,
                ha="center",
                va="center",
                fontsize=7,
                family="monospace",
            )
            if i == len(sel) - 1:
                ax.text(x + cell_w * 0.5, -0.4, f"GPU {k}", ha="center", fontsize=10)
        ax.text(
            -0.4, (len(sel) - 1 - i) * cell_h + cell_h * 0.5,
            f"L{layer_idx}",
            ha="right",
            va="center",
            fontsize=10
        )

    ax.set_xlim(-2.5, ep_size + 0.5)
    ax.set_ylim(-1.5, len(sel) + 0.5)
    ax.axis("off")
    ax.set_title(
        f"Permuted expert ids by (layer, GPU) -- EP={ep_size}, "
        f"showing first {len(sel)}/{L} layers"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _write_report(rmap: Dict, out_md: str, ep_sizes) -> None:
    lines: List[str] = []
    lines.append("# MoE Offline Repermute Report")
    lines.append("")
    lines.append(f"- algorithm: `{rmap.get('algorithm')}`")
    lines.append(f"- generated_at: `{rmap.get('generated_at')}`")
    lines.append(f"- num_layers: {rmap['num_layers']}, num_experts: {rmap['num_experts']}")
    lines.append(f"- source_dir: `{rmap['source_dir']}`")
    lines.append("")
    lines.append("## Multi-EP imbalance summary")
    lines.append("")
    lines.append(
        "| EP_size | before mean(max/min) | before max | "
        "after mean(max/min) | after max | mean improvement |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    mei = rmap["multi_ep_imbalance"]
    for k in sorted(mei.keys(), key=int):
        v = mei[k]
        lines.append(
            f"| {k} | {v['before_mean_max_over_min']:.3f} | "
            f"{v['before_max_layer_max_over_min']:.3f} | "
            f"{v['after_mean_max_over_min']:.3f} | "
            f"{v['after_max_layer_max_over_min']:.3f} | "
            f"{v['improvement_factor_mean']:.2f}x |"
        )
    lines.append("")

    weights = np.asarray(rmap["weights"], dtype=np.float64)
    perm = np.asarray(rmap["perm"], dtype=np.int64)
    L, N = weights.shape

    for ep in ep_sizes:
        if N % ep != 0:
            continue
        seg = N // ep
        load = (np.take_along_axis(weights, perm, axis=1).reshape(L, ep, seg).sum(axis=-1))
        ratio = load.max(axis=1) / np.where(load.min(axis=1) > 0, load.min(axis=1), 1.0)
        worst_idx = np.argsort(-ratio)[:5]
        lines.append(f"### Worst residual imbalance at EP={ep} (after)")
        lines.append("")
        lines.append("| layer | max/min |")
        lines.append("|---:|---:|")
        for L_id in worst_idx:
            lines.append(f"| {int(L_id)} | {ratio[int(L_id)]:.3f} |")
        lines.append("")

    with open(out_md, "w") as fp:
        fp.write("\n".join(lines))


if __name__ == "__main__":
    main()
