# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Visualise ``DEBUG router select`` blocks from a repermute verify log.

Consumes ``verify_equivalence``'s full-forward pass log, which contains
``DEBUG router select router_indices=tensor(...)`` dumps -- first ``num_layers``
blocks for the source checkpoint forward, then ``num_layers`` blocks for the
re-permuted checkpoint forward (split on the last ``Loading weights:`` line).

Emits:

* ``top8_table.md``                       -- per-layer hot top-8 (src / dst_new /
  dst->old) and whether the src vs dst->old sets are equal
* ``overlap_rate_per_layer.png``          -- per-layer mean/min/max of
  ``|src_top_k ∩ (dst->old)_top_k| / K`` across tokens
* ``overlap_rate_per_token_heatmap.png``  -- ``[L, T]`` heatmap of the same
  overlap rate
* ``overlap_rate_hist.png``               -- histogram of the ``L*T`` overlap
  rates, bucketed at multiples of ``1/K``
* ``router_report.md``                    -- summary statistics + worst 10
  (layer, token) mismatches + per-layer set-equal diff

Usage (from any cwd; no GPU required)::

    python -m tools.moe_offline_repermute.viz_debug_router \\
        --log           /work/wepsdl/gcore-dev/log/test_pipeline.log \\
        --routing-map   /mnt/.../moe_repermute_work/routing_map.json \\
        --out-dir       /mnt/.../moe_repermute_work/viz_debug_router

Convention reminder
-------------------
``perm[L, new_id] = old_id`` -- the dst checkpoint's slot ``new_id`` holds the
weights that used to live at ``old_id``. So when the router of the dst forward
selects ``new_id`` we recover the "old" expert via
``perm[L, new_id]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np

_BLOCK_RE = re.compile(
    r"DEBUG router select router_indices=tensor\(\[\[.*?\]\], device='cuda:\d+'\)",
    re.DOTALL,
)
_LOADING_WEIGHTS_RE = re.compile(r"Loading weights:")


def parse_router_blocks(log_path: str) -> Tuple[np.ndarray, List[int], int]:
    """Parse ``DEBUG router select`` blocks from ``log_path``.

    Parameters
    ----------
    log_path : str
        Path to a log file produced by ``verify_equivalence``.

    Returns
    -------
    blocks : np.ndarray, shape ``(num_blocks, T, K)``, int64
        Router top-K selections. ``T`` is the number of tokens printed and
        ``K`` the router's ``top_k``; both are inferred from the first block
        and asserted consistent across the rest.
    block_line_nos : list of int
        Line number (1-based) of each block's opening line, used for
        splitting src vs dst by log position.
    anchor_line_no : int
        Line number (1-based) of the LAST ``Loading weights:`` line. This is
        the boundary marker: blocks before it belong to the src forward,
        blocks after it to the dst forward.

    Raises
    ------
    FileNotFoundError
        If ``log_path`` does not exist.
    AssertionError
        If no blocks are found, or block shapes disagree, or no
        ``Loading weights:`` anchor is present.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(log_path)

    with open(log_path) as fp:
        text = fp.read()

    # Anchor: last `Loading weights:` line's line number (1-based).
    anchor_line_no = -1
    for i, line in enumerate(text.splitlines(), start=1):
        if _LOADING_WEIGHTS_RE.search(line):
            anchor_line_no = i
    assert anchor_line_no > 0, (
        "cannot locate 'Loading weights:' boundary in log; cannot split src/dst"
    )

    # Blocks: regex over the full text, plus compute each match's starting
    # line number via the newline count up to the match start.
    matches = list(_BLOCK_RE.finditer(text))
    assert matches, "no 'DEBUG router select' blocks found in log"

    block_line_nos: List[int] = []
    rows_per_block: List[np.ndarray] = []
    T, K = None, None
    for m in matches:
        # 1-based line number of the match's first char.
        line_no = text.count("\n", 0, m.start()) + 1
        block_line_nos.append(line_no)

        ints = [int(s) for s in re.findall(r"-?\d+", m.group(0))]
        # First numeric in the regex match is the `cuda:N` index, which we
        # must strip. The payload is the remaining `ints[:-1]`... actually
        # `device='cuda:0'` yields `0` as the LAST int -- and the regex
        # match starts with `[[` so no stray digits before the tensor data.
        assert ints, f"block at line {line_no} contains no integers"
        payload = ints[:-1]  # drop trailing cuda device index
        if T is None:
            # Infer (T, K) from first block: K = number of ints in the
            # first row (between `[[` and the first `]`). We parse the
            # first row explicitly to get K.
            first_row_match = re.search(r"\[\[\s*([-\d,\s]+?)\]", m.group(0))
            assert first_row_match, (f"cannot extract first row from block at line {line_no}")
            first_row_ints = [int(s) for s in re.findall(r"-?\d+", first_row_match.group(1))]
            K = len(first_row_ints)
            assert K > 0, f"first row of block at line {line_no} is empty"
            assert len(payload) % K == 0, (
                f"block at line {line_no}: payload length {len(payload)} "
                f"not divisible by K={K}"
            )
            T = len(payload) // K
            assert T > 0, f"block at line {line_no} has T=0"
        else:
            assert len(payload) == T * K, (
                f"block at line {line_no}: expected {T*K} ints (T={T} K={K}), "
                f"got {len(payload)}"
            )
        rows_per_block.append(np.asarray(payload, dtype=np.int64).reshape(T, K))

    blocks = np.stack(rows_per_block, axis=0)
    return blocks, block_line_nos, anchor_line_no


def split_src_dst(
    blocks: np.ndarray,
    block_line_nos: List[int],
    anchor_line_no: int,
    num_layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``blocks`` into src (before anchor) and dst (after anchor).

    Parameters
    ----------
    blocks : np.ndarray, shape ``(num_blocks, T, K)``
    block_line_nos : list of int
    anchor_line_no : int
    num_layers : int
        Expected number of layers -- used to sanity check each half.

    Returns
    -------
    src : np.ndarray, shape ``(num_layers, T, K)``
    dst : np.ndarray, shape ``(num_layers, T, K)``
    """
    assert len(block_line_nos) == len(blocks)
    src_mask = np.asarray(block_line_nos) < anchor_line_no
    dst_mask = ~src_mask
    src = blocks[src_mask]
    dst = blocks[dst_mask]
    assert src.shape[0] == num_layers, (
        f"src half has {src.shape[0]} blocks; expected num_layers={num_layers}. "
        f"anchor_line_no={anchor_line_no}, block_line_nos={block_line_nos[:5]}..."
    )
    assert dst.shape[0] == num_layers, (
        f"dst half has {dst.shape[0]} blocks; expected num_layers={num_layers}."
    )
    return src, dst


def dst_to_old(dst_new: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Map dst router's new_id selections back to old (src) expert ids.

    Parameters
    ----------
    dst_new : np.ndarray, shape ``(num_layers, T, K)``, int
    perm : np.ndarray, shape ``(num_layers, num_experts)``, int
        ``perm[L, new_id] = old_id``.

    Returns
    -------
    dst_old : np.ndarray, shape ``(num_layers, T, K)``, int
    """
    assert dst_new.ndim == 3 and perm.ndim == 2
    assert dst_new.shape[0] == perm.shape[0], (
        f"layer mismatch: dst_new has {dst_new.shape[0]} layers, "
        f"perm has {perm.shape[0]}"
    )
    num_experts = perm.shape[1]
    if dst_new.min() < 0 or dst_new.max() >= num_experts:
        raise IndexError(
            f"dst_new ids out of [0, {num_experts}): min={dst_new.min()} "
            f"max={dst_new.max()}"
        )
    # Gather per layer: dst_old[L, t, k] = perm[L, dst_new[L, t, k]].
    L, T, K = dst_new.shape
    out = np.empty_like(dst_new)
    for i in range(L):
        out[i] = perm[i][dst_new[i]]
    return out


def top_k_by_freq(ids_flat: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """Return the top-``k`` most frequent ids with their counts.

    Parameters
    ----------
    ids_flat : np.ndarray, 1-D, int
    k : int

    Returns
    -------
    list of (expert_id, count) tuples, length ``min(k, unique_count)``, sorted
    by count desc then expert_id asc (stable tie-break by id).
    """
    counts = Counter(int(x) for x in ids_flat.tolist())
    # sort by (-count, id) to get the requested tie-break.
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:k]


def overlap_rate(src: np.ndarray, dst_old: np.ndarray) -> np.ndarray:
    """Per-layer-per-token jaccard numerator divided by K.

    Parameters
    ----------
    src, dst_old : np.ndarray, shape ``(num_layers, T, K)``

    Returns
    -------
    np.ndarray, shape ``(num_layers, T)``, float64 in [0, 1]
        For each (layer, token), ``|set(src) ∩ set(dst_old)| / K``. Note K is
        the fixed top-k width, not the set size (so repeated ids inflate the
        denominator -- but top-k outputs don't repeat by construction).
    """
    assert src.shape == dst_old.shape
    L, T, K = src.shape
    out = np.empty((L, T), dtype=np.float64)
    for i in range(L):
        for t in range(T):
            inter = len(set(src[i, t].tolist()) & set(dst_old[i, t].tolist()))
            out[i, t] = inter / K
    return out


def sanity_check_layer0_top1(
    src: np.ndarray,
    dst_old: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """Raise if src vs dst->old layer-0 top-1 agreement is below ``threshold``.

    Validates both the src/dst ordering assumption (that the first half of
    blocks really is the src forward) and the token correspondence
    assumption (that src and dst forwards saw the same prompt).

    Parameters
    ----------
    src, dst_old : np.ndarray, shape ``(num_layers, T, K)``
    threshold : float, default 0.5
        Minimum required fraction of tokens where ``src[0, t, 0]`` equals
        ``dst_old[0, t, 0]``.

    Raises
    ------
    RuntimeError
        If the observed agreement is below ``threshold``.
    """
    assert src.ndim == 3 and dst_old.ndim == 3 and src.shape == dst_old.shape
    equal = (src[0, :, 0] == dst_old[0, :, 0]).mean()
    if equal < threshold:
        raise RuntimeError(
            f"layer-0 top-1 src vs dst->old agreement is {equal:.2%}, below "
            f"threshold {threshold:.0%}. This likely means the src/dst ordering "
            f"in the log is wrong, or the two forwards saw different prompts."
        )


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_top_list(items: List[Tuple[int, int]]) -> str:
    return ", ".join(f"{eid}×{cnt}" for eid, cnt in items)


def _write_top8_table(
    out_path: str,
    src: np.ndarray,
    dst_new: np.ndarray,
    dst_old: np.ndarray,
    k: int,
) -> None:
    L = src.shape[0]
    lines: List[str] = []
    lines.append(
        "| layer | src hot top-{k} (id×cnt) | dst_new hot top-{k} (id×cnt) | "
        "dst→old hot top-{k} (id×cnt) | set equal (src vs dst→old) |".format(k=k)
    )
    lines.append(
        "|------:|:------------------------|:-----------------------------|"
        ":-----------------------------|:-------------------------:|"
    )
    for i in range(L):
        src_top = top_k_by_freq(src[i].ravel(), k)
        dnew_top = top_k_by_freq(dst_new[i].ravel(), k)
        dold_top = top_k_by_freq(dst_old[i].ravel(), k)
        src_ids = {eid for eid, _ in src_top}
        dold_ids = {eid for eid, _ in dold_top}
        equal = src_ids == dold_ids
        mark = "✅" if equal else "❌"
        lines.append(
            f"| {i:>4d} | {_fmt_top_list(src_top)} | {_fmt_top_list(dnew_top)} | "
            f"{_fmt_top_list(dold_top)} | {mark} |"
        )
    with open(out_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")


def _write_overlap_plots(
    out_dir: str,
    overlap: np.ndarray,
    k: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, T = overlap.shape

    # 1) per-layer mean/min/max curve
    fig, ax = plt.subplots(figsize=(12, 4))
    mean = overlap.mean(axis=1)
    mn = overlap.min(axis=1)
    mx = overlap.max(axis=1)
    xs = np.arange(L)
    ax.fill_between(xs, mn, mx, alpha=0.2, label=f"min..max over {T} tokens")
    ax.plot(xs, mean, "-o", label="mean", linewidth=1.5, markersize=3)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"|src ∩ (dst→old)| / K (K={k})")
    ax.set_title("Per-layer router top-k overlap rate (src vs dst→old)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.5, L - 0.5)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlap_rate_per_layer.png"), dpi=120)
    plt.close(fig)

    # 2) [L, T] heatmap
    fig, ax = plt.subplots(figsize=(max(6, T * 0.4), max(6, L * 0.18)))
    im = ax.imshow(overlap, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xlabel("token index")
    ax.set_ylabel("layer")
    ax.set_title(f"Overlap rate heatmap (K={k})")
    fig.colorbar(im, ax=ax, label="overlap")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlap_rate_per_token_heatmap.png"), dpi=120)
    plt.close(fig)

    # 3) histogram with bins at multiples of 1/K
    fig, ax = plt.subplots(figsize=(8, 4))
    edges = np.arange(k + 2) / k - 0.5 / k  # center bins on i/K
    ax.hist(overlap.ravel(), bins=edges, edgecolor="black", alpha=0.85)
    ax.set_xlabel(f"overlap rate (K={k})")
    ax.set_ylabel("count of (layer, token) pairs")
    ax.set_title(f"Overlap-rate distribution over {L}×{T} = {L*T} pairs")
    ax.set_xticks(np.arange(k + 1) / k)
    ax.set_xticklabels([f"{i}/{k}" for i in range(k + 1)])
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlap_rate_hist.png"), dpi=120)
    plt.close(fig)


def _write_router_report(
    out_path: str,
    log_path: str,
    routing_map_path: str,
    rmap: dict,
    src: np.ndarray,
    dst_new: np.ndarray,
    dst_old: np.ndarray,
    overlap: np.ndarray,
    k: int,
) -> None:
    L, T = overlap.shape
    total = L * T
    eq1 = int((overlap == 1.0).sum())
    ge_km1 = int((overlap >= (k - 1) / k).sum())
    min_overlap = float(overlap.min())

    # worst 10 (layer, token) pairs.
    flat_idx = np.argsort(overlap, axis=None)
    worst10: List[str] = []
    for n in range(min(10, total)):
        ii = int(flat_idx[n])
        i, t = divmod(ii, T)
        src_ids = sorted(src[i, t].tolist())
        dold_ids = sorted(dst_old[i, t].tolist())
        only_src = sorted(set(src_ids) - set(dold_ids))
        only_dold = sorted(set(dold_ids) - set(src_ids))
        worst10.append(
            f"- L={i:>2d} t={t:>2d} overlap={overlap[i,t]*k:.0f}/{k} "
            f"src_only={only_src} dst→old_only={only_dold}"
        )

    # per-layer hot top-k set equal
    layer_diffs: List[str] = []
    for i in range(L):
        src_top = {e for e, _ in top_k_by_freq(src[i].ravel(), k)}
        dold_top = {e for e, _ in top_k_by_freq(dst_old[i].ravel(), k)}
        if src_top != dold_top:
            only_src = sorted(src_top - dold_top)
            only_dold = sorted(dold_top - src_top)
            layer_diffs.append(f"- L={i:>2d} src_only={only_src} dst→old_only={only_dold}")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_sha1 = _sha1_file(log_path)

    rmap_source_sha1 = rmap.get("source_sha1", {})

    with open(out_path, "w") as fp:
        fp.write(f"# viz_debug_router report\n\n")
        fp.write(f"Generated at: `{generated_at}`\n\n")
        fp.write(f"- log file: `{log_path}` (sha1 `{log_sha1}`)\n")
        fp.write(f"- routing_map: `{routing_map_path}`\n")
        if rmap_source_sha1:
            fp.write(f"- routing_map `source_sha1`:\n")
            for k2, v in rmap_source_sha1.items():
                fp.write(f"  - `{k2}` `{v}`\n")
        fp.write(
            f"- shapes: num_layers={L}, tokens_per_layer={T}, top_k={k}, "
            f"num_experts={len(rmap['perm'][0])}\n\n"
        )

        fp.write("## Two top-k definitions used here\n\n")
        fp.write(
            "- **`top8_table.md`** uses *flatten-then-frequency*: for each "
            "layer we flatten its `[T, K]` selections into `T*K` slots and "
            "take the `k` most frequent expert ids (tie-break: id asc).\n"
        )
        fp.write(
            "- **overlap rate** uses *per-token set intersection*: for each "
            "(layer, token) we compute `|set(src_top_k) ∩ "
            "set(dst→old_top_k)| / K`.\n\n"
        )

        fp.write("## Global overlap statistics\n\n")
        fp.write(f"- `overlap == 1.0` (perfect match): {eq1} / {total} = {eq1/total:.2%}\n")
        fp.write(
            f"- `overlap >= {k-1}/{k}` (off-by-at-most-one): {ge_km1} / "
            f"{total} = {ge_km1/total:.2%}\n"
        )
        fp.write(f"- min overlap: `{min_overlap*k:.0f}/{k}` = `{min_overlap:.4f}`\n\n")

        fp.write("## Worst 10 (layer, token) pairs\n\n")
        fp.write("\n".join(worst10) if worst10 else "(none)")
        fp.write("\n\n")

        fp.write("## Per-layer hot-top-k set mismatch (src vs dst→old)\n\n")
        if layer_diffs:
            fp.write(f"{len(layer_diffs)} / {L} layers differ:\n\n")
            fp.write("\n".join(layer_diffs))
            fp.write("\n")
        else:
            fp.write(f"All {L} layers have equal hot-top-{k} sets. ✅\n")


def run(
    *,
    log_path: str,
    routing_map_path: str,
    out_dir: str,
) -> None:
    """Parse log + routing_map and write all artifacts into ``out_dir``.

    Parameters
    ----------
    log_path : str
        Path to ``verify_equivalence`` log containing ``DEBUG router select``
        blocks.
    routing_map_path : str
        Path to ``routing_map.json`` produced by ``build_routing_map``.
    out_dir : str
        Directory to write artifacts to. Will be created if absent; files
        inside will be overwritten.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(routing_map_path) as fp:
        rmap = json.load(fp)
    perm = np.asarray(rmap["perm"], dtype=np.int64)
    num_layers, num_experts = perm.shape

    blocks, line_nos, anchor = parse_router_blocks(log_path)
    src, dst_new = split_src_dst(blocks, line_nos, anchor, num_layers)
    K = src.shape[2]
    assert src.shape[1] == dst_new.shape[1], (
        f"src/dst token counts differ: {src.shape[1]} vs {dst_new.shape[1]}"
    )
    assert src.shape[2] == dst_new.shape[2]

    dst_old = dst_to_old(dst_new, perm)

    sanity_check_layer0_top1(src, dst_old)

    overlap = overlap_rate(src, dst_old)

    _write_top8_table(os.path.join(out_dir, "top8_table.md"), src, dst_new, dst_old, k=K)
    _write_overlap_plots(out_dir, overlap, k=K)
    _write_router_report(
        os.path.join(out_dir, "router_report.md"),
        log_path=log_path,
        routing_map_path=routing_map_path,
        rmap=rmap,
        src=src,
        dst_new=dst_new,
        dst_old=dst_old,
        overlap=overlap,
        k=K,
    )

    print(f"[viz_debug_router] wrote {out_dir}/")
    print(f"[viz_debug_router]   top8_table.md")
    print(f"[viz_debug_router]   overlap_rate_per_layer.png")
    print(f"[viz_debug_router]   overlap_rate_per_token_heatmap.png")
    print(f"[viz_debug_router]   overlap_rate_hist.png")
    print(f"[viz_debug_router]   router_report.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--log", required=True, help="verify_equivalence log path")
    p.add_argument("--routing-map", required=True, help="routing_map.json path")
    p.add_argument("--out-dir", required=True, help="output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        log_path=args.log,
        routing_map_path=args.routing_map,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
