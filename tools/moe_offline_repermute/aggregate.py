# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Aggregate per-expert token counts from MoE routing-dump jsonl files.

Each line of the source jsonl is expected to have:

    {"step": int, "mbs": int, "layer": int, "tokens_per_expert": [N ints],
     "recompute_factor": int, "pp_rank": int}

The dumped tensor ``tokens_per_expert`` is already weighted by
``recompute_factor`` (rf=N record sums to N times the per-microbatch token
count). We deduplicate by keeping only ``recompute_factor == 1`` records.

Source jsonl uses 1-based layer ids, while HF safetensors use 0-based
(``layers.0`` .. ``layers.{N-1}``). The aggregator emits a numpy array
indexed by HF layer id.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def aggregate_jsonl_to_weights(
    source_dir: str,
    num_layers: int,
    num_experts: int,
    keep_recompute_factor: int = 1,
) -> Tuple[np.ndarray, Dict[int, int], Dict[str, str]]:
    """Read all ``step_1_pp*.jsonl`` files and aggregate token counts.

    Parameters
    ----------
    source_dir : str
        Directory containing ``step_1_pp{0..3}.jsonl`` files.
    num_layers : int
        Total number of MoE layers (e.g., 40).
    num_experts : int
        Number of routed experts per layer (e.g., 256).
    keep_recompute_factor : int, optional
        Only records with this ``recompute_factor`` value are aggregated.

    Returns
    -------
    weights : ``np.ndarray`` shape ``(num_layers, num_experts)``, float64
        Total token counts per expert per HF layer (HF layer id = jsonl
        layer id - 1).
    records_per_layer : dict[int, int]
        Number of effective records aggregated per HF layer id.
    metadata : dict[str, str]
        ``{filename: sha1_hex}`` of every jsonl read, plus aggregate stats.
    """
    files = sorted(
        f for f in os.listdir(source_dir) if f.startswith("step_1_pp") and f.endswith(".jsonl")
    )
    assert files, f"no step_1_pp*.jsonl found under {source_dir}"

    weights = np.zeros((num_layers, num_experts), dtype=np.float64)
    records: Dict[int, int] = defaultdict(int)
    file_sha: Dict[str, str] = {}

    for fname in files:
        path = os.path.join(source_dir, fname)
        sha = _sha1_of_file(path)
        file_sha[fname] = sha
        with open(path) as fp:
            for line in fp:
                rec = json.loads(line)
                if rec["recompute_factor"] != keep_recompute_factor:
                    continue
                hf_layer = rec["layer"] - 1
                assert 0 <= hf_layer < num_layers, (
                    f"layer={rec['layer']} (HF={hf_layer}) out of range "
                    f"[0, {num_layers})"
                )
                tpe = rec["tokens_per_expert"]
                assert len(tpe
                          ) == num_experts, (f"tokens_per_expert len={len(tpe)} != {num_experts}")
                weights[hf_layer] += np.asarray(tpe, dtype=np.float64)
                records[hf_layer] += 1

    assert len(records) == num_layers, (
        f"missing layers: got {sorted(records.keys())} "
        f"expected 0..{num_layers - 1}"
    )

    return weights, dict(records), file_sha


def _sha1_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


__all__ = ["aggregate_jsonl_to_weights"]
