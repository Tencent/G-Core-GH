"""Load DAPO / math examples from GDatasetV4 metadata + jsonl shards."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

_cache: Dict[str, List[Tuple[str, str]]] = {}


def _extract_question_target(obj: Dict[str, Any]) -> Tuple[str, str]:
    gt = obj.get("target", "")
    if gt is None:
        gt = ""
    else:
        gt = str(gt)
    q = obj.get("question", "")
    if isinstance(q, dict):
        problem = q.get("problem", "")
        example_prompt = problem if problem else str(q)
    elif isinstance(q, list) and len(q) > 0:
        if isinstance(q[0], dict) and "content" in q[0]:
            example_prompt = q[0]["content"]
        else:
            example_prompt = str(q)
    else:
        example_prompt = str(q)
    return example_prompt, gt


def load_examples_from_metadata(metadata_file: str) -> List[Tuple[str, str]]:
    metadata_file = os.path.abspath(metadata_file)
    if metadata_file in _cache:
        return _cache[metadata_file]
    base_dir = os.path.dirname(metadata_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    data_files = metadata.get("data_files", [])
    out: List[Tuple[str, str]] = []
    for entry in data_files:
        fp = entry.get("fp") if isinstance(entry, dict) else None
        if not fp:
            continue
        path = fp if os.path.isabs(fp) else os.path.join(base_dir, fp)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as df:
            for line in df:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(_extract_question_target(obj))
    if not out:
        raise ValueError(f"No examples loaded from metadata {metadata_file}")
    _cache[metadata_file] = out
    return out
