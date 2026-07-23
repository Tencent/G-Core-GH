"""Offline dump helpers for validating FSDP2 GRPO THD token alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch


def _cpu_clone(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = tensor.detach().to(device="cpu", dtype=dtype or tensor.dtype)
    return tensor.clone()


def _assert_logprob_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # The standard PPO dump stores source logprobs as bfloat16.
    if expected.dtype == torch.bfloat16:
        assert torch.equal(actual.to(torch.bfloat16), expected)
    else:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.0, atol=1e-6)


def build_grpo_thd_dump_records(
    batches: List[Dict[str, Any]],
    batch_data: Dict[str, Any],
    curr_log_probs: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Extract one padded packed segment per source sample for offline checks."""
    packed_token_ids = batch_data["packed_token_ids"]
    packed_target = batch_data["target"]
    packed_mask = batch_data["mask"]
    packed_prev = batch_data["prev_log_probs"]
    packed_rollout = batch_data.get("rollout_log_probs")
    cu = batch_data["cu_seqlens"].detach().cpu().tolist()
    cu_padded = batch_data["cu_seqlens_padded"].detach().cpu().tolist()

    assert len(cu) == len(batches) + 1
    assert len(cu_padded) == len(batches) + 1
    assert curr_log_probs.shape == packed_target.shape

    records = []
    for sample_idx in range(len(batches)):
        start = int(cu_padded[sample_idx])
        end = int(cu_padded[sample_idx + 1])
        actual_len = int(cu[sample_idx + 1] - cu[sample_idx])
        assert 0 <= actual_len <= end - start
        sl = slice(start, end)
        record = {
            "packed_segment_index": sample_idx,
            "packed_segment_count": len(batches),
            "packed_offset": start,
            "packed_actual_len": actual_len,
            "packed_padded_len": end - start,
            "packed_token_ids": _cpu_clone(packed_token_ids[0, sl], torch.int64),
            "packed_target": _cpu_clone(packed_target[0, sl], torch.int64),
            "packed_response_mask": _cpu_clone(packed_mask[0, sl], torch.float32),
            "packed_prev_logprobs": _cpu_clone(packed_prev[0, sl], torch.float32),
            "curr_logprobs": _cpu_clone(curr_log_probs[0, sl], torch.float32),
        }
        if packed_rollout is not None:
            record["packed_rollout_logprobs"] = _cpu_clone(packed_rollout[0, sl], torch.float32)
        records.append(record)
    return records


def validate_grpo_thd_dump_sample(sample: Dict[str, Any]) -> Dict[str, float]:
    """Assert exact source-to-packed mapping for one loaded dump record."""
    segment_index = int(sample["packed_segment_index"])
    dumped_segment_count = sample.get("packed_segment_count")
    segment_count = int(dumped_segment_count) if dumped_segment_count is not None else -1
    packed_offset = int(sample["packed_offset"])
    assert segment_index >= 0
    if segment_count >= 0:
        assert segment_index < segment_count
    assert (segment_index == 0) == (packed_offset == 0)

    tokens = torch.as_tensor(sample["tokens"]).reshape(-1).to(torch.int64)
    actual_len = tokens.numel() - 1
    dumped_actual_len = int(sample["packed_actual_len"])
    assert dumped_actual_len == actual_len, (
        f"packed_actual_len={dumped_actual_len} != tokens-1={actual_len}"
    )

    packed_ids = torch.as_tensor(sample["packed_token_ids"]).reshape(-1).to(torch.int64)
    packed_target = torch.as_tensor(sample["packed_target"]).reshape(-1).to(torch.int64)
    packed_mask = torch.as_tensor(sample["packed_response_mask"]).reshape(-1).float()
    packed_prev = torch.as_tensor(sample["packed_prev_logprobs"]).reshape(-1).float()
    curr = torch.as_tensor(sample["curr_logprobs"]).reshape(-1).float()
    padded_len = int(sample["packed_padded_len"])

    assert all(
        x.numel() == padded_len for x in (
            packed_ids,
            packed_target,
            packed_mask,
            packed_prev,
            curr,
        )
    )
    assert torch.equal(packed_ids[:actual_len], tokens[:-1])
    assert torch.equal(packed_target[:actual_len], tokens[1:])

    source_mask = torch.as_tensor(sample["response_mask"]).reshape(-1).float()
    source_prev = torch.as_tensor(sample["pre_logprobs"]).reshape(-1)
    assert source_mask.numel() == actual_len
    assert source_prev.numel() == actual_len
    assert torch.equal(packed_mask[:actual_len], source_mask)
    _assert_logprob_close(packed_prev[:actual_len], source_prev)

    if "packed_rollout_logprobs" in sample:
        packed_rollout = torch.as_tensor(sample["packed_rollout_logprobs"]).reshape(-1).float()
        source_rollout = torch.as_tensor(sample["rollout_logprobs"]).reshape(-1)
        assert source_rollout.numel() in (actual_len, actual_len + 1)
        _assert_logprob_close(packed_rollout[:actual_len], source_rollout[:actual_len])
        assert torch.count_nonzero(packed_rollout[actual_len:]) == 0

    assert torch.count_nonzero(packed_target[actual_len:]) == 0
    assert torch.count_nonzero(packed_mask[actual_len:]) == 0
    assert torch.count_nonzero(packed_prev[actual_len:]) == 0

    active = packed_mask[:actual_len].bool()
    aligned_mae = (
        (curr[:actual_len][active] -
         packed_prev[:actual_len][active]).abs().mean().item() if active.any() else 0.0
    )
    return {
        "segment_index": float(segment_index),
        "segment_count": float(segment_count),
        "packed_offset": float(packed_offset),
        "actual_len": float(actual_len),
        "padded_len": float(padded_len),
        "curr_prev_active_mae": aligned_mae,
    }


def validate_grpo_thd_dump_file(path: str | Path) -> List[Dict[str, float]]:
    """Load a standard PPO metrics dump and validate every sample."""
    samples = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(samples, list) and samples, f"{path} must contain a non-empty list"
    return [validate_grpo_thd_dump_sample(sample) for sample in samples]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="PPO .pt dump files")
    parser.add_argument(
        "--require-multi-segment",
        action="store_true",
        help="fail unless at least one validated micro-batch contains multiple segments",
    )
    args = parser.parse_args()
    all_stats = []
    for path in args.paths:
        stats = validate_grpo_thd_dump_file(path)
        all_stats.extend(stats)
        max_mae = max(item["curr_prev_active_mae"] for item in stats)
        print(f"{path}: validated {len(stats)} samples, max curr-prev MAE={max_mae:.6g}")
    if args.require_multi_segment:
        assert any(item["segment_index"] > 0 for item in all_stats
                  ), ("no multi-segment THD micro-batch found; this dump only covers segments=1")


if __name__ == "__main__":
    _main()
