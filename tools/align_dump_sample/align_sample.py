#!/usr/bin/env python3
"""
对齐 dump metrics sample 中各字段到训练阶段的 padded_seqlen。
统一支持 GRPO 和 SFT。

对齐后各字段长度：
  - tokens:           [S]          (padded_seqlen)
  - mask:             [S-1]        (shifted 预测位置的 loss mask)
  - per_token_entropy:[S-1]
  - curr_logprobs:    [S-1]        (GRPO only)
  - rollout_logprobs: [S-1]        (GRPO only)
  - pre_logprobs:     [S-1]        (GRPO only)
  - advantages:       [S-1]        (GRPO only)
  - ppo_ratio_*:      [S-1]        (GRPO only)
  - topk_logprobs:    [S-1, topk]  (截掉最后无意义位置)
  - topk_token_ids:   [S-1, topk]
  - moe_topk_info:    [S, moe_topk](路由信息对每个 token 本身有意义)

用法:
    python tools/align_dump_sample/align_sample.py \\
        --pt-file <path/to/dumped_metrics.pt> \\
        --sample-idx <int>
"""
import argparse

import torch


def pad_or_truncate_1d(t: torch.Tensor, target_len: int, pad_value=0) -> torch.Tensor:
    if len(t) >= target_len:
        return t[:target_len]
    return torch.nn.functional.pad(t, (0, target_len - len(t)), value=pad_value)


def pad_or_truncate_2d(t: torch.Tensor, target_rows: int, pad_value=0) -> torch.Tensor:
    if t.shape[0] >= target_rows:
        return t[:target_rows]
    pad_rows = target_rows - t.shape[0]
    padding = torch.full((pad_rows, t.shape[1]), pad_value, dtype=t.dtype)
    return torch.cat([t, padding], dim=0)


def _determine_padded_seqlen(sample: dict) -> int:
    """
    从字段推断 padded_seqlen S。

    优先使用无歧义的字段：
      topk first_dim = S, per_token_entropy len = S-1, moe first_dim = S
    mask 作为 fallback（GRPO mask=[S-1], SFT mask=[S]，需要区分）
    """
    for key in ("topk_logprobs", "topk_token_ids", "topk_logits"):
        if key in sample and torch.is_tensor(sample[key]) and sample[key].ndim == 2:
            return sample[key].shape[0]

    if "per_token_entropy" in sample and torch.is_tensor(sample["per_token_entropy"]):
        return len(sample["per_token_entropy"]) + 1

    if "moe_topk_info" in sample and isinstance(sample["moe_topk_info"], dict):
        for layer_dict in sample["moe_topk_info"].values():
            for v in layer_dict.values():
                if torch.is_tensor(v) and v.ndim == 2:
                    return v.shape[0]

    for key in ("curr_logprobs", "ppo_ratio_unclamped"):
        if key in sample and torch.is_tensor(sample[key]):
            return len(sample[key]) + 1

    if "mask" in sample and torch.is_tensor(sample["mask"]):
        is_grpo = "ppo_step" in sample
        mask_len = len(sample["mask"])
        return mask_len + 1 if is_grpo else mask_len

    raise ValueError("Cannot determine padded_seqlen: no reliable field found")


def align_sample(sample: dict) -> dict:
    """
    将 dump sample 中不同长度的字段对齐到训练阶段的 padded_seqlen。
    自动检测 GRPO / SFT，统一对齐策略。
    """
    S = _determine_padded_seqlen(sample)

    shifted_1d_fields = {
        "rollout_logprobs", "pre_logprobs", "advantages",
        "curr_logprobs", "per_token_entropy", "ppo_ratio_unclamped",
    }
    bool_shifted_fields = {"mask", "is_ppo_ratio_clamped"}
    topk_2d_fields = {"topk_logprobs", "topk_logits", "topk_token_ids"}

    aligned = {}

    for key, val in sample.items():
        if key == "tokens":
            aligned[key] = pad_or_truncate_1d(val, S, pad_value=0)
        elif key in shifted_1d_fields:
            aligned[key] = pad_or_truncate_1d(val, S - 1, pad_value=0)
        elif key in bool_shifted_fields:
            aligned[key] = pad_or_truncate_1d(val, S - 1, pad_value=False)
        elif key in topk_2d_fields:
            aligned[key] = pad_or_truncate_2d(val, S - 1, pad_value=0)
        elif key == "moe_topk_info":
            aligned[key] = {}
            for layer_name, layer_dict in val.items():
                aligned[key][layer_name] = {}
                for k, v in layer_dict.items():
                    aligned[key][layer_name][k] = pad_or_truncate_2d(v, S, pad_value=0)
        else:
            aligned[key] = val

    return aligned


def print_sample(obj, prefix="", level=0):
    """递归打印 dict/tensor 结构，tensor 显示 shape 和中间 10 个元素。"""
    indent = "  " * level

    if isinstance(obj, dict):
        print(f"{indent}{prefix}dict ({len(obj)} keys)")
        for k, v in obj.items():
            print_sample(v, prefix=f"[{k!r}] ", level=level + 1)

    elif isinstance(obj, torch.Tensor):
        shape_str = list(obj.shape)
        dtype_str = str(obj.dtype).replace("torch.", "")
        flat = obj.flatten()
        total = flat.numel()
        if total == 0:
            mid_str = "[]"
        elif total <= 10:
            mid_vals = flat.tolist()
            mid_str = ", ".join(_fmt(v) for v in mid_vals)
        else:
            mid_start = max(0, total // 2 - 5)
            mid_vals = flat[mid_start:mid_start + 10].tolist()
            mid_str = ", ".join(_fmt(v) for v in mid_vals)
        print(f"{indent}{prefix}Tensor shape={shape_str} dtype={dtype_str}")
        print(f"{indent}  mid10: [{mid_str}]")

    elif isinstance(obj, list):
        print(f"{indent}{prefix}list (len={len(obj)})")
        if len(obj) > 0 and len(obj) <= 3:
            for i, item in enumerate(obj):
                print_sample(item, prefix=f"[{i}] ", level=level + 1)
        elif len(obj) > 3:
            print_sample(obj[0], prefix="[0] ", level=level + 1)
            print(f"{indent}  ... ({len(obj) - 2} more)")
            print_sample(obj[-1], prefix=f"[{len(obj)-1}] ", level=level + 1)

    elif isinstance(obj, (int, float, bool, type(None))):
        print(f"{indent}{prefix}{type(obj).__name__} = {obj}")

    else:
        print(f"{indent}{prefix}{type(obj).__name__}: (not displayed)")


def _fmt(v) -> str:
    if isinstance(v, float):
        if abs(v) < 1e-4 and v != 0:
            return f"{v:.6e}"
        return f"{v:.6g}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(
        description="对齐 dump metrics sample 各字段到训练 padded_seqlen 并打印"
    )
    parser.add_argument("--pt-file", required=True, help="Path to .pt dump file")
    parser.add_argument("--sample-idx", type=int, required=True, help="Sample index")
    args = parser.parse_args()

    print(f"Loading {args.pt_file} ...")
    data = torch.load(args.pt_file, map_location="cpu", weights_only=False)

    if not isinstance(data, list):
        raise ValueError(f"Expected .pt to contain a list, got {type(data).__name__}")

    if args.sample_idx < 0 or args.sample_idx >= len(data):
        raise IndexError(
            f"sample_idx={args.sample_idx} out of range [0, {len(data) - 1}] "
            f"(total {len(data)} samples)"
        )

    sample = data[args.sample_idx]
    del data

    print(f"Selected sample idx={args.sample_idx}\n")

    print("=" * 60)
    print("ALIGNED SAMPLE")
    print("=" * 60)
    aligned = align_sample(sample)
    print_sample(aligned)


if __name__ == "__main__":
    main()
