# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""Dump one BAD-aligned and one GOOD-aligned FP8 (128×128) block from the
official DSV4-Flash checkpoint, do dequant → re-quant on each, and pretty-
print every observable value side-by-side.

Why this exists. ``TestDeepseekCheckpointQuantDequant`` shows that ~7% of
on-disk FP8 blocks pick a ``disk_scale = 2 × our_scale`` representation —
numerically equivalent but byte-different from what our quantizer
chooses. To investigate without bothering the test infrastructure, this
script captures a single concrete BAD block (worst-aligned tensor:
``layers.20.attn.wq_a.weight``, block (0, 3)) and a single GOOD block
(same tensor, block (0, 0)) with **identical post-dequant max_abs =
0.109375** so the comparison isolates the upstream-vs-ours choice.

Outputs:

  1. Prints a human-readable side-by-side report to stdout.
  2. Saves the full raw tensor contents (disk bytes, our re-quant bytes,
     dequant values) into a ``.pt`` file for offline inspection
     (``torch.load(...)``).

Usage::

    cd /work/wepsdl/gcore-dev
    python tests/test_gfused/dump_fp8_bad_block.py
    # → writes ./tests/test_gfused/bad_block_dumps/fp8_bad_block_dump.pt
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import safe_open

# Import the production quantizer / dequantizer by file path so the
# script remains runnable without any package-import gymnastics.
_FP_QUANT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "gpatch_v4/kernel/quantize/eager_quant_kernels.py"
)
_spec = importlib.util.spec_from_file_location("fp_quantize", _FP_QUANT_PATH)
assert _spec is not None and _spec.loader is not None
fp_quantize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp_quantize)

CKPT_DIR = "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/deepseek-ai/DeepSeek-V4-Flash"
TARGET_KEY = "layers.20.attn.wq_a.weight"
TARGET_SCALE_KEY = TARGET_KEY.replace(".weight", ".scale")
BLOCK_M, BLOCK_N = 128, 128
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_full_tensor(key: str) -> torch.Tensor:
    """Stream a single tensor off the checkpoint's safetensors shards."""
    with open(os.path.join(CKPT_DIR, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    shard = idx["weight_map"][key]
    with safe_open(os.path.join(CKPT_DIR, shard), framework="pt", device="cpu") as fh:
        return fh.get_tensor(key)


def _find_bad_good_blocks(
    w_disk: torch.Tensor, s_disk: torch.Tensor
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Locate one BAD and one GOOD block, both with max_abs == 0.109375.

    BAD: our re-quantized scale disagrees with disk scale (specifically
         ``disk_scale = 2 × our_scale``).
    GOOD: our re-quantized scale agrees byte-exact with disk scale, AND
          max_abs happens to be the same 0.109375 ⇒ the two blocks
          differ only in upstream's representation choice, not in any
          observable post-dequant statistic.
    """
    v = fp_quantize.dequant_fp4_e2m1_fp8_scale_e8m0_packed(w_disk, s_disk).float()
    _, s_re = fp_quantize.quant_fp8_e4m3_scale_e8m0(v, block_size=(BLOCK_M, BLOCK_N))

    sR, sC = s_disk.shape
    blk_v = v.reshape(sR, BLOCK_M, sC, BLOCK_N)
    ma = blk_v.abs().amax(dim=(1, 3))                       # (sR, sC)
    sf = s_disk.float()
    sref = s_re.float()
    bad_mask = sf != sref

    # Pick one BAD block whose max_abs == 0.109375 (the canonical case).
    target_max = 0.109375
    bad_locs = (
        (bad_mask & (ma.sub(target_max).abs() < 1e-12)).nonzero().tolist()
    )
    good_locs = (
        ((~bad_mask) & (ma.sub(target_max).abs() < 1e-12)).nonzero().tolist()
    )
    assert bad_locs, "no BAD block with max_abs=0.109375 found"
    assert good_locs, "no GOOD block with max_abs=0.109375 found"
    return tuple(bad_locs[0]), tuple(good_locs[0])  # type: ignore[return-value]


def _slice_block(
    t: torch.Tensor, r: int, c: int, bm: int = BLOCK_M, bn: int = BLOCK_N
) -> torch.Tensor:
    """Slice a single (bm, bn) block out of a 2D tensor."""
    return t[r * bm : (r + 1) * bm, c * bn : (c + 1) * bn].clone().contiguous()


def _describe_block(
    label: str,
    w_disk_blk: torch.Tensor,    # e4m3fn, (bm, bn)
    s_disk_blk: torch.Tensor,    # e8m0fnu, scalar (0-dim or 1-elt)
    w_re_blk: torch.Tensor,      # e4m3fn, (bm, bn)
    s_re_blk: torch.Tensor,      # e8m0fnu, scalar
    v_blk: torch.Tensor,         # fp32, (bm, bn) — dequant of disk
    v_re_blk: torch.Tensor,      # fp32, (bm, bn) — dequant of (re_w, re_s)
    block_idx: tuple[int, int],
) -> None:
    """Print every observable for one block: scale, codes, dequant values,
    raw bytes, sanity counters."""
    bm, bn = w_disk_blk.shape
    print(f"\n{'='*70}")
    print(f"=== {label} block at scale-grid index {block_idx}  ===")
    print(f"{'='*70}")

    # Scales.
    sd = s_disk_blk.float().item()
    sr = s_re_blk.float().item()
    print("\n[Scale]")
    print(f"  disk_scale = {sd:.10g}   = 2^{math.log2(sd):.4f}")
    print(f"  our_scale  = {sr:.10g}   = 2^{math.log2(sr):.4f}")
    print(f"  ratio disk/our = {sd / sr:.6f}")
    # E8M0 byte (single byte).
    sd_byte = s_disk_blk.view(torch.uint8).item()
    sr_byte = s_re_blk.view(torch.uint8).item()
    print(f"  disk_scale byte = 0x{sd_byte:02x} ({sd_byte})")
    print(f"  our_scale  byte = 0x{sr_byte:02x} ({sr_byte})")

    # Block summary statistics — post-dequant fp32 view.
    print("\n[Dequant value distribution]")
    abs_v = v_blk.abs()
    max_abs = abs_v.max().item()
    cnt_at_max = (abs_v == max_abs).sum().item()
    print(f"  max_abs   = {max_abs}")
    print(f"  count_at_max = {cnt_at_max} (out of {bm*bn} elements)")
    print(f"  mean_abs  = {abs_v.mean().item():.6e}")
    print(f"  rms       = {(abs_v * abs_v).mean().sqrt().item():.6e}")
    print(f"  ratio max_abs / disk_scale = {max_abs / sd:.4f}   (= max code disk reads as e4m3)")
    print(f"  ratio max_abs / our_scale  = {max_abs / sr:.4f}   (= max code we      read as e4m3)")

    # Raw fp8_e4m3fn code distribution (top 8 most common bytes, ignoring
    # sign bit so e.g. +0.5 and -0.5 stack).
    print("\n[Raw FP8 e4m3fn code histogram (block, top 8 by abs-value)]")
    u8_disk = w_disk_blk.view(torch.uint8).flatten().cpu()
    u8_re = w_re_blk.view(torch.uint8).flatten().cpu()
    abs_disk = u8_disk & 0x7F
    abs_re = u8_re & 0x7F
    # Decode each unique abs-code to its fp32 magnitude via e4m3 cast.
    def _decode_e4m3(code: int) -> float:
        t = torch.tensor([code], dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        return abs(t.item())
    for tag, abs_codes, u8_full in [("disk", abs_disk, u8_disk), ("ours", abs_re, u8_re)]:
        uniq, cnt = torch.unique(abs_codes, return_counts=True)
        pairs = sorted(
            zip(uniq.tolist(), cnt.tolist()), key=lambda p: -p[1]
        )[:8]
        formatted = ", ".join(
            f"(abs_code=0x{u:02x}={_decode_e4m3(u):g}, n={n})" for u, n in pairs
        )
        print(f"  {tag}: {formatted}")
        # Max abs code (the "biggest fp8 number" stored, ignoring sign):
        max_abs_code = int(abs_codes.max().item())
        max_abs_val = _decode_e4m3(max_abs_code)
        print(
            f"    → max |code| = 0x{max_abs_code:02x} = {max_abs_val}  "
            f"(decodes to {max_abs_val} × {tag}_scale = {max_abs_val * (sd if tag=='disk' else sr):.6g})"
        )

    # Round-trip equivalence: was the rebuilt dequant equal to disk's?
    print("\n[Round-trip numerical equivalence]")
    print(f"  v_disk == v_re ?  {torch.equal(v_blk, v_re_blk)}")
    diff = (v_blk - v_re_blk).abs()
    print(f"  max abs diff = {diff.max().item():.6e}  (mean {diff.mean().item():.6e})")

    # Show first 8 elements both side-by-side (interleaved decoded).
    print("\n[First row, first 8 elements — fp8 code / fp32 dequant]")
    print("  idx | disk(code)  disk(deq)        | ours(code) ours(deq)        | equal?")
    print("  ----+----------------------------------+----------------------------------+-------")
    for j in range(8):
        c_disk = int(u8_disk[j].item())
        c_re = int(u8_re[j].item())
        vd = v_blk[0, j].item()
        vr = v_re_blk[0, j].item()
        eq = "✓" if vd == vr else "✗"
        print(
            f"  {j:>3} | 0x{c_disk:02x} ({_decode_e4m3(c_disk & 0x7F) * (-1 if c_disk & 0x80 else 1):>+9.6g})"
            f"        | 0x{c_re:02x} ({_decode_e4m3(c_re & 0x7F) * (-1 if c_re & 0x80 else 1):>+9.6g})"
            f"        | {eq}"
        )


def main() -> None:
    assert os.path.isfile(
        os.path.join(CKPT_DIR, "model.safetensors.index.json")
    ), f"checkpoint not found at {CKPT_DIR}"

    print(f"Loading {TARGET_KEY} from disk ...")
    w_disk_full = _load_full_tensor(TARGET_KEY).to(DEVICE)
    s_disk_full = _load_full_tensor(TARGET_SCALE_KEY).to(DEVICE)
    print(f"  w shape={tuple(w_disk_full.shape)} dtype={w_disk_full.dtype}")
    print(f"  s shape={tuple(s_disk_full.shape)} dtype={s_disk_full.dtype}")

    # Find one BAD and one GOOD block whose max_abs is identical.
    bad_rc, good_rc = _find_bad_good_blocks(w_disk_full, s_disk_full)
    print(f"\nSelected BAD block at (r, c) = {bad_rc}")
    print(f"Selected GOOD block at (r, c) = {good_rc}")

    # Build per-block tensors (single block each).
    v_full = fp_quantize.dequant_fp4_e2m1_fp8_scale_e8m0_packed(
        w_disk_full, s_disk_full
    ).float()
    w_re_full, s_re_full = fp_quantize.quant_fp8_e4m3_scale_e8m0(
        v_full, block_size=(BLOCK_M, BLOCK_N)
    )
    v_re_full = fp_quantize.dequant_fp4_e2m1_fp8_scale_e8m0_packed(
        w_re_full, s_re_full
    ).float()

    dumps: dict[str, dict[str, torch.Tensor | tuple | str]] = {}
    for label, rc in [("BAD", bad_rc), ("GOOD", good_rc)]:
        r, c = rc
        block = {
            "label": label,
            "tensor_key": TARGET_KEY,
            "block_idx": rc,
            "block_size": (BLOCK_M, BLOCK_N),
            # disk side
            "w_disk_block": _slice_block(w_disk_full, r, c).cpu(),
            "s_disk_block": s_disk_full[r, c].clone().cpu(),
            "v_disk_block": _slice_block(v_full, r, c).cpu(),
            # our re-quant
            "w_re_block": _slice_block(w_re_full, r, c).cpu(),
            "s_re_block": s_re_full[r, c].clone().cpu(),
            "v_re_block": _slice_block(v_re_full, r, c).cpu(),
        }
        dumps[label] = block

    # Pretty print for human eyes.
    for label in ("BAD", "GOOD"):
        d = dumps[label]
        _describe_block(
            label,
            d["w_disk_block"], d["s_disk_block"],
            d["w_re_block"],   d["s_re_block"],
            d["v_disk_block"], d["v_re_block"],
            d["block_idx"],
        )

    # Save .pt for offline inspection.
    out_dir = Path(__file__).resolve().parent / "bad_block_dumps"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fp8_bad_block_dump.pt"
    torch.save(dumps, out_path)
    print(f"\n\n>>> dumped to {out_path}")
    print(">>> reload with:  d = torch.load(...) ; d['BAD']['w_disk_block'] ...")


if __name__ == "__main__":
    main()
