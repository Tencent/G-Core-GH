# coding=utf-8
# Adapted from DeepSeek-V4-Flash/inference/convert.py
# Goal: dequantize quantized checkpoint into plain bfloat16 for training.
# Diff vs the original is intentionally kept small; see comments tagged "CHANGE".
import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


# CHANGE: replace cast_e2m1fn_to_e4m3fn with cast_e2m1fn_to_bf16.
# Same nibble-unpack as the original; we just multiply by the e8m0 scale and
# return bf16 directly instead of producing a (fp8, scale) pair.
def cast_e2m1fn_to_bf16(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 (int8-packed) weight + e8m0 block scale to bf16."""
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp4_block_size = 32
    assert in_dim % fp4_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low  = x & 0x0F
    high = (x >> 4) & 0x0F
    # Same interleave as the original: low at even positions, high at odd.
    # CHANGE vs original: flatten(1) instead of flatten(2) — original keeps the
    # extra nibble dim because it later reshapes via .view(bOut,128,bIn,128);
    # we go straight to [out, in_dim] for the scale broadcast below.
    x = torch.stack([FP4_TABLE.to(x.device)[low.long()],
                     FP4_TABLE.to(x.device)[high.long()]], dim=-1).flatten(1)

    # CHANGE: directly multiply by per-32 scale (no fp8 detour). E8M0 -> fp32 via .float().
    scale = scale.float().repeat_interleave(fp4_block_size, dim=-1)
    return (x * scale).bfloat16()


# CHANGE: new helper for the fp8 weights (attn / shared_experts).
# Original convert.py keeps these as fp8 + scale; we want pure bf16.
def cast_e4m3fn_to_bf16(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 (e4m3fn) weight + e8m0 2D block scale to bf16."""
    assert x.dtype == torch.float8_e4m3fn
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    fp8_block_size = 128
    assert out_dim % fp8_block_size == 0 and in_dim % fp8_block_size == 0
    assert scale.size() == (out_dim // fp8_block_size, in_dim // fp8_block_size)

    s = (scale.float()
              .repeat_interleave(fp8_block_size, dim=0)
              .repeat_interleave(fp8_block_size, dim=1))
    return (x.float() * s).bfloat16()


# CHANGE: drop the `mapping` dict entirely. We keep the original ckpt's key
# names (no rename to wq_a/etc); training code can read them as-is.


# CHANGE: signature lost `n_experts`/`mp`/`expert_dtype`; gained `layers`.
def main(hf_ckpt_path, save_path, layers=None):
    """Read every shard, dequantize quantized weights to bf16, write new shards.

    Args:
        hf_ckpt_path: source DeepSeek-V4-Flash safetensors directory.
        save_path:    destination directory (created if missing).
        layers:       optional set[int] to keep only these decoder layers
                      (layer-agnostic keys like embed/head/norm are always kept).
    """
    torch.set_num_threads(8)
    os.makedirs(save_path, exist_ok=True)
    state_dict = {}                                       # CHANGE: single dict, no mp shards.

    # CHANGE: gather everything first, defer dequant to second pass (need both
    # weight and its .scale, which may live in different shards).
    for file_path in tqdm(sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors"))),
                          desc="load"):
        shard_name = os.path.basename(file_path)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if layers is not None and name.startswith("layers."):
                    lid = int(name.split(".")[1])
                    if lid not in layers:
                        continue
                state_dict[name] = (shard_name, f.get_tensor(name))

    names = list(state_dict.keys())
    new_state_dict = {}                                   # name -> (shard, tensor) for output
    for name in tqdm(names, desc="dequant"):
        if name.endswith(".scale"):
            continue                                      # consumed below
        shard, w = state_dict[name]
        scale_name = name.replace(".weight", ".scale") if name.endswith(".weight") else name + ".scale"

        if w.dtype == torch.int8 and scale_name in state_dict:
            _, s = state_dict[scale_name]
            w = cast_e2m1fn_to_bf16(w, s)                 # CHANGE: was view(float4_e2m1fn_x2)
        elif w.dtype == torch.float8_e4m3fn and scale_name in state_dict:
            _, s = state_dict[scale_name]
            w = cast_e4m3fn_to_bf16(w, s)                 # CHANGE: original kept fp8 here
        # else: bf16/fp32/int64 buffer -> passthrough.
        new_state_dict[name] = (shard, w.contiguous())

    # CHANGE: regroup by source shard, save with same filenames so layout looks
    # like the original (and downstream HF index loaders find the same files).
    by_shard = {}
    for name, (shard, t) in new_state_dict.items():
        by_shard.setdefault(shard, {})[name] = t
    weight_map, total_size = {}, 0
    for shard, sd in tqdm(sorted(by_shard.items()), desc="save"):
        save_file(sd, os.path.join(save_path, shard))
        for name, t in sd.items():
            weight_map[name] = shard
            total_size += t.numel() * t.element_size()

    # CHANGE: write HF-style index so AutoModel.from_pretrained can load it.
    import json
    with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map},
                  f, indent=2, sort_keys=True)

    # CHANGE: copy *all* aux files (config / tokenizer / generation_config / ...),
    # not just tokenizer.{json,_config.json}. Skip safetensors (already rewritten).
    for item in os.listdir(hf_ckpt_path):
        src = os.path.join(hf_ckpt_path, item)
        if not os.path.isfile(src) or item.endswith(".safetensors") \
                or item == "model.safetensors.index.json":
            continue
        shutil.copy2(src, os.path.join(save_path, item))


def _parse_layers(spec):
    """'0,1,3-5' -> {0,1,3,4,5}. None means keep all layers."""
    if spec is None:
        return None
    out = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    # CHANGE: dropped --n-experts, --model-parallel, --expert-dtype; added --layers.
    parser.add_argument("--layers", type=str, default=None,
                        help="restrict to a layer subset, e.g. '0,1' or '0-2'")
    args = parser.parse_args()
    main(args.hf_ckpt_path, args.save_path, layers=_parse_layers(args.layers))
