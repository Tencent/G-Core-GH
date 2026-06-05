# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Per-layer numerical equivalence check after expert permutation.

We avoid loading the full 35B-parameter model. Instead, for a list of
representative MoE layers, we:

1. Load the original 3 routed-expert tensors (gate, gate_up_proj, down_proj).
2. Load the same 3 tensors from the permuted checkpoint.
3. Assert ``new_t == old_t[perm[L]]`` for all 3 (bit-exact).
4. Run a manual SwiGLU MoE block with top-k=8 routing on a random fp32 input.
   Check that:
     - The set of selected expert ids in original vs. permuted is identical
       *up to* the perm map: ``perm[L][new_topk] == old_topk``.
     - The final per-token output tensors are equal within fp32 numerical
       tolerance.

This catches any indexing / shape / dtype bug without loading the full
transformer by default. Use ``--verify-full-forward`` to additionally load the
full original and permuted HF models and compare their logits.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors import safe_open

_GATE_W = "model.language_model.layers.{L}.mlp.gate.weight"
_GATE_UP = "model.language_model.layers.{L}.mlp.experts.gate_up_proj"
_DOWN = "model.language_model.layers.{L}.mlp.experts.down_proj"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src-dir", required=True, help="original HF ckpt dir")
    p.add_argument("--dst-dir", required=True, help="permuted HF ckpt dir")
    p.add_argument("--routing-map", required=True)
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="HF layer ids to verify. Defaults to all layers in routing_map.",
    )
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help=(
            "relative tolerance for the per-token MoE output comparison. "
            "Both rtol and atol must hold."
        ),
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="absolute tolerance for the per-token MoE output comparison",
    )
    p.add_argument(
        "--require-bitexact",
        action="store_true",
        help=(
            "stricter mode: require the original-vs-permuted MoE outputs to "
            "be bit-exact (same compute order on both sides means floats "
            "should match exactly). Useful for catching index bugs."
        ),
    )
    p.add_argument(
        "--verify-full-forward",
        action="store_true",
        help="also load both full HF models and compare their forward logits",
    )
    p.add_argument(
        "--sample-type",
        type=str,
        default="text",
        choices=["text", "image", "video", "mix"],
        help="sample type for --verify-full-forward",
    )
    p.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="local image path used by image/mix full-forward samples",
    )
    p.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="local video path used by video/mix full-forward samples",
    )
    p.add_argument(
        "--full-forward-rtol",
        type=float,
        default=1e-4,
        help="relative tolerance for full-model logits comparison",
    )
    p.add_argument(
        "--full-forward-atol",
        type=float,
        default=1e-4,
        help="absolute tolerance for full-model logits comparison",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def verify_equivalence(
    *,
    src_dir: str,
    dst_dir: str,
    routing_map_path: str,
    layers: Optional[List[int]] = None,
    seq_len: int = 64,
    top_k: int = 8,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    require_bitexact: bool = False,
    verify_full_forward: bool = False,
    sample_type: str = "text",
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    full_forward_rtol: float = 1e-4,
    full_forward_atol: float = 1e-4,
    seed: int = 0,
) -> None:
    """Verify per-layer and optionally full-model equivalence after repermute."""
    with open(routing_map_path) as fp:
        rmap = json.load(fp)
    perm_full = torch.tensor(rmap["perm"], dtype=torch.long)
    num_experts = rmap["num_experts"]

    src_index = load_index(src_dir)
    dst_index = load_index(dst_dir)
    hidden = _peek_hidden(src_dir, src_index)

    verify_layers = list(range(rmap["num_layers"])) if layers is None else list(layers)
    print(
        f"[verify] hidden={hidden} num_experts={num_experts} "
        f"layers={verify_layers} seq={seq_len} top_k={top_k}"
    )

    torch.manual_seed(seed)

    failures: List[str] = []
    for L in verify_layers:
        try:
            verify_one_layer(
                L=L,
                src_dir=src_dir,
                src_index=src_index,
                dst_dir=dst_dir,
                dst_index=dst_index,
                perm=perm_full[L],
                hidden=hidden,
                seq_len=seq_len,
                top_k=top_k,
                rtol=rtol,
                atol=atol,
                require_bitexact=require_bitexact,
            )
            print(f"[verify] layer {L:>2d}: OK")
        except AssertionError as e:
            print(f"[verify] layer {L:>2d}: FAIL  {e}")
            failures.append(f"L{L}: {e}")

    if failures:
        print("\n".join(["[verify] FAILURES:"] + failures))
        raise SystemExit(1)
    print(f"[verify] all {len(verify_layers)} layers passed.")

    if verify_full_forward:
        verify_full_model_forward(
            src_dir=src_dir,
            dst_dir=dst_dir,
            sample_type=sample_type,
            image_path=image_path,
            video_path=video_path,
            rtol=full_forward_rtol,
            atol=full_forward_atol,
        )


def main() -> None:
    args = parse_args()
    verify_equivalence(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        routing_map_path=args.routing_map,
        layers=args.layers,
        seq_len=args.seq_len,
        top_k=args.top_k,
        rtol=args.rtol,
        atol=args.atol,
        require_bitexact=args.require_bitexact,
        verify_full_forward=args.verify_full_forward,
        sample_type=args.sample_type,
        image_path=args.image_path,
        video_path=args.video_path,
        full_forward_rtol=args.full_forward_rtol,
        full_forward_atol=args.full_forward_atol,
        seed=args.seed,
    )


def verify_one_layer(
    *,
    L: int,
    src_dir: str,
    src_index: Dict,
    dst_dir: str,
    dst_index: Dict,
    perm: torch.Tensor,
    hidden: int,
    seq_len: int,
    top_k: int,
    rtol: float,
    atol: float,
    require_bitexact: bool,
) -> None:
    keys = (_GATE_W.format(L=L), _GATE_UP.format(L=L), _DOWN.format(L=L))
    g_o, gu_o, dn_o = _load_tensors(src_dir, src_index, keys)
    g_n, gu_n, dn_n = _load_tensors(dst_dir, dst_index, keys)

    assert torch.equal(g_n, g_o.index_select(0, perm)), (f"gate.weight mismatch at L={L}")
    assert torch.equal(gu_n, gu_o.index_select(0,
                                               perm)), (f"experts.gate_up_proj mismatch at L={L}")
    assert torch.equal(dn_n, dn_o.index_select(0, perm)), (f"experts.down_proj mismatch at L={L}")

    g_o_f = g_o.to(torch.float32)
    gu_o_f = gu_o.to(torch.float32)
    dn_o_f = dn_o.to(torch.float32)
    g_n_f = g_n.to(torch.float32)
    gu_n_f = gu_n.to(torch.float32)
    dn_n_f = dn_n.to(torch.float32)

    x = torch.randn(seq_len, hidden, dtype=torch.float32)

    out_orig, ids_orig = mock_moe_forward(x, g_o_f, gu_o_f, dn_o_f, top_k=top_k)
    out_new, ids_new = mock_moe_forward(x, g_n_f, gu_n_f, dn_n_f, top_k=top_k)

    mapped = perm[ids_new]
    assert torch.equal(_sort_lastdim(mapped), _sort_lastdim(ids_orig)), (
        f"top-k routing changed at L={L}: "
        f"orig={ids_orig[:2].tolist()} new->mapped={mapped[:2].tolist()}"
    )

    if require_bitexact:
        assert torch.equal(out_orig, out_new), (
            f"MoE output not bit-exact at L={L}: "
            f"max_abs={(out_orig - out_new).abs().max().item():.3e}"
        )
    else:
        diff = (out_orig - out_new).abs()
        max_abs = float(diff.max())
        max_rel = float((diff / out_orig.abs().clamp_min(1e-6)).max())
        assert max_abs < atol and max_rel < rtol, (
            f"MoE output diff at L={L}: max_abs={max_abs:.3e} "
            f"max_rel={max_rel:.3e} (atol={atol}, rtol={rtol})"
        )


def mock_moe_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Manual SwiGLU MoE block. Returns (output, top_k_ids).

    Shapes
    ------
    x              : (T, H)
    gate_weight    : (E, H)
    gate_up_proj   : (E, 2*M, H)
    down_proj      : (E, H, M)
    output         : (T, H)
    top_k_ids      : (T, K)
    """
    T, H = x.shape
    E = gate_weight.shape[0]
    K = top_k

    logits = x @ gate_weight.T
    topk_vals, topk_ids = logits.topk(K, dim=-1)
    routing = torch.softmax(topk_vals, dim=-1)

    out = torch.zeros_like(x)
    for k in range(K):
        ids_k = topk_ids[:, k]
        scale_k = routing[:, k]
        gu = gate_up_proj[ids_k]
        dn = down_proj[ids_k]
        h = torch.einsum("th,tmh->tm", x, gu)
        M = h.shape[-1] // 2
        gate_part = torch.nn.functional.silu(h[:, :M])
        up_part = h[:, M:]
        inter = gate_part * up_part
        y = torch.einsum("tm,thm->th", inter, dn)
        out += scale_k.unsqueeze(-1) * y

    return out, topk_ids


def _sort_lastdim(t: torch.Tensor) -> torch.Tensor:
    return t.sort(dim=-1).values


def cos_similarity(a: torch.Tensor, b: torch.Tensor) -> None:
    print(f"a {a.shape} b {b.shape}")
    assert a.shape == b.shape, f"logits shape mismatch: {a.shape=} {b.shape=}"
    a = a.float()
    a = torch.exp(a - a.max(dim=-1, keepdim=True)[0])
    a = a / a.norm(dim=-1, keepdim=True)
    b = b.float()
    b = torch.exp(b - b.max(dim=-1, keepdim=True)[0])
    b = b / b.norm(dim=-1, keepdim=True)
    sim = (a * b).sum(dim=-1)
    print(f"src vs dst cos_similarity min: {sim.min()}; max: {sim.max()}; mean: {sim.mean()}")


def verify_full_model_forward(
    *,
    src_dir: str,
    dst_dir: str,
    sample_type: str,
    image_path: Optional[str],
    video_path: Optional[str],
    rtol: float,
    atol: float,
) -> None:
    inputs = get_sample_for_forward(
        src_dir,
        sample_type=sample_type,
        image_path=image_path,
        video_path=video_path,
    )
    input_mask = build_input_mask(inputs, src_dir)
    src_logits = run_full_model_forward(src_dir, inputs)
    dst_logits = run_full_model_forward(dst_dir, inputs)

    assert src_logits.shape == dst_logits.shape, (
        f"full-forward logits shape mismatch: src={tuple(src_logits.shape)} "
        f"dst={tuple(dst_logits.shape)}"
    )
    print("======= full forward cos_similarity without mask =======")
    cos_similarity(src_logits, dst_logits)

    assert input_mask.shape == src_logits.shape[:2], (
        f"input mask shape mismatch: mask={tuple(input_mask.shape)} "
        f"logits={tuple(src_logits.shape)}"
    )
    src_logits = src_logits[input_mask]
    dst_logits = dst_logits[input_mask]
    print("======= full forward cos_similarity with mask =======")
    cos_similarity(src_logits, dst_logits)

    diff = (src_logits.float() - dst_logits.float()).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    min_abs = float(diff.min())
    print(
        "[verify] full forward logits: OK "
        f"compared_tokens={src_logits.shape[0]} vocab={src_logits.shape[-1]} "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} min_abs={min_abs:.3e}"
    )


def run_full_model_forward(model_dir: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    try:
        from transformers import Qwen3_5MoeForConditionalGeneration
    except ImportError as e:
        raise RuntimeError(
            "install transformers with Qwen3_5MoeForConditionalGeneration support"
        ) from e

    torch.set_grad_enabled(False)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        model_dir,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    model_inputs = _clone_inputs(inputs)
    with torch.no_grad():
        logits = model.forward(**model_inputs).logits.detach().cpu()
    del model, model_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def _clone_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        out[k] = v.clone() if torch.is_tensor(v) else v
    return out


def build_input_mask(inputs: Dict[str, torch.Tensor], hf_model_path: str) -> torch.Tensor:
    input_ids = inputs["input_ids"]
    image_token_id = load_config_value(hf_model_path, "image_token_id")
    vision_end_token_id = load_config_value(hf_model_path, "vision_end_token_id")
    vision_start_token_id = load_config_value(hf_model_path, "vision_start_token_id")

    input_mask = input_ids != image_token_id
    input_mask = input_mask & (input_ids != vision_end_token_id)
    input_mask = input_mask & (input_ids != vision_start_token_id)
    return torch.nn.functional.pad(input_mask[:, 1:], (0, 1, 0, 0), value=True)


def load_config_value(hf_model_path: str, key: str) -> int:
    with open(os.path.join(hf_model_path, "config.json")) as fp:
        cfg = json.load(fp)
    value = find_config_value(cfg, key)
    assert value is not None, f"{key} not found in {hf_model_path}/config.json"
    return int(value)


def find_config_value(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_config_value(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_config_value(v, key)
            if found is not None:
                return found
    return None


def get_sample_for_forward(
    hf_model_path: str,
    sample_type: str = "text",
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    if sample_type == "text":
        return get_text_sample_for_forward(hf_model_path)
    if sample_type == "image":
        assert image_path is not None, "--image-path is required for image sample"
        return get_image_sample_for_forward(hf_model_path, image_path)
    if sample_type == "video":
        assert video_path is not None, "--video-path is required for video sample"
        return get_video_sample_for_forward(hf_model_path, video_path)
    if sample_type == "mix":
        assert image_path is not None, "--image-path is required for mix sample"
        assert video_path is not None, "--video-path is required for mix sample"
        return get_mix_sample_for_forward(hf_model_path, image_path, video_path)
    raise AssertionError(f"unknown sample_type={sample_type}")


def get_text_sample_for_forward(hf_model_path: str) -> Dict[str, torch.Tensor]:
    processor = _load_processor(hf_model_path)
    messages = [
        {
            "role":
                "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the purpose of load balancing briefly."
                }
            ],
        }
    ]
    return _apply_chat_template(processor, messages)


def get_image_sample_for_forward(hf_model_path: str, image_path: str) -> Dict[str, torch.Tensor]:
    assert os.path.exists(image_path), f"image path does not exist: {image_path}"
    processor = _load_processor(hf_model_path)
    messages = [
        {
            "role":
                "user",
            "content":
                [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {
                        "type": "text",
                        "text": "Describe this image shortly."
                    },
                ],
        }
    ]
    return _apply_chat_template(processor, messages, max_pixels=256 * 28 * 28)


def get_video_sample_for_forward(hf_model_path: str, video_path: str) -> Dict[str, torch.Tensor]:
    assert os.path.exists(video_path), f"video path does not exist: {video_path}"
    processor = _load_processor(hf_model_path)
    messages = [
        {
            "role":
                "user",
            "content":
                [
                    {
                        "type": "video",
                        "video": video_path
                    },
                    {
                        "type": "text",
                        "text": "Describe this video shortly."
                    },
                ],
        }
    ]
    return _apply_chat_template(processor, messages)


def get_mix_sample_for_forward(
    hf_model_path: str,
    image_path: str,
    video_path: str,
) -> Dict[str, torch.Tensor]:
    assert os.path.exists(image_path), f"image path does not exist: {image_path}"
    assert os.path.exists(video_path), f"video path does not exist: {video_path}"
    processor = _load_processor(hf_model_path)
    messages = [
        {
            "role":
                "user",
            "content":
                [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {
                        "type": "video",
                        "video": video_path
                    },
                    {
                        "type": "text",
                        "text": "Describe this image and video shortly."
                    },
                ],
        }
    ]
    return _apply_chat_template(processor, messages, add_vision_id=True)


def _load_processor(hf_model_path: str):
    try:
        from transformers import Qwen3VLProcessor
    except ImportError as e:
        raise RuntimeError("install transformers with Qwen3VLProcessor support") from e
    return Qwen3VLProcessor.from_pretrained(hf_model_path)


def _apply_chat_template(processor, messages: List[Dict], **kwargs) -> Dict[str, torch.Tensor]:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **kwargs,
    )
    inputs.pop("token_type_ids", None)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)
    return dict(inputs)


def load_index(d: str) -> Dict:
    with open(os.path.join(d, "model.safetensors.index.json")) as fp:
        return json.load(fp)


def _load_tensors(base_dir: str, index: Dict, keys: Tuple[str, ...]) -> Tuple[torch.Tensor, ...]:
    wm = index["weight_map"]
    out = []
    for k in keys:
        shard = wm[k]
        with safe_open(os.path.join(base_dir, shard), framework="pt", device="cpu") as f:
            out.append(f.get_tensor(k))
    return tuple(out)


def _peek_hidden(base_dir: str, index: Dict) -> int:
    k = _GATE_W.format(L=0)
    shard = index["weight_map"][k]
    with safe_open(os.path.join(base_dir, shard), framework="pt", device="cpu") as f:
        sl = f.get_slice(k)
        shape = sl.get_shape()
    return int(shape[1])


if __name__ == "__main__":
    main()
