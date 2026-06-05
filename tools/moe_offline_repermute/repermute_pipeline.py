# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""End-to-end Python pipeline for MoE offline expert re-permutation."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from .build_routing_map import build_routing_map
from .repermute_hf_ckpt import repermute_hf_ckpt
from .verify_equivalence import verify_equivalence
from .viz_repermute import visualize_repermute


@dataclass
class MoeOfflineRepermuteConfig:
    source_dump_dir: str
    src_ckpt_dir: str
    dst_ckpt_dir: str
    work_dir: str
    out_dir: Optional[str] = None
    num_layers: Optional[int] = None
    num_experts: Optional[int] = None
    ep_sizes: Tuple[int, ...] = (4, 8, 16)
    verify_full_forward: bool = True
    sample_type: str = "text"
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    force_refresh_aux: bool = False


@dataclass
class MoeOfflineRepermuteResult:
    routing_map_path: str
    imbalance_curve_path: str
    dst_ckpt_dir: str
    viz_out_dir: Optional[str]


def run_moe_offline_repermute(config: MoeOfflineRepermuteConfig) -> MoeOfflineRepermuteResult:
    """Run the full offline re-permutation workflow."""
    routing_map_path = os.path.join(config.work_dir, "routing_map.json")
    imbalance_curve_path = os.path.join(config.work_dir, "imbalance_curve.png")
    viz_out_dir = config.out_dir or os.path.join(config.work_dir, "viz")
    num_layers, num_experts = resolve_model_shape(
        src_ckpt_dir=config.src_ckpt_dir,
        num_layers=config.num_layers,
        num_experts=config.num_experts,
    )

    print("[pipeline] step=build_routing_map")
    build_routing_map(
        source_dir=config.source_dump_dir,
        output=routing_map_path,
        imbalance_curve=imbalance_curve_path,
        num_layers=num_layers,
        num_experts=num_experts,
    )

    print("[pipeline] step=repermute_hf_ckpt dry_run")
    repermute_hf_ckpt(
        src_dir=config.src_ckpt_dir,
        dst_dir=config.dst_ckpt_dir,
        routing_map_path=routing_map_path,
        dry_run=True,
        force_refresh_aux=config.force_refresh_aux,
    )

    print("[pipeline] step=repermute_hf_ckpt rewrite")
    repermute_hf_ckpt(
        src_dir=config.src_ckpt_dir,
        dst_dir=config.dst_ckpt_dir,
        routing_map_path=routing_map_path,
        dry_run=False,
        force_refresh_aux=config.force_refresh_aux,
    )

    print("[pipeline] step=verify_equivalence")
    verify_equivalence(
        src_dir=config.src_ckpt_dir,
        dst_dir=config.dst_ckpt_dir,
        routing_map_path=routing_map_path,
        require_bitexact=True,
        verify_full_forward=config.verify_full_forward,
        sample_type=config.sample_type,
        image_path=config.image_path,
        video_path=config.video_path,
    )

    print("[pipeline] step=visualize_repermute")
    visualize_repermute(
        routing_map_path=routing_map_path,
        out_dir=viz_out_dir,
        ep_sizes=config.ep_sizes,
    )

    return MoeOfflineRepermuteResult(
        routing_map_path=routing_map_path,
        imbalance_curve_path=imbalance_curve_path,
        dst_ckpt_dir=config.dst_ckpt_dir,
        viz_out_dir=viz_out_dir,
    )


def resolve_model_shape(
    *,
    src_ckpt_dir: str,
    num_layers: Optional[int] = None,
    num_experts: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve MoE layer/expert counts from ``src_ckpt_dir``.

    Uses ``transformers.AutoConfig`` so VL checkpoints (whose MoE fields live
    under ``text_config``) are handled uniformly with flat LM checkpoints via
    ``PretrainedConfig.get_text_config()`` -- on flat configs it returns
    ``self``; on composite VL configs it unwraps to the text sub-config.

    Parameters
    ----------
    src_ckpt_dir : str
        HF checkpoint directory containing ``config.json``. Must declare a
        ``model_type`` that transformers recognises; otherwise transformers
        raises ``ValueError: Unrecognized model in ...``.
    num_layers, num_experts : int, optional
        If both are provided, ``src_ckpt_dir`` is not read and transformers
        is not imported.

    Returns
    -------
    Tuple[int, int]
        ``(num_layers, num_experts)`` for the text model.
    """
    if num_layers is not None and num_experts is not None:
        return num_layers, num_experts

    # Local import: keeps `--help` fast and confines the transformers dep
    # to the resolve path (the rest of this module doesn't need it loaded).
    from transformers import AutoConfig

    text_config = AutoConfig.from_pretrained(src_ckpt_dir).get_text_config()

    if num_layers is None:
        num_layers = int(text_config.num_hidden_layers)
    if num_experts is None:
        # This tool targets Qwen3 MoE only (see repermute_hf_ckpt.py's
        # ``_LAYER_RE``); no speculative ``n_routed_experts`` fallback --
        # it would be dead code and violates the "avoid getattr" rule.
        # A missing attribute raises ``AttributeError`` naturally.
        num_experts = int(text_config.num_experts)
    return num_layers, num_experts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-dump", required=True, help="dir with step_1_pp*.jsonl")
    parser.add_argument("--src-ckpt", required=True, help="original HF checkpoint dir")
    parser.add_argument("--dst-ckpt", required=True, help="output permuted HF checkpoint dir")
    parser.add_argument("--work-dir", required=True, help="dir for routing_map and imbalance curve")
    parser.add_argument("--out-dir", default=None, help="dir for visualization outputs")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="override config.json num_hidden_layers",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="override config.json num_experts",
    )
    parser.add_argument("--ep-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument(
        "--no-verify-full-forward",
        action="store_false",
        dest="verify_full_forward",
        help="skip full HF model forward comparison",
    )
    parser.set_defaults(verify_full_forward=True)
    parser.add_argument(
        "--sample-type",
        type=str,
        default="text",
        choices=["text", "image", "video", "mix"],
        help="sample type for full-forward verification",
    )
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--force-refresh-aux", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_moe_offline_repermute(
        MoeOfflineRepermuteConfig(
            source_dump_dir=args.source_dump,
            src_ckpt_dir=args.src_ckpt,
            dst_ckpt_dir=args.dst_ckpt,
            work_dir=args.work_dir,
            out_dir=args.out_dir,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            ep_sizes=tuple(args.ep_sizes),
            verify_full_forward=args.verify_full_forward,
            sample_type=args.sample_type,
            image_path=args.image_path,
            video_path=args.video_path,
            force_refresh_aux=args.force_refresh_aux,
        )
    )
    print(f"[pipeline] done. routing_map={result.routing_map_path}")
    print(f"[pipeline] done. dst_ckpt={result.dst_ckpt_dir}")
    if result.viz_out_dir is not None:
        print(f"[pipeline] done. viz={result.viz_out_dir}")


if __name__ == "__main__":
    main()
