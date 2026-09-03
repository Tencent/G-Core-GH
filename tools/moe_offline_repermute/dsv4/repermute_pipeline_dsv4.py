# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""End-to-end offline expert re-permutation pipeline for DeepSeek-V4-Flash.

Chains the three DSV4 stages: build routing map -> rewrite checkpoint (dry-run
then real) -> verify equivalence.

Usage (run from ``gcore-dev/``)::

    python -m tools.moe_offline_repermute.dsv4.repermute_pipeline_dsv4 \\
        --config-path /abs/path/to/yaml --config-name your_config.yaml \\
        +counts=debug-tmp/debug_dsv4_router/counts.pt \\
        +dst_ckpt=moe_repermute/DeepSeek-V4-Flash-repermuted \\
        +work_dir=moe_repermute
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import hydra
from hydra.utils import get_original_cwd
from transformers import DeepseekV4Config

from .build_routing_map import build_routing_map
from .repermute_hf_ckpt import repermute_hf_ckpt
from .verify_equivalence_dsv4 import verify_equivalence_dsv4
from .verify_full_forward import run_forward_equivalence
from ..viz_repermute import visualize_repermute

_EP_SIZES = (2, 4, 8, 16)


@dataclass
class Dsv4RepermuteConfig:
    counts_path: str
    src_ckpt_dir: str
    dst_ckpt_dir: str
    work_dir: str
    enable_mtp: bool = False
    ep_sizes: tuple = _EP_SIZES
    debug_truncate_num_hidden_layers: Optional[int] = None
    verify: bool = True
    verify_full_forward: bool = True
    force_refresh_aux: bool = True


@dataclass
class Dsv4RepermuteResult:
    routing_map_path: str
    dst_ckpt_dir: str


def _truncate_config(hf_config: DeepseekV4Config, config: Dsv4RepermuteConfig) -> DeepseekV4Config:
    nl = config.debug_truncate_num_hidden_layers
    if nl is not None:
        hf_config.num_hidden_layers = nl
        hf_config.layer_types = hf_config.layer_types[:nl]
        hf_config.mlp_layer_types = hf_config.mlp_layer_types[:nl]
    if not config.enable_mtp:
        hf_config.num_nextn_predict_layers = 0


def run_dsv4_repermute(config: Dsv4RepermuteConfig) -> Dsv4RepermuteResult:
    """Run build -> rewrite -> verify."""
    routing_map_path = os.path.join(config.work_dir, "routing_map.json")
    imbalance_curve_path = os.path.join(config.work_dir, "imbalance_curve_dsv4.png")
    hf_config = DeepseekV4Config.from_pretrained(config.src_ckpt_dir)
    _truncate_config(hf_config, config)

    print("[pipeline_dsv4] step=build_routing_map")
    build_routing_map(
        counts_path=config.counts_path,
        hf_config=hf_config,
        output=routing_map_path,
        imbalance_curve=imbalance_curve_path,
        enable_mtp=config.enable_mtp,
        ep_sizes=config.ep_sizes,
    )

    print("[pipeline_dsv4] step=repermute_hf_ckpt dry_run")
    repermute_hf_ckpt(
        src_dir=config.src_ckpt_dir,
        dst_dir=config.dst_ckpt_dir,
        routing_map_path=routing_map_path,
        dry_run=True,
        force_refresh_aux=config.force_refresh_aux,
    )

    print("[pipeline_dsv4] step=repermute_hf_ckpt rewrite")
    phantom_keys = repermute_hf_ckpt(
        src_dir=config.src_ckpt_dir,
        dst_dir=config.dst_ckpt_dir,
        routing_map_path=routing_map_path,
        dry_run=False,
        force_refresh_aux=config.force_refresh_aux,
    )

    if config.verify:
        print("[pipeline_dsv4] step=verify_equivalence_dsv4")
        verify_equivalence_dsv4(
            src_dir=config.src_ckpt_dir,
            dst_dir=config.dst_ckpt_dir,
            routing_map_path=routing_map_path,
            hf_config=hf_config,
            phantom_keys=phantom_keys,
        )

    if config.verify_full_forward:
        print("[pipeline_dsv4] step=verify_full_forward")
        run_forward_equivalence(
            src_ckpt=config.src_ckpt_dir,
            dst_ckpt=config.dst_ckpt_dir,
            hf_config=hf_config,
            enable_mtp=config.enable_mtp,
        )

    print("[pipeline_dsv4] step=visualize_repermute")
    visualize_repermute(
        routing_map_path=routing_map_path,
        out_dir=config.work_dir,
        ep_sizes=config.ep_sizes,
    )

    return Dsv4RepermuteResult(
        routing_map_path=routing_map_path,
        dst_ckpt_dir=config.dst_ckpt_dir,
    )


def _resolve(path: str, base: str) -> str:
    """Absolutize `path` against `base`."""
    return path if os.path.isabs(path) else os.path.join(base, path)


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg) -> None:
    base = get_original_cwd()
    counts_path = cfg['counts']
    dst_ckpt = cfg['dst_ckpt']
    work_dir = cfg['work_dir']
    src_ckpt = cfg['policy']['hf_model_path']
    try:
        enable_mtp = cfg['training']['enable_mtp']
    except:
        enable_mtp = False
    try:
        debug_truncate_num_hidden_layers = int(cfg['debug']['debug_truncate_num_hidden_layers'])
    except:
        debug_truncate_num_hidden_layers = None

    counts_path = _resolve(counts_path, base)
    dst_ckpt = _resolve(dst_ckpt, base)
    work_dir = _resolve(work_dir, base)
    src_ckpt = _resolve(src_ckpt, base)
    os.makedirs(work_dir, exist_ok=True)


    result = run_dsv4_repermute(
        Dsv4RepermuteConfig(
            counts_path=counts_path,
            src_ckpt_dir=src_ckpt,
            dst_ckpt_dir=dst_ckpt,
            work_dir=work_dir,
            enable_mtp=enable_mtp,
            ep_sizes=tuple(cfg['ep_sizes']),
            debug_truncate_num_hidden_layers=debug_truncate_num_hidden_layers,
            verify=cfg['verify'],
            verify_full_forward=cfg['verify_full_forward'],
            force_refresh_aux=cfg['force_refresh_aux']
        )
    )
    print(f"[pipeline_dsv4] done. routing_map={result.routing_map_path}")
    print(f"[pipeline_dsv4] done. dst_ckpt={result.dst_ckpt_dir}")


if __name__ == "__main__":
    main()
