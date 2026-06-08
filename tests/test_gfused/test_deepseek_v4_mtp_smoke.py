# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""MTP finetune smoke test：truncated DSV4-Flash + 1 MTP depth on a small cluster.

Usage:

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=1200 tests/test_gfused/test_deepseek_v4_mtp_smoke.py
"""

from __future__ import annotations

import math
import os
import time
import unittest

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, DeepseekV4Config

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.models.deepseek_v4 import DeepseekV4ForCausalLM, apply_hp
from gpatch_v4.training_backend.fsdp2_backend.mtp_loss import calculate_mtp_loss

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"

# Smallest viable smoke config:
#   * 4 backbone layers + 1 MTP depth (DSV4-Flash ships with
#     num_nextn_predict_layers=1, which we keep as-is).
#   * EP=2 → at least 2 GPUs needed; world_size=4 leaves an FSDP shard
#     dim of 2 on top of EP=2, which exercises both meshes.
NUM_GPUS = 32
EP_SIZE = 4
NUM_BACKBONE_LAYERS = 4
SEQ_LEN = 128


def _truncate_config(config: DeepseekV4Config) -> DeepseekV4Config:
    """Truncate the backbone to ``NUM_BACKBONE_LAYERS`` layers.

    Mirrors ``_truncate_config`` in
    ``tests/test_gfused/test_deepseek_v4_ep_cp.py`` and the same helper in
    ``gpatch_v4/training_backend/fsdp2_backend/mixin.py`` (debug truncate
    path). ``num_nextn_predict_layers`` is intentionally untouched — the
    whole point of this test is to keep MTP in the loop while everything
    else shrinks.
    """
    assert config.layer_types is not None
    assert config.mlp_layer_types is not None
    config.num_hidden_layers = NUM_BACKBONE_LAYERS
    config.layer_types = config.layer_types[:NUM_BACKBONE_LAYERS]
    config.mlp_layer_types = config.mlp_layer_types[:NUM_BACKBONE_LAYERS]
    return config


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip() -> str:
    return ray.util.get_node_ip_address()


def _param_norms(model: nn.Module, ep_group) -> dict[str, float]:
    """Return per-parameter global norm; expert grads are EP-all-reduced.

    Mirrors ``_compute_global_param_norms`` in test_deepseek_v4_ep_cp.py;
    we keep the EP aggregation rule even for non-expert params (no-op,
    keeps the helper local to this file).
    """
    norms: dict[str, float] = {}
    for name, p in model.named_parameters():
        t = p.detach()
        if isinstance(t, DTensor):
            t = t.full_tensor()
        local_norm_sq = t.float().norm() ** 2
        if ".experts." in name and ep_group is not None:
            dist.all_reduce(local_norm_sq, group=ep_group)
        norms[name] = local_norm_sq.sqrt().item()
    return norms


def _grad_norms(model: nn.Module, ep_group) -> dict[str, float]:
    """Return per-parameter global grad norm; expert grads are EP-all-reduced."""
    norms: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if isinstance(g, DTensor):
            g = g.full_tensor()
        local_norm_sq = g.float().norm() ** 2
        if ".experts." in name and ep_group is not None:
            dist.all_reduce(local_norm_sq, group=ep_group)
        norms[name] = local_norm_sq.sqrt().item()
    return norms


@ray.remote(num_gpus=1)
def _mtp_smoke_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    seq_len: int = SEQ_LEN,
) -> dict:
    """Build truncated DSV4 (4 backbone + 1 MTP) on meta, load weights,
    run one fwd / MTP-loss bwd / step, and return MTP-focused diagnostics.

    The worker is intentionally minimal — no router replay, no CP, no
    baseline. The aim is to exercise the MTP load / forward / backward
    path with the smallest sensible GPU budget.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    assert world_size % ep_size == 0, (
        f"world_size {world_size} not divisible by ep_size {ep_size}"
    )
    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rng = torch.Generator().manual_seed(42 + rank)
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, seq_len), generator=rng,
    ).cuda()

    print(f"[mtp_smoke] rank {rank}: loading {hf_model_path} via meta-device ...")
    _t_load_start = time.time()

    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
    assert config.num_nextn_predict_layers > 0, (
        f"DSV4-Flash config expected num_nextn_predict_layers > 0, "
        f"got {config.num_nextn_predict_layers}; this smoke needs an MTP-enabled checkpoint."
    )
    # mtp_loss_scaling_factor is read by the model __init__ via
    # getattr(config, "mtp_loss_scaling_factor", 0.1); set it explicitly to
    # mirror the mixin's enable_mtp branch (fsdp2_backend/mixin.py:300-302).
    config.mtp_loss_scaling_factor = 0.1
    assert not config.tie_word_embeddings, (
        "meta-device init may hang with tie_word_embeddings=True; "
        "see fsdp2_backend/mixin.py:111"
    )

    dist.barrier()

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    # Sanity: MTP submodule must exist before load. If this fails the
    # cfg.num_nextn_predict_layers branch never triggered.
    assert model.mtp is not None, (
        f"model.mtp is None after construction with "
        f"num_nextn_predict_layers={config.num_nextn_predict_layers}; "
        f"MTP submodule was not built."
    )
    assert len(model.mtp.layers) == config.num_nextn_predict_layers, (
        f"mtp.layers count {len(model.mtp.layers)} != "
        f"num_nextn_predict_layers {config.num_nextn_predict_layers}"
    )

    model = apply_hp(model, ep_2d_mesh, cp_mesh=None, amp_fp32=False)
    model.gradient_checkpointing_enable()
    _t_apply_hp_done = time.time()
    model.load_checkpoint_hp(hf_model_path)
    _t_load_end = time.time()
    apply_hp_seconds = _t_apply_hp_done - _t_load_start
    load_state_seconds = _t_load_end - _t_apply_hp_done
    print(
        f"[mtp_smoke] rank {rank}: load done in "
        f"{_t_load_end - _t_load_start:.1f}s "
        f"(meta+apply_hp={apply_hp_seconds:.1f}s, "
        f"load_checkpoint_hp={load_state_seconds:.1f}s)"
    )

    # ----- Assertion 1: every mtp param is materialized (no meta) -----
    mtp_meta_params = [
        n for n, p in model.named_parameters()
        if n.startswith("mtp.") and p.is_meta
    ]
    assert not mtp_meta_params, (
        f"[rank {rank}] MTP params still on meta after load: {mtp_meta_params}"
    )
    mtp_param_count = sum(
        1 for n, _ in model.named_parameters() if n.startswith("mtp.")
    )
    assert mtp_param_count > 0, (
        f"[rank {rank}] no mtp.* params found after load; MTP submodule "
        f"was built but its parameters did not get registered."
    )

    # ----- Forward in train() mode (use_mtp gate requires self.training) -----
    model.train()
    torch.cuda.reset_peak_memory_stats()

    # Match the trainer's label-shift convention (see fsdp2_backend/mixin.py
    # ~ line 540 and mtp_loss.py docstring): MTP loss expects already-shifted
    # next-token labels.
    full_labels = input_ids.clone()
    full_labels[full_labels == tokenizer.pad_token_id] = -100
    full_labels = torch.roll(full_labels, shifts=-1, dims=-1)
    full_labels[:, -1] = -100

    outputs = model(input_ids=input_ids)
    assert outputs.mtp_per_depth_h is not None, (
        "model forward in train() mode returned mtp_per_depth_h=None; "
        "MTP path did not fire (check use_mtp gate)."
    )
    assert len(outputs.mtp_per_depth_h) == config.num_nextn_predict_layers, (
        f"mtp_per_depth_h has {len(outputs.mtp_per_depth_h)} entries, "
        f"expected num_nextn_predict_layers={config.num_nextn_predict_layers}"
    )
    for d, h in enumerate(outputs.mtp_per_depth_h):
        assert torch.isfinite(h).all(), (
            f"mtp_per_depth_h[{d}] contains NaN/Inf "
            f"(shape={list(h.shape)}, dtype={h.dtype})"
        )

    # ----- Compute MTP loss only (we are testing MTP, not the main LM) -----
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    mtp_total_loss, mtp_per_depth_loss, _, _ = calculate_mtp_loss(
        mtp_per_depth_h=outputs.mtp_per_depth_h,
        labels=full_labels,
        lm_head=model.lm_head,
        loss_fct=loss_fct,
        loss_mask=None,
        scaling_factor=float(config.mtp_loss_scaling_factor),
    )
    assert torch.isfinite(mtp_total_loss), (
        f"MTP total loss not finite: {mtp_total_loss.item()}"
    )

    # ----- Snapshot MTP param norms before bwd/step -----
    norms_before = _param_norms(model, model._ep_group)
    mtp_norms_before = {n: v for n, v in norms_before.items() if n.startswith("mtp.")}

    mtp_total_loss.backward()

    # ----- Assertion 2: every trainable mtp param got a non-zero grad -----
    grad_norms = _grad_norms(model, model._ep_group)
    mtp_grad_norms = {n: v for n, v in grad_norms.items() if n.startswith("mtp.")}
    missing_grad = [
        n for n, p in model.named_parameters()
        if n.startswith("mtp.") and p.requires_grad and p.grad is None
    ]
    assert not missing_grad, (
        f"[rank {rank}] MTP params with requires_grad=True but grad=None: "
        f"{missing_grad}"
    )
    zero_grad_norm = [
        n for n, v in mtp_grad_norms.items() if v <= 0.0 or math.isnan(v)
    ]
    assert not zero_grad_norm, (
        f"[rank {rank}] MTP params with zero / NaN grad norm: {zero_grad_norm}"
    )

    # ----- Assertion 3: optimizer.step actually moves MTP params -----
    # Use clip_grad_norm_ to mirror the real trainer (handles multi-mesh).
    total_grad_norm = model.clip_grad_norm_(2.0)
    has_nan = math.isnan(total_grad_norm)
    assert not has_nan, f"[rank {rank}] total grad norm is NaN"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()
    norms_after = _param_norms(model, model._ep_group)
    mtp_norms_after = {n: v for n, v in norms_after.items() if n.startswith("mtp.")}

    moved = [
        n for n in mtp_norms_before
        if abs(mtp_norms_after[n] - mtp_norms_before[n]) > 0.0
    ]
    assert moved, (
        f"[rank {rank}] optimizer.step did not change any MTP parameter "
        f"(before={mtp_norms_before}, after={mtp_norms_after})"
    )

    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[mtp_smoke] rank {rank}: PASS — "
        f"mtp_total_loss={mtp_total_loss.item():.4f}, "
        f"per_depth_loss={[round(x.item(), 4) for x in mtp_per_depth_loss]}, "
        f"total_grad_norm={total_grad_norm:.4f}, "
        f"#mtp_params_moved={len(moved)}/{len(mtp_norms_before)}, "
        f"mem_peak={mem_peak:.2f} GiB"
    )

    result = {
        "rank": rank,
        "mtp_total_loss": mtp_total_loss.item(),
        "mtp_per_depth_loss": [x.item() for x in mtp_per_depth_loss],
        "total_grad_norm": total_grad_norm,
        "mtp_param_count": mtp_param_count,
        "mtp_params_moved": len(moved),
        "mtp_params_total_tracked": len(mtp_norms_before),
        "mem_peak_gib": mem_peak,
        "apply_hp_seconds": apply_hp_seconds,
        "load_state_seconds": load_state_seconds,
    }

    del model, outputs, optimizer
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestDeepseekV4MtpSmoke(unittest.TestCase):
    """MTP finetune smoke — truncated DSV4 + 1 MTP depth on >= 4 GPUs."""

    def setUp(self):
        # Ensure Ray workers can import test modules and gcore-dev packages
        # by propagating PYTHONPATH via runtime_env. This mirrors the export
        # block in the Usage docstring and works even when the Ray head was
        # started without these paths.
        import sys
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        if not env_pythonpath:
            # Fallback: construct from the known workspace layout.
            rcdir = "/work/wepsdl"
            env_pythonpath = ":".join([
                f"{rcdir}/gcore-dev",
                f"{rcdir}/gcore-dev/tests",
                f"{rcdir}/gcore-dev/tests/test_gpatch_v4",
                f"{rcdir}/Megatron-LM",
                f"{rcdir}/mbridge",
                f"{rcdir}/Megatron-Bridge/src",
            ])
        ray.init(
            address="auto",
            runtime_env={"env_vars": {"PYTHONPATH": env_pythonpath}},
        )
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {NUM_GPUS} GPUs in Ray cluster, only {total_gpus}"
            )

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def test_mtp_load_forward_backward_step(
        self,
        world_size: int = NUM_GPUS,
        ep_size: int = EP_SIZE,
        master_port: int = 12700,
    ):
        """End-to-end MTP smoke: load → forward → MTP loss bwd → step.

        Passes iff every MTP assertion in ``_mtp_smoke_worker`` holds on
        every rank. The per-rank prints + the
        ``[load_checkpoint_hp] MTP routing`` / ``MTP materialized`` lines
        from ``_load_checkpoint_hp`` are the human-readable trace.
        """
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}; "
            f"download DeepSeek-V4-Flash into hf-hub/ first"
        )
        assert world_size % ep_size == 0, (
            f"world_size ({world_size}) must be divisible by ep_size ({ep_size})"
        )

        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
        )
        ray.get(pg.ready())

        try:
            master_addr = ray.get(
                _get_node_ip.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=0,
                    )
                ).remote()
            )

            print("=" * 60)
            print(
                f"Running MTP smoke "
                f"(world_size={world_size}, ep_size={ep_size}, "
                f"backbone_layers={NUM_BACKBONE_LAYERS}, seq_len={SEQ_LEN}) ..."
            )
            futures = [
                _mtp_smoke_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=r,
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    ep_size=ep_size,
                )
                for r in range(world_size)
            ]
            results = ray.get(futures)
        finally:
            remove_placement_group(pg)

        # --- Summary ---
        print("\n--- MTP smoke summary ---")
        for res in results:
            r = res["rank"]
            print(
                f"  rank {r}: mtp_total_loss={res['mtp_total_loss']:.4f}, "
                f"per_depth={res['mtp_per_depth_loss']}, "
                f"total_grad_norm={res['total_grad_norm']:.4f}, "
                f"#mtp_params_moved={res['mtp_params_moved']}/"
                f"{res['mtp_params_total_tracked']}, "
                f"mem_peak={res['mem_peak_gib']:.2f} GiB, "
                f"load: meta+hp={res['apply_hp_seconds']:.1f}s + "
                f"weights={res['load_state_seconds']:.1f}s"
            )

        # --- Cross-rank consistency: all ranks should see the same MTP
        # param count (FSDP names are global). ---
        counts = {r["mtp_param_count"] for r in results}
        self.assertEqual(
            len(counts), 1,
            f"MTP param count diverges across ranks: {counts}",
        )

        # --- Every rank must report at least one MTP param moved by step. ---
        for res in results:
            self.assertGreater(
                res["mtp_params_moved"], 0,
                f"rank {res['rank']}: optimizer.step moved zero MTP params",
            )

        print("\nPASSED")


if __name__ == "__main__":
    unittest.main()
