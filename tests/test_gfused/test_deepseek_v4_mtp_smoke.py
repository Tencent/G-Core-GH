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
    for _bi, _blk in enumerate(model.mtp.layers):
        assert _blk.gradient_checkpointing, (
            f"MTP block {_bi} gradient_checkpointing not enabled after "
            f"gradient_checkpointing_enable(); "
            f"DeepseekV4MTPBlock must inherit GradientCheckpointingLayer"
        )
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

    # 计数 MTP block 真实进入 forward 的次数：正常 forward 一次，checkpoint backward 再重算一次。
    mtp_forward_counts = [0 for _ in model.mtp.layers]
    mtp_forward_hook_handles = []
    for i, block in enumerate(model.mtp.layers):
        def _count_mtp_forward(_module, _args, i=i):
            mtp_forward_counts[i] += 1

        mtp_forward_hook_handles.append(block.register_forward_pre_hook(_count_mtp_forward))

    outputs = model(input_ids=input_ids)
    # forward 结束时每个 MTP block 应只跑过一次。
    expected_forward_counts = [1] * len(model.mtp.layers)
    assert mtp_forward_counts == expected_forward_counts, (
        f"MTP block forward counts after forward = {mtp_forward_counts}, "
        f"expected {expected_forward_counts}"
    )
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
    mtp_depth_nums = calculate_mtp_loss(
        mtp_per_depth_h=outputs.mtp_per_depth_h,
        labels=full_labels,
        lm_head=model.lm_head,
        loss_fct=loss_fct,
        loss_mask=None,
    )
    # Build per-depth loss and total from numerators (single rank, no CP,
    # so local valid-token count == global denominator).
    mtp_per_depth_loss = []
    for num in mtp_depth_nums:
        den = (full_labels != -100).sum().float().clamp_min(1.0)
        mtp_per_depth_loss.append(num / den)
    mtp_scale = float(config.mtp_loss_scaling_factor)
    mtp_total_loss = (
        torch.stack(mtp_per_depth_loss).sum()
        * (mtp_scale / max(len(mtp_per_depth_loss), 1))
    )
    assert torch.isfinite(mtp_total_loss), (
        f"MTP total loss not finite: {mtp_total_loss.item()}"
    )

    # ----- Snapshot MTP param norms before bwd/step -----
    norms_before = _param_norms(model, model._ep_group)
    mtp_norms_before = {n: v for n, v in norms_before.items() if n.startswith("mtp.")}

    mtp_total_loss.backward()
    # backward 若触发 activation checkpoint，每个 MTP block 会再进入一次 forward 重算。
    expected_recompute_counts = [2] * len(model.mtp.layers)
    assert mtp_forward_counts == expected_recompute_counts, (
        f"MTP block forward counts after backward = {mtp_forward_counts}, "
        f"expected {expected_recompute_counts}; MTP was not recomputed during backward"
    )
    for handle in mtp_forward_hook_handles:
        handle.remove()

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


def _make_fake_segments(fake_seq_lens, tokenizer, device, seed=42):
    """Generate deterministic fake segments (shared by BSHD and THD workers)."""
    pad_id = tokenizer.pad_token_id
    vocab = tokenizer.vocab_size
    ids_list, labels_list = [], []
    for i, s in enumerate(fake_seq_lens):
        rng = torch.Generator().manual_seed(seed + i)
        ids = torch.randint(0, vocab, (s,), generator=rng).to(device)
        ids = torch.where(ids == pad_id, (ids + 1) % vocab, ids)
        lab = torch.roll(ids, shifts=-1)
        lab[-1] = -100
        ids_list.append(ids)
        labels_list.append(lab)
    return ids_list, labels_list


def _build_model_for_mtp(hf_model_path, ep_2d_mesh, cp_mesh=None):
    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
    config.mtp_loss_scaling_factor = 0.1
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)
    model = apply_hp(model, ep_2d_mesh, cp_mesh=cp_mesh, amp_fp32=False)
    model.gradient_checkpointing_enable()
    model.load_checkpoint_hp(hf_model_path)
    model.train()
    return model, config


@ray.remote(num_gpus=1)
def _mtp_bshd_perseg_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    fake_seq_lens: list[int],
    seed: int = 42,
) -> dict:
    """BSHD baseline: per-segment independent forward, accumulate MTP loss.

    Captures per-segment routing decisions and concatenates them into
    ``[T_TOTAL, top_k]`` per router layer for THD replay.
    """
    from gpatch_v4.models.deepseek_v4.router_replay import capture_routing_decisions

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ids_list, labels_list = _make_fake_segments(
        fake_seq_lens, tokenizer, torch.device("cuda"), seed,
    )
    model, config = _build_model_for_mtp(hf_model_path, ep_2d_mesh)
    pad_mul = max(config.compress_rates.values())
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    mtp_scale = float(config.mtp_loss_scaling_factor)

    total_n_valid = torch.zeros((), device="cuda")
    for lab in labels_list:
        masked_lab = lab.clone()
        masked_lab[-1] = -100
        total_n_valid += (masked_lab != -100).sum()

    accum_depth_loss_vals: list[float] = []
    reported_loss = 0.0
    per_seg_routing: list[list[torch.Tensor]] = []
    for seg_i, (ids, lab) in enumerate(zip(ids_list, labels_list)):
        s = ids.shape[0]
        s_padded = ((s + pad_mul - 1) // pad_mul) * pad_mul
        seg_ids = torch.full((1, s_padded), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
        seg_labels = torch.full((1, s_padded), -100, dtype=torch.long, device="cuda")
        seg_ids[0, :s] = ids
        seg_labels[0, :s] = lab
        seg_labels[0, s - 1] = -100

        with capture_routing_decisions(model) as recorded:
            outputs = model(input_ids=seg_ids)
        assert outputs.mtp_per_depth_h is not None
        per_seg_routing.append([t.detach().cpu() for t in recorded if t is not None])

        depth_nums = calculate_mtp_loss(
            mtp_per_depth_h=outputs.mtp_per_depth_h,
            labels=seg_labels,
            lm_head=model.lm_head,
            loss_fct=loss_fct,
            loss_mask=None,
            cp_group=None,
        )
        seg_depth_losses = [num / total_n_valid for num in depth_nums]
        seg_mtp_loss = (
            torch.stack(seg_depth_losses).sum()
            * (mtp_scale / max(len(seg_depth_losses), 1))
        )
        seg_mtp_loss.backward()
        reported_loss += seg_mtp_loss.item()
        if not accum_depth_loss_vals:
            accum_depth_loss_vals = [d.item() for d in seg_depth_losses]
        else:
            for d_i, d in enumerate(seg_depth_losses):
                accum_depth_loss_vals[d_i] += d.item()

    total_grad_norm = model.clip_grad_norm_(2.0)

    # cat 段→T 维度：[seg][router_layer] → [router_layer][T_TOTAL, top_k]
    n_router_layers = len(per_seg_routing[0])
    recorded_routing = []
    for layer_i in range(n_router_layers):
        recorded_routing.append(
            torch.cat([per_seg_routing[s][layer_i] for s in range(len(per_seg_routing))], dim=0)
        )

    result = {
        "rank": rank,
        "mtp_total_loss": reported_loss,
        "mtp_per_depth_loss": accum_depth_loss_vals,
        "total_grad_norm": total_grad_norm,
        "recorded_routing": recorded_routing,
    }
    print(
        f"[mtp_bshd_perseg] rank {rank}: mtp_total_loss={reported_loss:.6f}, "
        f"total_grad_norm={total_grad_norm:.4f}, "
        f"n_router_layers={n_router_layers}"
    )
    del model, outputs
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


@ray.remote(num_gpus=1)
def _mtp_thd_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    fake_seq_lens: list[int],
    seed: int = 42,
    replay_indices: list[torch.Tensor] | None = None,
) -> dict:
    """MTP + THD pack-seq: pack multiple segments, run forward + MTP loss bwd.

    When ``replay_indices`` is provided (``[router_layer][T_TOTAL, top_k]``),
    forces the same routing decisions as the BSHD baseline.
    """
    from contextlib import nullcontext

    from gpatch_v4.models.deepseek_v4.router_replay import router_replay_ctx
    from gpatch_v4.models.deepseek_v4.thd import pack_sequences

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ids_list, labels_list = _make_fake_segments(
        fake_seq_lens, tokenizer, torch.device("cuda"), seed,
    )
    model, config = _build_model_for_mtp(hf_model_path, ep_2d_mesh)
    pad_mul = max(config.compress_rates.values())

    packed_ids, packed_pos, packed_labels, psp = pack_sequences(
        ids_list, labels_list,
        config=config,
        pad_to_multiple_of=pad_mul,
        cp_size=1,
        pad_token_id=tokenizer.pad_token_id,
        label_ignore_index=-100,
    )

    if replay_indices is not None and len(replay_indices) > 0:
        cuda_replay = [t.cuda() for t in replay_indices]
        ctx = router_replay_ctx(model, cuda_replay)
    else:
        ctx = nullcontext()

    with ctx:
        outputs = model(
            input_ids=packed_ids,
            position_ids=packed_pos,
            packed_seq_params=psp,
        )
        assert outputs.mtp_per_depth_h is not None

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        mtp_scale = float(config.mtp_loss_scaling_factor)
        n_valid = (packed_labels != -100).sum().float().clamp_min(1.0)

        mtp_depth_nums = calculate_mtp_loss(
            mtp_per_depth_h=outputs.mtp_per_depth_h,
            labels=packed_labels,
            lm_head=model.lm_head,
            loss_fct=loss_fct,
            loss_mask=None,
            cp_group=None,
            packed_seq_params=psp,
        )
        mtp_per_depth_loss = [num / n_valid for num in mtp_depth_nums]
        mtp_total_loss = (
            torch.stack(mtp_per_depth_loss).sum()
            * (mtp_scale / max(len(mtp_per_depth_loss), 1))
        )
        mtp_total_loss.backward()
    total_grad_norm = model.clip_grad_norm_(2.0)

    result = {
        "rank": rank,
        "mtp_total_loss": mtp_total_loss.item(),
        "mtp_per_depth_loss": [x.item() for x in mtp_per_depth_loss],
        "total_grad_norm": total_grad_norm,
    }
    print(
        f"[mtp_thd] rank {rank}: mtp_total_loss={mtp_total_loss.item():.6f}, "
        f"total_grad_norm={total_grad_norm:.4f}"
    )
    del model, outputs
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


CP_SIZE_FOR_MTP_CP_TEST = 4


@ray.remote(num_gpus=1)
def _mtp_thd_cp_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    cp_size: int,
    fake_seq_lens: list[int],
    seed: int = 42,
    replay_indices: list[torch.Tensor] | None = None,
) -> dict:
    """MTP + THD + CP: pack segments, CP-chunk, run forward + MTP loss bwd."""
    from contextlib import nullcontext

    from gpatch_v4.models.deepseek_v4.cp import cp_chunk_data
    from gpatch_v4.models.deepseek_v4.router_replay import router_replay_ctx
    from gpatch_v4.models.deepseek_v4.thd import pack_sequences

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    cp_full_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // cp_size, cp_size),
        mesh_dim_names=("dp", "cp"),
    )
    cp_mesh = cp_full_mesh["cp"]

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ids_list, labels_list = _make_fake_segments(
        fake_seq_lens, tokenizer, torch.device("cuda"), seed,
    )
    model, config = _build_model_for_mtp(hf_model_path, ep_2d_mesh, cp_mesh=cp_mesh)
    pad_mul = max(config.compress_rates.values())

    packed_ids, packed_pos, packed_labels, psp = pack_sequences(
        ids_list, labels_list,
        config=config,
        pad_to_multiple_of=pad_mul,
        cp_size=cp_size,
        pad_token_id=tokenizer.pad_token_id,
        label_ignore_index=-100,
    )

    cp_rank = dist.get_rank() % cp_size
    local_ids, local_labels, _, local_pos, local_psp = cp_chunk_data(
        cp_rank, cp_size,
        tokens=packed_ids,
        labels=packed_labels,
        position_ids=packed_pos,
        packed_seq_params=psp,
    )

    if replay_indices is not None and len(replay_indices) > 0:
        s_local = local_ids.shape[1]
        cp_replay = [
            t[cp_rank * s_local:(cp_rank + 1) * s_local].cuda()
            for t in replay_indices
        ]
        ctx = router_replay_ctx(model, cp_replay)
    else:
        ctx = nullcontext()

    with ctx:
        outputs = model(
            input_ids=local_ids,
            position_ids=local_pos,
            packed_seq_params=local_psp,
        )
        assert outputs.mtp_per_depth_h is not None

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        mtp_scale = float(config.mtp_loss_scaling_factor)

        n_valid = (local_labels != -100).sum()
        dist.all_reduce(n_valid, group=model._cp_group)
        n_valid = n_valid.float().clamp_min(1.0)

        mtp_depth_nums = calculate_mtp_loss(
            mtp_per_depth_h=outputs.mtp_per_depth_h,
            labels=local_labels,
            lm_head=model.lm_head,
            loss_fct=loss_fct,
            loss_mask=None,
            cp_group=model._cp_group,
            packed_seq_params=local_psp,
        )
        mtp_per_depth_loss = [num / n_valid for num in mtp_depth_nums]
        mtp_total_loss = (
            torch.stack(mtp_per_depth_loss).sum()
            * (mtp_scale / max(len(mtp_per_depth_loss), 1))
        )
        reported_loss = mtp_total_loss.detach().clone()
        dist.all_reduce(reported_loss, group=model._cp_group)
        mtp_total_loss.backward()

    total_grad_norm = model.clip_grad_norm_(2.0)

    result = {
        "rank": rank,
        "mtp_total_loss": reported_loss.item(),
        "mtp_per_depth_loss": [x.item() for x in mtp_per_depth_loss],
        "total_grad_norm": total_grad_norm,
    }
    print(
        f"[mtp_thd_cp] rank {rank}: mtp_total_loss={reported_loss.item():.6f}, "
        f"total_grad_norm={total_grad_norm:.4f}"
    )
    del model, outputs
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

    def test_mtp_bshd_vs_thd(
        self,
        world_size: int = NUM_GPUS,
        ep_size: int = EP_SIZE,
        master_port: int = 12700,
    ):
        """BSHD vs THD pack-seq: MTP loss and grad should be close."""
        assert os.path.isdir(HF_MODEL_PATH)

        # 3 段不等长，pad 到 128 倍数
        fake_seq_lens = [100, 80, 120]

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

            # 1. BSHD baseline — 每段独立 forward，累加 MTP loss
            bshd_futures = [
                _mtp_bshd_perseg_worker.options(
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
                    fake_seq_lens=fake_seq_lens,
                )
                for r in range(world_size)
            ]
            bshd_results = ray.get(bshd_futures)
        finally:
            remove_placement_group(pg)

        self.tearDown()

        # 重启 Ray
        self.setUp()

        pg2 = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
        )
        ray.get(pg2.ready())

        try:
            master_addr2 = ray.get(
                _get_node_ip.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg2, placement_group_bundle_index=0,
                    )
                ).remote()
            )

            # 2. THD pack-seq with router replay from BSHD rank 0
            replay = bshd_results[0].get("recorded_routing")
            thd_futures = [
                _mtp_thd_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg2, placement_group_bundle_index=r,
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr2,
                    master_port=master_port + 1,
                    ep_size=ep_size,
                    fake_seq_lens=fake_seq_lens,
                    replay_indices=replay,
                )
                for r in range(world_size)
            ]
            thd_results = ray.get(thd_futures)
        finally:
            remove_placement_group(pg2)

        bshd_r0 = bshd_results[0]
        thd_r0 = thd_results[0]

        print("\n" + "=" * 60)
        print("  MTP: BSHD vs THD pack-seq")
        print("=" * 60)
        print(f"  BSHD: mtp_total_loss={bshd_r0['mtp_total_loss']:.6f}  grad_norm={bshd_r0['total_grad_norm']:.4f}")
        print(f"  THD:  mtp_total_loss={thd_r0['mtp_total_loss']:.6f}  grad_norm={thd_r0['total_grad_norm']:.4f}")

        loss_rel = abs(bshd_r0["mtp_total_loss"] - thd_r0["mtp_total_loss"]) / (
            abs(bshd_r0["mtp_total_loss"]) + 1e-8
        )
        gn_rel = abs(bshd_r0["total_grad_norm"] - thd_r0["total_grad_norm"]) / (
            abs(bshd_r0["total_grad_norm"]) + 1e-8
        )
        print(f"  loss rel_diff={loss_rel:.6f}  grad_norm rel_diff={gn_rel:.6f}")
        print("=" * 60 + "\n")

        self.assertLess(loss_rel, 0.005, f"MTP loss rel_diff {loss_rel:.6f} > 0.5%")
        self.assertLess(gn_rel, 0.01, f"MTP grad_norm rel_diff {gn_rel:.6f} > 1%")

    def test_mtp_thd_vs_thd_cp(
        self,
        world_size: int = NUM_GPUS,
        ep_size: int = EP_SIZE,
        cp_size: int = CP_SIZE_FOR_MTP_CP_TEST,
        master_port: int = 12700,
    ):
        """THD CP=1 vs THD CP=4: MTP loss and grad should be close."""
        assert os.path.isdir(HF_MODEL_PATH)

        # segments must pad to multiples of 128 AND total T divisible by cp_size*128
        fake_seq_lens = [100, 200, 100]

        # 1. THD CP=1 (baseline, with router capture)
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
            thd_futures = [
                _mtp_bshd_perseg_worker.options(
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
                    fake_seq_lens=fake_seq_lens,
                )
                for r in range(world_size)
            ]
            thd_results = ray.get(thd_futures)
        finally:
            remove_placement_group(pg)

        self.tearDown()
        self.setUp()

        # 2. THD CP=4 (with router replay from baseline rank 0)
        pg2 = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
        )
        ray.get(pg2.ready())

        try:
            master_addr2 = ray.get(
                _get_node_ip.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg2, placement_group_bundle_index=0,
                    )
                ).remote()
            )
            replay = thd_results[0].get("recorded_routing")
            cp_futures = [
                _mtp_thd_cp_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg2, placement_group_bundle_index=r,
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr2,
                    master_port=master_port + 1,
                    ep_size=ep_size,
                    cp_size=cp_size,
                    fake_seq_lens=fake_seq_lens,
                    replay_indices=replay,
                )
                for r in range(world_size)
            ]
            cp_results = ray.get(cp_futures)
        finally:
            remove_placement_group(pg2)

        thd_r0 = thd_results[0]
        cp_r0 = cp_results[0]

        print("\n" + "=" * 60)
        print("  MTP: THD CP=1 vs THD CP=4")
        print("=" * 60)
        print(f"  THD:    mtp_total_loss={thd_r0['mtp_total_loss']:.6f}  grad_norm={thd_r0['total_grad_norm']:.4f}")
        print(f"  THD+CP: mtp_total_loss={cp_r0['mtp_total_loss']:.6f}  grad_norm={cp_r0['total_grad_norm']:.4f}")

        loss_rel = abs(thd_r0["mtp_total_loss"] - cp_r0["mtp_total_loss"]) / (
            abs(thd_r0["mtp_total_loss"]) + 1e-8
        )
        gn_rel = abs(thd_r0["total_grad_norm"] - cp_r0["total_grad_norm"]) / (
            abs(thd_r0["total_grad_norm"]) + 1e-8
        )
        print(f"  loss rel_diff={loss_rel:.6f}  grad_norm rel_diff={gn_rel:.6f}")
        print("=" * 60 + "\n")

        self.assertLess(loss_rel, 0.005, f"MTP loss rel_diff {loss_rel:.6f} > 0.5%")
        self.assertLess(gn_rel, 0.02, f"MTP grad_norm rel_diff {gn_rel:.6f} > 2%")


if __name__ == "__main__":
    unittest.main()
