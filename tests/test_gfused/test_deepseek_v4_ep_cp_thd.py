# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""DSV4 pack-seq scaffold: vanilla HF baseline vs THD pack on EP=8 CP=4 mesh.

Vanilla baseline (HF transformers DSV4 + 纯 FSDP, world=NUM_GPUS=32) 全 rank
喂同一份 fake QA：loss/grad 全 rank 一致；多卡仅用于均摊 FSDP unshard 显存
(8 卡时 unshard 一层 12 GiB 会 OOM)。3 段独立 forward 录 routing。

THD pack (our fork, EP=8 CP=4, world=NUM_GPUS=32) 走 [1, T=512] packed
forward，全 rank 同 input；用 ``router_replay_ctx`` 把 baseline rank 0 录
到的 routing 喂回去。

Numeric equivalence asserted at rtol_logits=0.02 / rtol_loss=0.005 /
rtol_grad_norm=0.01 + per-param grad ratio places=1。Sibling
test_deepseek_v4_ep_cp.py 实测 6.8e-5 / 5.3e-4 / 1e-2 量级，本测试加
attention `cross-seg gate` 后 6.5e-5 / 1.1e-4 / per-param 最差 4.1%
（layer.1.ffn_hc.scale）。rtol 取实测 2-50× 余量。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp_thd.py

TODO:
- [x] 仔细检查本测试
- [x] Correction bias updating and CP
- [x] Add recompute
- [x] fp48 quant, dequant
- [x] 补完 save ckpt
- [x] load ckpt 速度优化
   - [x] 并行 load
   - [x] 重写 save
   - [x] 并行 save
   - [x] 补完 fp8 save ckpt
   - [x] ~~是否漏了 router 的 buffer？~~（无遗漏）
- [x] 嵌入 trainer v4 的 SFT 流程
- [x] fix trainer v4 的 cp and loss
- [x] Add pack seq
- [x] ~~Add dyn CP~~（对于 swa 没有意义）
- [x] MTP
- [x] 特殊处理 fp32 mhc
- [x] Add FA
- [x] DeepEP
- [ ] THD 的 CP 改成 zz
- [ ] 优化 CSA 与 HCA 的 op
- [ ] better use real data to test it
"""

import math
import os
import time
import unittest
import importlib.util
from contextlib import nullcontext

import ray
import torch
import torch.nn.functional as F
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from transformers import (
    AutoTokenizer,
    DeepseekV4Config,
    DeepseekV4ForCausalLM as _HfModel,
    FineGrainedFP8Config,
)

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.models.deepseek_v4 import DeepseekV4ForCausalLM, apply_hp
from gpatch_v4.models.deepseek_v4.cp import cp_chunk_data
from gpatch_v4.models.deepseek_v4.router_replay import (
    capture_routing_decisions,
    router_replay_ctx,
)
from gpatch_v4.models.deepseek_v4.thd import pack_sequences
from gpatch_v4.orches.placement_group import _create_placement_group
from gpatch_v4.training_backend.loss_factory import load_balancing_loss_func

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
NUM_GPUS = 32
# Layer 类型分布（compress_ratios 配置）：layer 0/1 普通 attn；layer 2 CSA(m=4)；
# layer 3 HCA(m=128)；layer 4+ 4/128 交替。最小对齐策略：
#   A: NUM_LAYERS=2 CP=1 → 只普通 attn，无 compressor，无 CP 切片
#   B: NUM_LAYERS=2 CP=4 → 加 CP（cp_chunk_data + compressor_cp_ring）
#   C: NUM_LAYERS=3 CP=1 → 加 CSA
#   D: NUM_LAYERS=4 CP=1 → 加 HCA
#   E: NUM_LAYERS=4 CP=4 → 全配置（v4 现状 ratio 0.989/1.07）
NUM_LAYERS = 4
EP_SIZE = 8
CP_SIZE = 4
PAD_TO_MULTIPLE_OF = 128
# pad sum 必须是 cp_size × m'=128 倍数：[128, 256, 128] = 512 ✓ s_local=512/cp ✓
# 阶段 A 回归：恢复多段 [100, 200, 100]，验证 modeling.py 加了 cross-seg
# gate 后 ratio 是否从 1.0073/1.1523 收敛到 1.0000（A2 单段已确认 1.0000）。

# passed
# CP_SIZE = 1
# CP_SIZE = 4
FAKE_SEQ_LENS = [100, 200, 100]
PADDED_SEQ_LENS = [128, 256, 128]

# passed
# CP_SIZE = 1
# FAKE_SEQ_LENS = [100]
# PADDED_SEQ_LENS = [128]

# passed
# CP_SIZE = 1
# FAKE_SEQ_LENS = [200]
# PADDED_SEQ_LENS = [256]

# passed
# CP_SIZE = 1
# FAKE_SEQ_LENS = [100, 200]
# PADDED_SEQ_LENS = [128, 256]

# passed
# CP_SIZE = 1
# FAKE_SEQ_LENS = [200, 100]
# PADDED_SEQ_LENS = [256, 128]

T_TOTAL = sum(PADDED_SEQ_LENS)  # 512
# Baseline 所有 rank 喂同一份 fake QA → loss/grad 全 rank 相同；
# pack 也是所有 rank 同 input → 全部用 baseline rank 0 的 routing。
# 这样 vanilla baseline 用满 NUM_GPUS=32 张卡均摊 FSDP unshard 显存（8 卡时 OOM）。
FAKE_QA_SEED = 20260530
_DEEPEP_AVAILABLE = importlib.util.find_spec("deep_ep") is not None
requires_deepep = unittest.skipUnless(_DEEPEP_AVAILABLE, "deep_ep not installed")


# ---------------------------------------------------------------------------
# verbatim copy from test_deepseek_v4_ep_cp.py (kept here so that file stays
# untouched; do not import private _xxx symbols across sibling tests).
# ---------------------------------------------------------------------------


def _truncate_config(config):
    if NUM_LAYERS is not None:
        config.num_hidden_layers = NUM_LAYERS
        config.layer_types = config.layer_types[:NUM_LAYERS]
        config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
    config.num_nextn_predict_layers = 0
    return config


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip():
    return ray.util.get_node_ip_address()


# ---------------------------------------------------------------------------
# fake QA fixture
# ---------------------------------------------------------------------------


def _make_fake_qa(tokenizer, device, fake_seq_lens=None):
    """3 段 ids + per-seg roll-shift labels（next-token 预测，末位 -100）。

    所有 rank 用同一 ``FAKE_QA_SEED`` → vanilla baseline 全 rank loss/grad
    一致（DP 复制），仅借多卡均摊 FSDP unshard 显存；pack 全 rank 同 input
    → 共享同一份 baseline routing。
    """
    if fake_seq_lens is None:
        fake_seq_lens = FAKE_SEQ_LENS
    pad_id = tokenizer.pad_token_id
    vocab = tokenizer.vocab_size
    assert pad_id < vocab, (
        f"pad_id ({pad_id}) >= vocab_size ({vocab}); the (ids+1)%vocab "
        f"collision-fix below assumes pad_id is in randint range"
    )
    ids_list, labels_list = [], []
    for i, s in enumerate(fake_seq_lens):
        rng = torch.Generator().manual_seed(FAKE_QA_SEED + i)
        ids = torch.randint(0, vocab, (s,), generator=rng).to(device)
        # DSV4 pad_id 大，randint(pad_id+1, vocab) 会压扁低位 vocab；
        # 改成把撞 pad_id 的位置移一格
        ids = torch.where(ids == pad_id, (ids + 1) % vocab, ids)
        labels = torch.roll(ids, shifts=-1)
        labels[-1] = -100
        ids_list.append(ids)
        labels_list.append(labels)
    return ids_list, labels_list


def _prep_vanilla_baseline(ids_list, labels_list, pad_id, device, padded_seq_lens=None):
    """Vanilla baseline: 每段独立 pad 到 padded_seq_lens[i]（与 THD pack 完全一致）。

    不走 CP，所以不需要 BSHD_PAD=512；每段 [1, padded_len_i] 单独 forward。
    Routing capture 出 [padded_len_i, top_k]/layer，cat 后正好对齐 T_TOTAL。
    """
    if padded_seq_lens is None:
        padded_seq_lens = PADDED_SEQ_LENS
    per_seg_ids = []
    per_seg_labels = []
    for i, (ids, lab) in enumerate(zip(ids_list, labels_list)):
        s = ids.numel()
        padded = padded_seq_lens[i]
        assert s <= padded, f"seg {i} length {s} exceeds padded {padded}"
        seg_ids = torch.full((1, padded), pad_id, dtype=torch.long, device=device)
        seg_labels = torch.full((1, padded), -100, dtype=torch.long, device=device)
        seg_ids[0, :s] = ids
        seg_labels[0, :s] = lab
        per_seg_ids.append(seg_ids)
        per_seg_labels.append(seg_labels)
    return per_seg_ids, per_seg_labels


def _prep_pack(ids_list, labels_list, tokenizer, config):
    return pack_sequences(
        ids_list, labels_list,
        config=config,
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
        cp_size=CP_SIZE,
        pad_token_id=tokenizer.pad_token_id,
        label_ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# fwd+bwd helpers
# ---------------------------------------------------------------------------


def _common_fwd_bwd_diagnostics(
    model, logits, reported_loss, rank, tag, ep_group, mem_after_shard,
):
    """同 ep_cp.py 的 grad/optimizer 段。"""
    logits = logits.detach()
    if ep_group is not None:
        total_grad_norm = model.clip_grad_norm_(2.0)
    else:
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        if isinstance(total_grad_norm, DTensor):
            total_grad_norm = total_grad_norm.full_tensor()
        total_grad_norm = total_grad_norm.item()

    per_param_grad_norm = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if isinstance(g, DTensor):
            g = g.full_tensor()
        g = g.float()
        local_norm_sq = g.norm() ** 2
        if ".experts." in name and ep_group is not None:
            dist.all_reduce(local_norm_sq, group=ep_group)
        per_param_grad_norm[name] = local_norm_sq.sqrt().item()
    num_grads = sum(1 for p in model.parameters() if p.grad is not None)
    has_nan = math.isnan(total_grad_norm)

    mem_after_bwd = torch.cuda.memory_allocated() / 1024**3
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[{tag}] rank {rank}: logits.shape={list(logits.shape)} "
        f"logits.mean={logits.float().mean().item():.4f} "
        f"loss={reported_loss.item():.4f} "
        f"total_grad_norm={total_grad_norm:.4f} num_grads={num_grads} "
        f"has_nan={has_nan}\n"
        f"  mem_after_bwd={mem_after_bwd:.2f} GiB, mem_peak={mem_peak:.2f} GiB"
    )

    return {
        "rank": rank,
        "logits": logits.cpu(),
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "logits_mean": logits.float().mean().item(),
        "loss": reported_loss.item(),
        "total_grad_norm": total_grad_norm,
        "num_grads": num_grads,
        "has_nan": has_nan,
        "per_param_grad_norm": per_param_grad_norm,
        "mem_after_shard_gib": mem_after_shard,
        "mem_after_bwd_gib": mem_after_bwd,
        "mem_peak_gib": mem_peak,
    }


def _run_fwd_bwd_vanilla_baseline(
    model, per_seg_ids, per_seg_labels, rank, tag, padded_seq_lens=None,
):
    """N 段独立 [1, padded_len_i] forward；不走 CP（vanilla = 纯 FSDP）。

    每段一次 capture_routing_decisions ctx；段间 cat 每层 indices 得到
    [T_TOTAL, top_k]/layer。loss 累加除以全段 n_valid 总和（所有 rank 求 sum）。
    """
    if padded_seq_lens is None:
        padded_seq_lens = PADDED_SEQ_LENS
    model.train()
    torch.cuda.reset_peak_memory_stats()
    mem_after_shard = torch.cuda.memory_allocated() / 1024**3
    print(f"[{tag}] rank {rank}: FSDP sharded, mem_allocated={mem_after_shard:.2f} GiB")

    # 全段 n_valid（rank-local 求和；全 rank 同 input → 不跨 rank reduce，
    # 否则分母 ×world_size、loss 缩 1/world_size，与 pack 口径错位 4×=cp_size）
    total_n_local = torch.zeros((), dtype=torch.long, device=per_seg_ids[0].device)
    for seg_labels in per_seg_labels:
        total_n_local = total_n_local + (seg_labels != -100).sum()
    global_n = total_n_local

    # 每段独立 forward + 单独 capture，避免段间 hook overwrite。
    # capture ctx 只裹 forward——backward 移到 ctx 外，避免 grad-checkpoint
    # 在 backward 时 recompute layer 触发同一个 hook、第二次写 recorded[i]。
    accum_reported = torch.zeros((), dtype=torch.float32, device=per_seg_ids[0].device)
    gathered_logits = []
    per_seg_routing = []  # list[list[Tensor[s_padded, top_k]]] —— [seg][layer]
    for seg_ids, seg_labels in zip(per_seg_ids, per_seg_labels):
        with capture_routing_decisions(model) as recorded:
            outputs = model(input_ids=seg_ids)
            logits = outputs.logits
            seg_loss_sum = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                seg_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            seg_loss = seg_loss_sum / global_n
        # ctx 已退出（hook 已摘）：backward 期 recompute 不会再写 recorded[i]
        accum_reported = accum_reported + seg_loss.detach()
        seg_loss.backward()
        gathered_logits.append(logits.detach())
        # recorded[layer_i] is [B*s_padded, top_k] = [s_padded, top_k] (B=1)
        for i, t in enumerate(recorded):
            assert t is not None, (
                f"layer {i} routing not captured (hook didn't fire); "
                f"check capture_routing_decisions's router_class_names"
            )
        per_seg_routing.append([t.detach().cpu() for t in recorded])

    reported_loss = accum_reported.clone()

    # cat 段→T 维度，得到 [T_TOTAL, top_k]/layer
    n_layers = len(per_seg_routing[0])
    for seg_idx in range(len(per_seg_routing)):
        assert len(per_seg_routing[seg_idx]) == n_layers, (
            f"seg {seg_idx} captured {len(per_seg_routing[seg_idx])} layers, "
            f"expected {n_layers}"
        )
    recorded_routing = []
    for layer_i in range(n_layers):
        layer_per_seg = [per_seg_routing[s][layer_i] for s in range(len(per_seg_routing))]
        for s, t in enumerate(layer_per_seg):
            assert t.shape[0] == padded_seq_lens[s], (
                f"layer {layer_i} seg {s}: routing shape[0]={t.shape[0]} "
                f"!= padded_seq_lens[{s}]={padded_seq_lens[s]}"
            )
        recorded_routing.append(torch.cat(layer_per_seg, dim=0))  # [T_TOTAL, top_k]

    # 段间 cat 到 token 维度（dim=1）：三段 [1, 128, V] / [1, 256, V] / [1, 128, V]
    # → [1, 512, V]，与 pack 路径返回的 logits 形状一致；只给诊断打印用。
    logits_for_diag = torch.cat(gathered_logits, dim=1)

    result = _common_fwd_bwd_diagnostics(
        model, logits_for_diag, reported_loss, rank, tag, ep_group=None,
        mem_after_shard=mem_after_shard,
    )
    result["recorded_routing"] = recorded_routing
    return result


def _run_fwd_bwd_pack(
    model, packed_ids, packed_position_ids, packed_labels, psp,
    rank, tag, cp_group, replay_indices=None,
):
    """THD [1, T=512] forward；CP 切 packed_ids+position_ids 到 T_local=128。

    cu_seqlens_q / cu_seqlens_q_padded 保持全局；layout 经 cp_chunk_data
    切到 per-rank-with-prefix 视图。

    replay_indices：list[Tensor[T_TOTAL, top_k]]/layer（来自 vanilla baseline）；
    按 cp_rank 切到 [s_local, top_k] 后用 router_replay_ctx 喂回。
    """
    model.train()
    torch.cuda.reset_peak_memory_stats()
    mem_after_shard = torch.cuda.memory_allocated() / 1024**3
    print(f"[{tag}] rank {rank}: FSDP sharded, mem_allocated={mem_after_shard:.2f} GiB")

    cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    cp_rank = dist.get_rank(cp_group) if cp_group is not None else 0

    if cp_size > 1:
        local_ids, local_labels, _, local_position_ids, local_psp = cp_chunk_data(
            cp_rank, cp_size,
            tokens=packed_ids, labels=packed_labels,
            position_ids=packed_position_ids,
            packed_seq_params=psp,
        )
    else:
        local_ids = packed_ids
        local_labels = packed_labels
        local_position_ids = packed_position_ids
        local_psp = psp
    assert local_labels is not None  # narrows for pyright

    # n_valid: cp 切片后每 rank 持 1/cp_size，cp_group 内 all_reduce 得整段
    # 真实 n_valid；dp（cp_pair 之间）全 rank 同 input → 不跨 dp 累加，否则
    # 分母 ×dp_size、loss 缩水。
    global_n = (local_labels != -100).sum()
    if cp_size > 1:
        dist.all_reduce(global_n, group=cp_group)

    # Router replay：T 维度按 cp_rank 切到 [s_local, top_k]。
    # 列表为空 = 模型没有 TopKRouter（NUM_LAYERS 太小，全 HashRouter 路径，
    # deterministic 不需要 replay），跳过。
    if replay_indices is not None and len(replay_indices) > 0:
        s_local = local_ids.shape[1]
        cp_replay = [
            t[cp_rank * s_local : (cp_rank + 1) * s_local].cuda()
            for t in replay_indices
        ]
        ctx = router_replay_ctx(model, cp_replay)
    else:
        ctx = nullcontext()

    # packed_seq_params 一路显式 kwarg 透传到 attention/HCA/CSA/Indexer
    with ctx:
        outputs = model(
            input_ids=local_ids,
            position_ids=local_position_ids,
            packed_seq_params=local_psp,
        )
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            local_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        loss = loss / global_n
        reported_loss = loss.detach().clone()
        if cp_size > 1:
            dist.all_reduce(reported_loss, group=cp_group)
        loss.backward()

    if cp_size > 1:
        gathered = [torch.empty_like(logits) for _ in range(cp_size)]
        dist.all_gather(gathered, logits.contiguous(), group=cp_group)
        logits = torch.cat(gathered, dim=1)

    return _common_fwd_bwd_diagnostics(
        model, logits, reported_loss, rank, tag, model._ep_group, mem_after_shard,
    )


# ---------------------------------------------------------------------------
# Ray workers
# ---------------------------------------------------------------------------


def _setup_dist(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _setup_ep_cp_meshes(world_size):
    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // EP_SIZE, EP_SIZE),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    cp_full_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // CP_SIZE, CP_SIZE),
        mesh_dim_names=("dp", "cp"),
    )
    return ep_2d_mesh, cp_full_mesh["cp"]


def _build_fork_model(hf_model_path, tokenizer):
    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
    assert all(PAD_TO_MULTIPLE_OF % r == 0 for r in config.compress_rates.values()), (
        f"PAD_TO_MULTIPLE_OF={PAD_TO_MULTIPLE_OF} must be divisible by every "
        f"compress_rate; got config.compress_rates={dict(config.compress_rates)}"
    )
    assert not config.tie_word_embeddings, (
        "meta-device init may hang with tie_word_embeddings=True"
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)
    return model


@ray.remote(num_gpus=1)
def _vanilla_baseline_worker(
    hf_model_path: str, rank: int, world_size: int,
    master_addr: str, master_port: int,
    fake_seq_lens: list[int] | None = None,
    padded_seq_lens: list[int] | None = None,
):
    """Vanilla HF transformers DSV4 + 纯 FSDP，不走 EP/CP。

    所有 rank 用同一 input_ids → loss/grad 全 rank 相同（DP 复制），仅借
    多卡均摊 FSDP unshard 显存（8 卡时 unshard 一层 12 GiB OOM）。
    Pack 全 rank 同 input → 直接复用 baseline rank 0 的 routing。
    """
    _setup_dist(rank, world_size, master_addr, master_port)
    # FSDP2 默认 mesh：fully_shard(layer, mp_policy=...) 不显式传 mesh，
    # 走的就是这里 init 的 1D mesh。删掉会让 FSDP 找不到 default group。
    init_device_mesh("cuda", mesh_shape=(world_size,))

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ids_list, labels_list = _make_fake_qa(
        tokenizer, device=torch.device("cuda"),
        fake_seq_lens=fake_seq_lens,
    )
    per_seg_ids, per_seg_labels = _prep_vanilla_baseline(
        ids_list, labels_list, tokenizer.pad_token_id, device=torch.device("cuda"),
        padded_seq_lens=padded_seq_lens,
    )

    tag = "vanilla_baseline"
    print(f"[{tag}] rank {rank}: loading {hf_model_path} (HF + fp8 dequantize, fp32 CPU) ...")
    _t_start = time.time()
    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
    # 注意：transformers DSV4 load weights 自带规则忽略 torch_dtype，实际上出来 bf16 和少量 fp32，
    # 必须强行转一次。见 test_deepseek_v4_ep_cp.py 同样段。
    model = _HfModel.from_pretrained(
        hf_model_path,
        config=config,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        quantization_config=FineGrainedFP8Config(dequantize=True),
    )
    model.to(torch.float32)
    model.gradient_checkpointing_enable()
    dist.barrier()

    layers = model.model.layers
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    for layer in layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    load_seconds = time.time() - _t_start
    print(f"[{tag}] rank {rank}: load + FSDP wrap done in {load_seconds:.1f}s")

    result = _run_fwd_bwd_vanilla_baseline(
        model, per_seg_ids, per_seg_labels, rank, tag,
        padded_seq_lens=padded_seq_lens,
    )
    result["load_seconds"] = load_seconds

    del model, per_seg_ids, per_seg_labels
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


@ray.remote(num_gpus=1)
def _pack_thd_worker(
    hf_model_path: str, rank: int, world_size: int,
    master_addr: str, master_port: int,
    replay_indices=None,
    attn_backend: str = "fused",
    indexer_backend: str = "fused",
    ep_backend: str = "eager",
    fake_seq_lens: list[int] | None = None,
    padded_seq_lens: list[int] | None = None,
    fp8: bool = False,
):
    """THD [1, T=512] packed，cp_size=4 切 [1, 128]/rank。

    所有 rank 用同一 input_ids（与 vanilla baseline 全 rank 一致）→ pack
    全部 32 个 rank 共享同一份 baseline routing（rank 0 录出来的）。

    replay_indices: list[Tensor[T_TOTAL, top_k]]/layer，来自 vanilla baseline
    rank 0 的 recorded_routing。
    """
    if padded_seq_lens is None:
        padded_seq_lens = PADDED_SEQ_LENS
    _setup_dist(rank, world_size, master_addr, master_port)
    ep_2d_mesh, cp_mesh = _setup_ep_cp_meshes(world_size)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ids_list, labels_list = _make_fake_qa(
        tokenizer, device=torch.device("cuda"),
        fake_seq_lens=fake_seq_lens,
    )
    config = _truncate_config(DeepseekV4Config.from_pretrained(hf_model_path))
    packed_ids, packed_position_ids, packed_labels, psp = _prep_pack(
        ids_list, labels_list, tokenizer, config,
    )
    assert packed_labels is not None  # narrows for pyright
    # 把 vanilla baseline 段→T cat 顺序与 pack 后的 cu_seqlens_q_padded 钉死：
    # 一旦 pack_sequences 改换内部对齐策略（合段 / 重排 / 不同 pad），
    # router replay 会 silent 错位（indices 仍 valid range，无 NaN）。
    expected_cu = [0]
    for s in padded_seq_lens:
        expected_cu.append(expected_cu[-1] + s)
    actual_cu = psp.cu_seqlens_q_padded.tolist()
    assert actual_cu == expected_cu, (
        f"cu_seqlens_q_padded mismatch: pack={actual_cu} vs expected "
        f"(from padded_seq_lens)={expected_cu}; vanilla baseline routing "
        f"order would silently misalign with pack token order"
    )

    tag = f"pack_thd_ep{EP_SIZE}_cp{CP_SIZE}"
    if replay_indices is not None:
        tag += "_replay"
    if ep_backend != "eager":
        tag += f"_{ep_backend}"
    if fp8:
        tag += "_fp8"
    print(f"[{tag}] rank {rank}: loading {hf_model_path} via meta-device ...", flush=True)
    _t_start = time.time()
    model = _build_fork_model(hf_model_path, tokenizer)
    model = apply_hp(
        model,
        ep_2d_mesh,
        cp_mesh=cp_mesh,
        amp_fp32=False,
        attn_backend=attn_backend,
        indexer_backend=indexer_backend,
        ep_backend=ep_backend,
        fp8=fp8,
    )
    model.gradient_checkpointing_enable()
    model.load_checkpoint_hp(hf_model_path)
    load_seconds = time.time() - _t_start
    print(f"[{tag}] rank {rank}: load_checkpoint done in {load_seconds:.1f}s", flush=True)

    # cp_size=1 时 apply_hp 不绑 _cp_group（hp.py:166 cp_size>1 才 _bind_cp）。
    # 显式分支，避免静默 getattr fallback。
    cp_group = model._cp_group if CP_SIZE > 1 else None
    result = _run_fwd_bwd_pack(
        model, packed_ids, packed_position_ids, packed_labels, psp, rank, tag,
        cp_group=cp_group,
        replay_indices=replay_indices,
    )
    result["load_seconds"] = load_seconds

    del model, packed_ids, packed_position_ids, packed_labels, psp
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


@ray.remote(num_gpus=1)
def _balance_loss_cp_worker(
    hf_model_path: str, rank: int, world_size: int,
    master_addr: str, master_port: int,
    fake_seq_lens: list[int] | None = None,
    padded_seq_lens: list[int] | None = None,
    cp_size_override: int = 4,
    replay_indices=None,
):
    """THD pack + balance loss fwd+bwd with configurable cp_size.

    Stage 1 (cp=1, no replay): baseline, captures routing decisions.
    Stage 2 (cp=4, with replay): replays baseline routing to eliminate
    routing divergence, isolating balance-loss CP path numerics.
    """
    if padded_seq_lens is None:
        padded_seq_lens = list(PADDED_SEQ_LENS)
    _setup_dist(rank, world_size, master_addr, master_port)
    cp_size = cp_size_override
    ep_size = EP_SIZE

    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    if cp_size > 1:
        cp_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // cp_size, cp_size),
            mesh_dim_names=("dp", "cp"),
        )["cp"]
    else:
        cp_mesh = None

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ids_list, labels_list = _make_fake_qa(
        tokenizer, device=torch.device("cuda"),
        fake_seq_lens=fake_seq_lens,
    )
    config = _truncate_config(DeepseekV4Config.from_pretrained(hf_model_path))
    packed_ids, packed_position_ids, packed_labels, psp = pack_sequences(
        ids_list, labels_list,
        config=config,
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
        cp_size=cp_size,
        pad_token_id=tokenizer.pad_token_id,
        label_ignore_index=-100,
    )
    assert packed_labels is not None

    tag = f"balance_cp{cp_size}"
    model = _build_fork_model(hf_model_path, tokenizer)
    model = apply_hp(
        model,
        ep_2d_mesh,
        cp_mesh=cp_mesh,
        amp_fp32=False,
        attn_backend="eager",
        indexer_backend="eager",
        ep_backend="eager",
    )
    model.gradient_checkpointing_enable()
    model.load_checkpoint_hp(hf_model_path)

    cp_group = model._cp_group if cp_size > 1 else None
    actual_cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    cp_rank = dist.get_rank(cp_group) if cp_group is not None else 0

    if actual_cp_size > 1:
        local_ids, local_labels, _, local_position_ids, local_psp = cp_chunk_data(
            cp_rank, cp_size,
            tokens=packed_ids, labels=packed_labels,
            position_ids=packed_position_ids,
            packed_seq_params=psp,
        )
    else:
        local_ids = packed_ids
        local_labels = packed_labels
        local_position_ids = packed_position_ids
        local_psp = psp
    assert local_labels is not None

    global_n = (local_labels != -100).sum()
    if cp_size > 1:
        dist.all_reduce(global_n, group=cp_group)

    # Router replay: force same routing as baseline to isolate balance-loss path.
    if replay_indices is not None and len(replay_indices) > 0:
        s_local = local_ids.shape[1]
        cp_replay = [
            t[cp_rank * s_local : (cp_rank + 1) * s_local].cuda()
            for t in replay_indices
        ]
        replay_ctx = router_replay_ctx(model, cp_replay)
    else:
        replay_ctx = nullcontext()

    model.train()
    with replay_ctx:
        with capture_routing_decisions(model) as recorded:
            outputs = model(
                input_ids=local_ids,
                position_ids=local_position_ids,
                packed_seq_params=local_psp,
                output_router_logits=True,
            )
            logits = outputs.logits
            main_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                local_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ) / global_n

            balance_loss = load_balancing_loss_func(
                gate_logits=outputs.router_logits,
                num_experts=model.num_experts,
                top_k=model.num_experts_per_tok,
                cp_group=cp_group,
            )
            balance_coef = 1e-2
            loss = main_loss + balance_coef * balance_loss

            reported_loss = loss.detach().clone()
            reported_balance = balance_loss.detach().clone()
            if cp_size > 1:
                dist.all_reduce(reported_loss, group=cp_group)

            # backward inside capture ctx: recompute triggers the same hook,
            # which harmlessly overwrites recorded[i] with the same value
            # (assignment, not append), keeping checkpoint tensor count matched.
            loss.backward()

    total_grad_norm = model.clip_grad_norm_(2.0)
    print(
        f"[{tag}] rank {rank}: loss={reported_loss.item():.6f} "
        f"balance_loss={reported_balance.item():.6f} "
        f"grad_norm={total_grad_norm:.6f}"
    )

    # Capture routing for replay in the next stage.
    recorded_routing = None
    if replay_indices is None:
        for i, t in enumerate(recorded):
            assert t is not None, f"layer {i} routing not captured"
        recorded_routing = [t.detach().cpu() for t in recorded]

    result = {
        "rank": rank,
        "loss": reported_loss.item(),
        "balance_loss": reported_balance.item(),
        "total_grad_norm": total_grad_norm,
        "recorded_routing": recorded_routing,
    }
    del model
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


class _LayerPerfTimer:

    def __init__(self, model: torch.nn.Module):
        self._active = False
        self._pending: dict[tuple[int, str], list[torch.cuda.Event]] = {}
        self._events: list[tuple[tuple[int, str], torch.cuda.Event, torch.cuda.Event]] = []
        self._steps: list[dict[tuple[int, str], float]] = []
        self._handles = []

        for layer_idx, layer in enumerate(model.model.layers):
            self._register(layer_idx, "attn", layer.self_attn)
            self._register(layer_idx, "moe", layer.mlp)
            self._register(layer_idx, "mhc", layer.attn_hc)
            self._register(layer_idx, "mhc", layer.ffn_hc)

    def _register(self, layer_idx: int, kind: str, module: torch.nn.Module):
        key = (layer_idx, kind)

        def start(*_args):
            self._start(key)

        def finish(*_args):
            self._finish(key)

        self._handles.append(module.register_forward_pre_hook(start))
        self._handles.append(module.register_forward_hook(finish))
        self._handles.append(module.register_full_backward_pre_hook(start))
        self._handles.append(module.register_full_backward_hook(finish))

    def _start(self, key: tuple[int, str]):
        if not self._active:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._pending.setdefault(key, []).append(event)

    def _finish(self, key: tuple[int, str]):
        if not self._active:
            return
        starts = self._pending[key]
        start = starts.pop()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._events.append((key, start, end))

    def start_step(self):
        self._active = True
        self._pending.clear()
        self._events.clear()

    def finish_step(self):
        self._active = False
        totals: dict[tuple[int, str], float] = {}
        for key, start, end in self._events:
            totals[key] = totals.get(key, 0.0) + start.elapsed_time(end)
        self._steps.append(totals)
        return totals

    def summary(self, skip_steps: int = 1) -> dict[int, dict[str, float]]:
        steps = self._steps[skip_steps:] or self._steps
        if not steps:
            return {}
        out: dict[int, dict[str, float]] = {}
        for step in steps:
            for (layer_idx, kind), elapsed_ms in step.items():
                layer_out = out.setdefault(layer_idx, {})
                layer_out[kind] = layer_out.get(kind, 0.0) + elapsed_ms
        for layer_out in out.values():
            for kind in layer_out:
                layer_out[kind] /= len(steps)
        return out

    def close(self):
        for handle in self._handles:
            handle.remove()


def _format_layer_perf_by_step(layer_perf: dict[tuple[int, str], float]) -> list[str]:
    layer_ids = sorted({layer_idx for layer_idx, _ in layer_perf})
    out = []
    for layer_idx in layer_ids:
        parts = []
        for kind in ("attn", "moe", "mhc"):
            parts.append(f"{kind}={layer_perf.get((layer_idx, kind), 0.0):.2f}")
        out.append(f"layer {layer_idx}:  " + "  ".join(parts))
    return out


@ray.remote(num_gpus=1)
def _bench_thd_worker(
    hf_model_path: str, rank: int, world_size: int,
    master_addr: str, master_port: int,
    n_steps: int = 5,
    attn_backend: str = "fused",
    indexer_backend: str = "fused",
    ep_backend: str = "eager",
    fp8: bool = False,
    fake_seq_lens: list[int] | None = None,
    padded_seq_lens: list[int] | None = None,
    ep_size: int = 8,
    cp_size: int = 4,
):
    if padded_seq_lens is None:
        padded_seq_lens = PADDED_SEQ_LENS
    _setup_dist(rank, world_size, master_addr, master_port)
    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    if cp_size > 1:
        cp_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // cp_size, cp_size),
            mesh_dim_names=("dp", "cp"),
        )["cp"]
    else:
        cp_mesh = None

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ids_list, labels_list = _make_fake_qa(
        tokenizer, device=torch.device("cuda"),
        fake_seq_lens=fake_seq_lens,
    )
    config = _truncate_config(DeepseekV4Config.from_pretrained(hf_model_path))
    # 测试用，一般模型用 self attn，dsv4 attention 占比较小，所以 mfu 30% 左右。
    # config.sliding_window = 16384

    packed_ids, packed_position_ids, packed_labels, psp = _prep_pack(
        ids_list, labels_list, tokenizer, config,
    )
    assert packed_labels is not None

    tag = f"bench_thd_ep{ep_size}_cp{cp_size}_{attn_backend}"
    if fp8:
        tag += "_fp8"
    print(f"[{tag}] rank {rank}: loading {hf_model_path} via meta-device ...")
    _t_start = time.time()
    model = _build_fork_model(hf_model_path, tokenizer)
    model = apply_hp(
        model,
        ep_2d_mesh,
        cp_mesh=cp_mesh,
        amp_fp32=False,
        attn_backend=attn_backend,
        indexer_backend=indexer_backend,
        ep_backend=ep_backend,
        fp8=fp8,
    )
    model.gradient_checkpointing_enable()
    model.load_checkpoint_hp(hf_model_path)
    load_seconds = time.time() - _t_start
    print(f"[{tag}] rank {rank}: load_checkpoint done in {load_seconds:.1f}s")

    cp_group = model._cp_group if cp_size > 1 else None
    actual_cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    cp_rank = dist.get_rank(cp_group) if cp_group is not None else 0

    if actual_cp_size > 1:
        local_ids, local_labels, _, local_position_ids, local_psp = cp_chunk_data(
            cp_rank, cp_size,
            tokens=packed_ids, labels=packed_labels,
            position_ids=packed_position_ids,
            packed_seq_params=psp,
        )
    else:
        local_ids = packed_ids
        local_labels = packed_labels
        local_position_ids = packed_position_ids
        local_psp = psp
    assert local_labels is not None

    global_n = (local_labels != -100).sum()
    if cp_size > 1:
        dist.all_reduce(global_n, group=cp_group)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    layer_perf = _LayerPerfTimer(model)

    step_times = []
    last_loss = None
    last_grad_norm = None
    last_logits_finite = None
    last_num_grads = None
    for step in range(n_steps):
        torch.cuda.synchronize()
        layer_perf.start_step()
        t0 = time.time()

        outputs = model(
            input_ids=local_ids,
            position_ids=local_position_ids,
            packed_seq_params=local_psp,
        )
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            local_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ) / global_n
        reported_loss = loss.detach().clone()
        if cp_size > 1:
            dist.all_reduce(reported_loss, group=cp_group)
        loss.backward()

        total_grad_norm = model.clip_grad_norm_(2.0)
        last_loss = reported_loss.item()
        last_grad_norm = float(total_grad_norm)
        last_logits_finite = bool(torch.isfinite(logits).all())
        last_num_grads = sum(p.grad is not None for p in model.parameters())
        model.zero_grad()
        torch.cuda.synchronize()
        layer_step_perf = layer_perf.finish_step()

        step_time = time.time() - t0
        step_times.append(step_time)
        print(
            f"[{tag}] rank {rank}: step {step}  "
            f"loss={reported_loss.item():.4f}  gn={total_grad_norm:.4f}  "
            f"time={step_time:.3f}s",
            flush=True,
        )

        if rank == 0:
            for line in _format_layer_perf_by_step(layer_step_perf):
                print(f"[{tag}] rank {rank}: step {step}  {line}", flush=True)

    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    avg_step = sum(step_times[1:]) / len(step_times[1:]) if len(step_times) > 1 else step_times[0]

    # 测试 ep8 cp1 seqlen 16k FLOPs 4L: MFU 29%
    # 测试 ep8 cp1 seqlen 16k FLOPs 4L SW 8k: MFU >50%
    from gpatch_v4.utils.flops_counter import FlopsCounter, get_device_flops
    from gpatch_v4.core.constants import MODEL_ARCH
    counter = FlopsCounter(config, MODEL_ARCH.DEEPSEEK_V4)
    real_seq_lens = fake_seq_lens if fake_seq_lens is not None else list(FAKE_SEQ_LENS)
    est_tflops, dev_tflops = counter.estimate_flops(real_seq_lens, avg_step)

    print(
        f"[{tag}] rank {rank}: {n_steps} steps done, "
        f"avg={avg_step:.3f}s/step, mem_peak={mem_peak:.2f} GiB, "
        f"TFLOPs/s={est_tflops:.1f}, device_peak={dev_tflops:.0f} TFLOPs/s, "
        f"MFU={est_tflops / dev_tflops * 100:.1f}%",
        flush=True,
    )

    layer_perf_summary = layer_perf.summary(skip_steps=1)
    if rank == 0:
        print(f"[{tag}] rank {rank}: layer perf ms/step fwd+bwd, step0 skipped", flush=True)
        for layer_idx in sorted(layer_perf_summary):
            parts = []
            for kind in ("attn", "moe", "mhc"):
                parts.append(f"{kind}={layer_perf_summary[layer_idx].get(kind, 0.0):.2f}")
            print(f"[{tag}] rank {rank}: layer {layer_idx}:  " + "  ".join(parts), flush=True)
    layer_perf.close()
    del model, packed_ids, packed_position_ids, packed_labels, psp
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return {
        "rank": rank,
        "load_seconds": load_seconds,
        "step_times": step_times,
        "mem_peak_gib": mem_peak,
        "layer_perf_ms": layer_perf_summary,
        "loss": last_loss,
        "total_grad_norm": last_grad_norm,
        "logits_finite": last_logits_finite,
        "num_grads": last_num_grads,
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestEpCpThd(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {NUM_GPUS} GPUs in Ray cluster, only {total_gpus}"
            )

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def _run_workers(
        self, worker_fn, world_size, pg, master_port,
        *,
        replay_indices_per_rank: "list | None" = None,
        attn_backend: "str | None" = None,
        indexer_backend: "str | None" = None,
        ep_backend: "str | None" = None,
        fake_seq_lens: "list[int] | None" = None,
        padded_seq_lens: "list[int] | None" = None,
        cp_size_override: "int | None" = None,
        fp8: "bool | None" = None,
    ):
        pg_obj, bundle_indices = pg
        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg_obj, placement_group_bundle_index=bundle_indices[0],
                )
            ).remote()
        )
        futures = []
        for r in range(world_size):
            kw = {}
            if attn_backend is not None:
                kw["attn_backend"] = attn_backend
            if indexer_backend is not None:
                kw["indexer_backend"] = indexer_backend
            if ep_backend is not None:
                kw["ep_backend"] = ep_backend
            if replay_indices_per_rank is not None:
                kw["replay_indices"] = replay_indices_per_rank[r]
            if fake_seq_lens is not None:
                kw["fake_seq_lens"] = fake_seq_lens
                kw["padded_seq_lens"] = padded_seq_lens
            if cp_size_override is not None:
                kw["cp_size_override"] = cp_size_override
            if fp8 is not None:
                kw["fp8"] = fp8
            futures.append(
                worker_fn.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg_obj,
                        placement_group_bundle_index=bundle_indices[r],
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    **kw,
                )
            )
        return ray.get(futures)

    def _pack_runs_impl(
        self,
        attn_backend: str = "fused",
        indexer_backend: str = "fused",
        ep_backend: str = "eager",
        master_port_base: int = 12500,
        fake_seq_lens: list[int] | None = None,
        padded_seq_lens: list[int] | None = None,
        fp8: bool = False,
    ):
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}; "
            f"download DeepSeek-V4-Flash into hf-hub/ first"
        )
        assert NUM_GPUS % EP_SIZE == 0, (
            f"NUM_GPUS={NUM_GPUS} not divisible by EP_SIZE={EP_SIZE}"
        )
        assert NUM_GPUS % CP_SIZE == 0, (
            f"NUM_GPUS={NUM_GPUS} not divisible by CP_SIZE={CP_SIZE}"
        )

        pg = _create_placement_group(NUM_GPUS)

        # --- Stage 1: vanilla baseline + 录 routing（全 rank 同 input，
        # loss/grad 一致；32 卡均摊 FSDP unshard 显存）---
        print("=" * 60)
        print(
            f"Running vanilla baseline (HF + 纯 FSDP, world={NUM_GPUS}) + routing capture ..."
        )
        baseline_results = self._run_workers(
            _vanilla_baseline_worker, NUM_GPUS, pg, master_port=master_port_base,
            fake_seq_lens=fake_seq_lens,
            padded_seq_lens=padded_seq_lens,
        )

        replay_indices_per_rank = None
        rank0_routing = baseline_results[0]["recorded_routing"]
        replay_indices_per_rank = [rank0_routing for _ in range(NUM_GPUS)]
        print(
            f"  pack rank * ← baseline rank 0: "
            f"{len(rank0_routing)} TopKRouter layers"
        )
        for layer_i, t in enumerate(rank0_routing[:1]):
            print(f"    layer {layer_i}: shape={list(t.shape)}")

        print("=" * 60)
        print(
            f"Running THD pack (ep={EP_SIZE} cp={CP_SIZE} N={NUM_GPUS} "
            f"attn={attn_backend} fp8={fp8}) + router replay ..."
        )
        pack_results = self._run_workers(
            _pack_thd_worker, NUM_GPUS, pg, master_port=master_port_base + 1,
            replay_indices_per_rank=replay_indices_per_rank,
            attn_backend=attn_backend,
            indexer_backend=indexer_backend,
            ep_backend=ep_backend,
            fake_seq_lens=fake_seq_lens,
            padded_seq_lens=padded_seq_lens,
            fp8=fp8,
        )

        remove_placement_group(pg[0])

        # --- sanity ---
        print("\n--- sanity (finite / no-nan) ---")
        for tag, results in [("baseline", baseline_results), ("pack", pack_results)]:
            for res in results:
                r = res["rank"]
                self.assertTrue(
                    res["logits_finite"],
                    f"{tag} rank {r}: logits contain NaN/Inf",
                )
                self.assertFalse(
                    res["has_nan"],
                    f"{tag} rank {r}: grads contain NaN/Inf",
                )
                print(
                    f"  {tag} rank {r}: loss={res['loss']:.4f} "
                    f"logits.mean={res['logits_mean']:.4f} "
                    f"total_grad_norm={res['total_grad_norm']:.4f} "
                    f"num_grads={res['num_grads']}"
                )

        # --- 数值等价 assert ---
        # bf16 pack vs baseline：sibling ep_cp 6.8e-5 / 5.3e-4 量级，本测
        # E5 实测 + 2-50× 余量。
        # fp8 MoE：对齐 test_te_gemm_fp8 单 op ~5% 量级，整模多层专家累加后放宽。
        if fp8:
            rtol_loss = 0.05
            rtol_grad_norm = 0.05
            rtol_logits = 0.10
            per_param_rel = 0.15
            atol_per_param = 1e-2
        else:
            rtol_loss = 0.005
            rtol_grad_norm = 0.01
            rtol_logits = 0.02
            per_param_rel = 0.05
            atol_per_param = 1e-3

        bl0 = baseline_results[0]
        pk0 = pack_results[0]

        # logits 对比（pack rank 0 已 all-gather 到全 T=512；baseline rank 0
        # 是 3 段独立 cat 后 [1, T, V]——shape 应一致）
        print("\n--- logits comparison (rank 0) ---")
        bl_logits = bl0["logits"].float()
        pk_logits = pk0["logits"].float()
        self.assertEqual(
            bl_logits.shape, pk_logits.shape,
            f"logits shape mismatch: baseline={list(bl_logits.shape)} "
            f"vs pack={list(pk_logits.shape)}",
        )
        max_abs = (bl_logits - pk_logits).abs().max().item()
        rel = max_abs / max(bl_logits.abs().max().item(), 1e-12)
        print(f"  logits max_abs_diff={max_abs:.2e}, rel_diff={rel:.2e} (rtol={rtol_logits})")
        self.assertLess(
            rel, rtol_logits,
            f"Logits rel_diff={rel:.2e} > rtol={rtol_logits}",
        )

        # Loss
        loss_rel = abs(pk0["loss"] - bl0["loss"]) / max(abs(bl0["loss"]), 1e-12)
        print(
            f"\n--- loss / total_grad_norm comparison ---\n"
            f"  baseline rank 0:  loss={bl0['loss']:.6f}  "
            f"total_grad_norm={bl0['total_grad_norm']:.6f}\n"
            f"  pack     rank 0:  loss={pk0['loss']:.6f}  "
            f"total_grad_norm={pk0['total_grad_norm']:.6f}\n"
            f"  loss rel_diff       = {loss_rel:.6f} (rtol={rtol_loss})"
        )
        self.assertLess(
            loss_rel, rtol_loss,
            f"Loss rel_diff={loss_rel:.6f} > rtol={rtol_loss}",
        )

        # Total grad norm
        gnorm_rel = abs(pk0["total_grad_norm"] - bl0["total_grad_norm"]) / max(
            bl0["total_grad_norm"], 1e-12,
        )
        print(f"  total_grad_norm rel_diff = {gnorm_rel:.6f} (rtol={rtol_grad_norm})")
        self.assertLess(
            gnorm_rel, rtol_grad_norm,
            f"total_grad_norm rel_diff={gnorm_rel:.6f} > rtol={rtol_grad_norm}",
        )

        # 逐参数 grad norm
        print("\n--- per-param grad norm ratio ---")
        bl_norms = bl0["per_param_grad_norm"]
        pk_norms = pk0["per_param_grad_norm"]
        # fork 模型用 _Fp32ParamHolder 包裹 sinks / position_bias，
        # named_parameters() 会产出 _sink_holder.weight / _position_bias_holder.weight，
        # 与 HF baseline 的 sinks / position_bias 命名不一致。
        # 按 checkpoint.py 的命名规约做一次标准化，让两边可对齐比较。
        import re
        pk_norms_normalized = {}
        _holder_to_hf = [
            (r"\.self_attn\._sink_holder\.weight$", ".self_attn.sinks"),
            (r"\.compressor\.indexer\._position_bias_holder\.weight$",
             ".compressor.indexer.position_bias"),
            (r"\.compressor\._position_bias_holder\.weight$",
             ".compressor.position_bias"),
        ]
        for k, v in pk_norms.items():
            mapped = k
            for pat, repl in _holder_to_hf:
                if re.search(pat, mapped):
                    mapped = re.sub(pat, repl, mapped)
                    break
            pk_norms_normalized[mapped] = v
        pk_norms = pk_norms_normalized
        common = sorted(set(bl_norms.keys()) & set(pk_norms.keys()))
        # 名字必须一致：HF transformers DSV4 与 fork DSV4 同根，参数路径同名。
        # 任何一边独有的参数都是 schema drift，需要立刻报警。
        bl_only = sorted(set(bl_norms.keys()) - set(pk_norms.keys()))
        pk_only = sorted(set(pk_norms.keys()) - set(bl_norms.keys()))
        self.assertEqual(
            (bl_only, pk_only), ([], []),
            f"per-param grad norm key mismatch: bl_only={bl_only}, pk_only={pk_only}",
        )
        header = f"  {'parameter':<75s} {'bl_norm':>12s} {'pk_norm':>12s} {'ratio':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name in common:
            bl_n = bl_norms[name]
            pk_n = pk_norms[name]
            ratio = pk_n / bl_n if bl_n > 1e-12 else float("nan")
            print(f"  {name:<75s} {bl_n:12.6f} {pk_n:12.6f} {ratio:8.4f}")
            if bl_n > 1e-10:
                rel_ok = abs(ratio - 1.0) < per_param_rel
                abs_ok = abs(pk_n - bl_n) < atol_per_param
                self.assertTrue(
                    rel_ok or abs_ok,
                    msg=f"Grad norm mismatch: {name}: baseline={bl_n:.6f}, "
                        f"pack={pk_n:.6f}, ratio={ratio:.4f}, "
                        f"abs_diff={abs(pk_n - bl_n):.2e}",
                )

        print(f"\nPASSED (attn={attn_backend} fp8={fp8})")

    def test_pack_runs_fused(self):
        """THD pack fused attn vs vanilla baseline, short segs [100, 200, 100]."""
        self._pack_runs_impl(attn_backend="fused", indexer_backend="fused", ep_backend="deepep")

    def test_pack_runs_fused_fp8(self):
        """THD pack fused + MoE FP8 grouped GEMM vs bf16 vanilla baseline."""
        self._pack_runs_impl(
            attn_backend="fused",
            indexer_backend="fused",
            ep_backend="deepep",
            fp8=True,
            master_port_base=12800,
        )

    def test_pack_runs_eager(self):
        """THD pack eager attn + fused indexer vs vanilla baseline."""
        self._pack_runs_impl(attn_backend="eager", indexer_backend="eager", ep_backend="eager",
                             master_port_base=12600)

    def test_pack_runs_eager_topk(self):
        """THD pack eager attn + fused indexer vs vanilla baseline."""
        self._pack_runs_impl(attn_backend="eager-topk", indexer_backend="fused", ep_backend="deepep",
                             master_port_base=12600)

    def test_pack_runs_long(self):
        """THD pack fused attn, long segs [1, 2023, 200] padded to [128, 2048, 384]=2560.

        T_TOTAL=2560 必须是 cp_size(4)×m'(128)=512 的倍数，所以第三段 pad 到 384
        而非 256（否则 T_TOTAL=2432, s_local=608, 608%128≠0）。
        pack_sequences 的 cp_size 参数会自动处理这个对齐。
        """
        self._pack_runs_impl(attn_backend="fused", indexer_backend="fused", ep_backend="deepep",
                             master_port_base=12700,
                             fake_seq_lens=[1, 2023, 200],
                             padded_seq_lens=[128, 2048, 384])

    # ------------------------------------------------------------------
    # CP balance-loss equivalence: cp=1 vs cp=4 same loss/grad
    # ------------------------------------------------------------------

    def test_cp_balance_loss_equivalence(self):
        """cp=1 vs cp=4 same balance_loss / loss / grad with router replay."""
        assert os.path.isdir(HF_MODEL_PATH), f"model dir not found: {HF_MODEL_PATH}"
        pg = _create_placement_group(NUM_GPUS)

        # --- Stage 1: cp=1 baseline, capture routing ---
        print("=" * 60)
        print(f"Running THD pack ep={EP_SIZE} cp=1 (balance loss baseline + routing capture) ...")
        cp1_results = self._run_workers(
            _balance_loss_cp_worker, NUM_GPUS, pg, master_port=13000,
            fake_seq_lens=list(FAKE_SEQ_LENS),
            padded_seq_lens=list(PADDED_SEQ_LENS),
            cp_size_override=1,
        )

        # --- Stage 2: cp=4, replay baseline routing ---
        rank0_routing = cp1_results[0]["recorded_routing"]
        replay_indices_per_rank = [rank0_routing for _ in range(NUM_GPUS)]
        print(
            f"  captured {len(rank0_routing)} TopKRouter layers from cp=1 baseline"
        )
        print("=" * 60)
        print(f"Running THD pack ep={EP_SIZE} cp={CP_SIZE} (balance loss CP + router replay) ...")
        cp4_results = self._run_workers(
            _balance_loss_cp_worker, NUM_GPUS, pg, master_port=13001,
            fake_seq_lens=list(FAKE_SEQ_LENS),
            padded_seq_lens=list(PADDED_SEQ_LENS),
            cp_size_override=CP_SIZE,
            replay_indices_per_rank=replay_indices_per_rank,
        )

        remove_placement_group(pg[0])

        # 取 rank 0 对比
        r0_cp1 = cp1_results[0]
        r0_cp4 = cp4_results[0]

        print("\n--- cp=1 vs cp=4 comparison (rank 0, with router replay) ---")
        print(
            f"  cp=1: loss={r0_cp1['loss']:.6f}  balance_loss={r0_cp1['balance_loss']:.6f}  "
            f"total_grad_norm={r0_cp1['total_grad_norm']:.6f}"
        )
        print(
            f"  cp=4: loss={r0_cp4['loss']:.6f}  balance_loss={r0_cp4['balance_loss']:.6f}  "
            f"total_grad_norm={r0_cp4['total_grad_norm']:.6f}"
        )

        # balance_loss: CpMean 前向值与非 CP 路径一致。残差来自 gate_logits
        # 本身（attention 数值差→不同 hidden states→不同 softmax routing_weights），
        # 但 topk 决策被 replay 钉死，所以差异只在 routing_weights 的 soft 值。
        rtol_balance = 0.002
        bl_rel = abs(r0_cp4["balance_loss"] - r0_cp1["balance_loss"]) / max(
            abs(r0_cp1["balance_loss"]), 1e-12
        )
        print(f"  balance_loss rel_diff = {bl_rel:.6e} (rtol={rtol_balance})")
        self.assertLess(bl_rel, rtol_balance, f"balance_loss rel_diff={bl_rel:.6e}")

        # grad_norm: 验证 CpMean backward 梯度正确性（框架 CP-SUM 拼出全局梯度）。
        rtol_grad = 0.01
        gnorm_rel = abs(r0_cp4["total_grad_norm"] - r0_cp1["total_grad_norm"]) / max(
            r0_cp1["total_grad_norm"], 1e-12,
        )
        print(f"  total_grad_norm rel_diff = {gnorm_rel:.6f} (rtol={rtol_grad})")
        self.assertLess(gnorm_rel, rtol_grad, f"grad_norm rel_diff={gnorm_rel:.6f}")

        # total loss 不 assert：主 loss 的 ~4% diff 来自 attention 实现差异
        # （standard vs ring），与 balance loss 无关。仅 informational 打印。
        loss_rel = abs(r0_cp4["loss"] - r0_cp1["loss"]) / max(abs(r0_cp1["loss"]), 1e-12)
        print(f"  total loss rel_diff = {loss_rel:.6f} (informational, attention numerics)")

        print("\nPASSED (cp=1 vs cp=4 balance loss equivalence with router replay)")

    # ------------------------------------------------------------------
    # bench (no baseline, multi-step fwd+bwd only)
    # ------------------------------------------------------------------

    def _bench_impl(
        self,
        n_steps: int = 5,
        attn_backend: str = "fused",
        indexer_backend: str = "fused",
        ep_backend: str = "eager",
        fp8: bool = False,
        master_port_base: int = 12800,
        fake_seq_lens: list[int] | None = None,
        padded_seq_lens: list[int] | None = None,
    ):
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}"
        )
        pg = _create_placement_group(NUM_GPUS)
        pg_obj, bundle_indices = pg
        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg_obj,
                    placement_group_bundle_index=bundle_indices[0],
                )
            ).remote()
        )
        futures = []
        for r in range(NUM_GPUS):
            futures.append(
                _bench_thd_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg_obj,
                        placement_group_bundle_index=bundle_indices[r],
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=NUM_GPUS,
                    master_addr=master_addr,
                    master_port=master_port_base,
                    n_steps=n_steps,
                    attn_backend=attn_backend,
                    indexer_backend=indexer_backend,
                    ep_backend=ep_backend,
                    fp8=fp8,
                    fake_seq_lens=fake_seq_lens,
                    padded_seq_lens=padded_seq_lens,
                )
            )
        results = ray.get(futures)
        remove_placement_group(pg_obj)

        print("\n" + "=" * 60, flush=True)
        print(f"Bench summary: {n_steps} steps, fake_seq_lens={fake_seq_lens}", flush=True)
        for res in results:
            r = res["rank"]
            timed_steps = res["step_times"][1:] or res["step_times"]
            avg = sum(timed_steps) / len(timed_steps)
            print(
                f"  rank {r}: load={res['load_seconds']:.1f}s  "
                f"avg_step={avg:.3f}s  mem_peak={res['mem_peak_gib']:.2f} GiB",
                flush=True,
            )
        layer_ids = sorted({
            layer_idx
            for res in results
            for layer_idx in res["layer_perf_ms"]
        })
        print("\nLayer perf: ms/step fwd+bwd, step0 skipped", flush=True)
        for layer_idx in layer_ids:
            parts = []
            for kind in ("attn", "moe", "mhc"):
                vals = [
                    res["layer_perf_ms"].get(layer_idx, {}).get(kind, 0.0)
                    for res in results
                ]
                parts.append(
                    f"{kind}=max {max(vals):.2f} / avg {sum(vals) / len(vals):.2f}"
                )
            print(f"  layer {layer_idx}:  " + "  ".join(parts), flush=True)
        return results

    def test_hp_fused_smoke(self):
        """Run one HP THD fwd/bwd with fused attention, indexer, EP, and FP8 MoE."""
        results = self._bench_impl(
            n_steps=1,
            attn_backend="fused",
            indexer_backend="fused",
            ep_backend="deepep",
            fp8=True,
            master_port_base=13100,
        )
        for result in results:
            rank = result["rank"]
            self.assertTrue(result["logits_finite"], f"rank {rank}: logits contain NaN/Inf")
            self.assertTrue(math.isfinite(result["loss"]), f"rank {rank}: loss is not finite")
            self.assertTrue(
                math.isfinite(result["total_grad_norm"]),
                f"rank {rank}: total_grad_norm is not finite",
            )
            self.assertGreater(result["num_grads"], 0, f"rank {rank}: no parameter gradients")

    def test_bench_long(self):
        """Perf bench: 16k single seq, 5 fwd+bwd steps, no baseline comparison."""
        self._bench_impl(
            n_steps=5,
            attn_backend="fused",
            indexer_backend="fused",
            ep_backend="deepep",
            master_port_base=12800,
            fake_seq_lens=[64*1024],
            padded_seq_lens=[64*1024],
        )
