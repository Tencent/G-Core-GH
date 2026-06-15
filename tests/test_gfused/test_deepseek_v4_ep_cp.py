# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""正确性测试：HF 原生 FSDP vs 我们的 EP+CP 实现（DeepSeek-V4）。

验证 logits / loss / grad norm 在容差内一致（含 router replay）。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep_cp.py
"""

import math
import os
import time
import unittest
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
from gpatch_v4.orches.placement_group import _create_placement_group

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
NUM_GPUS = 32
NUM_LAYERS = 4
SEQ_LEN = 256


def _truncate_config(config):
    """截断到 NUM_LAYERS 层以加速测试。"""
    if NUM_LAYERS is not None:
        config.num_hidden_layers = NUM_LAYERS
        config.layer_types = config.layer_types[:NUM_LAYERS]
        config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
    config.num_nextn_predict_layers = 0
    return config


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip():
    return ray.util.get_node_ip_address()


def _compute_global_param_norms(model, ep_group):
    """计算每个参数的全局 norm，expert 参数跨 EP rank 聚合。"""
    norms = {}
    for name, p in model.named_parameters():
        t = p.detach()
        if isinstance(t, DTensor):
            t = t.full_tensor()
        t = t.float()
        local_norm_sq = t.norm() ** 2
        if ".experts." in name and ep_group is not None:
            dist.all_reduce(local_norm_sq, group=ep_group)
        norms[name] = local_norm_sq.sqrt().item()
    return norms


def _run_fwd_bwd(model, input_ids, tokenizer, rank, world_size, tag,
                 ep_group=None, ep_fsdp_mesh=None, cp_group=None,
                 memory_only: bool = False, label_ids=None):
    """前向 + 反向 + optimizer step，返回诊断 dict。"""
    model.train()
    assert model.device.type == "cuda", f"expected cuda, got {model.device}"
    torch.cuda.reset_peak_memory_stats()
    mem_after_shard = torch.cuda.memory_allocated() / 1024**3
    print(f"[{tag}] rank {rank}: FSDP sharded, mem_allocated={mem_after_shard:.2f} GiB")

    cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    cp_rank = dist.get_rank(cp_group) if cp_group is not None else 0
    s_full = input_ids.shape[1]

    full_labels = label_ids.clone() if label_ids is not None else input_ids.clone()
    full_labels[full_labels == tokenizer.pad_token_id] = -100
    # 预先 shift labels，避免 CP 切分时边界丢 token
    full_labels = torch.roll(full_labels, shifts=-1, dims=-1)
    full_labels[:, -1] = -100

    if cp_size > 1:
        s_local = s_full // cp_size
        labels = full_labels[:, cp_rank * s_local : (cp_rank + 1) * s_local]
    else:
        labels = full_labels

    # global_n 基于 local labels 求和：CP 下各 rank 持有不重叠切片，
    # all_reduce 后得到全局有效 token 数（不会 double-count）
    global_n = (labels != -100).sum()
    dist.all_reduce(global_n)

    if cp_size > 1:
        local_input_ids, _, _, local_position_ids, _ = cp_chunk_data(
            cp_rank, cp_size, tokens=input_ids,
        )
        outputs = model(input_ids=local_input_ids, position_ids=local_position_ids)
    else:
        outputs = model(input_ids=input_ids)
    logits = outputs.logits
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    loss = loss / global_n
    reported_loss = loss.detach().clone()
    if cp_size > 1:
        dist.all_reduce(reported_loss, group=cp_group)
    # CP 梯度修正由 apply_hp 的 set_gradient_divide_factor 处理，无需 loss * cp_size
    loss.backward()
    torch.cuda.synchronize()
    mem_after_bwd = torch.cuda.memory_allocated() / 1024**3
    mem_peak_fwd_bwd = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[{tag}] rank {rank}: fwd+bwd peak={mem_peak_fwd_bwd:.2f} GiB, "
        f"after_bwd={mem_after_bwd:.2f} GiB, loss={reported_loss.item():.4f}"
    )
    if memory_only:
        result = {
            "rank": rank,
            "loss": reported_loss.item(),
            "mem_after_shard_gib": mem_after_shard,
            "mem_after_bwd_gib": mem_after_bwd,
            "mem_peak_fwd_bwd_gib": mem_peak_fwd_bwd,
            "mem_peak_gib": mem_peak_fwd_bwd,
        }
        del model, outputs
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        return result
    logits = logits.detach()

    if cp_size > 1:
        gathered = [torch.empty_like(logits) for _ in range(cp_size)]
        dist.all_gather(gathered, logits.contiguous(), group=cp_group)
        logits = torch.cat(gathered, dim=1)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    norms_before = _compute_global_param_norms(model, ep_group)
    optimizer.step()

    for name, p in model.named_parameters():
        assert isinstance(p, DTensor), f"{name} is not DTensor after FSDP2 wrap"
        state = optimizer.state.get(p, {})
        for k in ("exp_avg", "exp_avg_sq"):
            if k in state:
                assert isinstance(state[k], DTensor), (
                    f"optimizer state {name}.{k} not DTensor"
                )

    optimizer.zero_grad()
    norms_after = _compute_global_param_norms(model, ep_group)

    any_nonzero_delta = False
    for name in norms_before:
        if name in norms_after:
            delta = abs(norms_after[name] - norms_before[name])
            if delta > 0:
                any_nonzero_delta = True
                break
    assert any_nonzero_delta, "optimizer.step() did not change any parameter"

    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[{tag}] rank {rank}: logits.shape={list(logits.shape)} "
        f"logits.mean={logits.float().mean().item():.4f} "
        f"loss={reported_loss.item():.4f} "
        f"total_grad_norm={total_grad_norm:.4f} num_grads={num_grads} "
        f"has_nan={has_nan}\n"
        f"  mem_after_bwd={mem_after_bwd:.2f} GiB, mem_peak={mem_peak:.2f} GiB"
    )

    result = {
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
        "param_norms_before": norms_before,
        "param_norms_after": norms_after,
        "mem_after_shard_gib": mem_after_shard,
        "mem_after_bwd_gib": mem_after_bwd,
        "mem_peak_fwd_bwd_gib": mem_peak_fwd_bwd,
        "mem_peak_gib": mem_peak,
    }

    del model, logits, outputs, optimizer
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


@ray.remote(num_gpus=1)
def _baseline_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    seq_len: int = SEQ_LEN,
    record_routing: bool = False,
):
    """Baseline: HF 原生 DSV4 + 纯 FSDP。每个 rank 独立 seed（42 + rank）。"""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    dp_mesh = init_device_mesh("cuda", mesh_shape=(world_size,))

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rng = torch.Generator().manual_seed(42 + rank)
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, seq_len), generator=rng
    ).cuda()

    print(f"[baseline] rank {rank}: loading {hf_model_path} ...")
    _t_load_start = time.time()
    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
    # 注意：transformers DSV4 load weights 自带规则忽略 torch_dtype， 实际上出来 bf16 和少量 fp32，必须强行转一次。
    # ```
    # model.layers.0.ffn_hc.scale torch.float32
    # model.layers.1.self_attn.sinks torch.float32
    # model.layers.1.self_attn.q_a_proj.weight torch.bfloat16
    # model.layers.1.self_attn.q_a_norm.weight torch.bfloat16
    # model.layers.1.self_attn.q_b_proj.weight torch.bfloat16
    # model.layers.1.self_attn.kv_proj.weight torch.bfloat16
    # ````
    model = _HfModel.from_pretrained(
        hf_model_path,
        config=config,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        quantization_config=FineGrainedFP8Config(dequantize=True),
    )
    # 经过这里 dtype 都是 fp32，满足 fsdp 要求
    # ```
    # model.layers.0.ffn_hc.scale torch.float32
    # model.layers.1.self_attn.sinks torch.float32
    # model.layers.1.self_attn.q_a_proj.weight torch.float32
    # model.layers.1.self_attn.q_a_norm.weight torch.float32
    # model.layers.1.self_attn.q_b_proj.weight torch.float32
    # ```
    model.to(torch.float32)
    _t_load_end = time.time()
    load_seconds = _t_load_end - _t_load_start
    print(
        f"[baseline] rank {rank}: load_checkpoint done in {load_seconds:.1f}s "
        f"(from_pretrained + fp8 dequantize, CPU)"
    )
    model.gradient_checkpointing_enable()
    torch.distributed.barrier()

    layers = model.model.layers
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    for layer in layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    if record_routing:
        ctx = capture_routing_decisions(model)
    else:
        ctx = nullcontext()

    with ctx as recorded:
        result = _run_fwd_bwd(
            model, input_ids, tokenizer, rank, world_size, "baseline",
        )
    result["load_seconds"] = load_seconds
    if recorded is not None:
        result["recorded_routing"] = [t.cpu() for t in recorded]
    return result


# ---------------------------------------------------------------------------
# Ray worker
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
def _ep_cp_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    cp_size: int,
    seq_len: int = SEQ_LEN,
    replay_indices: "list[torch.Tensor] | None" = None,
    attn_backend: str = "fused",
    indexer_backend: str = "fused",
    ep_backend: str = "eager",
    memory_only: bool = False,
    input_mode: str = "random",
):
    """EP+CP 实现：双正交 mesh (ep_fsdp, ep) × (dp, cp)。

    同一 cp-pair 内 rank 共享 full_input_ids（seed = 42 + rank // cp_size），
    与 baseline rank r//cp_size 对应。
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 构建两个正交 mesh
    # apply_hp 通过 set_gradient_divide_factor(world_size / cp_size) 修正梯度
    assert world_size % ep_size == 0, (
        f"world_size {world_size} not divisible by ep_size {ep_size}"
    )
    assert world_size % cp_size == 0, (
        f"world_size {world_size} not divisible by cp_size {cp_size}"
    )
    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    # cp_mesh: (dp, cp) 下 row-major，同 cp-group pairs 为 (0,1),(2,3),...
    cp_full_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // cp_size, cp_size),
        mesh_dim_names=("dp", "cp"),
    )
    cp_mesh = cp_full_mesh["cp"]

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # cp-pair 内共享 seed，与 baseline rank r//cp_size 对应
    rng = torch.Generator().manual_seed(42 + rank // cp_size)
    if input_mode == "random":
        full_input_ids = torch.randint(
            0, tokenizer.vocab_size, (1, seq_len), generator=rng
        ).cuda()
        label_ids = None
    elif input_mode == "all_pad":
        full_input_ids = torch.full(
            (1, seq_len), tokenizer.pad_token_id, dtype=torch.long,
            device="cuda",
        )
        label_token_id = 1 if tokenizer.pad_token_id == 0 else 0
        label_ids = torch.full_like(full_input_ids, label_token_id)
    else:
        raise ValueError(f"unknown input_mode: {input_mode}")

    tag = (
        f"ep{ep_size}_cp{cp_size}_{input_mode}"
        if replay_indices is None
        else f"ep{ep_size}_cp{cp_size}_{input_mode}_replay"
    )
    if ep_backend != "eager":
        tag += f"_{ep_backend}"
    print(f"[{tag}] rank {rank}: loading {hf_model_path} via meta-device ...")
    _t_load_start = time.time()

    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
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

    model = apply_hp(
        model,
        ep_2d_mesh,
        cp_mesh=cp_mesh,
        amp_fp32=False,
        attn_backend=attn_backend,
        indexer_backend=indexer_backend,
        ep_backend=ep_backend,
    )
    # recompute 时 CP 通信（SWA-ring / compressor）会在 backward 重新执行，NCCL 配对有效
    model.gradient_checkpointing_enable()
    _t_apply_hp_done = time.time()
    model.load_checkpoint_hp(hf_model_path)
    _t_load_end = time.time()
    apply_hp_seconds = _t_apply_hp_done - _t_load_start
    load_state_seconds = _t_load_end - _t_apply_hp_done
    load_seconds = _t_load_end - _t_load_start
    print(
        f"[{tag}] rank {rank}: load_checkpoint done in {load_seconds:.1f}s "
        f"(meta-construct+apply_hp={apply_hp_seconds:.1f}s, "
        f"load_checkpoint_hp={load_state_seconds:.1f}s)"
    )

    # Router replay：将 baseline 录制的 [s_full, top_k] 按 cp_rank 切片
    if replay_indices is not None:
        cp_rank = model._cp_rank
        s_full = full_input_ids.shape[1]
        s_local = s_full // cp_size
        cp_replay = [
            t[cp_rank * s_local : (cp_rank + 1) * s_local].cuda()
            for t in replay_indices
        ]
        ctx = router_replay_ctx(model, cp_replay)
    else:
        ctx = nullcontext()

    # _run_fwd_bwd 在 cp_size > 1 时调 cp_chunk_data() 切到 s_local
    with ctx:
        result = _run_fwd_bwd(
            model, full_input_ids, tokenizer, rank, world_size, tag,
            ep_group=model._ep_group, ep_fsdp_mesh=model._ep_fsdp_mesh,
            cp_group=model._cp_group,
            memory_only=memory_only,
            label_ids=label_ids,
        )
    result["load_seconds"] = load_seconds
    result["apply_hp_seconds"] = apply_hp_seconds
    result["load_state_seconds"] = load_state_seconds
    return result


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestFsdpVsEpCp(unittest.TestCase):
    """HF 原生 FSDP vs EP+CP 实现正确性对比。需要 >= 32 GPU。"""

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

    def _run_baseline(
        self,
        world_size: int,
        pg: "ray.util.placement_group.PlacementGroup",
        master_port: int,
        *,
        seq_len: int = SEQ_LEN,
        record_routing: bool = False,
    ) -> list[dict]:
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
            futures.append(
                _baseline_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg_obj, placement_group_bundle_index=bundle_indices[r],
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    seq_len=seq_len,
                    record_routing=record_routing,
                )
            )
        return ray.get(futures)

    def _run_ep_cp(
        self,
        world_size: int,
        pg: "ray.util.placement_group.PlacementGroup",
        master_port: int,
        ep_size: int,
        cp_size: int,
        *,
        seq_len: int = SEQ_LEN,
        replay_indices_per_rank: "list[list[torch.Tensor]] | None" = None,
        attn_backend: str = "fused",
        indexer_backend: str = "fused",
        ep_backend: str = "eager",
        memory_only: bool = False,
        input_mode: str = "random",
    ) -> list[dict]:
        """启动 EP+CP workers。"""
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
            futures.append(
                _ep_cp_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg_obj, placement_group_bundle_index=bundle_indices[r],
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    ep_size=ep_size,
                    cp_size=cp_size,
                    seq_len=seq_len,
                    replay_indices=(
                        replay_indices_per_rank[r]
                        if replay_indices_per_rank is not None
                        else None
                    ),
                    attn_backend=attn_backend,
                    indexer_backend=indexer_backend,
                    ep_backend=ep_backend,
                    memory_only=memory_only,
                    input_mode=input_mode,
                )
            )
        return ray.get(futures)

    def _assert_results(
        self,
        baseline_results: list[dict],
        ep_results: list[dict],
        ep_tag: str,
        cp_size: int,
        rtol: float,
    ):
        """断言 baseline 和 EP+CP 结果一致。EP+CP rank r 对应 baseline rank r//cp_size。"""
        # --- 基本检查：无 NaN/Inf ---
        print("\n--- sanity checks ---")
        for tag, results in [("baseline", baseline_results), (ep_tag, ep_results)]:
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

        # --- 内存使用汇总 ---
        print("\n--- memory usage summary ---")
        for tag, results in [("baseline", baseline_results), (ep_tag, ep_results)]:
            for res in results:
                r = res["rank"]
                print(
                    f"  {tag} rank {r}: "
                    f"after_shard={res['mem_after_shard_gib']:.2f} GiB, "
                    f"after_bwd={res['mem_after_bwd_gib']:.2f} GiB, "
                    f"peak={res['mem_peak_gib']:.2f} GiB"
                )

        # --- 加载耗时汇总 ---
        print("\n--- checkpoint load time summary ---")
        bl_loads = [r["load_seconds"] for r in baseline_results]
        ep_loads = [r["load_seconds"] for r in ep_results]
        print(
            f"  baseline (from_pretrained + fp8 dequantize): "
            f"min={min(bl_loads):.1f}s, max={max(bl_loads):.1f}s, "
            f"mean={sum(bl_loads)/len(bl_loads):.1f}s"
        )
        ep_apply = [r["apply_hp_seconds"] for r in ep_results]
        ep_lsh = [r["load_state_seconds"] for r in ep_results]
        print(
            f"  {ep_tag} (meta-construct + apply_hp + load_checkpoint_hp): "
            f"min={min(ep_loads):.1f}s, max={max(ep_loads):.1f}s, "
            f"mean={sum(ep_loads)/len(ep_loads):.1f}s"
        )
        print(
            f"    breakdown — meta+apply_hp: mean={sum(ep_apply)/len(ep_apply):.1f}s; "
            f"load_checkpoint_hp: mean={sum(ep_lsh)/len(ep_lsh):.1f}s"
        )

        # --- 逐 rank logits 对比（EP+CP rank r ↔ baseline rank r//cp）---
        print("\n--- logits comparison (per rank) ---")
        for r_ep in range(len(ep_results)):
            r_bl = r_ep // cp_size
            bl_logits = baseline_results[r_bl]["logits"].float()
            ep_logits = ep_results[r_ep]["logits"].float()
            max_abs = (bl_logits - ep_logits).abs().max().item()
            rel = max_abs / max(bl_logits.abs().max().item(), 1e-12)
            print(
                f"  ep rank {r_ep} vs baseline rank {r_bl}: "
                f"logits max_abs_diff={max_abs:.2e}, rel_diff={rel:.2e}"
            )
            self.assertLess(
                rel,
                rtol,
                f"Logits mismatch ep rank {r_ep} (vs baseline {r_bl}): "
                f"rel_diff={rel:.2e} > rtol={rtol}",
            )

        # --- Loss 对比 ---
        bl_loss = baseline_results[0]["loss"]
        ep_loss = ep_results[0]["loss"]
        loss_rel = abs(bl_loss - ep_loss) / max(abs(bl_loss), 1e-12)
        print(
            f"\n  baseline loss       = {bl_loss:.6f}\n"
            f"  {ep_tag} loss          = {ep_loss:.6f}\n"
            f"  loss rel_diff       = {loss_rel:.6f} (rtol={rtol})"
        )
        self.assertLess(
            loss_rel,
            rtol,
            f"Loss mismatch: baseline={bl_loss:.6f} vs {ep_tag}={ep_loss:.6f} "
            f"(rel_diff={loss_rel:.6f} > rtol={rtol})",
        )

        # --- 逐参数 grad norm 对比 ---
        print("\n--- per-param grad norm table ---")
        bl_norms = baseline_results[0]["per_param_grad_norm"]
        # EP 端把 ``sinks`` / ``position_bias`` wrap 进了 ``_Fp32ParamHolder``；
        # 把 holder 路径名归一化回 baseline 端的扁平名再比对。
        def _strip_holder(name: str) -> str:
            return (
                name.replace("._sink_holder.weight", ".sinks")
                    .replace("._position_bias_holder.weight", ".position_bias")
            )
        ep_norms = {_strip_holder(k): v for k, v in ep_results[0]["per_param_grad_norm"].items()}

        header = f"  {'parameter':<75s} {'bl_norm':>12s} {'ep_norm':>12s} {'ratio':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name in sorted(bl_norms.keys()):
            bl_n = bl_norms[name]
            ep_n = ep_norms[name]
            ratio = ep_n / bl_n if bl_n > 1e-12 else float("nan")
            print(f"  {name:<75s} {bl_n:12.6f} {ep_n:12.6f} {ratio:8.4f}")
            if bl_n > 1e-10:
                self.assertAlmostEqual(
                    ratio, 1.0, places=1,
                    msg=f"Grad norm mismatch: {name}: "
                        f"baseline={bl_n:.6f}, ep={ep_n:.6f}, ratio={ratio:.4f}",
                )

        # --- Total grad norm ---
        bl_gnorm = baseline_results[0]["total_grad_norm"]
        ep_gnorm = ep_results[0]["total_grad_norm"]
        gnorm_rel = abs(bl_gnorm - ep_gnorm) / max(bl_gnorm, 1e-12)
        print(
            f"\n  baseline total_grad_norm   = {bl_gnorm:.6f}\n"
            f"  {ep_tag} total_grad_norm      = {ep_gnorm:.6f}\n"
            f"  rel_diff                   = {gnorm_rel:.6f}"
        )
        self.assertLess(
            gnorm_rel, rtol,
            f"Total grad norm mismatch: baseline={bl_gnorm:.6f} vs {ep_tag}={ep_gnorm:.6f} "
            f"(rel_diff={gnorm_rel:.6f} > rtol={rtol})",
        )

        # --- Optimizer Δnorm 诊断 ---
        print("\n--- optimizer Δnorm diagnostics (rank 0) ---")
        bl_before = baseline_results[0]["param_norms_before"]
        bl_after = baseline_results[0]["param_norms_after"]
        ep_before = ep_results[0]["param_norms_before"]
        ep_after = ep_results[0]["param_norms_after"]
        header = f"  {'parameter':<75s} {'bl_Δ':>12s} {'ep_Δ':>12s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name in sorted(bl_before.keys()):
            bl_d = bl_after.get(name, 0) - bl_before.get(name, 0)
            ep_d = ep_after.get(name, 0) - ep_before.get(name, 0)
            print(f"  {name:<75s} {bl_d:12.6e} {ep_d:12.6e}")

        print("\nPASSED")

    def _fsdp_vs_ep_cp_impl(
        self,
        seq_len: int = SEQ_LEN,
        attn_backend: str = "fused",
        indexer_backend: str = "fused",
        baseline_world_size: int = NUM_GPUS // 2,
        epcp_world_size: int = NUM_GPUS,
        ep_size: int = 4,
        cp_size: int = 2,
        master_port_base: int = 12500,
        rtol: float = 0.02,
        ep_backend: str = "eager",
    ):
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}; "
            f"download DeepSeek-V4-Flash into hf-hub/ first"
        )
        assert seq_len % cp_size == 0, (
            f"seq_len ({seq_len}) must be divisible by cp_size ({cp_size})"
        )
        assert epcp_world_size == baseline_world_size * cp_size, (
            f"epcp_world_size ({epcp_world_size}) must equal "
            f"baseline_world_size ({baseline_world_size}) * cp_size ({cp_size}); "
            f"the EP+CP run pairs every cp_size ranks to one baseline sample"
        )
        assert epcp_world_size % ep_size == 0, (
            f"epcp_world_size ({epcp_world_size}) must be divisible by "
            f"ep_size ({ep_size})"
        )

        pg = _create_placement_group(epcp_world_size)

        print("=" * 60)
        print(
            f"Running baseline + routing capture "
            f"(baseline_world_size={baseline_world_size}, seq_len={seq_len}) ..."
        )
        baseline_results = self._run_baseline(
            baseline_world_size, pg, master_port_base,
            seq_len=seq_len,
            record_routing=True,
        )

        replay_indices_per_rank = [
            baseline_results[r // cp_size]["recorded_routing"]
            for r in range(epcp_world_size)
        ]
        for r in (0, epcp_world_size - 1):
            print(
                f"  ep rank {r} ← baseline rank {r // cp_size}: "
                f"{len(replay_indices_per_rank[r])} TopKRouter layers of routing indices"
            )

        print("=" * 60)
        print(
            f"Running EP+CP + router replay (ep_size={ep_size}, "
            f"cp_size={cp_size}, attn={attn_backend}, seq_len={seq_len}) ..."
        )
        ep_results = self._run_ep_cp(
            epcp_world_size, pg, master_port_base + 1,
            ep_size=ep_size, cp_size=cp_size,
            seq_len=seq_len,
            replay_indices_per_rank=replay_indices_per_rank,
            attn_backend=attn_backend,
            indexer_backend=indexer_backend,
            ep_backend=ep_backend,
        )

        remove_placement_group(pg[0])

        self._assert_results(
            baseline_results, ep_results,
            f"ep{ep_size}_cp{cp_size}_{attn_backend}_{ep_backend}_s{seq_len}", cp_size, rtol,
        )

    def test_fsdp_vs_ep_cp(self):
        """EP+CP fused attn vs baseline, seq_len=256."""
        self._fsdp_vs_ep_cp_impl(seq_len=256, attn_backend="fused", indexer_backend="fused")

    def test_fsdp_vs_ep_cp_eager(self):
        """EP+CP eager attn + fused indexer vs baseline, seq_len=256."""
        self._fsdp_vs_ep_cp_impl(seq_len=256, attn_backend="eager", indexer_backend="eager", ep_backend="eager",
                                  master_port_base=12600)

    def test_fsdp_vs_ep_cp_s512(self):
        """EP+CP fused attn vs baseline, seq_len=512."""
        self._fsdp_vs_ep_cp_impl(
            seq_len=512,
            attn_backend="fused",
            indexer_backend="fused",
            ep_backend="deepep",
            master_port_base=12700,
        )

    def test_fsdp_vs_ep_cp_s1024(self):
        """EP+CP fused attn vs baseline, seq_len=1024."""
        self._fsdp_vs_ep_cp_impl(
            seq_len=1024,
            attn_backend="fused",
            indexer_backend="fused",
            ep_backend="deepep",
            master_port_base=12800,
        )

    def test_cp_memory_scaling(
        self,
        ep_size: int = 8,
        master_port_base: int = 13000,
    ):
        """验证 cp_size 增大时 peak 显存单调下降。

        固定 ep_size=8, seq_len=4096, world_size=32，扫 cp_size ∈ {2,4,8,16,32}。
        seq_len=4096 保证最大 cp_size=32 时 s_local=128 满足 HCA compress_ratio=128 整除约束。
        """
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}"
        )
        world_size = NUM_GPUS
        seq_len = 64 * 1024
        cp_sizes = [4, 8, 16, 32]
        input_modes = ["random", "all_pad"]

        summaries = {}

        for mode_i, input_mode in enumerate(input_modes):
            summary = {}
            summaries[input_mode] = summary
            for i, cp_size in enumerate(cp_sizes):
                s_local = seq_len // cp_size
                assert s_local % 128 == 0, (
                    f"s_local={s_local} (seq_len={seq_len}/cp_size={cp_size}) "
                    f"not divisible by 128 (HCA compress_ratio)"
                )
                assert world_size % cp_size == 0
                assert world_size % ep_size == 0

                pg = placement_group(
                    [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
                )
                ray.get(pg.ready())

                print("=" * 60)
                print(
                    f"[mem-scaling] mode={input_mode}, cp_size={cp_size}, "
                    f"ep_size={ep_size}, seq_len={seq_len}, "
                    f"s_local={s_local}, world_size={world_size}"
                )

                results = self._run_ep_cp(
                    world_size, pg, master_port_base + mode_i * 100 + i,
                    ep_size=ep_size, cp_size=cp_size,
                    seq_len=seq_len,
                    memory_only=True,
                    input_mode=input_mode,
                )

                remove_placement_group(pg)

                peaks = [r["mem_peak_fwd_bwd_gib"] for r in results]
                after_shard = [r["mem_after_shard_gib"] for r in results]
                avg_peak = sum(peaks) / len(peaks)
                max_peak = max(peaks)
                avg_shard = sum(after_shard) / len(after_shard)
                loss = results[0]["loss"]

                summary[cp_size] = {
                    "avg_peak": avg_peak,
                    "max_peak": max_peak,
                    "avg_shard": avg_shard,
                    "loss": loss,
                }
                print(
                    f"[mem-scaling] mode={input_mode} cp={cp_size}: "
                    f"fwd_bwd_peak avg={avg_peak:.2f}G max={max_peak:.2f}G "
                    f"shard={avg_shard:.2f}G loss={loss:.4f}"
                )

        # 汇总表
        for input_mode, summary in summaries.items():
            print("\n" + "=" * 60)
            print(
                f"CP Memory Scaling Summary ({input_mode}) "
                f"(ep={ep_size}, seq_len={seq_len}, layers={NUM_LAYERS}, ws={world_size})"
            )
            print("=" * 60)
            header = (
                f"{'cp':>4s} {'s_local':>8s} "
                f"{'peak_avg':>10s} {'peak_max':>10s} {'shard':>8s} "
                f"{'vs_base':>7s} {'loss':>10s}"
            )
            print(header)
            print("-" * len(header))
            base_peak = summary[cp_sizes[0]]["avg_peak"]
            for cp_size in cp_sizes:
                s = summary[cp_size]
                ratio = s["avg_peak"] / base_peak
                print(
                    f"{cp_size:>4d} {seq_len // cp_size:>8d} "
                    f"{s['avg_peak']:>9.2f}G {s['max_peak']:>9.2f}G "
                    f"{s['avg_shard']:>7.2f}G "
                    f"{ratio:>6.2f}x {s['loss']:>10.4f}"
                )
            print("=" * 60)

            prev_peak = None
            for cp_size in cp_sizes:
                cur_peak = summary[cp_size]["avg_peak"]
                if prev_peak is not None:
                    self.assertLess(
                        cur_peak, prev_peak,
                        f"{input_mode}: peak mem did not decrease: cp={cp_size} "
                        f"({cur_peak:.2f}G) >= previous ({prev_peak:.2f}G)",
                    )
                prev_peak = cur_peak
