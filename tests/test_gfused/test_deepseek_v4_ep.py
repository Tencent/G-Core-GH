# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""正确性测试：HF 原生 FSDP vs 我们的 EP 实现（DeepSeek-V4）。

验证 logits / loss / grad norm 在容差内一致。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_ep.py
"""

import math
import os
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
from gpatch_v4.models.deepseek_v4.router_replay import (
    capture_routing_decisions,
    router_replay_ctx,
)

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
NUM_GPUS = 32
NUM_LAYERS = 4  # 最小前缀，覆盖 3 种 attn + 2 种 moe 类型
SEQ_LEN = 256  # HCA m'=128 需要 s_local>=128，cp=2 时 s_local=SEQ_LEN/2


def _truncate_config(config):
    """截断到 NUM_LAYERS 层以加速测试。"""
    if NUM_LAYERS is not None:
        config.num_hidden_layers = NUM_LAYERS
        config.layer_types = config.layer_types[:NUM_LAYERS]
        config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
    config.num_nextn_predict_layers = 0
    return config


def _get_layers(model):
    return model.model.layers


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip():
    return ray.util.get_node_ip_address()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Ray workers
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
def _baseline_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    seq_len: int = SEQ_LEN,
    record_routing: bool = False,
    dp_size: int = 1,
):
    """Baseline: HF 原生 DSV4 + 纯 FSDP。seed = 42 + rank // dp_size。"""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    dp_mesh = init_device_mesh("cuda", mesh_shape=(world_size,))

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rng = torch.Generator().manual_seed(42 + (rank // dp_size))
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, seq_len), generator=rng
    ).cuda()

    print(f"[baseline] rank {rank}: loading {hf_model_path} ...")
    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
    model = _HfModel.from_pretrained(
        hf_model_path,
        config=config,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        quantization_config=FineGrainedFP8Config(dequantize=True),
    )
    # E8M0 dequant 输出 bf16，强制转 fp32 以满足 FSDP2 uniform-dtype 要求
    model.to(torch.float32)
    model.gradient_checkpointing_enable()

    layers = _get_layers(model)

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
    if recorded is not None:
        result["recorded_routing"] = [t.cpu() for t in recorded]
    return result


@ray.remote(num_gpus=1)
def _ep_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    seq_len: int = SEQ_LEN,
    replay_indices: "list[torch.Tensor] | None" = None,
    dp_size: int = 1,
):
    """我们的 EP 实现：meta-device init + load_checkpoint_hp。"""

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

    rng = torch.Generator().manual_seed(42 + (rank // dp_size))
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, seq_len), generator=rng
    ).cuda()

    tag = f"ep{ep_size}" if replay_indices is None else f"ep{ep_size}_replay"
    print(f"[{tag}] rank {rank}: loading {hf_model_path} via meta-device ...")

    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)

    assert not config.tie_word_embeddings, (
        "meta-device init may hang with tie_word_embeddings=True; "
        "see fsdp2_backend/mixin.py:111"
    )

    if rank == 0:
        _verify_stacked_experts(hf_model_path, config)

    dist.barrier()

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    model.gradient_checkpointing_enable()

    model = apply_hp(model, ep_2d_mesh, amp_fp32=False)
    model.load_checkpoint_hp(hf_model_path)

    if replay_indices is not None:
        ctx = router_replay_ctx(model, [t.cuda() for t in replay_indices])
    else:
        ctx = nullcontext()

    with ctx:
        return _run_fwd_bwd(
            model, input_ids, tokenizer, rank, world_size, tag,
            ep_group=model._ep_group, ep_fsdp_mesh=model._ep_fsdp_mesh,
        )



def _verify_stacked_experts(hf_model_path: str, config) -> None:
    """快速校验 checkpoint 中 per-expert FP4 key 格式是否正确。"""
    from safetensors import safe_open

    index_path = os.path.join(hf_model_path, "model.safetensors.index.json")
    single_path = os.path.join(hf_model_path, "model.safetensors")

    import json

    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    elif os.path.isfile(single_path):
        weight_map = None
    else:
        return

    probe_key = "layers.0.ffn.experts.0.w1.weight"

    if weight_map is not None:
        if probe_key not in weight_map:
            return
        shard_file = weight_map[probe_key]
    else:
        shard_file = "model.safetensors"

    shard_path = os.path.join(hf_model_path, shard_file)
    with safe_open(shard_path, framework="pt") as f:
        if probe_key not in f.keys():
            return
        shape = tuple(f.get_tensor(probe_key).shape)

    # FP4 每字节打包 2 个 e2m1，磁盘 last dim = hidden_size / 2
    expected_packed_hidden = config.hidden_size // 2
    assert shape[-1] == expected_packed_hidden, (
        f"Unexpected packed FP4 shape for {probe_key}: got {shape}, "
        f"expected last dim = hidden_size/2 = {expected_packed_hidden}. "
        f"Checkpoint layout may have changed."
    )


def _run_fwd_bwd(model, input_ids, tokenizer, rank, world_size, tag,
                 ep_group=None, ep_fsdp_mesh=None, cp_group=None):
    """前向 + 反向 + optimizer step，返回诊断 dict。"""
    assert cp_group is None, (
        "test_deepseek_v4_ep.py is non-CP only; use test_deepseek_v4_ep_cp.py for CP coverage"
    )
    model.train()
    assert model.device.type == "cuda", f"expected cuda, got {model.device}"
    torch.cuda.reset_peak_memory_stats()
    mem_after_shard = torch.cuda.memory_allocated() / 1024**3
    print(f"[{tag}] rank {rank}: FSDP sharded, mem_allocated={mem_after_shard:.2f} GiB")

    full_labels = input_ids.clone()
    full_labels[full_labels == tokenizer.pad_token_id] = -100
    global_n = (full_labels != -100).sum()
    dist.all_reduce(global_n)

    labels = full_labels

    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    loss = loss / global_n
    reported_loss = loss.detach().clone()
    loss.backward()
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

    # --- Optimizer 验证 ---
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
        "mem_peak_gib": mem_peak,
    }

    del model, logits, outputs, optimizer
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestFsdpVsEp(unittest.TestCase):
    """HF 原生 FSDP vs EP 实现正确性对比。需要 >= 32 GPU。"""

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
        record_routing: bool = False,
        dp_size: int = 1,
    ) -> list[dict]:
        """启动 baseline workers（纯 FSDP）。"""
        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0,
                )
            ).remote()
        )
        futures = [
            _baseline_worker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=r,
                )
            ).remote(
                HF_MODEL_PATH,
                rank=r,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                record_routing=record_routing,
                dp_size=dp_size,
            )
            for r in range(world_size)
        ]
        return ray.get(futures)

    def _run_ep(
        self,
        world_size: int,
        pg: "ray.util.placement_group.PlacementGroup",
        master_port: int,
        ep_size: int,
        *,
        replay_indices_per_rank: "list[list[torch.Tensor]] | None" = None,
        dp_size: int = 1,
    ) -> list[dict]:
        """启动 EP workers。"""
        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0,
                )
            ).remote()
        )
        futures = [
            _ep_worker.options(
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
                replay_indices=(
                    replay_indices_per_rank[r]
                    if replay_indices_per_rank is not None
                    else None
                ),
                dp_size=dp_size,
            )
            for r in range(world_size)
        ]
        return ray.get(futures)

    def _assert_results(
        self,
        baseline_results: list[dict],
        ep_results: list[dict],
        ep_tag: str,
        world_size: int,
        rtol: float,
        rtol_per_param: "float | None" = None,
    ):
        """断言 baseline 和 EP 结果一致：logits / loss / grad norm / optimizer。"""
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

        # --- 逐 rank logits 对比 ---
        print("\n--- logits comparison (per rank) ---")
        for r in range(world_size):
            bl_logits = baseline_results[r]["logits"].float()
            ep_logits = ep_results[r]["logits"].float()
            max_abs = (bl_logits - ep_logits).abs().max().item()
            rel = max_abs / max(bl_logits.abs().max().item(), 1e-12)
            print(f"  rank {r}: logits max_abs_diff={max_abs:.2e}, rel_diff={rel:.2e}")
            if "_replay" in ep_tag and rel >= rtol:
                print(
                    f"    [triage] rank {r}: replay rel_diff={rel:.2e} exceeds "
                    f"rtol={rtol:.0e}; routing is NOT the dominant error source "
                    f"(expert/all-to-all/bf16 residual)."
                )
            self.assertLess(
                rel,
                rtol,
                f"Logits mismatch rank {r}: rel_diff={rel:.2e} > rtol={rtol}",
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
                if rtol_per_param is None:
                    self.assertAlmostEqual(
                        ratio, 1.0, places=1,
                        msg=f"Grad norm mismatch: {name}: "
                            f"baseline={bl_n:.6f}, ep={ep_n:.6f}, ratio={ratio:.4f}",
                    )
                else:
                    self.assertLess(
                        abs(ratio - 1.0),
                        rtol_per_param,
                        f"Grad norm mismatch: {name}: "
                        f"baseline={bl_n:.6f}, ep={ep_n:.6f}, ratio={ratio:.4f} "
                        f"(rtol_per_param={rtol_per_param})",
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

    def test_fsdp_vs_ep(
        self,
        world_size: int = NUM_GPUS,
        ep_size: int = 8,
        master_port_base: int = 12460,
        rtol: float = 0.05,
    ):
        """EP vs baseline：验证 logits / loss / grad norm 一致。"""
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}; "
            f"download DeepSeek-V4-Flash into hf-hub/ first"
        )

        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
        )
        ray.get(pg.ready())

        # --- 运行 baseline ---
        print("=" * 60)
        print(f"Running baseline (transformers native, world_size={world_size}) ...")
        baseline_results = self._run_baseline(
            world_size, pg, master_port_base
        )

        # --- 运行 EP ---
        print("=" * 60)
        print(f"Running EP (ep_size={ep_size}, world_size={world_size}) ...")
        ep_results = self._run_ep(
            world_size, pg, master_port_base + 1, ep_size=ep_size
        )

        remove_placement_group(pg)

        self._assert_results(
            baseline_results, ep_results, f"ep{ep_size}", world_size, rtol,
        )

    def test_fsdp_vs_ep_with_router_replay(
        self,
        world_size: int = NUM_GPUS,
        ep_size: int = 8,
        master_port_base: int = 12480,
        rtol: float = 5e-3,
    ):
        self.skipTest("skipped") # 重复测试

        """带 router replay 的 EP vs baseline，隔离路由漂移。"""
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}; "
            f"download DeepSeek-V4-Flash into hf-hub/ first"
        )

        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
        )
        ray.get(pg.ready())

        # --- Baseline + 录制路由 ---
        print("=" * 60)
        print(f"Running baseline + routing capture (world_size={world_size}) ...")
        baseline_results = self._run_baseline(
            world_size, pg, master_port_base, record_routing=True,
        )

        # 收集路由索引
        replay_indices_per_rank = [
            res["recorded_routing"] for res in baseline_results
        ]
        for r, indices in enumerate(replay_indices_per_rank):
            print(f"  rank {r}: captured {len(indices)} layers of routing indices")

        # --- EP + router replay ---
        print("=" * 60)
        print(f"Running EP + router replay (ep_size={ep_size}, world_size={world_size}) ...")
        ep_results = self._run_ep(
            world_size,
            pg,
            master_port_base + 1,
            ep_size=ep_size,
            replay_indices_per_rank=replay_indices_per_rank,
        )

        remove_placement_group(pg)

        self._assert_results(
            baseline_results, ep_results,
            f"ep{ep_size}_replay", world_size, rtol,
        )


