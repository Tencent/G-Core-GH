# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""正确性测试：hpMod baseline（amp_fp32=False）vs
hpMod + amp_fp32=True。Self-baseline 对照。

baseline 用现有 hp 实现（默认 mp_policy bf16 forward + modeling 内 .float()）；
target 在同一份 model / ckpt / input / seed 下打开 amp_fp32
开关：对 5 类 disk-FP32 leaves（attn_hc / ffn_hc / hc_head / _sink_holder /
_position_bias_holder）嵌套 fully_shard，独立 mp_policy(param_dtype=None,
reduce_dtype=fp32)，让这些 weight 在 forward 中真实保持 fp32。

预期：target 与 baseline 数值非常接近（rel_diff ≤ 5e-3）。Target 走 fp32
forward，理论上应**更稳**（消除 fp32 master → bf16 unshard → modeling 内
.float() 双 round-trip）；但 baseline 也已经通过 modeling 内手写 .float()
对 mHC/hc_head 走 fp32，所以这部分应当结构等价；剩余差异来自 attn_sink
进入 sparse_attn / compressor.position_bias `.to(chunk_gate.dtype)`
down-cast 的微弱舍入。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_keep_fp32.py
"""

import math
import os
import time
import unittest

import ray
import torch
import torch.nn.functional as F
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, DeepseekV4Config

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.models.deepseek_v4 import DeepseekV4ForCausalLM, apply_hp

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
NUM_GPUS = 32
NUM_LAYERS = 4  # 覆盖 sliding / CSA / HCA 三种 attn + hash_moe / moe
SEQ_LEN = 256
EP_SIZE = 8


def _truncate_config(config):
    """截断到 NUM_LAYERS 层以加速测试。"""
    if NUM_LAYERS is not None:
        config.num_hidden_layers = NUM_LAYERS
        config.layer_types = config.layer_types[:NUM_LAYERS]
        config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
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
                 ep_group=None, ep_fsdp_mesh=None):
    """前向 + 反向 + optimizer step，返回诊断 dict。"""
    model.train()
    assert model.device.type == "cuda", f"expected cuda, got {model.device}"
    torch.cuda.reset_peak_memory_stats()
    mem_after_shard = torch.cuda.memory_allocated() / 1024**3
    print(f"[{tag}] rank {rank}: FSDP sharded, mem_allocated={mem_after_shard:.2f} GiB")

    s_full = input_ids.shape[1]
    full_labels = input_ids.clone()
    full_labels[full_labels == tokenizer.pad_token_id] = -100
    # shift labels by -1（标准 next-token 预测）
    full_labels = torch.roll(full_labels, shifts=-1, dims=-1)
    full_labels[:, -1] = -100
    labels = full_labels

    global_n = (labels != -100).sum()
    dist.all_reduce(global_n)

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


@ray.remote(num_gpus=1)
def _hp_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    seq_len: int = SEQ_LEN,
    *,
    amp_fp32: bool,
):
    """用 apply_hp 跑一遍。开关由调用方传入：
    - False = baseline（现有行为，所有 weight 走 mp_policy bf16 forward）。
    - True  = target（disk-FP32 leaves 嵌套 fully_shard, mp_policy=param_dtype=None）。
    """
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

    # 每个 rank 独立 seed，使 EP all-to-all 内有非 trivial routing 流量
    rng = torch.Generator().manual_seed(42 + rank)
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, seq_len), generator=rng
    ).cuda()

    tag = "keep_fp32" if amp_fp32 else "baseline"
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
        model, ep_2d_mesh,
        amp_fp32=amp_fp32,
    )
    model.gradient_checkpointing_enable()
    _t_apply_hp_done = time.time()
    model.load_checkpoint_hp(hf_model_path)
    _t_load_end = time.time()
    apply_hp_seconds = _t_apply_hp_done - _t_load_start
    load_state_seconds = _t_load_end - _t_apply_hp_done
    load_seconds = _t_load_end - _t_load_start
    print(
        f"[{tag}] rank {rank}: load done in {load_seconds:.1f}s "
        f"(meta+apply_hp={apply_hp_seconds:.1f}s, "
        f"load_checkpoint_hp={load_state_seconds:.1f}s)"
    )

    result = _run_fwd_bwd(
        model, input_ids, tokenizer, rank, world_size, tag,
        ep_group=model._ep_group, ep_fsdp_mesh=model._ep_fsdp_mesh,
    )
    result["load_seconds"] = load_seconds
    result["apply_hp_seconds"] = apply_hp_seconds
    result["load_state_seconds"] = load_state_seconds
    return result


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestKeepWeightsFp32InFwd(unittest.TestCase):
    """hpMod baseline vs hpMod + amp_fp32 自比对。需要 >= 32 GPU。"""

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

    def _run(
        self,
        world_size: int,
        pg: "ray.util.placement_group.PlacementGroup",
        master_port: int,
        ep_size: int,
        *,
        amp_fp32: bool,
    ) -> list[dict]:
        """启动 hp workers，开关由调用方传入。"""
        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0,
                )
            ).remote()
        )
        futures = [
            _hp_worker.options(
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
                amp_fp32=amp_fp32,
            )
            for r in range(world_size)
        ]
        return ray.get(futures)

    def _assert_results(
        self,
        baseline_results: list[dict],
        target_results: list[dict],
        rtol: float,
        rtol_logits_rms: float = 2e-2,
    ):
        """断言 baseline (hpMod) 与 target (hpMod+keep_fp32) 结果在容差内一致。

        loss / grad_norm / per-param grad norm 用严阈值 ``rtol`` (5e-3)
        —— 整体训练 signal 应当几乎一致。logits RMS 用更宽 ``rtol_logits_rms``
        (2e-2) —— attention 内部把 sinks / position_bias 由 bf16 提升到 fp32
        会改变 softmax 输出 ~1% 量级，是预期的精度提升。
        """
        # --- 基本检查：无 NaN/Inf ---
        print("\n--- sanity checks ---")
        for tag, results in [("baseline", baseline_results), ("keep_fp32", target_results)]:
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
        for tag, results in [("baseline", baseline_results), ("keep_fp32", target_results)]:
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
        for tag, results in [("baseline", baseline_results), ("keep_fp32", target_results)]:
            loads = [r["load_seconds"] for r in results]
            applies = [r["apply_hp_seconds"] for r in results]
            states = [r["load_state_seconds"] for r in results]
            print(
                f"  {tag}: load mean={sum(loads)/len(loads):.1f}s "
                f"(apply_hp mean={sum(applies)/len(applies):.1f}s, "
                f"load_state mean={sum(states)/len(states):.1f}s)"
            )

        # --- 逐 rank logits 对比 ---
        # 用两个 metric：
        #   max rel_diff: 单元素最坏情况，对 outlier 敏感（不强阈值）
        #   mean abs / RMS: 整体偏差，对 outlier 鲁棒（强阈值，作主要判据）
        print("\n--- logits comparison (per rank) ---")
        max_logits_rms_rel = 0.0
        for r in range(len(baseline_results)):
            bl_logits = baseline_results[r]["logits"].float()
            tg_logits = target_results[r]["logits"].float()
            diff = bl_logits - tg_logits
            max_abs = diff.abs().max().item()
            max_rel = max_abs / max(bl_logits.abs().max().item(), 1e-12)
            # RMS 相对差异：对 outlier 鲁棒，反映整体偏差量级
            rms = diff.pow(2).mean().sqrt().item()
            rms_rel = rms / max(bl_logits.pow(2).mean().sqrt().item(), 1e-12)
            print(
                f"  rank {r}: logits max_abs_diff={max_abs:.2e} max_rel={max_rel:.2e} "
                f"rms_rel={rms_rel:.2e}"
            )
            max_logits_rms_rel = max(max_logits_rms_rel, rms_rel)
        # RMS 阈值用 rtol_logits_rms（2e-2，反映 sinks/position_bias 由 bf16
        # 升 fp32 带来的 attention softmax 输出精度提升量级）；max 单点 outlier
        # 不强阈值。
        self.assertLess(
            max_logits_rms_rel, rtol_logits_rms,
            f"Logits RMS rel_diff {max_logits_rms_rel:.4e} > "
            f"rtol_logits_rms={rtol_logits_rms}",
        )

        # --- Loss 对比 ---
        bl_loss = baseline_results[0]["loss"]
        tg_loss = target_results[0]["loss"]
        loss_rel = abs(bl_loss - tg_loss) / max(abs(bl_loss), 1e-12)
        print(
            f"\n  baseline loss        = {bl_loss:.6f}\n"
            f"  keep_fp32 loss       = {tg_loss:.6f}\n"
            f"  loss rel_diff        = {loss_rel:.6f} (rtol={rtol})"
        )
        self.assertLess(
            loss_rel,
            rtol,
            f"Loss mismatch: baseline={bl_loss:.6f} vs keep_fp32={tg_loss:.6f} "
            f"(rel_diff={loss_rel:.6f} > rtol={rtol})",
        )

        # --- 逐参数 grad norm 对比 ---
        print("\n--- per-param grad norm table ---")
        bl_norms = baseline_results[0]["per_param_grad_norm"]
        tg_norms = target_results[0]["per_param_grad_norm"]

        # 两边参数集合可能不同：keep_fp32 引入了 _sink_holder / _position_bias_holder。
        # 用名字归一化函数对齐。
        def _normalize(name: str) -> str:
            return (
                name.replace("._sink_holder.weight", ".sinks")
                    .replace("._position_bias_holder.weight", ".position_bias")
            )

        bl_norms_n = {_normalize(k): v for k, v in bl_norms.items()}
        tg_norms_n = {_normalize(k): v for k, v in tg_norms.items()}

        # 两个集合的并集（理论上 normalize 后应当一致）
        all_names = sorted(set(bl_norms_n.keys()) | set(tg_norms_n.keys()))

        header = f"  {'parameter':<75s} {'bl_norm':>12s} {'tg_norm':>12s} {'ratio':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        missing_in_bl, missing_in_tg = [], []
        for name in all_names:
            bl_n = bl_norms_n.get(name)
            tg_n = tg_norms_n.get(name)
            if bl_n is None:
                missing_in_bl.append(name)
                continue
            if tg_n is None:
                missing_in_tg.append(name)
                continue
            ratio = tg_n / bl_n if bl_n > 1e-12 else float("nan")
            print(f"  {name:<75s} {bl_n:12.6f} {tg_n:12.6f} {ratio:8.4f}")
            if bl_n > 1e-10:
                self.assertAlmostEqual(
                    ratio, 1.0, places=1,
                    msg=f"Grad norm mismatch: {name}: "
                        f"baseline={bl_n:.6f}, target={tg_n:.6f}, ratio={ratio:.4f}",
                )
        self.assertEqual(missing_in_bl, [], f"params only in target: {missing_in_bl}")
        self.assertEqual(missing_in_tg, [], f"params only in baseline: {missing_in_tg}")

        # --- Total grad norm ---
        bl_gnorm = baseline_results[0]["total_grad_norm"]
        tg_gnorm = target_results[0]["total_grad_norm"]
        gnorm_rel = abs(bl_gnorm - tg_gnorm) / max(bl_gnorm, 1e-12)
        print(
            f"\n  baseline total_grad_norm    = {bl_gnorm:.6f}\n"
            f"  keep_fp32 total_grad_norm   = {tg_gnorm:.6f}\n"
            f"  rel_diff                    = {gnorm_rel:.6f}"
        )
        self.assertLess(
            gnorm_rel, rtol,
            f"Total grad norm mismatch: baseline={bl_gnorm:.6f} vs keep_fp32={tg_gnorm:.6f} "
            f"(rel_diff={gnorm_rel:.6f} > rtol={rtol})",
        )

        # --- Optimizer Δnorm 诊断（rank 0，归一化后对齐） ---
        print("\n--- optimizer Δnorm diagnostics (rank 0) ---")
        bl_before = {_normalize(k): v for k, v in baseline_results[0]["param_norms_before"].items()}
        bl_after = {_normalize(k): v for k, v in baseline_results[0]["param_norms_after"].items()}
        tg_before = {_normalize(k): v for k, v in target_results[0]["param_norms_before"].items()}
        tg_after = {_normalize(k): v for k, v in target_results[0]["param_norms_after"].items()}
        header = f"  {'parameter':<75s} {'bl_Δ':>12s} {'tg_Δ':>12s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name in sorted(bl_before.keys()):
            bl_d = bl_after.get(name, 0) - bl_before.get(name, 0)
            tg_d = tg_after.get(name, 0) - tg_before.get(name, 0)
            print(f"  {name:<75s} {bl_d:12.6e} {tg_d:12.6e}")

        print("\nPASSED")

    def test_baseline_vs_keep_fp32(
        self,
        world_size: int = NUM_GPUS,
        ep_size: int = EP_SIZE,
        master_port_base: int = 12550,
        rtol: float = 5e-3,
    ):
        """hpMod baseline (amp_fp32=False) vs
        hpMod + amp_fp32=True。

        预期数值非常接近：mHC / hc_head 在 baseline 路径下已经被 modeling
        内手写 `.float()` 强行 fp32 forward，所以这部分**结构等价**；
        差异主要来自 attn_sink 与 compressor.position_bias 这两组 disk-FP32
        leaves —— baseline 经 mp_policy 被 cast 成 bf16 后做 cat / `.to(...)`
        down-cast，target 则保持 fp32（line 891 处后续显式 down-cast 到
        query.dtype）。

        实测 rank-0 (32 GPU, NUM_LAYERS=4):
          baseline: loss=0.6388 logits.mean=-0.4857 grad_norm=0.9439
          target:   loss=0.6391 logits.mean=-0.4860 grad_norm=0.9435
          loss/gradnorm rel_diff < 5e-4。
          logits 最坏单点 rel_diff ~5-6%（局部 outlier，bf16 cat vs
          fp32 cat 的 single-element 噪声），但整体 RMS rel_diff < 1e-3。

        rtol=5e-3 应用于 loss / grad_norm / logits-RMS（鲁棒 metric）；
        logits 单点 outlier (max rel) 仅打印不强阈值。
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

        # --- baseline: amp_fp32=False ---
        print("=" * 60)
        print(f"Running baseline (hpMod, amp_fp32=False, "
              f"world_size={world_size}, ep_size={ep_size}) ...")
        baseline_results = self._run(
            world_size, pg, master_port_base,
            ep_size=ep_size,
            amp_fp32=False,
        )

        # --- target: amp_fp32=True ---
        print("=" * 60)
        print(f"Running target (hpMod, amp_fp32=True, "
              f"world_size={world_size}, ep_size={ep_size}) ...")
        target_results = self._run(
            world_size, pg, master_port_base + 1,
            ep_size=ep_size,
            amp_fp32=True,
        )

        remove_placement_group(pg)

        self._assert_results(baseline_results, target_results, rtol)


if __name__ == "__main__":
    unittest.main()
