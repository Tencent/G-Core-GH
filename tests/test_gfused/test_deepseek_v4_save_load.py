# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""save_checkpoint_hp 的 round-trip 测试（DSV4 EP+CP+FSDP2, bf16）。

验证 bf16 保存路径的正确性：
  0. Path-A: HF from_pretrained(orig, FineGrainedFP8Config(dequantize=True)) → sd_ref
  1. load_checkpoint_hp → sd1，验证 sd1 ≡ sd_ref（bit-equal）
  3. save_checkpoint_hp → 磁盘
  4. 重新 load_checkpoint_hp → sd2，验证 sd2 ≡ sd_ref（bit-equal）
  6. Path-B: from_pretrained(save_dir, bf16) → 逐 key 比对 sd_ref（用户验收）

任何非零 diff 都表示 bug（EP gather 顺序 / weight_map / quantization_config 未剥离等）。

另外新增 e_score_correction_bias 专项 round-trip（``test_e_score_bias_roundtrip_*``）：
  1. 加载真实权重后，把所有 router 的 ``e_score_correction_bias`` 覆写成
     distinct dummy values（不同层 router / 同 router 不同 expert 位置数值都不同）；
  2. ``save_checkpoint_hp`` → 重新 ``load_checkpoint_hp`` → 逐 buffer bit-equal 比较。
分别覆盖 quantized / bf16 两条保存路径（该 buffer 均走 f32_passthrough，保 fp32）。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=3600 tests/test_gfused/test_deepseek_v4_save_load.py
"""

import os
import gc
import re
import shutil
import unittest
from packaging.version import Version
from importlib.metadata import version as pkg_version

import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM, apply_hp
# Path-A / Path-B baseline 用 HF 原版 modeling，跟 gpatch_v4 patched 版本解耦：
# patched 版把 ``sinks`` / ``position_bias`` 包进了 ``_Fp32ParamHolder`` 让
# FSDP2 能 wrap 单 Parameter，导致 vanilla HF rename target (``self_attn.sinks``)
# 在 patched modeling 里是 ``@property`` 不是 ``nn.Parameter``，HF load 时 weight
# 落不进去。Phase 0 / Phase 6 的语义本来就是「开源用户用 HF 原版能不能 load」，
# 所以这里**必须**用 ``transformers.DeepseekV4ForCausalLM``。
from transformers import DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
NUM_GPUS = 32
EP_SIZE = 8
CP_SIZE = 2
SEQ_LEN = 256

NUM_LAYERS: "int | None" = 4

# 共享 FS 上的临时存储路径，setUp/tearDown 负责清理
SAVE_DIR = "dsv4_save_load_test"


def _truncate_config(config):
    """截断到 NUM_LAYERS 层。"""
    if NUM_LAYERS is not None:
        config.num_hidden_layers = NUM_LAYERS
        config.layer_types = config.layer_types[:NUM_LAYERS]
        config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
    return config


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip():
    return ray.util.get_node_ip_address()


def _print_rss(tag: str) -> None:
    """打印 rank-0 进程 RSS + 节点可用内存（用于 Stage-2 内存追踪）。"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    vmrss_kb = int(line.split()[1])
                    break
            else:
                vmrss_kb = -1
        with open("/proc/meminfo") as f:
            mi = {ln.split(":")[0]: int(ln.split()[1]) for ln in f if ":" in ln}
        avail_kb = mi.get("MemAvailable", 0)
        used_kb = mi.get("MemTotal", 0) - avail_kb
        print(
            f"{tag}  [proc RSS={vmrss_kb / 1024 / 1024:.1f} GB; "
            f"node used/total={used_kb / 1024 / 1024:.1f}/{mi['MemTotal'] / 1024 / 1024:.1f} GB]",
            flush=True,
        )
    except Exception as e:
        print(f"{tag}  [_print_rss failed: {e!r}]", flush=True)


# ---------------------------------------------------------------------------
# 逐 key round-trip 断言（bf16 bit-equal）
# ---------------------------------------------------------------------------


def assert_round_trip(name: str, t1: torch.Tensor, t2: torch.Tensor) -> dict:
    """比较两个 tensor 快照，要求 bit-equal。不一致时抛 AssertionError 并附诊断信息。"""
    if not t1.dtype.is_floating_point:
        if not torch.equal(t1, t2):
            raise AssertionError(
                f"{name}: integer-buffer mismatch (dtype={t1.dtype})"
            )
        return {"name": name, "ok": True, "max_abs_diff": 0.0}

    # fp32 storage: bit-equal expected. torch.equal does element-wise eq
    # with no tolerance — catches any sub-ULP drift.
    if torch.equal(t1, t2):
        return {"name": name, "ok": True, "max_abs_diff": 0.0}

    # 只在失败路径构建诊断，避免每个 key 都 fp32 拷贝导致 2x 内存
    diff = (t1.float() - t2.float()).abs()
    max_abs_diff = diff.max().item()
    max_abs_t1 = t1.abs().max().item()
    rel = max_abs_diff / max(max_abs_t1, 1e-12)

    extra = ""
    # 对 stacked-experts tensor 输出 per-expert 诊断，方便定位 EP gather 顺序 bug
    if (".experts.gate_up_proj" in name or ".experts.down_proj" in name) and (
        t1.ndim >= 2 and t1.shape[0] >= 8
    ):
        per_expert = diff.reshape(t1.shape[0], -1)
        per_expert_max = per_expert.max(dim=1).values
        zeros = (per_expert_max == 0).sum().item()
        worst = per_expert_max.argsort(descending=True)[:8].tolist()
        extra = (
            f"\n  per-expert: bit-equal={zeros}/{t1.shape[0]}; "
            "worst 8 (idx, max_abs_diff): "
            + ", ".join(f"({i},{per_expert_max[i]:.3e})" for i in worst)
        )

    raise AssertionError(
        f"{name}: fp32 bit-equal expected, got max_abs_diff={max_abs_diff:.4e} "
        f"rel={rel:.4e} (dtype={t1.dtype}, shape={list(t1.shape)})" + extra
    )


# ---------------------------------------------------------------------------
# 共享 helper（模块顶层定义以便 Ray pickle）
# ---------------------------------------------------------------------------


def _stream_full_state_dict(model: torch.nn.Module, on_rank0=None) -> int:
    """逐 key gather 完整 state_dict，rank 0 回调 on_rank0(name, t_cpu)。

    expert 参数通过两步 ep_fsdp → ep 聚合。内存峰值为单个 tensor（~5 GB）。
    """
    is_rank0 = dist.get_rank() == 0
    n = 0
    for name, p in model.state_dict().items():
        is_expert = ".experts.gate_up_proj" in name or ".experts.down_proj" in name
        if isinstance(p, DTensor):
            if is_expert and model._ep_size > 1:
                local = p.to_local().contiguous()
                dt_fsdp = DTensor.from_local(
                    local, device_mesh=model._ep_fsdp_mesh, placements=[Shard(0)],
                )
                ep_local_full = dt_fsdp.full_tensor()
                ep_chunks = [torch.empty_like(ep_local_full) for _ in range(model._ep_size)]
                dist.all_gather(ep_chunks, ep_local_full, group=model._ep_group)
                t = torch.cat(ep_chunks, dim=0)
                del ep_local_full, ep_chunks
            else:
                t = p.full_tensor()
        else:
            t = p.detach()

        if is_rank0 and on_rank0 is not None:
            t_cpu = t.cpu().contiguous()
            on_rank0(name, t_cpu)
            del t_cpu

        del t
        torch.cuda.empty_cache()
        n += 1
    return n


# ---------------------------------------------------------------------------
# e_score_correction_bias round-trip helpers
# ---------------------------------------------------------------------------
#
# ``e_score_correction_bias`` 是 DeepseekV4TopKRouter 上的 persistent buffer
# （replicated 普通 fp32 tensor，非 DTensor）。save 时映射到 disk key
# ``layers.{i}.ffn.gate.bias`` 走 f32_passthrough（quantized / bf16 两条路径
# 都保 fp32），因此 dummy 值可 bit-equal round-trip。
#

def _iter_e_score_bias_names(model: torch.nn.Module) -> list[str]:
    """按确定性（排序后）顺序返回所有 TopKRouter e_score_correction_bias buffer 名。

    排序保证跨 rank 的枚举顺序一致（dummy 值与 buffer 的绑定对每个 rank 相同）。
    """
    names = [
        name for name, _ in model.named_buffers()
        if name.endswith("e_score_correction_bias")
    ]
    return sorted(names)


def _make_dummy_e_score_bias(idx: int, num_experts: int) -> torch.Tensor:
    """为第 ``idx`` 个 router 生成一批 distinct fp32 dummy。

    * ``base = 100 * (idx + 1)`` —— 区分不同层  router；
    * ``ramp = arange(E) * 0.01`` —— 区分同一 router 内不同 expert 位置。

    不同 router 的取值区间互不相交（间隔 100 ≫ ramp 最大值），且每个元素的
    间距（0.01）远大于该量级下的 fp32 ULP，任何"层错配 / expert 错序"都会被
    bit-equal 比较捕获。
    """
    base = 100.0 * (idx + 1)
    ramp = torch.arange(num_experts, dtype=torch.float32) * 0.01
    return base + ramp


def _set_dummy_e_score_bias(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """把 model 内所有 e_score_correction_bias 覆写成 distinct dummy。

    返回 ``{name: cpu fp32 tensor}`` 作为期望值。buffer 是 replicated，每个
    rank 写入相同的确定性值。
    """
    buffers = dict(model.named_buffers())
    expected: dict[str, torch.Tensor] = {}
    for idx, name in enumerate(_iter_e_score_bias_names(model)):
        buf = buffers[name]
        assert not isinstance(buf, DTensor), (
            f"{name}: expected replicated plain-tensor buffer, got DTensor"
        )
        assert buf.dtype == torch.float32, f"{name}: expected fp32 buffer, got {buf.dtype}"
        dummy = _make_dummy_e_score_bias(idx, buf.shape[-1])
        with torch.no_grad():
            buf.copy_(dummy.to(device=buf.device))
        # 存实际写进 buffer 的值（fp32 cpu 快照），避免任何隐性 dtype 偏差。
        expected[name] = buf.detach().float().cpu().clone()
    return expected


def _collect_e_score_bias(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """收集 model 内所有 e_score_correction_bias 的 fp32 cpu 快照。"""
    buffers = dict(model.named_buffers())
    return {
        name: buffers[name].detach().float().cpu().clone()
        for name in _iter_e_score_bias_names(model)
    }


# ---------------------------------------------------------------------------
# Ray worker
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
def _save_load_worker(
    hf_model_path: str,
    save_dir: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    cp_size: int,
    dtype_format: str,
):
    """每个 GPU 一个 Ray task：load → save → reload → compare。

    dtype_format
        "quantized" — FP4/FP8/E8M0 + 保留 quantization_config，Phase 6 走
        HF FineGrainedFP8 dequant reload。
        "bf16" — 折叠成 bf16 + strip quantization_config，Phase 6 走 vanilla
        bf16 reload（与原 §9/§10 实现一致）。
    """
    from datetime import timedelta

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    # Stage 2 全量模型 Phase 0 可能 30-60 min，加大 NCCL timeout
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=2),
    )

    assert world_size % ep_size == 0
    assert world_size % cp_size == 0
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

    # NCCL communicator warmup：在 Phase 0 之前强制完成 handshake，
    # 否则 rank 0 的长时间 from_pretrained 会导致其他 rank NCCL 超时
    dist.barrier()

    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
    assert not config.tie_word_embeddings, (
        "meta-device init may hang with tie_word_embeddings=True"
    )

    # ------------------------------------------------------------------
    # Phase 0 (rank 0 only): Path-A 参考，rank 1..N-1 在下方 barrier 等待
    # ------------------------------------------------------------------
    sd_ref: dict[str, torch.Tensor] = {}
    if rank == 0:
        print(f"[rank 0] phase 0: capturing Path-A reference (HF dequant from orig) ...", flush=True)
        from transformers import FineGrainedFP8Config

        rconfig = DeepseekV4Config.from_pretrained(hf_model_path)
        _truncate_config(rconfig)
        rmodel = HFDeepseekV4ForCausalLM.from_pretrained(
            hf_model_path,
            config=rconfig,
            device_map="cpu",
            torch_dtype=torch.float32,
            quantization_config=FineGrainedFP8Config(dequantize=True),
        )
        rmodel.float()
        for _, param in rmodel.named_parameters():
            assert param.dtype == torch.float32
        for k, v in rmodel.state_dict().items():
            t = v.detach()
            sd_ref[k] = t.cpu().contiguous()
        del rmodel
        gc.collect()
        _print_rss(f"[rank 0] sd_ref captured: {len(sd_ref)} tensors (Path A)")

    dist.barrier()
    # return {"rank": rank, "ok": True}

    print(f"[rank {rank}] phase 1: building meta model + load_checkpoint_hp ...", flush=True)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model1 = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    model1 = apply_hp(model1, ep_2d_mesh, cp_mesh=cp_mesh, amp_fp32=False)
    model1.load_checkpoint_hp(hf_model_path)

    print(f"[rank {rank}] phase 2 + 2.5: streaming sd1 capture + bit-equal compare against sd_ref ...", flush=True)
    # Phase 2+2.5 合并：流式 gather，逐 key 比对 sd_ref，避免 ~800 GB sd1 全量存在内存
    seen_keys_sd1: set[str] = set()

    # gpatch_v4 modeling 把 ``sinks`` / ``position_bias`` wrap 进了
    # ``_Fp32ParamHolder``；归一化回 HF modeling 的扁平名后才能跟 sd_ref 比对。
    # transformers>=5.10.1 把 ``weights_proj.weight`` 移到了 ``scorer`` 层。
    def _to_hf_key(name: str) -> str:
        name = (
            name.replace("._sink_holder.weight", ".sinks")
                .replace("._position_bias_holder.weight", ".position_bias")
        )
        if Version(pkg_version("transformers")) >= Version("5.10.1"):
            name = name.replace(".compressor.indexer.weights_proj.weight", ".compressor.indexer.scorer.weights_proj.weight")
        return name

    def _check_sd1(name: str, t1: torch.Tensor) -> None:
        name = _to_hf_key(name)
        if name.startswith('mtp.'):
            return
        seen_keys_sd1.add(name)
        tr = sd_ref[name]
        assert t1.shape == tr.shape, (
            f"sd1 vs sd_ref shape mismatch {name}: {t1.shape} vs {tr.shape}"
        )
        assert t1.dtype == tr.dtype, (
            f"sd1 vs sd_ref dtype mismatch {name}: {t1.dtype} vs {tr.dtype}"
        )
        assert_round_trip(name, t1, tr)

    n_keys_sd1 = _stream_full_state_dict(model1, on_rank0=_check_sd1)
    if rank == 0:
        # 验证 sd1 完整覆盖 sd_ref 的所有 key
        only_in_ref = set(sd_ref) - seen_keys_sd1
        assert not only_in_ref, (
            f"sd1 missed {len(only_in_ref)} Path-A keys (first 10): "
            f"{sorted(only_in_ref)[:10]}"
        )
        _print_rss(
            f"[rank 0] phase 2.5 PASS: sd1 ≡ sd_ref ({len(seen_keys_sd1)} tensors bit-equal)"
        )

    print(f"[rank {rank}] phase 3: save_checkpoint_hp({save_dir}, orig_ckpt_dir={hf_model_path}, dtype_format={dtype_format!r}) ...", flush=True)
    model1.save_checkpoint_hp(save_dir, orig_ckpt_dir=hf_model_path, dtype_format=dtype_format)
    dist.barrier()

    # 释放 model1，避免与 model2 同时占用 GPU
    del model1
    torch.cuda.empty_cache()

    print(f"[rank {rank}] phase 4: building meta model2 + load_checkpoint_hp(save_dir) ...", flush=True)
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model2 = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    model2 = apply_hp(model2, ep_2d_mesh, cp_mesh=cp_mesh, amp_fp32=False)
    model2.load_checkpoint_hp(save_dir)  # default fp32 to match sd_ref (post rmodel.float())

    print(f"[rank {rank}] phase 5 + 5.5: streaming sd2 capture + bit-equal compare against sd_ref ...", flush=True)
    # Phase 5+5.5 合并：sd2 vs sd_ref（由传递性等价于 sd2 vs sd1，无需 sd1 常驻内存）
    seen_keys_sd2: set[str] = set()

    def _check_sd2(name: str, t2: torch.Tensor) -> None:
        name = _to_hf_key(name)
        if name.startswith('mtp.'):
            return
        seen_keys_sd2.add(name)
        tr = sd_ref[name]
        assert t2.shape == tr.shape, (
            f"sd2 vs sd_ref shape mismatch {name}: {t2.shape} vs {tr.shape}"
        )
        assert t2.dtype == tr.dtype, (
            f"sd2 vs sd_ref dtype mismatch {name}: {t2.dtype} vs {tr.dtype}"
        )
        assert_round_trip(name, t2, tr)

    n_keys_sd2 = _stream_full_state_dict(model2, on_rank0=_check_sd2)
    if rank == 0:
        only_in_ref = set(sd_ref) - seen_keys_sd2
        assert not only_in_ref, (
            f"sd2 missed {len(only_in_ref)} Path-A keys (first 10): "
            f"{sorted(only_in_ref)[:10]}"
        )
        _print_rss(
            f"[rank 0] phase 5.5 PASS: sd2 ≡ sd_ref (transitively ≡ sd1) "
            f"({len(seen_keys_sd2)} tensors bit-equal)"
        )

    # Phase 6 prep：释放 model2，保留 sd_ref 供 Phase 6 比对
    del model2
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        # Phase 6 (rank 0 only): Path-B 用户验收——按 dtype_format 选 reload 路径
        if dtype_format == "quantized":
            # 与 Phase 0 完全对称——HF FineGrainedFP8 dequant
            _print_rss(f"[rank 0] phase 6 (quantized): HF FineGrainedFP8 reload (Path B) ...")

            vconfig = DeepseekV4Config.from_pretrained(save_dir)
            _truncate_config(vconfig)
            # 直接读 JSON 确认 quantization_config 被保留并与原 ckpt byte-equal
            import json as _json
            with open(os.path.join(save_dir, "config.json")) as _f:
                _saved_cfg_json = _json.load(_f)
            with open(os.path.join(hf_model_path, "config.json")) as _f:
                _orig_cfg_json = _json.load(_f)
            assert _saved_cfg_json.get("quantization_config") == _orig_cfg_json.get("quantization_config"), (
                f"saved config.json quantization_config drifted from orig:\n"
                f"  orig:  {_orig_cfg_json.get('quantization_config')!r}\n"
                f"  saved: {_saved_cfg_json.get('quantization_config')!r}"
            )

            from transformers import FineGrainedFP8Config
            vmodel = HFDeepseekV4ForCausalLM.from_pretrained(
                save_dir,
                config=vconfig,
                device_map="cpu",
                torch_dtype=torch.float32,
                quantization_config=FineGrainedFP8Config(dequantize=True),
            )
            # 与 Phase 0 (sd_ref 路径) 完全对称：FineGrainedFP8 dequant + .float() 强 cast
            # 让所有 weight（含非量化的 embed/norm bf16）都到 fp32，避免与 sd_ref bf16/fp32 错配。
            vmodel.float()
        else:  # bf16
            # 与原 §9 实现一致——vanilla bf16 加载（保留原 strip 校验断言）
            _print_rss(f"[rank 0] phase 6 (bf16): vanilla DSV4 from_pretrained smoke load (Path B) ...")

            vconfig = DeepseekV4Config.from_pretrained(save_dir)
            _truncate_config(vconfig)
            # 直接读 JSON 确认 quantization_config 已被剥离（from_pretrained 会合成属性，不可靠）
            import json as _json
            with open(os.path.join(save_dir, "config.json")) as _f:
                _saved_cfg_json = _json.load(_f)
            assert "quantization_config" not in _saved_cfg_json, (
                f"saved config.json still contains quantization_config="
                f"{_saved_cfg_json.get('quantization_config')!r}; "
                f"saver finalize did not strip it."
            )
            vmodel = HFDeepseekV4ForCausalLM.from_pretrained(
                save_dir,
                config=vconfig,
                device_map="cpu",
                torch_dtype=torch.float32,
            )

        # 流式逐 key 比对，避免同时持有 vmodel state_dict + sd_ref 两份完整拷贝
        v_keys: set[str] = set()
        n_compared = 0
        for k, v in vmodel.state_dict().items():
            v_keys.add(k)
            if k not in sd_ref:
                # vmodel 可能携带额外 key（如 mtp placeholder），跳过
                continue
            tr = sd_ref[k]
            tv = v.detach().contiguous()
            assert tr.shape == tv.shape, (
                f"Path-B vs Path-A shape mismatch {k}: {tr.shape} vs {tv.shape}"
            )
            assert tr.dtype == tv.dtype, (
                f"Path-B vs Path-A dtype mismatch {k}: {tr.dtype} vs {tv.dtype}"
            )
            assert_round_trip(k, tr, tv)
            n_compared += 1
            del tv

        del vmodel
        gc.collect()
        _print_rss(f"[rank 0] phase 6: vmodel freed after streaming compare")

        # 验证 sd_ref 所有 key 都在 vmodel 中（sd_ref ⊆ v_keys）
        missing = [k for k in sd_ref if k not in v_keys]
        assert not missing, (
            f"vanilla state_dict missing {len(missing)} Path-A keys; "
            f"first 10: {missing[:10]}"
        )

        print(
            f"[rank 0] phase 6 PASS: Path B ≡ Path A ({n_compared} tensors "
            f"bit-equal, {len(v_keys) - n_compared} extra Path-B keys skipped)",
            flush=True,
        )
        del v_keys
        gc.collect()

    dist.barrier()
    dist.destroy_process_group()
    return {"rank": rank, "ok": True, "n_keys": len(sd_ref) if rank == 0 else None}


@ray.remote(num_gpus=1)
def _bias_roundtrip_worker(
    hf_model_path: str,
    save_dir: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    cp_size: int,
    dtype_format: str,
):
    """e_score_correction_bias 专项 round-trip：set dummy → save → reload → compare。

    流程
    ----
    1. meta 构造 DeepseekV4ForCausalLM + apply_hp（绑定 EP/CP/FSDP2），再
       ``load_checkpoint_hp(hf_model_path)`` 加载真实权重（save 需要非 meta 权重）。
    2. 把所有 ``e_score_correction_bias`` 覆写成 distinct dummy（不同层 router /
       同 router 不同 expert 位置都用不同数值）。
    3. ``save_checkpoint_hp`` 落盘。
    4. 重新 meta 构造 + apply_hp + ``load_checkpoint_hp(save_dir)``。
    5. 逐 buffer 比较 reload 后的值是否与 dummy bit-equal。
    """
    from datetime import timedelta

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=2),
    )

    assert world_size % ep_size == 0
    assert world_size % cp_size == 0
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

    # NCCL warmup 前置 handshake，避免 rank 0 长时间 load 期间其他 rank 超时。
    dist.barrier()

    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
    assert not config.tie_word_embeddings, (
        "meta-device init may hang with tie_word_embeddings=True"
    )

    prev_dtype = torch.get_default_dtype()

    # ------------------------------------------------------------------
    # Phase 1: 构造 model1 + apply_hp + load 真实权重
    # ------------------------------------------------------------------
    print(f"[rank {rank}] bias phase 1: build meta model1 + load_checkpoint_hp(orig) ...", flush=True)
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model1 = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)
    model1 = apply_hp(model1, ep_2d_mesh, cp_mesh=cp_mesh, amp_fp32=False)
    model1.load_checkpoint_hp(hf_model_path)

    # ------------------------------------------------------------------
    # Phase 2: 覆写所有 e_score_correction_bias 为 distinct dummy
    # ------------------------------------------------------------------
    expected = _set_dummy_e_score_bias(model1)
    assert len(expected) > 0, (
        f"no e_score_correction_bias buffers found; bump NUM_LAYERS "
        f"(={NUM_LAYERS}) so at least one TopK-MoE layer is included"
    )
    if rank == 0:
        preview = sorted(expected)[:4]
        print(
            f"[rank 0] bias phase 2: set dummy on {len(expected)} "
            f"e_score_correction_bias buffers, e.g. {preview}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Phase 3: save
    # ------------------------------------------------------------------
    print(f"[rank {rank}] bias phase 3: save_checkpoint_hp(dtype_format={dtype_format!r}) ...", flush=True)
    model1.save_checkpoint_hp(save_dir, orig_ckpt_dir=hf_model_path, dtype_format=dtype_format)
    dist.barrier()
    del model1
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 4: 重新构造 model2 + load from save_dir
    # ------------------------------------------------------------------
    print(f"[rank {rank}] bias phase 4: build meta model2 + load_checkpoint_hp(save_dir) ...", flush=True)
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model2 = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)
    model2 = apply_hp(model2, ep_2d_mesh, cp_mesh=cp_mesh, amp_fp32=False)
    model2.load_checkpoint_hp(save_dir)

    # ------------------------------------------------------------------
    # Phase 5: 比对（buffer 是 replicated，每个 rank 独立验证自己的副本）
    # ------------------------------------------------------------------
    got = _collect_e_score_bias(model2)
    assert set(got) == set(expected), (
        f"bias buffer name set changed after round-trip:\n"
        f"  only in expected: {sorted(set(expected) - set(got))[:8]}\n"
        f"  only in got:      {sorted(set(got) - set(expected))[:8]}"
    )
    mismatches = []
    for name in sorted(expected):
        e, g = expected[name], got[name]
        if e.shape != g.shape or not torch.equal(e, g):
            diff = (e - g).abs().max().item() if e.shape == g.shape else float("nan")
            mismatches.append((name, diff, e[:4].tolist(), g[:4].tolist()))
    assert not mismatches, (
        f"[rank {rank}] e_score_correction_bias round-trip mismatch on "
        f"{len(mismatches)}/{len(expected)} buffers (dtype_format={dtype_format!r}); "
        f"first few:\n" + "\n".join(
            f"  {n}: max_abs_diff={d:.3e} expected[:4]={ev} got[:4]={gv}"
            for n, d, ev, gv in mismatches[:8]
        )
    )
    if rank == 0:
        print(
            f"[rank 0] bias phase 5 PASS: {len(expected)} e_score_correction_bias "
            f"buffers bit-equal after {dtype_format!r} round-trip",
            flush=True,
        )

    del model2
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()
    return {"rank": rank, "ok": True, "n_bias": len(expected)}


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestDeepseekV4SaveLoad(unittest.TestCase):
    """save_checkpoint_hp 端到端 round-trip 测试。需要 >= 32 GPU。"""

    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {NUM_GPUS} GPUs in Ray cluster, only {total_gpus}"
            )

        # 每次测试前清空 SAVE_DIR，防止残留 ckpt 污染结果
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR, exist_ok=True)

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()
        # 清理 SAVE_DIR，避免在共享 FS 上积累垃圾
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)

    def test_save_load_quantized(self):
        """Quant 路径（默认）：FP4/FP8/E8M0 + 保留 quantization_config + HF FineGrainedFP8 reload。"""
        self._run_with_dtype_format("quantized")

    def test_save_load_bf16(self):
        """BF16 路径：与原 §9/§10 实现行为一致（回归保护）。"""
        # NOTE: it saves bf16 (mostly) and fp32
        self._run_with_dtype_format("bf16")

    def test_e_score_bias_roundtrip_quantized(self):
        """e_score_correction_bias dummy → save/load(quantized) → bit-equal。"""
        self._run_bias_roundtrip("quantized", master_port_base=12800)

    def test_e_score_bias_roundtrip_bf16(self):
        """e_score_correction_bias dummy → save/load(bf16) → bit-equal。"""
        self._run_bias_roundtrip("bf16", master_port_base=12810)

    def _run_with_dtype_format(
        self,
        dtype_format: str,
        world_size: int = NUM_GPUS,
        ep_size: int = EP_SIZE,
        cp_size: int = CP_SIZE,
        master_port_base: int = 12700,
    ):
        """load → save → reload，逐 key bit-equal 验证。"""
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}; "
            f"download DeepSeek-V4-Flash into hf-hub/ first"
        )

        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
        )
        ray.get(pg.ready())

        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0,
                )
            ).remote()
        )

        futures = [
            _save_load_worker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=r,
                )
            ).remote(
                HF_MODEL_PATH,
                SAVE_DIR,
                rank=r,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port_base,
                ep_size=ep_size,
                cp_size=cp_size,
                dtype_format=dtype_format,
            )
            for r in range(world_size)
        ]
        results = ray.get(futures)
        remove_placement_group(pg)

        for r, res in enumerate(results):
            self.assertTrue(res["ok"], f"rank {r} returned {res}")

        # 验证磁盘上的文件格式符合 HF 标准
        assert os.path.isfile(os.path.join(SAVE_DIR, "config.json"))
        assert os.path.isfile(os.path.join(SAVE_DIR, "model.safetensors.index.json"))
        shards = [
            f for f in os.listdir(SAVE_DIR)
            if re.match(r"model-\d{5}-of-\d{5}\.safetensors$", f)
        ]
        self.assertGreater(len(shards), 0, "no safetensors shards were written")
        print(f"\nSAVE_DIR layout: {len(shards)} shard(s), index.json + config.json present")
        print(f"PASSED (dtype_format={dtype_format!r})")

    def _run_bias_roundtrip(
        self,
        dtype_format: str,
        world_size: int = NUM_GPUS,
        ep_size: int = EP_SIZE,
        cp_size: int = CP_SIZE,
        master_port_base: int = 12800,
    ):
        """set dummy e_score_correction_bias → save → reload → 逐 buffer bit-equal。"""
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}; "
            f"download DeepSeek-V4-Flash into hf-hub/ first"
        )

        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
        )
        ray.get(pg.ready())

        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0,
                )
            ).remote()
        )

        futures = [
            _bias_roundtrip_worker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=r,
                )
            ).remote(
                HF_MODEL_PATH,
                SAVE_DIR,
                rank=r,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port_base,
                ep_size=ep_size,
                cp_size=cp_size,
                dtype_format=dtype_format,
            )
            for r in range(world_size)
        ]
        results = ray.get(futures)
        remove_placement_group(pg)

        for r, res in enumerate(results):
            self.assertTrue(res["ok"], f"rank {r} returned {res}")

        n_bias = results[0]["n_bias"]
        self.assertGreater(
            n_bias, 0, "no e_score_correction_bias buffers were exercised"
        )
        print(
            f"\nPASSED e_score_correction_bias round-trip "
            f"(dtype_format={dtype_format!r}): {n_bias} buffers bit-equal"
        )
