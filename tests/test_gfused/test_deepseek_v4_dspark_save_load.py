# coding=utf-8
"""DSpark ``load_checkpoint_hp`` / ``save_checkpoint_hp`` round-trip (0731).

现有 ``test_deepseek_v4_save_load.py`` 刻意 skip ``mtp.*``；本文件专门覆盖
DSpark draft 权重，并额外做 **save 目录 vs source 磁盘** 逐 key bit-equal。

流程：
  1. ``load_checkpoint_hp(0731)`` → 流式 gather ``sd1``（含 ``mtp.*``）
  2. ``save_checkpoint_hp(save_dir, quantized, preserve_mtp=True)``
  3. rank0：``save_dir`` 每个 weight ⊆ source；passthrough bit-equal，
     量化 weight 两侧 dequant 后 bit-equal；强制覆盖 DSpark 关键 key
  4. ``load_checkpoint_hp(save_dir)`` → ``sd2 ≡ sd1``（含 ``mtp.*``）

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=3600 tests/test_gfused/test_deepseek_v4_dspark_save_load.py
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import unittest
from datetime import timedelta

import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from safetensors import safe_open
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from gpatch_v4.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM, apply_hp
from gpatch_v4.models.deepseek_v4.checkpoint import infer_dspark_num_layers
from test_gfused.test_deepseek_v4_save_load import (
    CP_SIZE,
    EP_SIZE,
    NUM_GPUS,
    _get_node_ip,
    _print_rss,
    _stream_full_state_dict,
    assert_round_trip,
)
from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

HF_MODEL_PATH = os.environ.get(
    "DSPARK_HF_MODEL_PATH",
    "hf-hub/deepseek-ai/DeepSeek-V4-Flash-0731",
)
NUM_LAYERS = 4
SAVE_DIR = "dsv4_dspark_save_load_test"

REQUIRED_DSPARK_DISK_KEYS = (
    "mtp.0.main_proj.weight",
    "mtp.0.main_proj.scale",
    "mtp.0.main_norm.weight",
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.confidence_head.proj.weight",
    "mtp.2.norm.weight",
)


def _prepare_dspark_config(hf_model_path: str) -> DeepseekV4Config:
    config = DeepseekV4Config.from_pretrained(hf_model_path)
    config.dspark_num_layers = infer_dspark_num_layers(hf_model_path)
    # Training-only knob; HF config does not carry it. Smoke yaml uses 2.
    config.dspark_num_anchors = 2
    assert config.dspark_target_layer_ids
    config.num_hidden_layers = NUM_LAYERS
    if config.layer_types is not None:
        config.layer_types = config.layer_types[:NUM_LAYERS]
    if config.mlp_layer_types is not None:
        config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
    num_target_layers = len(config.dspark_target_layer_ids)
    assert NUM_LAYERS >= num_target_layers
    config.dspark_target_layer_ids = list(
        range(NUM_LAYERS - num_target_layers, NUM_LAYERS)
    )
    assert not config.tie_word_embeddings, (
        "meta-device init may hang with tie_word_embeddings=True"
    )
    return config


def _strip_holder(name: str) -> str:
    return (
        name.replace("._sink_holder.weight", ".sinks")
        .replace("._position_bias_holder.weight", ".position_bias")
    )


def _load_weight_map(ckpt_dir: str) -> dict[str, str]:
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        return json.load(f)["weight_map"]


def _read_disk_tensor(ckpt_dir: str, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = weight_map[key]
    with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as handle:
        return handle.get_tensor(key).contiguous()


def _is_quantized_weight_key(key: str, weight_map: dict[str, str]) -> bool:
    return key.endswith(".weight") and (key[: -len(".weight")] + ".scale") in weight_map


def _assert_saved_ckpt_matches_source(save_dir: str, source_dir: str) -> int:
    """对比 save 产物与 source：key 集合 + 语义相等。

    - passthrough（无 ``.scale``）：磁盘 tensor bit-equal
    - 量化 ``.weight+.scale``：两侧 dequant 后 bit-equal（load→save 会重量化，
      raw FP8/FP4 code 不保证与 source 字节相同）
    - ``.scale`` 单独跳过（跟对应 weight 一起验）
    """
    from gpatch_v4.kernel.quantize.eager_quant_kernels import (
        dequant_fp4_e2m1_fp8_scale_e8m0_packed,
    )

    save_map = _load_weight_map(save_dir)
    source_map = _load_weight_map(source_dir)

    missing_required = [k for k in REQUIRED_DSPARK_DISK_KEYS if k not in save_map]
    assert not missing_required, (
        f"saved ckpt missing required DSpark keys: {missing_required}"
    )

    only_in_save = sorted(set(save_map) - set(source_map))
    assert not only_in_save, (
        f"saved ckpt has {len(only_in_save)} keys absent from source "
        f"(first 10): {only_in_save[:10]}"
    )

    n_compared = 0
    n_dequant = 0
    for key in sorted(save_map):
        if key.endswith(".scale"):
            continue

        t_save = _read_disk_tensor(save_dir, save_map, key)
        t_src = _read_disk_tensor(source_dir, source_map, key)
        assert t_save.shape == t_src.shape, (
            f"{key}: shape mismatch save={tuple(t_save.shape)} "
            f"source={tuple(t_src.shape)}"
        )
        assert t_save.dtype == t_src.dtype, (
            f"{key}: dtype mismatch save={t_save.dtype} source={t_src.dtype}"
        )

        if _is_quantized_weight_key(key, save_map):
            scale_key = key[: -len(".weight")] + ".scale"
            s_save = _read_disk_tensor(save_dir, save_map, scale_key)
            s_src = _read_disk_tensor(source_dir, source_map, scale_key)
            assert s_save.shape == s_src.shape, (
                f"{scale_key}: shape mismatch save={tuple(s_save.shape)} "
                f"source={tuple(s_src.shape)}"
            )
            v_save = dequant_fp4_e2m1_fp8_scale_e8m0_packed(t_save, s_save).float()
            v_src = dequant_fp4_e2m1_fp8_scale_e8m0_packed(t_src, s_src).float()
            if not torch.equal(v_save, v_src):
                diff = (v_save - v_src).abs().max().item()
                raise AssertionError(
                    f"{key}: dequant(save) != dequant(source) "
                    f"(max_abs_diff={diff:.4e})"
                )
            n_dequant += 1
            del s_save, s_src, v_save, v_src
        else:
            if not torch.equal(t_save, t_src):
                if t_save.dtype.is_floating_point:
                    diff = (t_save.float() - t_src.float()).abs().max().item()
                else:
                    diff = float("nan")
                raise AssertionError(
                    f"{key}: saved ckpt != source ckpt "
                    f"(dtype={t_save.dtype}, max_abs_diff={diff:.4e})"
                )

        n_compared += 1
        del t_save, t_src

    _print_rss(
        f"[rank 0] disk compare PASS: {n_compared} weights "
        f"({n_dequant} via dequant, "
        f"{n_compared - n_dequant} passthrough bit-equal; "
        f"{len(REQUIRED_DSPARK_DISK_KEYS)} DSpark required present)"
    )
    return n_compared


@ray.remote(num_gpus=1)
def _dspark_save_load_worker(
    hf_model_path: str,
    save_dir: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    cp_size: int,
):
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
    dist.barrier()

    config = _prepare_dspark_config(hf_model_path)
    assert config.dspark_num_layers > 0

    print(
        f"[rank {rank}] phase 1: meta build + load_checkpoint_hp({hf_model_path}) ...",
        flush=True,
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model1 = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    assert model1.mtp is not None
    model1 = apply_hp(model1, ep_2d_mesh, cp_mesh=cp_mesh, amp_fp32=False)
    model1.load_checkpoint_hp(hf_model_path)

    print(f"[rank {rank}] phase 2: capture sd1 (incl. mtp.*) ...", flush=True)
    sd1: dict[str, torch.Tensor] = {}

    def _capture_sd1(name: str, t1: torch.Tensor) -> None:
        sd1[_strip_holder(name)] = t1

    n_keys_sd1 = _stream_full_state_dict(model1, on_rank0=_capture_sd1)
    if rank == 0:
        mtp_keys = [k for k in sd1 if k.startswith("mtp.")]
        assert mtp_keys, "sd1 has no mtp.* keys after DSpark load"
        _print_rss(
            f"[rank 0] sd1 captured: {len(sd1)} tensors "
            f"({len(mtp_keys)} mtp.*, stream_n={n_keys_sd1})"
        )

    print(
        f"[rank {rank}] phase 3: save_checkpoint_hp({save_dir}, quantized) ...",
        flush=True,
    )
    model1.save_checkpoint_hp(
        save_dir,
        orig_ckpt_dir=hf_model_path,
        dtype_format="quantized",
        preserve_mtp=True,
    )
    dist.barrier()

    if rank == 0:
        print(
            f"[rank 0] phase 3.5: disk compare save_dir vs source {hf_model_path} ...",
            flush=True,
        )
        n_disk = _assert_saved_ckpt_matches_source(save_dir, hf_model_path)
    else:
        n_disk = None
    dist.barrier()

    del model1
    torch.cuda.empty_cache()

    print(
        f"[rank {rank}] phase 4: meta build + load_checkpoint_hp({save_dir}) ...",
        flush=True,
    )
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model2 = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    model2 = apply_hp(model2, ep_2d_mesh, cp_mesh=cp_mesh, amp_fp32=False)
    model2.load_checkpoint_hp(save_dir)

    print(f"[rank {rank}] phase 5: sd2 ≡ sd1 (incl. mtp.*) ...", flush=True)
    seen_sd2: set[str] = set()

    def _check_sd2(name: str, t2: torch.Tensor) -> None:
        name = _strip_holder(name)
        seen_sd2.add(name)
        assert name in sd1, f"sd2 has unexpected key absent from sd1: {name}"
        t1 = sd1[name]
        assert t2.shape == t1.shape, (
            f"sd2 vs sd1 shape mismatch {name}: {t2.shape} vs {t1.shape}"
        )
        assert t2.dtype == t1.dtype, (
            f"sd2 vs sd1 dtype mismatch {name}: {t2.dtype} vs {t1.dtype}"
        )
        assert_round_trip(name, t1, t2)

    n_keys_sd2 = _stream_full_state_dict(model2, on_rank0=_check_sd2)
    if rank == 0:
        only_in_sd1 = set(sd1) - seen_sd2
        assert not only_in_sd1, (
            f"sd2 missed {len(only_in_sd1)} sd1 keys (first 10): "
            f"{sorted(only_in_sd1)[:10]}"
        )
        _print_rss(
            f"[rank 0] phase 5 PASS: sd2 ≡ sd1 "
            f"({len(seen_sd2)} tensors, stream_n={n_keys_sd2})"
        )

    del model2, sd1
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()
    return {"rank": rank, "ok": True, "n_disk": n_disk}


class TestDeepseekV4DSparkSaveLoad(unittest.TestCase):
    """DSpark 0731：save vs source 磁盘 bit-equal + reload bit-equal。"""

    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {NUM_GPUS} GPUs in Ray cluster, only {total_gpus}"
            )
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR, exist_ok=True)

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)

    def test_dspark_save_load_quantized_matches_source(self):
        hf_model_path = os.path.abspath(HF_MODEL_PATH)
        assert os.path.isdir(hf_model_path), (
            f"model dir not found: {hf_model_path}; set DSPARK_HF_MODEL_PATH"
        )

        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * NUM_GPUS,
            strategy="PACK",
        )
        ray.get(pg.ready())

        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                )
            ).remote()
        )

        futures = [
            _dspark_save_load_worker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=r,
                )
            ).remote(
                hf_model_path,
                SAVE_DIR,
                rank=r,
                world_size=NUM_GPUS,
                master_addr=master_addr,
                master_port=12900,
                ep_size=EP_SIZE,
                cp_size=CP_SIZE,
            )
            for r in range(NUM_GPUS)
        ]
        results = ray.get(futures)
        remove_placement_group(pg)

        for r, res in enumerate(results):
            self.assertTrue(res["ok"], f"rank {r} returned {res}")

        assert os.path.isfile(os.path.join(SAVE_DIR, "config.json"))
        assert os.path.isfile(os.path.join(SAVE_DIR, "model.safetensors.index.json"))
        shards = [
            f for f in os.listdir(SAVE_DIR)
            if re.match(r"model-\d{5}-of-\d{5}\.safetensors$", f)
        ]
        self.assertGreater(len(shards), 0, "no safetensors shards were written")
        n_disk = next(res["n_disk"] for res in results if res["n_disk"] is not None)
        print(
            f"\nSAVE_DIR layout: {len(shards)} shard(s); "
            f"disk compared {n_disk} keys vs source 0731"
        )
        print("PASSED (dspark quantized save≡source + reload≡sd1)")


if __name__ == "__main__":
    unittest.main()
