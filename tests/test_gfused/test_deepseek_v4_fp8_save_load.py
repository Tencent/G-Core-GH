# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.

import gc
import os
import shutil
from datetime import timedelta

import pytest
import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.device_mesh import init_device_mesh

from gpatch_v4.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    apply_hp,
)
from gpatch_v4.models.deepseek_v4.fp8_tensor import Fp8TensorAg
from gpatch_v4.orches.placement_group import _create_placement_group
from test_gfused.test_deepseek_v4_save_load import (
    CP_SIZE,
    EP_SIZE,
    HF_MODEL_PATH,
    NUM_GPUS,
    _truncate_config,
)
from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray


SAVE_ROOT = "dsv4_fp8_save_load_test"


def _assert_fp8_expert_params(model: torch.nn.Module) -> int:
    count = 0
    for name, param in model.named_parameters():
        if ".experts.gate_up_proj" not in name and ".experts.down_proj" not in name:
            continue
        local = param.to_local()
        assert isinstance(local, Fp8TensorAg), (name, type(local))
        assert local._tensor.dtype == torch.float32, (name, local._tensor.dtype)
        count += 1
    assert count > 0
    return count


def _build_fp8_model(config, ep_2d_mesh, cp_mesh, checkpoint_path):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model = apply_hp(
        model,
        ep_2d_mesh,
        cp_mesh=cp_mesh,
        amp_fp32=False,
        fp8=True,
        fsdp_fp8_gather=True,
    )
    model.load_checkpoint_hp(checkpoint_path)
    return model


@ray.remote(num_gpus=1)
def _fp8_save_load_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    save_dir: str,
    dtype_format: str,
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=2),
    )
    try:
        ep_2d_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // EP_SIZE, EP_SIZE),
            mesh_dim_names=("ep_fsdp", "ep"),
        )
        cp_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // CP_SIZE, CP_SIZE),
            mesh_dim_names=("dp", "cp"),
        )["cp"]
        config = _truncate_config(DeepseekV4Config.from_pretrained(HF_MODEL_PATH))

        model = _build_fp8_model(config, ep_2d_mesh, cp_mesh, HF_MODEL_PATH)
        wrapped_before = _assert_fp8_expert_params(model)
        model.save_checkpoint_hp(
            save_dir,
            orig_ckpt_dir=HF_MODEL_PATH,
            dtype_format=dtype_format,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

        reloaded = _build_fp8_model(config, ep_2d_mesh, cp_mesh, save_dir)
        wrapped_after = _assert_fp8_expert_params(reloaded)
        assert wrapped_after == wrapped_before
        return {
            "rank": rank,
            "wrapped_params": wrapped_after,
        }
    finally:
        dist.destroy_process_group()


@ray.remote(num_gpus=0)
def _get_node_ip() -> str:
    return ray.util.get_node_ip_address()


@pytest.mark.parametrize("dtype_format", ("quantized", "bf16"))
def test_fp8_save_load_preserves_wrapped_master(dtype_format: str) -> None:
    assert os.path.isdir(HF_MODEL_PATH), HF_MODEL_PATH
    ray.init(address="auto")
    save_dir = os.path.join(SAVE_ROOT, dtype_format)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    placement_group, bundle_indices = _create_placement_group(NUM_GPUS)
    try:
        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_indices[0],
                )
            ).remote()
        )
        futures = [
            _fp8_save_load_worker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_indices[rank],
                )
            ).remote(
                rank,
                NUM_GPUS,
                master_addr,
                12900 if dtype_format == "quantized" else 12910,
                save_dir,
                dtype_format,
            )
            for rank in range(NUM_GPUS)
        ]
        results = ray.get(futures, timeout=3600)
        assert all(result["wrapped_params"] > 0 for result in results)
        assert os.path.isfile(os.path.join(save_dir, "config.json"))
        assert os.path.isfile(
            os.path.join(save_dir, "model.safetensors.index.json")
        )
    finally:
        remove_placement_group(placement_group)
        kill_all_actors_and_shutdown_ray()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
