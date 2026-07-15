# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com

from __future__ import annotations

import importlib.util
import os
import socket
import time
import unittest
from pathlib import Path

import ray
import torch
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import DeepseekV4Config

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.models.deepseek_v4 import DeepseekV4ForCausalLM, apply_hp
from gpatch_v4.models.deepseek_v4.cp import cp_chunk_data
from gpatch_v4.models.deepseek_v4.thd import pack_sequences
from gpatch_v4.orches.placement_group import _create_placement_group

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
WORLD_SIZE = 32
EP_SIZE = 8
CP_SIZE = 8
SEQ_LEN = 256 * 1024
PAD_TO_MULTIPLE_OF = 128
WARMUP_STEPS = 2
INPUT_SEED = 20260714
_DEEPEP_AVAILABLE = importlib.util.find_spec("deep_ep") is not None


def _truncate_config(config: DeepseekV4Config) -> DeepseekV4Config:
    config.num_hidden_layers = 8
    config.layer_types = config.layer_types[:config.num_hidden_layers]
    config.mlp_layer_types = config.mlp_layer_types[:config.num_hidden_layers]
    config.num_nextn_predict_layers = 0
    return config


def _build_model(hf_model_path: str) -> DeepseekV4ForCausalLM:
    config = _truncate_config(DeepseekV4Config.from_pretrained(hf_model_path))
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    return model


@ray.remote(num_cpus=0, num_gpus=0)
def _get_master_endpoint() -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return ray.util.get_node_ip_address(), sock.getsockname()[1]


@ray.remote(num_gpus=1)
def _profile_forward_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    trace_dir: str,
) -> dict[str, int | float | str]:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

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
        config = _truncate_config(DeepseekV4Config.from_pretrained(hf_model_path))
        assert SEQ_LEN % (CP_SIZE * PAD_TO_MULTIPLE_OF) == 0

        input_generator = torch.Generator(device="cuda").manual_seed(INPUT_SEED)
        input_ids = torch.randint(
            config.vocab_size,
            (SEQ_LEN,),
            device="cuda",
            generator=input_generator,
        )
        packed_ids, packed_position_ids, _, packed_seq_params = pack_sequences(
            [input_ids],
            None,
            config=config,
            pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
            cp_size=CP_SIZE,
            pad_token_id=0,
            label_ignore_index=-100,
        )

        # TODO enable deepEP
        model = _build_model(hf_model_path)
        model = apply_hp(
            model,
            ep_2d_mesh,
            cp_mesh=cp_mesh,
            amp_fp32=False,
            attn_backend="fused",
            indexer_backend="fused",
            ep_backend="deepep",
            fp8=False,
        )
        model.load_checkpoint_hp(hf_model_path)
        model.train()

        cp_group = model._cp_group
        cp_rank = dist.get_rank(cp_group)
        local_ids, _, _, local_position_ids, local_packed_seq_params = cp_chunk_data(
            cp_rank,
            CP_SIZE,
            tokens=packed_ids,
            labels=None,
            position_ids=packed_position_ids,
            packed_seq_params=packed_seq_params,
        )
        assert local_ids.shape == (1, SEQ_LEN // CP_SIZE)

        with torch.no_grad():
            for _ in range(WARMUP_STEPS):
                model(
                    input_ids=local_ids,
                    position_ids=local_position_ids,
                    packed_seq_params=local_packed_seq_params,
                )
        torch.cuda.synchronize()
        dist.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        trace_path = Path(trace_dir) / f"rank-{rank}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
        ) as profiler:
            with torch.no_grad(), torch.profiler.record_function("dsv4_forward"):
                start.record()
                outputs = model(
                    input_ids=local_ids,
                    position_ids=local_position_ids,
                    packed_seq_params=local_packed_seq_params,
                )
                end.record()
        torch.cuda.synchronize()
        forward_ms = start.elapsed_time(end)
        profiler.export_chrome_trace(str(trace_path))
        assert outputs.logits.shape[:2] == (1, SEQ_LEN // CP_SIZE)

        print(
            f"[dsv4_perf] rank {rank}: forward_ms={forward_ms:.3f} "
            f"trace={trace_path}",
            flush=True,
        )
        return {
            "rank": rank,
            "forward_ms": forward_ms,
            "trace_path": str(trace_path),
        }
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(_DEEPEP_AVAILABLE, "deep_ep not installed")
class TestDeepseekV4Perf(unittest.TestCase):
    def setUp(self) -> None:
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {WORLD_SIZE} GPUs in Ray cluster, only {total_gpus}"
            )

    def tearDown(self) -> None:
        kill_all_actors_and_shutdown_ray()

    def test_hp_fused_forward_timeline(self) -> None:
        assert os.path.isdir(HF_MODEL_PATH), f"model dir not found: {HF_MODEL_PATH}"
        trace_dir = str(Path.cwd() / "dsv4_perf")
        placement_group, bundle_indices = _create_placement_group(WORLD_SIZE)

        try:
            master_addr, master_port = ray.get(
                _get_master_endpoint.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[0],
                    )
                ).remote()
            )
            futures = [
                _profile_forward_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[rank],
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank,
                    WORLD_SIZE,
                    master_addr,
                    master_port,
                    trace_dir,
                )
                for rank in range(WORLD_SIZE)
            ]
            results = ray.get(futures)
        finally:
            remove_placement_group(placement_group)

        for result in results:
            self.assertGreater(result["forward_ms"], 0.0)
            self.assertTrue(os.path.isfile(result["trace_path"]))


if __name__ == "__main__":
    unittest.main()
