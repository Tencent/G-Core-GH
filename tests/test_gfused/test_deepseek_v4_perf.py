# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com

from __future__ import annotations

import importlib.util
import pprint
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
_FLASH_ATTN_AVAILABLE = importlib.util.find_spec("flash_attn") is not None
ULYSSES_CP_SIZES = (8, 16)
ULYSSES_BATCH_SIZE = int(os.environ.get("DSV4_ULYSSES_BATCH_SIZE", "1"))
ULYSSES_NUM_HEADS = int(os.environ.get("DSV4_ULYSSES_NUM_HEADS", "16"))
ULYSSES_HEAD_DIM = int(os.environ.get("DSV4_ULYSSES_HEAD_DIM", "256"))
ULYSSES_WARMUP_STEPS = int(os.environ.get("DSV4_ULYSSES_WARMUP", "2"))
ULYSSES_BENCH_STEPS = int(os.environ.get("DSV4_ULYSSES_ITERS", "5"))


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
            (SEQ_LEN, ),
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
            moe_router_force_load_balancing=True,
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


def _ulysses_seq_to_head(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    batch, local_seq_len, num_heads, head_dim = tensor.shape
    cp_size = dist.get_world_size(group)
    assert num_heads % cp_size == 0
    local_num_heads = num_heads // cp_size
    send = (
        tensor.view(batch, local_seq_len, cp_size, local_num_heads,
                    head_dim).permute(2, 0, 1, 3, 4).contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.permute(1, 0, 2, 3,
                     4).reshape(batch, local_seq_len * cp_size, local_num_heads,
                                head_dim).contiguous()
    )


def _ulysses_head_to_seq(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    batch, seq_len, local_num_heads, head_dim = tensor.shape
    cp_size = dist.get_world_size(group)
    assert seq_len % cp_size == 0
    local_seq_len = seq_len // cp_size
    send = (
        tensor.view(batch, cp_size, local_seq_len, local_num_heads,
                    head_dim).permute(1, 0, 2, 3, 4).contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.permute(1, 2, 0, 3,
                     4).reshape(batch, local_seq_len, local_num_heads * cp_size,
                                head_dim).contiguous()
    )


@ray.remote(num_gpus=1)
def _profile_ulysses_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> list[dict[str, int | float | bool]]:
    flash_attn_func = importlib.import_module("flash_attn").flash_attn_func

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    try:
        results = []
        for cp_size in ULYSSES_CP_SIZES:
            assert world_size % cp_size == 0
            assert SEQ_LEN % cp_size == 0
            assert ULYSSES_NUM_HEADS % cp_size == 0
            cp_group = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // cp_size, cp_size),
                mesh_dim_names=("dp", "cp"),
            )["cp"].get_group()
            local_seq_len = SEQ_LEN // cp_size
            shape = (
                ULYSSES_BATCH_SIZE,
                local_seq_len,
                ULYSSES_NUM_HEADS,
                ULYSSES_HEAD_DIM,
            )
            q = torch.empty(shape, device="cuda", dtype=torch.bfloat16).normal_(std=0.02)
            k = torch.empty_like(q).normal_(std=0.02)
            v = torch.empty_like(q).normal_(std=0.02)

            q_full = _ulysses_seq_to_head(q, cp_group)
            k_full = _ulysses_seq_to_head(k, cp_group)
            v_full = _ulysses_seq_to_head(v, cp_group)
            q_roundtrip = _ulysses_head_to_seq(q_full, cp_group)
            roundtrip_ok = torch.equal(q_roundtrip, q)
            assert roundtrip_ok
            del q_roundtrip

            def communication_only() -> torch.Tensor:
                q_exchanged = _ulysses_seq_to_head(q, cp_group)
                k_exchanged = _ulysses_seq_to_head(k, cp_group)
                v_exchanged = _ulysses_seq_to_head(v, cp_group)
                output = _ulysses_head_to_seq(q_exchanged, cp_group)
                del k_exchanged, v_exchanged
                return output

            def computation_only() -> torch.Tensor:
                return flash_attn_func(
                    q_full,
                    k_full,
                    v_full,
                    causal=True,
                )

            def end_to_end() -> torch.Tensor:
                q_exchanged = _ulysses_seq_to_head(q, cp_group)
                k_exchanged = _ulysses_seq_to_head(k, cp_group)
                v_exchanged = _ulysses_seq_to_head(v, cp_group)
                output = flash_attn_func(
                    q_exchanged,
                    k_exchanged,
                    v_exchanged,
                    causal=True,
                )
                return _ulysses_head_to_seq(output, cp_group)

            def time_fn(fn) -> float:
                output = None
                for _ in range(ULYSSES_WARMUP_STEPS):
                    output = fn()
                torch.cuda.synchronize()
                dist.barrier(group=cp_group)
                start = time.perf_counter()
                for _ in range(ULYSSES_BENCH_STEPS):
                    output = fn()
                torch.cuda.synchronize()
                elapsed_ms = ((time.perf_counter() - start) * 1000.0 / ULYSSES_BENCH_STEPS)
                dist.barrier(group=cp_group)
                assert output is not None
                return elapsed_ms

            communication_ms = time_fn(communication_only)
            computation_ms = time_fn(computation_only)
            end_to_end_ms = time_fn(end_to_end)
            del q, k, v, q_full, k_full, v_full
            torch.cuda.empty_cache()

            auxiliary = torch.empty(
                (local_seq_len, SEQ_LEN),
                device="cuda",
                dtype=torch.bfloat16,
            )
            auxiliary_recv = torch.empty_like(auxiliary)

            def auxiliary_all_to_all() -> torch.Tensor:
                dist.all_to_all_single(
                    auxiliary_recv,
                    auxiliary,
                    group=cp_group,
                )
                return auxiliary_recv

            auxiliary_all_to_all_ms = time_fn(auxiliary_all_to_all)
            results.append(
                {
                    "rank": rank,
                    "cp_size": cp_size,
                    "local_seq_len": local_seq_len,
                    "local_num_heads": ULYSSES_NUM_HEADS // cp_size,
                    "auxiliary_gib": auxiliary.numel() * auxiliary.element_size() / 1024**3,
                    "roundtrip_ok": roundtrip_ok,
                    "communication_ms": communication_ms,
                    "computation_ms": computation_ms,
                    "end_to_end_ms": end_to_end_ms,
                    "auxiliary_all_to_all_ms": auxiliary_all_to_all_ms,
                }
            )
            del auxiliary, auxiliary_recv
            torch.cuda.empty_cache()
            dist.barrier()
        return results
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(_DEEPEP_AVAILABLE, "deep_ep not installed")
class TestDeepseekV4Perf(unittest.TestCase):
    def setUp(self) -> None:
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"need >= {WORLD_SIZE} GPUs in Ray cluster, only {total_gpus}")

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
                ) for rank in range(WORLD_SIZE)
            ]
            results = ray.get(futures)
        finally:
            remove_placement_group(placement_group)

        for result in results:
            self.assertGreater(result["forward_ms"], 0.0)
            self.assertTrue(os.path.isfile(result["trace_path"]))


# Ulysses forward benchmark: global_seq_len=262144, batch=1, heads=16, head_dim=256, causal=True, warmup=2, iters=5
# CP=8: local_shape=[1, 32768, 16, 256] -> [1, 262144, 2, 256]
#   phase-critical communication=5.820 ms (rank=5, 0.7%), computation=783.421 ms (rank=26, 99.3%), end_to_end=789.240 ms (rank=13)
#   overhead=e2e/compute-1=0.7%, e2e-(comm+compute)=-0.001 ms
#   auxiliary all_to_all bf16 [32768, 262144]: 16.00 GiB/rank, phase-critical=51.631 ms (rank=14)
# CP=16: local_shape=[1, 16384, 16, 256] -> [1, 262144, 1, 256]
#   phase-critical communication=10.901 ms (rank=9, 2.7%), computation=396.243 ms (rank=9, 97.3%), end_to_end=407.113 ms (rank=3)
#   overhead=e2e/compute-1=2.7%, e2e-(comm+compute)=-0.031 ms
#   auxiliary all_to_all bf16 [16384, 262144]: 8.00 GiB/rank, phase-critical=147.647 ms (rank=7)


@unittest.skipUnless(_FLASH_ATTN_AVAILABLE, "flash_attn not installed")
class TestUlyssesOverhead(unittest.TestCase):
    def setUp(self) -> None:
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"need >= {WORLD_SIZE} GPUs in Ray cluster, only {total_gpus}")

    def tearDown(self) -> None:
        kill_all_actors_and_shutdown_ray()

    def test_forward_computation_communication_overhead(self) -> None:
        self.assertGreater(ULYSSES_BENCH_STEPS, 0)
        self.assertGreaterEqual(ULYSSES_WARMUP_STEPS, 0)
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
                _profile_ulysses_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[rank],
                    )
                ).remote(
                    rank,
                    WORLD_SIZE,
                    master_addr,
                    master_port,
                ) for rank in range(WORLD_SIZE)
            ]
            worker_results = ray.get(futures)
        finally:
            remove_placement_group(placement_group)

        results = [result for rank_results in worker_results for result in rank_results]
        print(
            "\nUlysses forward benchmark: "
            f"global_seq_len={SEQ_LEN}, batch={ULYSSES_BATCH_SIZE}, "
            f"heads={ULYSSES_NUM_HEADS}, head_dim={ULYSSES_HEAD_DIM}, "
            f"causal=True, warmup={ULYSSES_WARMUP_STEPS}, "
            f"iters={ULYSSES_BENCH_STEPS}"
        )
        for cp_size in ULYSSES_CP_SIZES:
            case_results = [result for result in results if result["cp_size"] == cp_size]
            self.assertEqual(len(case_results), WORLD_SIZE)
            self.assertTrue(all(result["roundtrip_ok"] for result in case_results))
            communication_result = max(
                case_results, key=lambda result: float(result["communication_ms"])
            )
            computation_result = max(
                case_results, key=lambda result: float(result["computation_ms"])
            )
            end_to_end_result = max(case_results, key=lambda result: float(result["end_to_end_ms"]))
            auxiliary_result = max(
                case_results,
                key=lambda result: float(result["auxiliary_all_to_all_ms"]),
            )
            communication_ms = float(communication_result["communication_ms"])
            computation_ms = float(computation_result["computation_ms"])
            end_to_end_ms = float(end_to_end_result["end_to_end_ms"])
            auxiliary_all_to_all_ms = float(auxiliary_result["auxiliary_all_to_all_ms"])
            isolated_total_ms = communication_ms + computation_ms
            communication_share = communication_ms / isolated_total_ms
            computation_share = computation_ms / isolated_total_ms
            overhead = end_to_end_ms / computation_ms - 1.0
            serial_gap_ms = end_to_end_ms - isolated_total_ms
            print(
                f"CP={cp_size}: local_shape="
                f"[{ULYSSES_BATCH_SIZE}, {SEQ_LEN // cp_size}, "
                f"{ULYSSES_NUM_HEADS}, {ULYSSES_HEAD_DIM}] -> "
                f"[{ULYSSES_BATCH_SIZE}, {SEQ_LEN}, "
                f"{ULYSSES_NUM_HEADS // cp_size}, {ULYSSES_HEAD_DIM}]"
            )
            print(
                f"  phase-critical communication={communication_ms:.3f} ms "
                f"(rank={communication_result['rank']}, {communication_share:.1%}), "
                f"computation={computation_ms:.3f} ms "
                f"(rank={computation_result['rank']}, {computation_share:.1%}), "
                f"end_to_end={end_to_end_ms:.3f} ms "
                f"(rank={end_to_end_result['rank']})"
            )
            print(
                f"  overhead=e2e/compute-1={overhead:.1%}, "
                f"e2e-(comm+compute)={serial_gap_ms:+.3f} ms"
            )
            print(
                f"  auxiliary all_to_all bf16 [{SEQ_LEN // cp_size}, {SEQ_LEN}]: "
                f"{float(auxiliary_result['auxiliary_gib']):.2f} GiB/rank, "
                f"phase-critical={auxiliary_all_to_all_ms:.3f} ms "
                f"(rank={auxiliary_result['rank']})"
            )
            self.assertGreater(communication_ms, 0.0)
            self.assertGreater(computation_ms, 0.0)
            self.assertGreater(end_to_end_ms, 0.0)
            self.assertGreater(auxiliary_all_to_all_ms, 0.0)


@ray.remote(num_gpus=1)
def test_contiguous_order_to_zz_worker(
    cp_rank: int,
    cp_size: int,
    master_addr: str,
    master_port: int,
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=cp_rank, world_size=cp_size)

    # cu_seqlens_padded = torch.tensor([0, 16, 64, 800, 1200], device="cuda", dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([0, 8, 32, 64], device="cuda", dtype=torch.int32)

    cp_rank_id_zz_ord_per_token = []
    for i in range(len(cu_seqlens_padded) - 1):
        s = int((cu_seqlens_padded[i + 1] - cu_seqlens_padded[i]).item())
        assert s % (2 * cp_size) == 0
        local_s = s // (2 * cp_size)
        for j in range(cp_size):
            cp_rank_id_zz_ord_per_token.append(
                torch.full((local_s, ), j, device="cuda", dtype=torch.int32)
            )
        for j in reversed(range(cp_size)):
            cp_rank_id_zz_ord_per_token.append(
                torch.full((local_s, ), j, device="cuda", dtype=torch.int32)
            )
    cp_rank_id_zz_ord_per_token = torch.cat(cp_rank_id_zz_ord_per_token, dim=0)
    # pprint.pprint(cp_rank_id_zz_ord_per_token)

    total_s = int(cu_seqlens_padded[-1].item())
    local_s = total_s // cp_size
    my_cp_rank_id_zz_ord_per_token = cp_rank_id_zz_ord_per_token[cp_rank * local_s:(cp_rank + 1) * local_s]
    # print(f"cp_rank={cp_rank}, my_cp_rank_id_zz_ord_per_token={my_cp_rank_id_zz_ord_per_token}", flush=True)

    split_sizes_nat_to_zz = []
    sum_input_split_sizes = 0
    for i in range(cp_size):
        n = (my_cp_rank_id_zz_ord_per_token == i).sum().item()
        split_sizes_nat_to_zz.append(n)
        sum_input_split_sizes += n
    assert sum_input_split_sizes == local_s

    split_sizes_zz_to_nat = []
    sum_output_split_sizes = 0
    for i in range(cp_size):
        other = cp_rank_id_zz_ord_per_token[i * local_s:(i + 1) * local_s]
        n = (other == cp_rank).sum().item()
        split_sizes_zz_to_nat.append(n)
        sum_output_split_sizes += n
    assert sum_output_split_sizes == local_s

    global_token_ids = torch.arange(
        cp_rank * local_s,
        (cp_rank + 1) * local_s,
        device="cuda",
        dtype=torch.int64,
    )
    hidden_states = torch.stack(
        [
            global_token_ids,
            torch.full_like(global_token_ids, cp_rank),
            torch.full_like(global_token_ids, -1),
        ],
        dim=1,
    )

    send_order = torch.argsort(
        my_cp_rank_id_zz_ord_per_token,
        stable=True,
    )
    to_send_hidden_states = hidden_states[send_order]
    recved_hidden_states = torch.empty_like(to_send_hidden_states)
    dist.all_to_all_single(
        recved_hidden_states,
        to_send_hidden_states,
        input_split_sizes=split_sizes_nat_to_zz,
        output_split_sizes=split_sizes_zz_to_nat,
    )

    expected_global_token_ids = torch.where(
        cp_rank_id_zz_ord_per_token == cp_rank
    )[0]
    expected_recved_hidden_states = torch.stack(
        [
            expected_global_token_ids,
            expected_global_token_ids // local_s,
            torch.full_like(expected_global_token_ids, -1),
        ],
        dim=1,
    )
    torch.testing.assert_close(
        recved_hidden_states,
        expected_recved_hidden_states,
        rtol=0,
        atol=0,
    )

    computed_hidden_states = recved_hidden_states.clone()
    computed_hidden_states[:, 2] = cp_rank
    out_hidden_states = torch.empty_like(computed_hidden_states)
    dist.all_to_all_single(
        out_hidden_states,
        computed_hidden_states,
        input_split_sizes=split_sizes_zz_to_nat,
        output_split_sizes=split_sizes_nat_to_zz,
    )
    tmp = out_hidden_states.clone()
    out_hidden_states[send_order] = tmp

    expected_out_hidden_states = hidden_states.clone()
    expected_out_hidden_states[:, 2] = my_cp_rank_id_zz_ord_per_token
    torch.testing.assert_close(
        out_hidden_states,
        expected_out_hidden_states,
        rtol=0,
        atol=0,
    )

    ret = f'{cp_rank=}\n\t{my_cp_rank_id_zz_ord_per_token=}\n\t{split_sizes_zz_to_nat=}\n\t{split_sizes_nat_to_zz=}\n\t{send_order.tolist()=}'
    return ret


class TestContiguousOrderToZz(unittest.TestCase):

    def setUp(self) -> None:
        ray.init(address="auto", log_to_driver=True)
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"need >= {WORLD_SIZE} GPUs in Ray cluster, only {total_gpus}")

    def tearDown(self) -> None:
        kill_all_actors_and_shutdown_ray()

    def test_1(self) -> None:
        cp_size = 4
        placement_group, bundle_indices = _create_placement_group(cp_size)

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
                test_contiguous_order_to_zz_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[cp_rank],
                    )
                ).remote(
                    cp_rank,
                    cp_size,
                    master_addr,
                    master_port,
                ) for cp_rank in range(cp_size)
            ]
            worker_results = ray.get(futures)
            for result in worker_results:
                print(result, flush=True)
        finally:
            remove_placement_group(placement_group)