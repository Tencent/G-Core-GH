# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com

from __future__ import annotations

import functools
import importlib.util
import os
import socket
import unittest
from pathlib import Path

import ray
import torch
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.orches.placement_group import _create_placement_group
from gpatch_v4.orches.utils import build_actor_env_vars

WORLD_SIZE = 16
SEQ_LEN = 32 * 1024
_FA3_AVAILABLE = importlib.util.find_spec("flash_attn_interface") is not None
ULYSSES_CP_SIZES = (16,)
ULYSSES_BATCH_SIZE = int(os.environ.get("DSV4_ULYSSES_BATCH_SIZE", "1"))
ULYSSES_NUM_HEADS = int(os.environ.get("DSV4_ULYSSES_NUM_HEADS", "128"))
ULYSSES_HEAD_DIM = int(os.environ.get("DSV4_ULYSSES_HEAD_DIM", "256"))
ULYSSES_WARMUP_STEPS = int(os.environ.get("DSV4_ULYSSES_WARMUP", "2"))
# Pipeline depth. Prefer NUM_GROUPS; HEADS_PER_GROUP overrides if > 0.
# heads_per_group MUST be divisible by cp_size.
ULYSSES_NUM_GROUPS = int(os.environ.get("DSV4_ULYSSES_NUM_GROUPS", "4"))
ULYSSES_HEADS_PER_GROUP = int(os.environ.get("DSV4_ULYSSES_HEADS_PER_GROUP", "0"))
# FA3 的 sm_margin 留出 SM 给通信；margin=0 时 FA 占满 SM，a2a 会被饿死。
ULYSSES_SM_MARGIN = int(os.environ.get("DSV4_ULYSSES_SM_MARGIN", "4"))
SM_MARGINS = tuple(
    int(x) for x in os.environ.get("DSV4_SP_SM_MARGINS", "0,4,8,16").split(",") if x.strip()
)
# 低于此值说明通信基本没被隐藏，overlap 能力不成立。
OVERLAP_EFF_GATE = float(os.environ.get("DSV4_SP_OVERLAP_GATE", "0.5"))


@ray.remote(num_cpus=0, num_gpus=0)
def _get_master_endpoint() -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return ray.util.get_node_ip_address(), sock.getsockname()[1]


def _seq_to_head_send(tensor: torch.Tensor, cp_size: int) -> torch.Tensor:
    batch, local_seq_len, num_heads, head_dim = tensor.shape
    assert num_heads % cp_size == 0
    local_num_heads = num_heads // cp_size
    return (
        tensor.view(batch, local_seq_len, cp_size, local_num_heads,
                    head_dim).permute(2, 0, 1, 3, 4).contiguous()
    )


def _seq_to_head_finalize(
    recv: torch.Tensor,
    batch: int,
    local_seq_len: int,
    local_num_heads: int,
    head_dim: int,
    cp_size: int,
) -> torch.Tensor:
    return (
        recv.permute(1, 0, 2, 3,
                     4).reshape(batch, local_seq_len * cp_size, local_num_heads,
                                head_dim).contiguous()
    )


def _head_to_seq_send(tensor: torch.Tensor, cp_size: int) -> torch.Tensor:
    batch, seq_len, local_num_heads, head_dim = tensor.shape
    assert seq_len % cp_size == 0
    local_seq_len = seq_len // cp_size
    return (
        tensor.view(batch, cp_size, local_seq_len, local_num_heads,
                    head_dim).permute(1, 0, 2, 3, 4).contiguous()
    )


def _head_to_seq_finalize(
    recv: torch.Tensor,
    batch: int,
    local_seq_len: int,
    local_num_heads: int,
    head_dim: int,
    cp_size: int,
) -> torch.Tensor:
    return (
        recv.permute(1, 2, 0, 3,
                     4).reshape(batch, local_seq_len, local_num_heads * cp_size,
                                head_dim).contiguous()
    )


def _ulysses_seq_to_head(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    cp_size = dist.get_world_size(group)
    batch, local_seq_len, num_heads, head_dim = tensor.shape
    local_num_heads = num_heads // cp_size
    send = _seq_to_head_send(tensor, cp_size)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return _seq_to_head_finalize(recv, batch, local_seq_len, local_num_heads, head_dim, cp_size)


def _ulysses_head_to_seq(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    cp_size = dist.get_world_size(group)
    batch, seq_len, local_num_heads, head_dim = tensor.shape
    local_seq_len = seq_len // cp_size
    send = _head_to_seq_send(tensor, cp_size)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return _head_to_seq_finalize(recv, batch, local_seq_len, local_num_heads, head_dim, cp_size)


def _launch_seq_to_head_async(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, dist.Work, tuple[int, int, int, int, int]]:
    cp_size = dist.get_world_size(group)
    batch, local_seq_len, num_heads, head_dim = tensor.shape
    local_num_heads = num_heads // cp_size
    send = _seq_to_head_send(tensor, cp_size)
    recv = torch.empty_like(send)
    work = dist.all_to_all_single(recv, send, group=group, async_op=True)
    return recv, work, (batch, local_seq_len, local_num_heads, head_dim, cp_size)


def _launch_head_to_seq_async(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, dist.Work, tuple[int, int, int, int, int]]:
    cp_size = dist.get_world_size(group)
    batch, seq_len, local_num_heads, head_dim = tensor.shape
    local_seq_len = seq_len // cp_size
    send = _head_to_seq_send(tensor, cp_size)
    recv = torch.empty_like(send)
    work = dist.all_to_all_single(recv, send, group=group, async_op=True)
    return recv, work, (batch, local_seq_len, local_num_heads, head_dim, cp_size)


def _ulysses_e2e_serial(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    flash_attn_func,
) -> torch.Tensor:
    with torch.cuda.nvtx.range("ulysses_a2a_q"):
        q_exchanged = _ulysses_seq_to_head(q, cp_group)
    with torch.cuda.nvtx.range("ulysses_a2a_k"):
        k_exchanged = _ulysses_seq_to_head(k, cp_group)
    with torch.cuda.nvtx.range("ulysses_a2a_v"):
        v_exchanged = _ulysses_seq_to_head(v, cp_group)
    with torch.cuda.nvtx.range("ulysses_flash_attn"):
        output = flash_attn_func(
            q_exchanged,
            k_exchanged,
            v_exchanged,
            causal=True,
        )
    with torch.cuda.nvtx.range("ulysses_a2a_out"):
        return _ulysses_head_to_seq(output, cp_group)


def _heads_per_group(cp_size: int) -> int:
    if ULYSSES_HEADS_PER_GROUP > 0:
        g = ULYSSES_HEADS_PER_GROUP
    else:
        assert ULYSSES_NUM_GROUPS > 0
        assert ULYSSES_NUM_HEADS % ULYSSES_NUM_GROUPS == 0, (
            f"num_heads={ULYSSES_NUM_HEADS} must be divisible by "
            f"num_groups={ULYSSES_NUM_GROUPS}"
        )
        g = ULYSSES_NUM_HEADS // ULYSSES_NUM_GROUPS
    assert g % cp_size == 0, f"heads_per_group={g} must be divisible by cp_size={cp_size}"
    assert ULYSSES_NUM_HEADS % g == 0, f"num_heads={ULYSSES_NUM_HEADS} must be divisible by {g}"
    return g


def _ulysses_e2e_head_pipeline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    flash_attn_func,
    heads_per_group: int,
) -> torch.Tensor:
    """Pipeline Ulysses by head groups with eager input a2a.

    Launch ALL groups' QKV all-to-all first (so out-a2a cannot block later in-a2a
    on the single NCCL stream), then for each group: wait_in -> FA -> launch_out.
    Remaining in-a2a overlaps FA; out-a2a overlaps later FA / trailing compute.
    """
    num_heads = q.shape[2]
    num_groups = num_heads // heads_per_group
    out_local = torch.empty_like(q)

    def slice_heads(tensor: torch.Tensor, group_idx: int) -> torch.Tensor:
        start = group_idx * heads_per_group
        end = start + heads_per_group
        return tensor[:, :, start:end, :].contiguous()

    def launch_in(group_idx: int):
        with torch.cuda.nvtx.range(f"ulysses_launch_in_g{group_idx}"):
            q_g = slice_heads(q, group_idx)
            k_g = slice_heads(k, group_idx)
            v_g = slice_heads(v, group_idx)
            q_recv, q_work, q_meta = _launch_seq_to_head_async(q_g, cp_group)
            k_recv, k_work, k_meta = _launch_seq_to_head_async(k_g, cp_group)
            v_recv, v_work, v_meta = _launch_seq_to_head_async(v_g, cp_group)
        return (q_recv, k_recv, v_recv), (q_work, k_work, v_work), (q_meta, k_meta, v_meta)

    def wait_in(recvs, works, metas):
        with torch.cuda.nvtx.range("ulysses_wait_in"):
            for work in works:
                work.wait()
            q_x = _seq_to_head_finalize(recvs[0], *metas[0])
            k_x = _seq_to_head_finalize(recvs[1], *metas[1])
            v_x = _seq_to_head_finalize(recvs[2], *metas[2])
        return q_x, k_x, v_x

    pending_out: list[tuple[torch.Tensor, dist.Work, tuple, int]] = []

    def launch_out(attn_out: torch.Tensor, group_idx: int):
        with torch.cuda.nvtx.range(f"ulysses_launch_out_g{group_idx}"):
            recv, work, meta = _launch_head_to_seq_async(attn_out, cp_group)
        pending_out.append((recv, work, meta, group_idx))

    # 1) Eager-launch every group's input a2a before any FA / out a2a.
    in_states = [launch_in(group_idx) for group_idx in range(num_groups)]

    # 2) Compute + output: wait_in(i) only needs in(i); later in(*) stay on NCCL.
    for group_idx in range(num_groups):
        q_x, k_x, v_x = wait_in(*in_states[group_idx])
        with torch.cuda.nvtx.range(f"ulysses_flash_attn_g{group_idx}"):
            attn_out = flash_attn_func(q_x, k_x, v_x, causal=True)
        launch_out(attn_out, group_idx)

    with torch.cuda.nvtx.range("ulysses_drain_out"):
        for recv, work, meta, group_idx in pending_out:
            work.wait()
            local = _head_to_seq_finalize(recv, *meta)
            start = group_idx * heads_per_group
            end = start + heads_per_group
            out_local[:, :, start:end, :] = local
    return out_local


@ray.remote(num_gpus=1, max_calls=1)
def _profile_ulysses_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    enable_nsys: bool = False,
    trace_dir: str | None = None,
) -> list[dict[str, int | float | bool | str]]:
    flash_attn_func = functools.partial(
        importlib.import_module("flash_attn_interface").flash_attn_func,
        sm_margin=ULYSSES_SM_MARGIN,
    )

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    gpu_cnt = torch.cuda.device_count()
    assert gpu_cnt > 0
    device = torch.device("cuda", rank % gpu_cnt)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    try:
        results = []
        for cp_size in ULYSSES_CP_SIZES:
            assert world_size % cp_size == 0
            assert SEQ_LEN % cp_size == 0
            assert ULYSSES_NUM_HEADS % cp_size == 0
            heads_per_group = _heads_per_group(cp_size)
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
            q_roundtrip = _ulysses_head_to_seq(q_full, cp_group)
            roundtrip_ok = torch.equal(q_roundtrip, q)
            assert roundtrip_ok
            del q_full, q_roundtrip

            for _ in range(ULYSSES_WARMUP_STEPS):
                _ulysses_e2e_serial(q, k, v, cp_group, flash_attn_func)
                _ulysses_e2e_head_pipeline(
                    q, k, v, cp_group, flash_attn_func, heads_per_group
                )
            torch.cuda.synchronize()
            dist.barrier(group=cp_group)

            with torch.no_grad():
                serial_out = _ulysses_e2e_serial(q, k, v, cp_group, flash_attn_func)
                pipeline_out = _ulysses_e2e_head_pipeline(
                    q, k, v, cp_group, flash_attn_func, heads_per_group
                )
            torch.cuda.synchronize()
            # Heads are independent; grouped FA vs monolithic FA may differ at bf16 ULPs.
            torch.testing.assert_close(pipeline_out, serial_out, rtol=1.6e-2, atol=1e-2)
            del serial_out, pipeline_out

            timings = {}
            trace_paths: dict[str, str] = {}
            for mode, fn in (
                # 这个先不用看了
                (
                    "serial",
                    lambda: _ulysses_e2e_serial(q, k, v, cp_group, flash_attn_func),
                ),
                (
                    "overlap",
                    lambda: _ulysses_e2e_head_pipeline(
                        q, k, v, cp_group, flash_attn_func, heads_per_group
                    ),
                ),
            ):
                for _ in range(ULYSSES_WARMUP_STEPS):
                    fn()
                torch.cuda.synchronize()
                dist.barrier(group=cp_group)
                profile_with_nsys = (
                    enable_nsys
                    and rank == 0
                    and cp_size == ULYSSES_CP_SIZES[-1]
                    and mode == "overlap"
                )
                # 所有 rank 一起开 profiler：只插桩 rank 0 会让它比 peers 慢，
                # skew 沿 NCCL stream 传播，污染 a2a 的 kernel duration。
                profile_chrome = trace_dir is not None and not enable_nsys
                if profile_with_nsys:
                    torch.cuda.profiler.start()
                dist.barrier(group=cp_group)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                if profile_chrome:
                    chrome_warmup = 2
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch.profiler.schedule(
                            wait=0, warmup=chrome_warmup, active=1
                        ),
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                    ) as profiler:
                        # CUPTI 已开启但事件被 schedule 丢弃，先跑完懒初始化再进 active
                        for _ in range(chrome_warmup):
                            fn()
                            torch.cuda.synchronize()
                            profiler.step()
                        dist.barrier(group=cp_group)
                        with torch.cuda.nvtx.range(f"ulysses_e2e_{mode}_cp{cp_size}"):
                            start.record()
                            fn()
                            end.record()
                        torch.cuda.synchronize()
                    if rank == 0:
                        trace_path = Path(trace_dir) / f"rank-0-cp{cp_size}-{mode}.json"
                        profiler.export_chrome_trace(str(trace_path))
                        trace_paths[mode] = str(trace_path)
                        print(f"[ulysses] chrome trace: {trace_path}", flush=True)
                else:
                    with torch.cuda.nvtx.range(f"ulysses_e2e_{mode}_cp{cp_size}"):
                        start.record()
                        fn()
                        end.record()
                    torch.cuda.synchronize()
                if profile_with_nsys:
                    torch.cuda.profiler.stop()
                timings[mode] = start.elapsed_time(end)
                dist.barrier(group=cp_group)

            result: dict[str, int | float | bool | str] = {
                "rank": rank,
                "cp_size": cp_size,
                "heads_per_group": heads_per_group,
                "num_groups": ULYSSES_NUM_HEADS // heads_per_group,
                "local_seq_len": local_seq_len,
                "local_num_heads": ULYSSES_NUM_HEADS // cp_size,
                "roundtrip_ok": roundtrip_ok,
                "serial_ms": timings["serial"],
                "overlap_ms": timings["overlap"],
            }
            if "overlap" in trace_paths:
                result["trace_overlap"] = trace_paths["overlap"]
            if "serial" in trace_paths:
                result["trace_serial"] = trace_paths["serial"]
            results.append(result)
            del q, k, v
            torch.cuda.empty_cache()
            dist.barrier()
        return results
    finally:
        dist.destroy_process_group()


@ray.remote(num_gpus=1, max_calls=1)
def _overlap_probe_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> list[dict[str, int | float | str]]:
    """量化 a2a 与 FA3 的 overlap 能力；a2a 与 FA 之间故意无数据依赖。"""
    fa3_func = importlib.import_module("flash_attn_interface").flash_attn_func

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    gpu_cnt = torch.cuda.device_count()
    assert gpu_cnt > 0
    device = torch.device("cuda", rank % gpu_cnt)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    try:
        cp_size = ULYSSES_CP_SIZES[-1]
        assert world_size % cp_size == 0
        assert SEQ_LEN % cp_size == 0
        assert ULYSSES_NUM_HEADS % cp_size == 0
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        cp_group = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // cp_size, cp_size),
            mesh_dim_names=("dp", "cp"),
        )["cp"].get_group()

        local_seq_len = SEQ_LEN // cp_size
        shape = (ULYSSES_BATCH_SIZE, local_seq_len, ULYSSES_NUM_HEADS, ULYSSES_HEAD_DIM)
        q = torch.empty(shape, device="cuda", dtype=torch.bfloat16).normal_(std=0.02)
        k = torch.empty_like(q).normal_(std=0.02)
        v = torch.empty_like(q).normal_(std=0.02)
        q_x = _ulysses_seq_to_head(q, cp_group)
        k_x = _ulysses_seq_to_head(k, cp_group)
        v_x = _ulysses_seq_to_head(v, cp_group)

        def comm_only() -> None:
            # pending 持有 recv buffer，防止 a2a 在飞时被回收
            pending = [_launch_seq_to_head_async(t, cp_group) for t in (q, k, v)]
            for _, work, _ in pending:
                work.wait()

        def make_comp(sm_margin: int):
            def comp() -> None:
                fa3_func(q_x, k_x, v_x, causal=True, sm_margin=sm_margin)

            return comp

        def make_both(sm_margin: int):
            def both() -> None:
                pending = [_launch_seq_to_head_async(t, cp_group) for t in (q, k, v)]
                fa3_func(q_x, k_x, v_x, causal=True, sm_margin=sm_margin)
                for _, work, _ in pending:
                    work.wait()

            return both

        def bench_ms(fn) -> float:
            for _ in range(ULYSSES_WARMUP_STEPS):
                fn()
            torch.cuda.synchronize()
            dist.barrier(group=cp_group)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            dist.barrier(group=cp_group)
            return start.elapsed_time(end)

        results = []
        with torch.no_grad():
            comm_ms = bench_ms(comm_only)
            for sm_margin in SM_MARGINS:
                assert 0 <= sm_margin < sm_count, (
                    f"sm_margin={sm_margin} out of range for {sm_count} SMs"
                )
                comp_ms = bench_ms(make_comp(sm_margin))
                both_ms = bench_ms(make_both(sm_margin))
                results.append(
                    {
                        "rank": rank,
                        "sm_count": sm_count,
                        "sm_margin": sm_margin,
                        "comm_ms": comm_ms,
                        "comp_ms": comp_ms,
                        "both_ms": both_ms,
                    }
                )
        return results
    finally:
        dist.destroy_process_group()


def _overlap_eff(comm_ms: float, comp_ms: float, both_ms: float) -> float:
    return (comm_ms + comp_ms - both_ms) / min(comm_ms, comp_ms)


def _assert_ulysses_worker_results(
    test_case: unittest.TestCase,
    worker_results: list[list[dict[str, int | float | bool | str]]],
) -> None:
    results = [result for rank_results in worker_results for result in rank_results]
    print(
        "\nUlysses head-pipeline overlap demo: "
        f"global_seq_len={SEQ_LEN}, batch={ULYSSES_BATCH_SIZE}, "
        f"heads={ULYSSES_NUM_HEADS}, head_dim={ULYSSES_HEAD_DIM}, "
        f"causal=True, warmup={ULYSSES_WARMUP_STEPS}"
    )
    for cp_size in ULYSSES_CP_SIZES:
        case_results = [result for result in results if result["cp_size"] == cp_size]
        test_case.assertEqual(len(case_results), WORLD_SIZE)
        test_case.assertTrue(all(result["roundtrip_ok"] for result in case_results))
        serial_result = max(case_results, key=lambda r: float(r["serial_ms"]))
        overlap_result = max(case_results, key=lambda r: float(r["overlap_ms"]))
        serial_ms = float(serial_result["serial_ms"])
        overlap_ms = float(overlap_result["overlap_ms"])
        heads_per_group = int(case_results[0]["heads_per_group"])
        num_groups = int(case_results[0]["num_groups"])
        print(
            f"CP={cp_size}: heads_per_group={heads_per_group}, "
            f"num_groups={num_groups}, local_shape="
            f"[{ULYSSES_BATCH_SIZE}, {SEQ_LEN // cp_size}, "
            f"{ULYSSES_NUM_HEADS}, {ULYSSES_HEAD_DIM}] -> "
            f"[{ULYSSES_BATCH_SIZE}, {SEQ_LEN}, "
            f"{ULYSSES_NUM_HEADS // cp_size}, {ULYSSES_HEAD_DIM}]"
        )
        print(
            f"  serial={serial_ms:.3f} ms (rank={serial_result['rank']}), "
            f"overlap={overlap_ms:.3f} ms (rank={overlap_result['rank']})"
        )
        rank0 = next(r for r in case_results if int(r["rank"]) == 0)
        if "trace_serial" in rank0:
            print(f"  chrome serial: {rank0['trace_serial']}")
        if "trace_overlap" in rank0:
            print(f"  chrome overlap: {rank0['trace_overlap']}")
        test_case.assertGreater(serial_ms, 0.0)
        test_case.assertGreater(overlap_ms, 0.0)


@unittest.skipUnless(_FA3_AVAILABLE, "flash_attn_interface (FA3) not installed")
class Test1(unittest.TestCase):
    def setUp(self) -> None:
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"need >= {WORLD_SIZE} GPUs in Ray cluster, only {total_gpus}")

    def tearDown(self) -> None:
        kill_all_actors_and_shutdown_ray()

    def test_ulysses_e2e(self) -> None:
        self.assertGreaterEqual(ULYSSES_WARMUP_STEPS, 0)
        trace_dir = (Path.cwd() / "ulysses_sp_overlap").resolve()
        trace_dir.mkdir(exist_ok=True)
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
            actor_env_vars = build_actor_env_vars()
            futures = [
                _profile_ulysses_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[rank],
                    ),
                    runtime_env={"env_vars": actor_env_vars},
                ).remote(
                    rank,
                    WORLD_SIZE,
                    master_addr,
                    master_port,
                    False,
                    str(trace_dir),
                ) for rank in range(WORLD_SIZE)
            ]
            worker_results = ray.get(futures)
        finally:
            remove_placement_group(placement_group)

        _assert_ulysses_worker_results(self, worker_results)

    def test_a2a_fa3_overlap_capability(self) -> None:
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
            actor_env_vars = build_actor_env_vars()
            futures = [
                _overlap_probe_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[rank],
                    ),
                    runtime_env={"env_vars": actor_env_vars},
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

        results = [r for rank_results in worker_results for r in rank_results]
        cp_size = ULYSSES_CP_SIZES[-1]
        print(
            f"\na2a x3 vs FA3 overlap probe: cp={cp_size}, "
            f"local=[{ULYSSES_BATCH_SIZE}, {SEQ_LEN // cp_size}, {ULYSSES_NUM_HEADS}, "
            f"{ULYSSES_HEAD_DIM}] -> [{ULYSSES_BATCH_SIZE}, {SEQ_LEN}, "
            f"{ULYSSES_NUM_HEADS // cp_size}, {ULYSSES_HEAD_DIM}], "
            f"SMs={int(results[0]['sm_count'])}"
        )
        print("  sm_margin    comm_ms    comp_ms    both_ms   overlap_eff")

        best_eff = float("-inf")
        for sm_margin in SM_MARGINS:
            case = [r for r in results if int(r["sm_margin"]) == sm_margin]
            self.assertEqual(len(case), WORLD_SIZE)
            # 取最慢 rank：overlap 由全局最慢者决定
            comm_ms = max(float(r["comm_ms"]) for r in case)
            comp_ms = max(float(r["comp_ms"]) for r in case)
            both_ms = max(float(r["both_ms"]) for r in case)
            self.assertGreater(comm_ms, 0.0)
            self.assertGreater(comp_ms, 0.0)
            self.assertGreater(both_ms, 0.0)
            eff = _overlap_eff(comm_ms, comp_ms, both_ms)
            best_eff = max(best_eff, eff)
            print(
                f"  {sm_margin:>9d} {comm_ms:10.3f} {comp_ms:10.3f} "
                f"{both_ms:10.3f} {eff:13.3f}"
            )

        self.assertGreater(
            best_eff,
            OVERLAP_EFF_GATE,
            f"no sm_margin hides >{OVERLAP_EFF_GATE:.0%} of the a2a; "
            f"best overlap_eff={best_eff:.3f} over margins {SM_MARGINS}",
        )
