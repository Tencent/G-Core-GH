# coding=utf-8
"""Convert DeepSeek-V4-Flash checkpoints between official and SGLang FP8 layouts.

Directions
----------
* ``official2sgl`` : ``deepseek-ai/DeepSeek-V4-Flash`` -> ``sgl-project/DeepSeek-V4-Flash-FP8``
    - MoE experts: FP4 (int8-packed) -> FP8 via DeepSeek ``cast_e2m1fn_to_e4m3fn``
    - Dense FP8 scales: ``float8_e8m0fnu`` -> ``float32`` (weight bytes unchanged)
    - ``attn.wo_a``: FP8+scale -> BF16 (no scale written)
* ``sgl2official`` : reverse of the above
    - MoE experts: FP8 -> requantize to FP4 packed + E8M0 (1x32)
    - Dense FP8 scales: ``float32`` -> ``float8_e8m0fnu``
    - ``attn.wo_a``: BF16 -> FP8 + E8M0 scale

Example
-------
mpirun -np 8 python3 tools/dsv4_quant_and_dequant/convert.py \\
  --direction official2sgl \\
  --src hf-hub/deepseek-ai/DeepSeek-V4-Flash \\
  --dst /tmp/dsv4_sgl_fp8 \\
  --device cuda

Launch with ``mpirun -np <num_gpus>``: each rank binds to one GPU (via node-
local rank), processes a size-balanced subset of shards, and gathers only the
output index metadata back to rank 0. Weight tensors never cross ranks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Any

import torch
from safetensors import safe_open
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.dsv4_quant_and_dequant.kernels import (  # noqa: E402
    is_expert_weight,
    is_wo_a_weight,
    official_dense_to_sgl,
    official_expert_to_sgl,
    official_wo_a_to_sgl,
    scale_key,
    sgl_dense_to_official,
    sgl_expert_to_official,
    sgl_wo_a_to_official,
)

from gpatch_v4.utils.safetensor_io import save_file  # noqa: E402

DIRECTIONS = ("official2sgl", "sgl2official")
_LAYER_RE = re.compile(r"^(layers\.\d+|mtp\.\d+)")


def get_current_time():
    torch.cuda.synchronize()
    return time.time()


def _init_mpi():
    """Return (comm, rank, world_size). Falls back to a single-rank stub."""
    try:
        from mpi4py import MPI  # noqa: PLC0415

        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        return None, 0, 1


def _local_rank(rank: int) -> int:
    """Node-local rank for GPU binding, from common MPI launcher env vars."""
    for key in (
        "OMPI_COMM_WORLD_LOCAL_RANK",  # OpenMPI
        "MV2_COMM_WORLD_LOCAL_RANK",  # MVAPICH2
        "MPI_LOCALRANKID",  # MPICH / Intel MPI
        "PMI_LOCAL_RANK",
        "SLURM_LOCALID",
        "LOCAL_RANK",
    ):
        val = os.environ.get(key)
        if val is not None:
            return int(val)
    return rank


def _resolve_device(device: str, rank: int) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    lr = _local_rank(rank)
    n = torch.cuda.device_count()
    assert n > 0, "no visible CUDA devices"
    idx = lr % n
    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")


def _partition_shards(shard_sizes: list[tuple[str, int]], world: int) -> list[list[str]]:
    """Greedy longest-processing-time bin-packing by shard byte size.

    Deterministic across ranks (same input -> same partition), so no
    communication is needed to agree on the work split.
    """
    buckets: list[list[str]] = [[] for _ in range(world)]
    loads = [0] * world
    for shard, size in sorted(shard_sizes, key=lambda x: (-x[1], x[0])):
        r = min(range(world), key=lambda i: (loads[i], i))
        buckets[r].append(shard)
        loads[r] += size
    return buckets


def _layer_of(name: str) -> str:
    m = _LAYER_RE.match(name)
    return m.group(1) if m else "non-layer"


class _ShardReader:
    """Lazy per-worker safetensors handle cache."""
    def __init__(self, root: str, weight_map: dict[str, str]):
        self.root = root
        self.weight_map = weight_map
        self._open: dict[str, Any] = {}

    def _handle(self, fname: str):
        if fname not in self._open:
            self._open[fname] = safe_open(
                os.path.join(self.root, fname), framework="pt", device="cpu"
            )
        return self._open[fname]

    def has(self, key: str) -> bool:
        if key not in self.weight_map:
            return False
        fname = self.weight_map[key]
        try:
            return key in self._handle(fname).keys()
        except Exception:
            return False

    def get(self, key: str) -> torch.Tensor:
        return self._handle(self.weight_map[key]).get_tensor(key)

    def close(self) -> None:
        self._open.clear()


def _copy_aux_files(src: str, dst: str) -> None:
    for name in sorted(os.listdir(src)):
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        elif os.path.isfile(s):
            shutil.copy2(s, d)


def _update_config(src: str, dst: str, direction: str) -> None:
    cfg_path = os.path.join(src, "config.json")
    if not os.path.isfile(cfg_path):
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    if direction == "official2sgl":
        cfg.pop("expert_dtype", None)
    else:
        cfg["expert_dtype"] = "fp4"
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _convert_one_weight(
    name: str,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    direction: str,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}

    if direction == "official2sgl":
        if is_expert_weight(name):
            assert scale is not None, f"missing scale for expert {name}"
            assert weight.dtype == torch.int8, (name, weight.dtype)
            q, s = official_expert_to_sgl(weight, scale)
            out[name] = q
            out[scale_key(name)] = s
        elif is_wo_a_weight(name):
            assert scale is not None, f"missing scale for wo_a {name}"
            out[name] = official_wo_a_to_sgl(weight, scale)
        elif scale is not None and weight.dtype == torch.float8_e4m3fn:
            q, s = official_dense_to_sgl(weight, scale)
            out[name] = q
            out[scale_key(name)] = s
        else:
            out[name] = weight
            if scale is not None:
                out[scale_key(name)] = scale
        return out

    if is_expert_weight(name):
        assert scale is not None, f"missing scale for expert {name}"
        assert weight.dtype == torch.float8_e4m3fn, (name, weight.dtype)
        q, s = sgl_expert_to_official(weight, scale)
        out[name] = q
        out[scale_key(name)] = s
    elif is_wo_a_weight(name):
        if weight.dtype == torch.bfloat16:
            q, s = sgl_wo_a_to_official(weight)
            out[name] = q
            out[scale_key(name)] = s
        elif scale is not None and weight.dtype == torch.float8_e4m3fn:
            q, s = sgl_dense_to_official(weight, scale)
            out[name] = q
            out[scale_key(name)] = s
        else:
            raise TypeError(f"unexpected wo_a dtype {weight.dtype} for {name}")
    elif scale is not None and weight.dtype == torch.float8_e4m3fn:
        q, s = sgl_dense_to_official(weight, scale)
        out[name] = q
        out[scale_key(name)] = s
    else:
        out[name] = weight
        if scale is not None:
            if scale.dtype == torch.float32:
                out[scale_key(name)] = scale.to(torch.float8_e8m0fnu)
            else:
                out[scale_key(name)] = scale
    return out


def _op_label(
    name: str, weight: torch.Tensor, scale: torch.Tensor | None, direction: str
) -> str | None:
    """Return a short op description if this weight needs quant/dequant; else None."""
    if direction == "official2sgl":
        if is_expert_weight(name):
            return "expert FP4->FP8"
        if is_wo_a_weight(name):
            return "wo_a FP8->BF16"
        if scale is not None and weight.dtype == torch.float8_e4m3fn:
            return "dense scale E8M0->f32"
        return None
    if is_expert_weight(name):
        return "expert FP8->FP4"
    if is_wo_a_weight(name) and weight.dtype == torch.bfloat16:
        return "wo_a BF16->FP8"
    if scale is not None and weight.dtype == torch.float8_e4m3fn:
        return "dense scale f32->E8M0"
    return None


def _process_shard(
    src: str,
    dst: str,
    shard_name: str,
    direction: str,
    weight_map: dict[str, str],
    device: torch.device,
) -> tuple[str, dict[str, str], int, int, int, list[str], dict[str, float]]:
    on_gpu = device.type == "cuda"
    t_shard0 = get_current_time()
    timing = {"load_s": 0.0, "convert_s": 0.0, "save_s": 0.0, "total_s": 0.0}

    reader = _ShardReader(src, weight_map)
    out_tensors: dict[str, torch.Tensor] = {}
    n_converted = 0
    n_passthrough = 0
    notes: list[str] = []
    last_layer: str | None = None

    try:
        t_load0 = get_current_time()
        with safe_open(os.path.join(src, shard_name), framework="pt", device="cpu") as sf:
            for name in sf.keys():
                if name.endswith(".scale"):
                    continue

                tensor = sf.get_tensor(name)
                if not name.endswith(".weight"):
                    out_tensors[name] = tensor
                    n_passthrough += 1
                    continue

                sk = scale_key(name)
                scale = None
                if reader.has(sk):
                    scale = reader.get(sk)
                elif sk in weight_map:
                    notes.append(
                        f"WARN: index lists {sk} in {weight_map[sk]} but tensor "
                        f"missing; treating as no-scale for {name}"
                    )

                op = _op_label(name, tensor, scale, direction)
                if op is not None:
                    layer = _layer_of(name)
                    if layer != last_layer:
                        print(
                            f"[convert][{direction}] shard={shard_name} layer={layer}",
                            flush=True,
                        )
                        last_layer = layer
                    # experts are numerous — only print layer once; print name for others
                    if not is_expert_weight(name):
                        print(
                            f"[convert][{direction}] layer={layer} {op}: {name}",
                            flush=True,
                        )

                t_c0 = get_current_time()
                if on_gpu and op is not None:
                    tensor_c = tensor.to(device, non_blocking=True)
                    scale_c = scale.to(device, non_blocking=True) if scale is not None else None
                else:
                    tensor_c = tensor
                    scale_c = scale
                converted = _convert_one_weight(name, tensor_c, scale_c, direction)
                if on_gpu:
                    converted = {
                        k: (v.to("cpu").contiguous() if v.is_cuda else v)
                        for k, v in converted.items()
                    }
                    torch.cuda.synchronize()
                timing["convert_s"] += get_current_time() - t_c0
                out_tensors.update(converted)
                if list(converted.keys()) == [name] and scale is None:
                    n_passthrough += 1
                else:
                    n_converted += 1
        timing["load_s"] = get_current_time() - t_load0 - timing["convert_s"]
    finally:
        reader.close()

    if not out_tensors:
        notes.append(
            f"WARN: shard {shard_name} produced no tensors after conversion "
            f"(likely scale-only shard); skipping write"
        )
        timing["total_s"] = get_current_time() - t_shard0
        return shard_name, {}, 0, n_converted, n_passthrough, notes, timing

    t_s0 = get_current_time()
    save_file(out_tensors, os.path.join(dst, shard_name))
    timing["save_s"] = get_current_time() - t_s0
    shard_map = {k: shard_name for k in out_tensors}
    total_size = sum(t.numel() * t.element_size() for t in out_tensors.values())
    timing["total_s"] = get_current_time() - t_shard0
    print(
        f"[convert][{direction}] shard={shard_name} done "
        f"converted={n_converted} passthrough={n_passthrough} "
        f"load={timing['load_s']:.1f}s convert={timing['convert_s']:.1f}s "
        f"save={timing['save_s']:.1f}s total={timing['total_s']:.1f}s",
        flush=True,
    )
    return shard_name, shard_map, total_size, n_converted, n_passthrough, notes, timing


def _write_index(dst: str, weight_map: dict[str, str], total_size: int) -> None:
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")


def convert_checkpoint(
    src: str,
    dst: str,
    direction: str,
    device: str = "cuda",
) -> None:
    assert direction in DIRECTIONS, direction
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    assert src != dst, "src and dst must differ"

    comm, rank, world = _init_mpi()
    is_root = rank == 0
    dev = _resolve_device(device, rank)
    t_all0 = get_current_time()

    index_path = os.path.join(src, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    # Deterministic size-balanced partition — every rank computes the same split.
    shard_sizes = [(s, os.path.getsize(os.path.join(src, s))) for s in shards]
    buckets = _partition_shards(shard_sizes, world)
    my_shards = buckets[rank]

    if is_root:
        print(f"[convert] direction={direction}")
        print(f"[convert] src={src}")
        print(f"[convert] dst={dst}")
        print(
            f"[convert] shards={len(shards)} keys={len(weight_map)} "
            f"world={world} device={device}",
            flush=True,
        )

    # Only the root sets up the destination dir / aux files / config, then all
    # ranks wait so they write into a consistent directory.
    aux_s = 0.0
    if is_root:
        os.makedirs(dst, exist_ok=True)
        t_aux0 = get_current_time()
        _copy_aux_files(src, dst)
        _update_config(src, dst, direction)
        aux_s = get_current_time() - t_aux0
    if comm is not None:
        comm.Barrier()

    print(
        f"[convert][rank{rank}] device={dev} shards={len(my_shards)} "
        f"(of {len(shards)})",
        flush=True,
    )

    local = {
        "weight_map": {},
        "bytes": 0,
        "converted": 0,
        "passthrough": 0,
        "notes": [],
        "timings": [],
    }

    t_shards0 = get_current_time()
    iterator = tqdm(my_shards, desc=f"rank{rank} shards") if is_root else my_shards
    for shard in iterator:
        result = _process_shard(src, dst, shard, direction, weight_map, dev)
        shard_name, shard_map, nbytes, n_c, n_p, notes, timing = result
        local["weight_map"].update(shard_map)
        local["bytes"] += nbytes
        local["converted"] += n_c
        local["passthrough"] += n_p
        local["notes"].extend(notes)
        local["timings"].append((shard_name, timing))
    local["wall_s"] = get_current_time() - t_shards0

    # Gather per-rank metadata to root; the heavy tensors never cross ranks.
    gathered = comm.gather(local, root=0) if (comm is not None and world > 1) else [local]
    if not is_root:
        return

    out_weight_map: dict[str, str] = {}
    totals = {"converted": 0, "passthrough": 0, "bytes": 0}
    all_notes: list[str] = []
    shard_timings: list[tuple[str, dict[str, float]]] = []
    rank_walls: list[float] = []
    for part in gathered:
        out_weight_map.update(part["weight_map"])
        totals["bytes"] += part["bytes"]
        totals["converted"] += part["converted"]
        totals["passthrough"] += part["passthrough"]
        all_notes.extend(part["notes"])
        shard_timings.extend(part["timings"])
        rank_walls.append(part["wall_s"])
    shards_wall_s = max(rank_walls) if rank_walls else 0.0

    print(f"[convert] writing index ({len(out_weight_map)} keys) ...")
    t_idx0 = get_current_time()
    _write_index(dst, out_weight_map, totals["bytes"])
    index_s = get_current_time() - t_idx0

    for note in all_notes[:20]:
        print(f"[convert] {note}")
    if len(all_notes) > 20:
        print(f"[convert] ... and {len(all_notes) - 20} more warnings")

    sum_load = sum(t["load_s"] for _, t in shard_timings)
    sum_convert = sum(t["convert_s"] for _, t in shard_timings)
    sum_save = sum(t["save_s"] for _, t in shard_timings)
    sum_total = sum(t["total_s"] for _, t in shard_timings)
    slowest = max(shard_timings, key=lambda x: x[1]["total_s"]) if shard_timings else None
    fastest = min(shard_timings, key=lambda x: x[1]["total_s"]) if shard_timings else None

    print("\n================ TIMING ================", flush=True)
    print(f"aux copy/config     : {aux_s:.1f}s", flush=True)
    print(
        f"shards wall-clock   : {shards_wall_s:.1f}s  (world={world}, slowest rank)",
        flush=True,
    )
    print(f"  sum load          : {sum_load:.1f}s", flush=True)
    print(f"  sum convert       : {sum_convert:.1f}s", flush=True)
    print(f"  sum save          : {sum_save:.1f}s", flush=True)
    print(f"  sum shard total   : {sum_total:.1f}s", flush=True)
    print(
        "  per-rank wall     : " + ", ".join(f"r{i}={w:.1f}s" for i, w in enumerate(rank_walls)),
        flush=True,
    )
    if slowest is not None and fastest is not None:
        print(
            f"  slowest shard     : {slowest[0]}  {slowest[1]['total_s']:.1f}s "
            f"(load={slowest[1]['load_s']:.1f} convert={slowest[1]['convert_s']:.1f} "
            f"save={slowest[1]['save_s']:.1f})",
            flush=True,
        )
        print(
            f"  fastest shard     : {fastest[0]}  {fastest[1]['total_s']:.1f}s",
            flush=True,
        )
    print(f"write index         : {index_s:.1f}s", flush=True)
    print(f"total wall-clock    : {get_current_time() - t_all0:.1f}s", flush=True)
    print(
        f"[convert] done: converted_weights={totals['converted']} "
        f"passthrough={totals['passthrough']}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--direction",
        required=True,
        choices=DIRECTIONS,
        help="official2sgl | sgl2official",
    )
    ap.add_argument("--src", required=True, help="source HF checkpoint directory")
    ap.add_argument("--dst", required=True, help="destination checkpoint directory")
    ap.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="compute device for quant/dequant (launch with mpirun for multi-GPU)",
    )
    args = ap.parse_args(argv)
    convert_checkpoint(args.src, args.dst, args.direction, device=args.device)


if __name__ == "__main__":
    main()
