# coding=utf-8
"""Compare two HF DeepSeek-V4 checkpoints: dtypes first, then dequantized values.

Typical uses after conversion:
  1. official2sgl output  vs  published SGL FP8
  2. sgl2official output  vs  published official

Reports:
  * key coverage (only-in-A / only-in-B / phantom index entries)
  * dtype pair histogram (A.dtype vs B.dtype)
  * role x dtype pairs (expert / wo_a / dense_fp8 / scale / other)
  * shape mismatches
  * value comparison after per-side dequant (FP4/FP8/.scale aware)

Example
-------
# converted SGL vs published SGL (8 GPUs, one rank per GPU)
mpirun -np 8 python3 tools/dsv4_quant_and_dequant/compare_hf_ckpts.py \
  --a test_convert/official2sgl_full/ \
  --b hf-hub/sgl-project/DeepSeek-V4-Flash-FP8 \
  --a-name converted --b-name published_sgl --device cuda

mpirun -np 8 python3 tools/dsv4_quant_and_dequant/compare_hf_ckpts.py \
  --a test_convert/sgl2official_full/ \
  --b hf-hub/deepseek-ai/DeepSeek-V4-Flash_bak/ \
  --a-name converted --b-name published_official --device cuda

# dtype overview only (fast)
mpirun -np 8 python3 tools/dsv4_quant_and_dequant/compare_hf_ckpts.py \
  --a hf-hub/deepseek-ai/DeepSeek-V4-Flash \
  --b hf-hub/sgl-project/DeepSeek-V4-Flash-FP8 \
  --dtype-only --device cuda

Launch with ``mpirun -np <num_gpus>``: rank 0 computes key coverage and
broadcasts it; each rank binds one GPU and compares a round-robin slice of the
shared keys on that device; per-rank stats are gathered back to rank 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, OrderedDict
from typing import Any

import torch
from safetensors import safe_open

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.dsv4_quant_and_dequant.kernels import (  # noqa: E402
    dequant_fp8_block,
    is_expert_weight,
    is_wo_a_weight,
    scale_key,
)

FP8 = torch.float8_e4m3fn


def _init_mpi():
    """Return (comm, rank, world_size). Falls back to a single-rank stub."""
    try:
        from mpi4py import MPI  # noqa: PLC0415

        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        return None, 0, 1


def _local_rank(rank: int) -> int:
    for key in (
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
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
    n = torch.cuda.device_count()
    assert n > 0, "no visible CUDA devices"
    idx = _local_rank(rank) % n
    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")


def _norm_key(k: str) -> str:
    return re.sub(r"experts\.\d+\.", "experts.E.", k)


def _role(k: str, dtype: torch.dtype) -> str:
    if k.endswith(".scale"):
        return f"scale/{dtype}"
    if is_expert_weight(k):
        return f"expert/{dtype}"
    if is_wo_a_weight(k):
        return f"wo_a/{dtype}"
    if dtype == FP8:
        return f"dense_fp8/{dtype}"
    return f"other/{dtype}"


def _new_stat():
    return {
        "n": 0,
        "raw_equal": 0,
        "bad": 0,
        "max_abs_diff": 0.0,
        "min_cos": 1.0,
        "max_rel": 0.0,
    }


def _upd(st, mad, cos, rel, raw_equal, bad):
    st["n"] += 1
    st["raw_equal"] += int(raw_equal)
    st["bad"] += int(bad)
    st["max_abs_diff"] = max(st["max_abs_diff"], mad)
    st["min_cos"] = min(st["min_cos"], cos)
    st["max_rel"] = max(st["max_rel"], rel)


def _merge(dst, src):
    for kk in ("n", "raw_equal", "bad"):
        dst[kk] += src[kk]
    dst["max_abs_diff"] = max(dst["max_abs_diff"], src["max_abs_diff"])
    dst["min_cos"] = min(dst["min_cos"], src["min_cos"])
    dst["max_rel"] = max(dst["max_rel"], src["max_rel"])


def _metrics(a: torch.Tensor, b: torch.Tensor):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    if a.numel() != b.numel():
        return float("inf"), 0.0, float("inf")
    mad = (a - b).abs().max().item() if a.numel() else 0.0
    na, nb = a.norm(), b.norm()
    cos = float(torch.dot(a, b) / (na * nb + 1e-12)) if a.numel() else 1.0
    rel = float((a - b).norm() / (na + 1e-12))
    return mad, cos, rel


class HF:
    def __init__(self, d: str):
        self.dir = d
        self.wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self._open: dict[str, Any] = {}
        self._keys: dict[str, set[str]] = {}

    def _f(self, fname: str):
        if fname not in self._open:
            self._open[fname] = safe_open(
                os.path.join(self.dir, fname), framework="pt", device="cpu"
            )
        return self._open[fname]

    def _keyset(self, fname: str) -> set[str]:
        if fname not in self._keys:
            self._keys[fname] = set(self._f(fname).keys())
        return self._keys[fname]

    def has(self, k: str) -> bool:
        if k not in self.wm:
            return False
        try:
            return k in self._keyset(self.wm[k])
        except Exception:
            return False

    def raw(self, k: str) -> torch.Tensor:
        return self._f(self.wm[k]).get_tensor(k)

    def scale_of(self, k: str) -> torch.Tensor | None:
        """Companion ``.scale`` tensor for a ``.weight`` key, or None."""
        sk = scale_key(k)
        return self.raw(sk) if self.has(sk) else None


def _deq(k: str, w: torch.Tensor, scale: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    """Dequantize an already-loaded raw weight on ``device`` (float32)."""
    w = w.to(device)
    if k.endswith(".weight") and scale is not None and w.dtype in (FP8, torch.int8):
        return dequant_fp8_block(w, scale.to(device)).float()
    return w.float()


def _present_keys(wm: dict[str, str], d: str):
    present = set()
    missing = []
    opened: dict[str, set[str]] = {}
    for k, fname in wm.items():
        if fname not in opened:
            opened[fname] = set(safe_open(os.path.join(d, fname), framework="pt").keys())
        if k in opened[fname]:
            present.add(k)
        else:
            missing.append(k)
    return present, missing


def _dtype_probe(keys, A: "HF", B: "HF"):
    dtype_pairs = Counter()
    role_pairs = Counter()
    shape_mismatch = []
    for k in keys:
        try:
            if not (A.has(k) and B.has(k)):
                continue
            a, b = A.raw(k), B.raw(k)
            da, db = str(a.dtype), str(b.dtype)
            sa, sb = tuple(a.shape), tuple(b.shape)
            dtype_pairs[(da, db)] += 1
            role_pairs[(_role(k, a.dtype), da, db)] += 1
            if sa != sb:
                shape_mismatch.append(f"{k}: {sa} vs {sb}")
        except Exception as ex:  # noqa: BLE001
            shape_mismatch.append(f"{k}: {type(ex).__name__}: {ex}")
    return dtype_pairs, role_pairs, shape_mismatch


def _compare_values(keys, A: "HF", B: "HF", device: torch.device, cos_tol, abs_tol, plain_tol):
    stats = OrderedDict()
    reports = []
    for k in keys:
        if k.endswith(".scale"):
            continue
        nk = _norm_key(k)
        st = stats.setdefault(nk, _new_stat())
        try:
            a_raw, b_raw = A.raw(k), B.raw(k)
            a_val = _deq(k, a_raw, A.scale_of(k) if k.endswith(".weight") else None, device)
            b_val = _deq(k, b_raw, B.scale_of(k) if k.endswith(".weight") else None, device)
            is_quant = (
                a_raw.dtype in (FP8, torch.int8) or b_raw.dtype in (FP8, torch.int8) or
                a_raw.dtype != b_raw.dtype
            )
            mad, cos, rel = _metrics(a_val, b_val)
            if is_quant:
                bad = (cos < cos_tol) or (mad > abs_tol)
            else:
                bad = mad > plain_tol
            if a_raw.dtype == b_raw.dtype and tuple(a_raw.shape) == tuple(b_raw.shape):
                if a_raw.dtype == FP8:
                    raw_bytes_eq = bool(
                        torch.equal(a_raw.view(torch.uint8), b_raw.view(torch.uint8))
                    )
                else:
                    raw_bytes_eq = bool(torch.equal(a_raw, b_raw))
            else:
                raw_bytes_eq = False
            _upd(st, mad, cos, rel, raw_bytes_eq, bad)
            if bad and len(reports) < 500:
                reports.append(
                    f"[DIFF] {k} dtype=({a_raw.dtype},{b_raw.dtype}) "
                    f"shape=({tuple(a_raw.shape)},{tuple(b_raw.shape)}) "
                    f"mad={mad:.4g} cos={cos:.6f} rel={rel:.4g}"
                )
        except Exception as ex:  # noqa: BLE001
            st["bad"] += 1
            if len(reports) < 500:
                reports.append(f"[ERROR] {k}: {type(ex).__name__}: {ex}")
    return stats, reports


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", required=True, help="first HF ckpt dir")
    ap.add_argument("--b", required=True, help="second HF ckpt dir")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--cos-tol", type=float, default=0.999)
    ap.add_argument("--abs-tol", type=float, default=1e-2)
    ap.add_argument("--plain-tol", type=float, default=1e-6)
    ap.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=8,
        help="deprecated: parallelism is now the mpirun world size (kept for compat)",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="compute device for dequant/metrics (launch with mpirun for multi-GPU)",
    )
    ap.add_argument("--max-report", type=int, default=50)
    ap.add_argument(
        "--dtype-only",
        action="store_true",
        help="only report dtype/shape coverage, skip value compare",
    )
    args = ap.parse_args(argv)

    comm, rank, world = _init_mpi()
    is_root = rank == 0
    dev = _resolve_device(args.device, rank)
    t0 = time.time()

    # Rank 0 computes key coverage once, then broadcasts the key lists so every
    # rank agrees on the work set without each re-scanning all shard headers.
    if is_root:
        wa = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))["weight_map"]
        wb = json.load(open(os.path.join(args.b, "model.safetensors.index.json")))["weight_map"]
        keys_a, miss_a = _present_keys(wa, args.a)
        keys_b, miss_b = _present_keys(wb, args.b)
        cov = {
            "shared": sorted(keys_a & keys_b),
            "only_a": sorted(keys_a - keys_b),
            "only_b": sorted(keys_b - keys_a),
            "miss_a": miss_a,
            "miss_b": miss_b,
            "n_a": len(keys_a),
            "n_b": len(keys_b),
        }
        print(f"[compare] A({args.a_name})={args.a}", flush=True)
        print(f"[compare] B({args.b_name})={args.b}", flush=True)
        print(
            f"[compare] present A={cov['n_a']} B={cov['n_b']} "
            f"shared={len(cov['shared'])} only_A={len(cov['only_a'])} "
            f"only_B={len(cov['only_b'])} phantom_A={len(miss_a)} "
            f"phantom_B={len(miss_b)} world={world} device={args.device}",
            flush=True,
        )
        for k in miss_a[:args.max_report]:
            print(f"  [PHANTOM {args.a_name}] {k}", flush=True)
        for k in miss_b[:args.max_report]:
            print(f"  [PHANTOM {args.b_name}] {k}", flush=True)
        for k in cov["only_a"][:args.max_report]:
            print(f"  [ONLY {args.a_name}] {k}", flush=True)
        for k in cov["only_b"][:args.max_report]:
            print(f"  [ONLY {args.b_name}] {k}", flush=True)
    else:
        cov = None
    if comm is not None and world > 1:
        cov = comm.bcast(cov, root=0)

    shared = cov["shared"]
    only_a, only_b = cov["only_a"], cov["only_b"]
    miss_a, miss_b = cov["miss_a"], cov["miss_b"]

    A, B = HF(args.a), HF(args.b)
    # Round-robin over sorted keys: heavy experts (grouped by layer) get spread
    # evenly across ranks, avoiding the long tail of contiguous chunking.
    my_keys = shared[rank::world]

    # ---- dtype / shape probe ----
    my_dp, my_rp, my_sm = _dtype_probe(my_keys, A, B)
    if comm is not None and world > 1:
        parts = comm.gather((my_dp, my_rp, my_sm), root=0)
    else:
        parts = [(my_dp, my_rp, my_sm)]

    if is_root:
        dtype_pairs = Counter()
        role_pairs = Counter()
        shape_mismatch = []
        for dp, rp, sm in parts:
            dtype_pairs.update(dp)
            role_pairs.update(rp)
            shape_mismatch.extend(sm)

        print("\n================ DTYPE PAIRS (A vs B) ================", flush=True)
        for (da, db), n in sorted(dtype_pairs.items(), key=lambda x: -x[1]):
            tag = "SAME" if da == db else "DIFF"
            print(f"  [{tag}] {n:>6}  {da}  vs  {db}", flush=True)

        print("\n================ ROLE x DTYPE PAIRS ================", flush=True)
        for (role, da, db), n in sorted(role_pairs.items(), key=lambda x: (x[0][0], -x[1])):
            tag = "SAME" if da == db else "DIFF"
            print(f"  [{tag}] {n:>6}  {role:<48} {da} vs {db}", flush=True)

        if shape_mismatch:
            print("\n[shape mismatches]", flush=True)
            for r in shape_mismatch[:args.max_report]:
                print("  " + r, flush=True)

    if args.dtype_only:
        if is_root:
            print(f"\nelapsed={time.time() - t0:.1f}s", flush=True)
        return

    # ---- value compare (dequant + metrics on `dev`) ----
    my_stats, my_reports = _compare_values(
        my_keys, A, B, dev, args.cos_tol, args.abs_tol, args.plain_tol
    )
    if comm is not None and world > 1:
        vparts = comm.gather((my_stats, my_reports), root=0)
    else:
        vparts = [(my_stats, my_reports)]
    if not is_root:
        return

    stats = OrderedDict()
    all_reports = []
    for st_part, rep_part in vparts:
        all_reports.extend(rep_part)
        for nk, st in st_part.items():
            _merge(stats.setdefault(nk, _new_stat()), st)

    print(
        "\n================ per-tensor (experts.E collapsed, dequant compare) ================",
        flush=True,
    )
    hdr = (
        f"{'key':<56}{'n':>5}{'rawEq':>6}{'bad':>5}"
        f"{'max_abs':>11}{'min_cos':>10}{'max_rel':>11}"
    )
    print(hdr, flush=True)
    tot_n = tot_bad = tot_raw = 0
    for nk, st in sorted(stats.items()):
        tot_n += st["n"]
        tot_bad += st["bad"]
        tot_raw += st["raw_equal"]
        print(
            f"{nk:<56}{st['n']:>5}{st['raw_equal']:>6}{st['bad']:>5}"
            f"{st['max_abs_diff']:>11.4g}{st['min_cos']:>10.6f}{st['max_rel']:>11.4g}",
            flush=True,
        )

    if all_reports:
        print("\n[examples]", flush=True)
        for r in all_reports[:args.max_report]:
            print("  " + r, flush=True)

    print("\n================ SUMMARY ================", flush=True)
    print(f"compared tensors : {tot_n}", flush=True)
    print(f"raw byte-equal   : {tot_raw}", flush=True)
    print(f"mismatches (bad) : {tot_bad}", flush=True)
    print(f"only_A / only_B  : {len(only_a)} / {len(only_b)}", flush=True)
    print(f"phantom_A/B      : {len(miss_a)} / {len(miss_b)}", flush=True)
    print(f"elapsed={time.time() - t0:.1f}s", flush=True)
    ok = tot_bad == 0 and len(only_a) == 0 and len(only_b) == 0
    print("VERDICT: ALL CONSISTENT" if ok else "VERDICT: PROBLEMS FOUND", flush=True)


if __name__ == "__main__":
    main()
