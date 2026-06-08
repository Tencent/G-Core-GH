"""SGLang burn test -- per-node hardware health probe.

Uses Ray to dispatch one ``_burner_worker`` (``num_gpus=8``) per node.
Each worker launches ``sgl.Engine(tp=8)`` with an identical random
workload (54k prompt + forced 10k decode), reports wall-clock timing,
and the driver prints a summary table ranking nodes slowest-to-fastest.

Forces exact output length (``ignore_eos`` + ``min/max_new_tokens``) so
any hardware-induced slowdown (degraded GPU, NVLink, ECC errors) shows
up as elevated decode latency rather than being masked by early EOS.

Env knobs
---------
BURNER_MODEL_PATH     model path              (default: hf-hub/Qwen/Qwen3.6-35B-A3B)
BURNER_PROMPT_TOKENS  prompt length in tokens  (default: 54000)
BURNER_GEN_TOKENS     forced output tokens     (default: 10000)
BURNER_BATCH          batch size               (default: 4)
BURNER_TP_SIZE        tensor parallel size     (default: 8)
BURNER_EP_SIZE        expert parallel size     (default: BURNER_TP_SIZE)
BURNER_MEM_FRACTION   mem_fraction_static      (default: 0.7)
BURNER_TIMEOUT        ray.get timeout seconds  (default: 7200)

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=7200 \\
        tests/test_gpatch_v4/test_sgl_burner.py
"""

import json
import os
import random
import socket
import time
import unittest
from typing import Dict, List

import ray
import torch

from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

try:
    import sglang

    _SGLANG_AVAILABLE = True
except ImportError:
    _SGLANG_AVAILABLE = False

_GPUS_PER_NODE = 8


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _read_vocab_size(model_path: str) -> int:
    config_path = os.path.join(model_path, "config.json")
    assert os.path.isfile(config_path), (
        f"missing model config at {config_path}; "
        f"download the model into {model_path} first"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    vs = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size")
    assert isinstance(vs, int) and vs > 1, (f"cannot find a valid vocab_size in {config_path}")
    return vs


# ---------------------------------------------------------------------------
# Ray remote worker (one per node, occupies all 8 GPUs)
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=_GPUS_PER_NODE)
def _burner_worker(
    model_path: str,
    prompt_tokens: int,
    gen_tokens: int,
    batch: int,
    tp_size: int,
    ep_size: int,
    mem_fraction: float,
) -> Dict:
    """Run sgl.Engine on the local node and return timing result."""
    import sglang as sgl

    hostname = socket.gethostname()
    try:
        node_ip = ray.util.get_node_ip_address()
    except Exception:
        node_ip = "unknown"
    node_id = f"{hostname}/{node_ip}"
    vocab_size = _read_vocab_size(model_path)

    # Deterministic seed: all nodes get identical workload so timing
    # differences reflect hardware, not input variance.
    rng = random.Random(42)
    batched_input_ids: List[List[int]] = [
        [rng.randrange(1, vocab_size) for _ in range(prompt_tokens)] for _ in range(batch)
    ]
    batched_sp: List[Dict] = [
        {
            "temperature": 1.0,
            "top_p": 0.9,
            "min_new_tokens": gen_tokens,
            "max_new_tokens": gen_tokens,
            "ignore_eos": True,
        } for _ in range(batch)
    ]

    total_prompt = prompt_tokens * batch
    total_gen = gen_tokens * batch

    print(f"\n{'=' * 70}", flush=True)
    print(f"  [{node_id}] SGLang Burn Test", flush=True)
    print(f"  [{node_id}] model={model_path}", flush=True)
    print(
        f"  [{node_id}] tp={tp_size}  ep={ep_size}  batch={batch}",
        flush=True,
    )
    print(
        f"  [{node_id}] prompt={prompt_tokens}  gen={gen_tokens}  "
        f"total_prompt={total_prompt}  total_gen={total_gen}",
        flush=True,
    )
    print(f"{'=' * 70}\n", flush=True)

    # Let the OS pick a free port to avoid EADDRINUSE collisions.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.bind(("", 0))
        port = _s.getsockname()[1]
    server_args = sgl.ServerArgs(
        model_path=model_path,
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=1,
        pp_size=1,
        enable_dp_attention=False,
        dist_init_addr=f"127.0.0.1:{port}",
        nnodes=1,
        node_rank=0,
        base_gpu_id=0,
        mem_fraction_static=mem_fraction,
        trust_remote_code=True,
    )

    engine_t0 = time.perf_counter()
    llm = sgl.Engine(server_args=server_args)
    engine_elapsed = time.perf_counter() - engine_t0
    print(f"  [{node_id}] Engine ready in {engine_elapsed:.1f}s", flush=True)
    time.sleep(2)

    try:
        torch.cuda.synchronize()
        gen_t0 = time.perf_counter()
        outputs = llm.generate(
            input_ids=batched_input_ids,
            sampling_params=batched_sp,
        )
        torch.cuda.synchronize()
        gen_elapsed = time.perf_counter() - gen_t0

        assert isinstance(outputs, list) and len(outputs) == batch, (
            f"[{node_id}] expected {batch} outputs, "
            f"got {type(outputs)} "
            f"len={len(outputs) if isinstance(outputs, list) else 'N/A'}"
        )
        for oi, output in enumerate(outputs):
            got = len(output["output_ids"])
            assert got == gen_tokens, (
                f"[{node_id}] output[{oi}]: expected {gen_tokens} "
                f"tokens, got {got} -- EOS short-decode would "
                f"invalidate the burn test"
            )

        decode_tps = total_gen / gen_elapsed
        total_tps = (total_prompt + total_gen) / gen_elapsed

        print(f"\n{'=' * 70}", flush=True)
        print(f"  [{node_id}] BURN RESULT", flush=True)
        print(
            f"  [{node_id}]   gen_elapsed  = {gen_elapsed:.2f}s",
            flush=True,
        )
        print(
            f"  [{node_id}]   decode tok/s = {decode_tps:.1f}",
            flush=True,
        )
        print(
            f"  [{node_id}]   total  tok/s = {total_tps:.1f}",
            flush=True,
        )
        print(
            f"  [{node_id}]   engine_start = {engine_elapsed:.1f}s",
            flush=True,
        )
        print(f"{'=' * 70}\n", flush=True)

        return {
            "hostname": node_id,
            "gen_elapsed_s": round(gen_elapsed, 2),
            "engine_startup_s": round(engine_elapsed, 2),
            "decode_tok_per_s": round(decode_tps, 1),
            "total_tok_per_s": round(total_tps, 1),
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "batch": batch,
            "total_prompt_tokens": total_prompt,
            "total_gen_tokens": total_gen,
            "tp_size": tp_size,
            "ep_size": ep_size,
        }

    finally:
        llm.shutdown()
        time.sleep(2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(results: List[Dict]) -> None:
    """Print a summary table sorted by elapsed time (slowest first)."""
    if not results:
        print("[burner] No results to summarize.", flush=True)
        return

    n = len(results)
    print(f"\n{'=' * 80}", flush=True)
    print(
        f"  SGLang BURN TEST SUMMARY  ({n} node{'s' if n > 1 else ''})",
        flush=True,
    )
    print(f"{'=' * 80}", flush=True)
    print(
        f"  {'Hostname':<30} {'Gen Time (s)':>14} "
        f"{'Decode tok/s':>14} {'Note':>10}",
        flush=True,
    )
    print(f"  {'-' * 68}", flush=True)

    elapsed_list = [r["gen_elapsed_s"] for r in results]
    avg_elapsed = sum(elapsed_list) / len(elapsed_list)
    slowest = max(elapsed_list)
    fastest = min(elapsed_list)

    for r in sorted(results, key=lambda x: x["gen_elapsed_s"], reverse=True):
        e = r["gen_elapsed_s"]
        note = ""
        if n > 1:
            if e == slowest:
                note = "SLOWEST"
            elif e == fastest:
                note = "fastest"
        print(
            f"  {r['hostname']:<30} {e:>14.2f} "
            f"{r['decode_tok_per_s']:>14.1f} {note:>10}",
            flush=True,
        )

    print(f"  {'-' * 68}", flush=True)
    print(f"  {'Average':<30} {avg_elapsed:>14.2f}", flush=True)
    print(f"  {'Fastest':<30} {fastest:>14.2f}", flush=True)
    print(f"  {'Slowest':<30} {slowest:>14.2f}", flush=True)
    if n > 1:
        spread = slowest - fastest
        pct = (spread / fastest) * 100 if fastest > 0 else 0
        print(
            f"  {'Spread (slow-fast)':<30} {spread:>14.2f}  ({pct:.1f}%)",
            flush=True,
        )
    print(f"{'=' * 80}\n", flush=True)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@unittest.skipUnless(_SGLANG_AVAILABLE, "sglang not installed")
class SglBurnerTest(unittest.TestCase):
    """Per-node SGLang burn test for hardware fault detection.

    Dispatches one Ray task per node (``num_gpus=8``), each launching
    ``sgl.Engine(tp=8)`` with an identical random workload.  Compares
    wall-clock timing across nodes to surface degraded hardware.
    """
    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < _GPUS_PER_NODE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {_GPUS_PER_NODE} GPUs in Ray cluster, "
                f"only {total_gpus}"
            )

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def test_burn(self):
        model_path = _env_str("BURNER_MODEL_PATH", "hf-hub/Qwen/Qwen3.6-35B-A3B")
        prompt_tokens = _env_int("BURNER_PROMPT_TOKENS", 54000)
        gen_tokens = _env_int("BURNER_GEN_TOKENS", 10000)
        batch = _env_int("BURNER_BATCH", 4)
        tp_size = _env_int("BURNER_TP_SIZE", 8)
        ep_size = _env_int("BURNER_EP_SIZE", tp_size)
        mem_fraction = _env_float("BURNER_MEM_FRACTION", 0.7)
        timeout = _env_int("BURNER_TIMEOUT", 7200)

        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        num_nodes = total_gpus // _GPUS_PER_NODE
        assert num_nodes >= 1, (
            f"need at least {_GPUS_PER_NODE} GPUs for one node, "
            f"only {total_gpus} available"
        )

        total_prompt = prompt_tokens * batch
        total_gen = gen_tokens * batch

        print(f"\n{'=' * 70}", flush=True)
        print(
            f"  SGLang Burn Test: dispatching to {num_nodes} node(s)",
            flush=True,
        )
        print(f"  model      = {model_path}", flush=True)
        print(f"  tp={tp_size}  ep={ep_size}  batch={batch}", flush=True)
        print(
            f"  prompt={prompt_tokens}  gen={gen_tokens}  "
            f"total_prompt={total_prompt}  total_gen={total_gen}",
            flush=True,
        )
        print(f"{'=' * 70}\n", flush=True)

        futures = [
            _burner_worker.remote(
                model_path=model_path,
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                batch=batch,
                tp_size=tp_size,
                ep_size=ep_size,
                mem_fraction=mem_fraction,
            ) for _ in range(num_nodes)
        ]

        results = ray.get(futures, timeout=timeout)
        _print_summary(results)


if __name__ == "__main__":
    unittest.main()
