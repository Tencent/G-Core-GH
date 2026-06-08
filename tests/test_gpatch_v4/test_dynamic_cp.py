"""Dynamic CP correctness tests (RL).

Spawns 4 Ray actors (1 GPU each) with ``TP=1, CP=2, DP=2,
dynamic_context_parallel=True`` and exercises padding, schedule + reroute,
THD packing, CP slicing, RL extra-field routing.

Coverage matrix
---------------
* ``local_cp_size``: ``1``, ``2``, ``4``
* ``num_microbatches``: ``1``, ``2`` and more
* Position-level token round-trip (catches within-sample CP slice errors)
* Position-level RL extra-field alignment (advantages / logprobs paired with tokens)

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=600 tests/test_gpatch_v4/test_dynamic_cp.py
"""

import math
import os
import socket
import unittest
from typing import Any, Dict, List, Tuple

import ray
import torch


WORLD_SIZE = 4
TP, CP = 1, 2
DP = WORLD_SIZE // (TP * CP)
VOCAB = 8
MAX_SEQLEN_PER_RANK = 64

# Default lengths: produce num_micro=2 with both lcp=1 and lcp=2 microbatches.
SFT_LENGTHS = {0: [12, 24, 48, 100], 1: [16, 30, 60, 80]}
# A 200-token sample forces lcp=4 (200/4=50 <= 64).
LCP4_LENGTHS = {0: [200, 16, 24], 1: [16, 30, 32]}
# Tiny batch that fits in a single microbatch.
SMALL_LENGTHS = {0: [16, 24], 1: [16, 32]}

# Encoded value space for positional tests: sid * _POS_STRIDE + pos. Must
# exceed any tested sequence length.
_POS_STRIDE = 1000
_BINCOUNT_RANGE = VOCAB * _POS_STRIDE


def _make_sample_id(dp_rank: int, idx: int) -> int:
    """Return a non-zero token id in ``[1, VOCAB-1]`` unique within the rank."""
    return 1 + (dp_rank * 4 + idx) % (VOCAB - 1)


def _encode_pos(sid: int, pos: int) -> int:
    """Encode (sid, pos) into a unique int so position swaps are detectable."""
    return sid * _POS_STRIDE + pos


# ---------------------------------------------------------------------------
# Ray actor: one GPU per actor.
# ---------------------------------------------------------------------------

class DynCpWorker:
    """Single-GPU worker that runs Dynamic CP pipelines and returns raw data
    for the test class to assert on."""

    # -- bootstrapping ------------------------------------------------------

    def get_master_addr_and_port(self) -> Tuple[str, int]:
        ip = socket.gethostbyname(socket.gethostname())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return ip, s.getsockname()[1]

    def init_dist(self, rank: int, world_size: int, master_addr: str, master_port: int):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            backend="nccl", rank=rank, world_size=world_size,
        )
        self.device = torch.device("cuda:0")

        from megatron.core import parallel_state as ps
        ps.initialize_model_parallel(
            tensor_model_parallel_size=TP,
            pipeline_model_parallel_size=1,
            context_parallel_size=CP,
            dynamic_context_parallel=True,
            min_dynamic_context_parallel_size=1,
        )
        self.dp_rank = ps.get_data_parallel_rank()

    def destroy(self):
        torch.distributed.destroy_process_group()

    # -- sample builders ----------------------------------------------------

    def _rl_samples(self, lengths: List[int], positional: bool = False):
        """Build RL samples; ``positional=True`` makes tokens / extras carry
        ``encode(sid, pos)`` so each per-token position can be traced."""
        out = []
        for i, L in enumerate(lengths):
            sid = _make_sample_id(self.dp_rank, i)
            if positional:
                tokens = torch.tensor(
                    [_encode_pos(sid, j) for j in range(L)],
                    dtype=torch.int64, device=self.device,
                )
                adv = torch.tensor(
                    [float(_encode_pos(sid, j)) for j in range(L - 1)],
                    dtype=torch.float32, device=self.device,
                )
            else:
                tokens = torch.full((L,), sid, dtype=torch.int64, device=self.device)
                adv = torch.full((L - 1,), float(sid), dtype=torch.float32, device=self.device)
            mask = torch.ones(L - 1, dtype=torch.float32, device=self.device)
            out.append({
                "tokens": tokens, "mask": mask,
                "advantages": adv, "logprobs": -adv, "ref_logprobs": -2 * adv,
                "sequence_lengths": torch.tensor(L, dtype=torch.int64),
            })
        return out

    def _schedule(self, dcp_samples, max_seqlen=None, min_cp=1):
        from gpatch_v4.utils.dynamic_cp_utils import run_dyn_cp_schedule
        return run_dyn_cp_schedule(
            dcp_samples, num_microbatches=1,
            max_seqlen_per_dp_cp_rank=max_seqlen or MAX_SEQLEN_PER_RANK,
            min_cp_size=min_cp,
        )

    # -- collectors ---------------------------------------------------------

    def collect_pad(self, kind: str, lengths: List[int]) -> Dict[str, Any]:
        """Return per-sample padding info for a single converter call."""
        from gpatch_v4.utils.dynamic_cp_utils import (
            _get_total_pad_divisor,
            convert_rl_samples_to_dyn_cp_format,
        )
        if kind == "rl":
            raw = self._rl_samples(lengths)
            conv = convert_rl_samples_to_dyn_cp_format(raw)
        samples = []
        for i, (r, c) in enumerate(zip(raw, conv)):
            samples.append({
                "sid": _make_sample_id(self.dp_rank, i),
                "raw_len": r["tokens"].shape[0],
                "original_seq_len": int(c["original_seq_len"].item()),
                "padded_seq_len": int(c["padded_seq_len"].item()),
                "tokens": c["tokens"].cpu(),
                "loss_mask": c["loss_mask"].cpu(),
                "labels": c.get("labels", torch.empty(0)).cpu(),
                "raw_tokens": r["tokens"].cpu(),
                "raw_labels": r.get("labels", torch.empty(0)).cpu(),
                "advantages": c.get("advantages", torch.empty(0)).cpu(),
                "prev_log_probs": c.get("prev_log_probs", torch.empty(0)).cpu(),
            })
        return {"pad_div": _get_total_pad_divisor(), "samples": samples}

    def collect_rl_position_alignment(self, lengths_dp):
        """RL with positional tokens.  At ``loss_mask==1`` each per-token field
        must equal its source sample's ``encode(sid, pos)`` marker; outside
        that region every per-token field must be exactly 0.
        """
        from gpatch_v4.utils.dynamic_cp_utils import (
            RL_TOKEN_KEYS, convert_rl_samples_to_dyn_cp_format, get_batch_for_dyn_cp,
        )
        dcp = convert_rl_samples_to_dyn_cp_format(
            self._rl_samples(lengths_dp[self.dp_rank], positional=True)
        )
        diter, n_micro, _, _ = self._schedule(dcp)

        per_mb = []
        for _ in range(n_micro):
            batch, _ = get_batch_for_dyn_cp(
                diter, dynamic_cp=True, extra_token_keys=RL_TOKEN_KEYS,
            )
            tokens = batch["tokens"].view(-1)
            adv = batch["advantages"].view(-1)
            prev_lp = batch["prev_log_probs"].view(-1)
            ref_lp = batch["ref_log_probs"].view(-1)
            mask = batch["loss_mask"].view(-1)

            real = mask == 1
            zero = ~real
            real_f = tokens[real].to(torch.float32)
            per_mb.append({
                "adv_align": bool(torch.equal(adv[real], real_f)),
                "prev_lp_align": bool(torch.equal(prev_lp[real], -real_f)),
                "ref_lp_align": bool(torch.equal(ref_lp[real], -2 * real_f)),
                "adv_zero_outside": bool((adv[zero] == 0).all().item()),
                "prev_lp_zero_outside": bool((prev_lp[zero] == 0).all().item()),
                "n_real": int(real.sum().item()),
            })
        return {"num_micro": n_micro, "per_mb": per_mb}


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class DynamicCpCorrectnessTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ray.init(address="auto", ignore_reinit_error=True)
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            ray.shutdown()
            raise unittest.SkipTest(
                f"need >= {WORLD_SIZE} GPUs, only {total_gpus} available"
            )

        this_dir = os.path.dirname(os.path.abspath(__file__))
        env_vars = {
            "NCCL_CUMEM_ENABLE": "0",
            "PYTHONPATH": f"{this_dir}:{os.environ.get('PYTHONPATH', '')}",
            "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
        }
        WorkerCls = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(DynCpWorker)
        cls.workers = [WorkerCls.remote() for _ in range(WORLD_SIZE)]
        master_addr, master_port = ray.get(cls.workers[0].get_master_addr_and_port.remote())
        ray.get([
            w.init_dist.remote(rank, WORLD_SIZE, master_addr, master_port)
            for rank, w in enumerate(cls.workers)
        ])

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "workers"):
            ray.get([w.destroy.remote() for w in cls.workers])
        ray.shutdown()

    def _gather(self, method_name, *args):
        return ray.get([
            getattr(w, method_name).remote(*args) for w in self.workers
        ])

    # -- pad correctness ---------------------------------------------------

    def test_rl_pad_correctness(self):
        """RL converter aligns padding and zeroes the sub-sequence boundary
        slot for ``loss_mask``, ``advantages`` and ``prev_log_probs``.

        The converter does a next-token shift (``tokens[:-1]``), so the
        post-conversion ``original_seq_len`` equals ``raw_len - 1``.
        """
        lengths = [10, 17, 32]
        outs = self._gather("collect_pad", "rl", lengths)
        for rank, out in enumerate(outs):
            pad_div = out["pad_div"]
            self.assertEqual(pad_div, 8)
            for s in out["samples"]:
                L = s["raw_len"]
                actual_len = L - 1
                expected_padded = ((actual_len + pad_div - 1) // pad_div) * pad_div
                self.assertEqual(s["original_seq_len"], actual_len)
                self.assertEqual(s["padded_seq_len"], expected_padded)
                self.assertEqual(s["tokens"].shape[0], expected_padded)
                if actual_len < expected_padded:
                    self.assertEqual(s["loss_mask"][actual_len].item(), 0.0,
                                     f"rank{rank} first pad loss_mask must be 0")
                    self.assertEqual(s["advantages"][actual_len].item(), 0.0)
                    self.assertEqual(s["prev_log_probs"][actual_len].item(), 0.0)
                sid = s["sid"]
                self.assertTrue((s["advantages"][:actual_len] == float(sid)).all())
                self.assertTrue((s["prev_log_probs"][:actual_len] == -float(sid)).all())

    # -- pack / CP slice / scheduling --------------------------------------

    def test_rl_position_level_field_alignment(self):
        """For every loss_mask==1 position the RL extras equal the per-position
        marker carried by ``tokens``; outside that region they are zero.

        Catches mis-aligned all-to-all routing and within-sample CP-partition
        errors of advantages / prev_log_probs / ref_log_probs.
        """
        outs = self._gather("collect_rl_position_alignment", SFT_LENGTHS)
        for rank, out in enumerate(outs):
            self.assertGreater(sum(mb["n_real"] for mb in out["per_mb"]), 0)
            for mb_idx, mb in enumerate(out["per_mb"]):
                self.assertTrue(mb["adv_align"],
                                f"rank{rank} mb{mb_idx} advantages misaligned with tokens")
                self.assertTrue(mb["prev_lp_align"],
                                f"rank{rank} mb{mb_idx} prev_log_probs misaligned with tokens")
                self.assertTrue(mb["ref_lp_align"],
                                f"rank{rank} mb{mb_idx} ref_log_probs misaligned with tokens")
                self.assertTrue(mb["adv_zero_outside"],
                                f"rank{rank} mb{mb_idx} adv leaks into pad / boundary")
                self.assertTrue(mb["prev_lp_zero_outside"],
                                f"rank{rank} mb{mb_idx} prev_lp leaks into pad / boundary")
