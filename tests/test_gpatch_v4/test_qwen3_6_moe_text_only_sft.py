"""Text-only mbridge load + forward + backward tests.

Two test suites:

**TestQwen36MoETextOnlyFwd** — Qwen3.6-35B-A3B (MoE, 8 GPUs)
    Loads checkpoint via ``AutoBridge.from_pretrained``, runs forward + backward
    with random 2k-token input, verifies logits and gradients are finite.

    - test_fwd_bwd:         TP=2, EP=4
    - test_fwd_bwd_with_cp: TP=2, CP=2, EP=2  (FLA CP path in gated_delta_net)

**TestHFvsMegatron** — Qwen3.5-0.8B (dense, 2–3 GPUs)
    Compares HF transformers ground truth against mbridge/Megatron output for the
    small dense model (has GDN with ``full_attention_interval=4``).

    - test_hf_vs_megatron:         TP=1, CP=1 (2 GPUs: 1 HF + 1 mcore)
    - test_hf_vs_megatron_with_cp: TP=1, CP=2 (3 GPUs: 1 HF + 2 mcore)

Manual-run only -- not in CI. Requires:
- Ray cluster with GPUs
- Model checkpoints downloaded into ``hf-hub/``

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=1800 \\
        tests/test_gpatch_v4/test_qwen3_6_moe_text_only_sft.py

Set ``QWEN36_FWD_SAVE_DIR`` to save logits and grad norms for offline comparison::

    QWEN36_FWD_SAVE_DIR=/tmp/gdn_check pytest -v -s ...
"""

import os
import socket
import unittest

import ray
import torch

from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

HF_MODEL_PATH = "hf-hub/Qwen/Qwen3.6-35B-A3B"
HF_MODEL_PATH_SMALL = "hf-hub/Qwen/Qwen3.5-0.8B"
NUM_GPUS = 8
SEQ_LEN = 2048


from qwen36_text_only_sft_workers import (
    _hf_fwd_bwd_worker,
    _logits_cos_sim,
    _mbridge_fwd_bwd_worker,
)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    torch.cuda.device_count() >= NUM_GPUS,
    f"needs {NUM_GPUS} GPUs, have {torch.cuda.device_count()}",
)
class TestQwen36MoETextOnlyFwd(unittest.TestCase):
    """Load Qwen3.6-35B-A3B via mbridge, forward + backward."""
    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"need >= {NUM_GPUS} GPUs in Ray cluster, only {total_gpus}")

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    SAVE_DIR = os.environ.get("QWEN36_FWD_SAVE_DIR", None)

    def _run(self, tp: int, ep: int, cp: int = 1, master_port: int = 12355,
             mcore_extra_config: dict = None):
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}; "
            f"download Qwen3.6-35B-A3B into hf-hub/ first"
        )

        world_size = NUM_GPUS
        master_addr = socket.gethostbyname(ray.util.get_node_ip_address())

        futures = [
            _mbridge_fwd_bwd_worker.remote(
                rank=r,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                hf_model_path=HF_MODEL_PATH,
                tp=tp,
                ep=ep,
                cp=cp,
                save_dir=self.SAVE_DIR,
                mcore_extra_config=mcore_extra_config,
            ) for r in range(world_size)
        ]
        results = ray.get(futures)

        # check forward
        logits_results = [r for r in results if "logits_shape" in r]
        assert len(logits_results) > 0, "no rank produced logits"
        for r in logits_results:
            assert r["logits_finite"], (f"rank {r['rank']}: logits contain NaN/Inf")
            assert len(r["logits_shape"]
                      ) == 3, (f"rank {r['rank']}: expected 3-d logits, got {r['logits_shape']}")

        # check backward
        for r in results:
            assert not r["has_nan"], f"rank {r['rank']}: grads contain NaN/Inf"
            assert not r["all_zero"], f"rank {r['rank']}: all grads are zero"
            assert r["num_grads"] > 0, f"rank {r['rank']}: no grads computed"
            print(
                f"  rank {r['rank']}: "
                f"logits_mean={r.get('logits_mean', 'N/A')} "
                f"total_grad_norm={r['total_grad_norm']:.4f} "
                f"num_grads={r['num_grads']}"
            )

    def test_fwd_bwd(self):
        """TP=2, EP=4, CP=1."""
        self._run(tp=2, ep=4)

    def test_fwd_bwd_with_cp(self):
        """TP=2, CP=2, EP=2 — exercises FLA context-parallel path in gated_delta_net."""
        self._run(tp=2, ep=2, cp=2, master_port=12356)


# ---------------------------------------------------------------------------
# HF vs Megatron comparison tests (Qwen3.5-0.8B dense)
# ---------------------------------------------------------------------------


class TestHFvsMegatron(unittest.TestCase):
    """Compare Qwen3.5-0.8B logits/grads: HF transformers vs mbridge/Megatron.

    Uses the small dense Qwen3.5-0.8B model (has GDN with
    ``full_attention_interval=4``) so both HF and Megatron sides can
    run on a small number of GPUs.
    """
    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < 2:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"need >= 2 GPUs in Ray cluster, only {total_gpus}")

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def _run_hf_vs_mcore(
        self,
        cp: int = 1,
        master_port: int = 12357,
        grad_norm_rtol: float = 0.05,
        mcore_extra_config: dict = None,
    ):
        assert os.path.isdir(HF_MODEL_PATH_SMALL), (
            f"model dir not found: {HF_MODEL_PATH_SMALL}; "
            f"download Qwen3.5-0.8B into hf-hub/ first"
        )

        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        world_size = cp
        gpus_needed = 1 + world_size
        assert total_gpus >= gpus_needed, (
            f"need >= {gpus_needed} GPUs (1 HF + {world_size} mcore), "
            f"only {total_gpus}"
        )

        master_addr = socket.gethostbyname(ray.util.get_node_ip_address())

        hf_future = _hf_fwd_bwd_worker.remote(HF_MODEL_PATH_SMALL)
        mcore_futures = [
            _mbridge_fwd_bwd_worker.remote(
                rank=r,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                hf_model_path=HF_MODEL_PATH_SMALL,
                tp=1,
                ep=1,
                cp=cp,
                return_logits=True,
                return_grad_norms=True,
                mcore_extra_config=mcore_extra_config,
            ) for r in range(world_size)
        ]

        hf_result = ray.get(hf_future)
        mcore_results = ray.get(mcore_futures)

        # -- validate HF result --
        assert hf_result["logits_finite"], "HF logits contain NaN/Inf"
        assert not hf_result["has_nan"], "HF grads contain NaN/Inf"
        print(
            f"  HF: logits.shape={hf_result['logits_shape']} "
            f"logits.mean={hf_result['logits_mean']:.4f} "
            f"total_grad_norm={hf_result['total_grad_norm']:.4f} "
            f"num_grads={hf_result['num_grads']}"
        )

        # -- validate mcore results --
        for r in mcore_results:
            assert not r["has_nan"], f"mcore rank {r['rank']}: grads contain NaN"
            assert not r["all_zero"], f"mcore rank {r['rank']}: all grads zero"

        # -- assert mcore_extra_config override propagated to bridge.config --
        if mcore_extra_config is not None and "cp_comm_type" in mcore_extra_config:
            expected = mcore_extra_config["cp_comm_type"]
            for r in mcore_results:
                self.assertEqual(
                    r["cp_comm_type"],
                    expected,
                    f"mcore rank {r['rank']}: mcore_extra_config['cp_comm_type'] "
                    f"override didn't propagate: expected {expected!r}, "
                    f"got {r['cp_comm_type']!r}",
                )
            print(
                f"  mcore_extra_config['cp_comm_type']={expected!r} "
                f"propagated to bridge.config on all {len(mcore_results)} ranks"
            )

        mcore_with_logits = [r for r in mcore_results if "logits" in r]
        assert len(mcore_with_logits) > 0, "no mcore worker produced logits"
        mcore_result = mcore_with_logits[0]
        print(
            f"  mcore: logits.shape={mcore_result['logits_shape']} "
            f"logits.mean={mcore_result['logits_mean']:.4f} "
            f"total_grad_norm={mcore_result['total_grad_norm']:.4f} "
            f"num_grads={mcore_result['num_grads']}"
        )

        # -- compare logits (cosine similarity) --
        hf_logits = hf_result["logits"]
        mcore_logits = mcore_result["logits"]
        vocab_size = min(hf_logits.shape[-1], mcore_logits.shape[-1])
        hf_logits = hf_logits[..., :vocab_size]
        mcore_logits = mcore_logits[..., :vocab_size]

        cos_sim = _logits_cos_sim(hf_logits, mcore_logits)
        mean_sim = cos_sim.mean().item()
        min_sim = cos_sim.min().item()
        print(f"  logits cos_sim: mean={mean_sim:.6f} min={min_sim:.6f}")
        self.assertGreater(
            mean_sim,
            0.99,
            f"logits cos_sim mean too low: {mean_sim:.6f}",
        )

        # -- compare total grad norm --
        hf_gnorm = hf_result["total_grad_norm"]
        mcore_gnorm = mcore_result["total_grad_norm"]
        denom = max(hf_gnorm, mcore_gnorm, 1e-8)
        rel_diff = abs(hf_gnorm - mcore_gnorm) / denom
        print(
            f"  total_grad_norm: HF={hf_gnorm:.4f} mcore={mcore_gnorm:.4f} "
            f"rel_diff={rel_diff:.4%}"
        )
        self.assertLess(
            rel_diff,
            grad_norm_rtol,
            f"total grad norm rel diff too large: {rel_diff:.4%} "
            f"(threshold={grad_norm_rtol:.0%})",
        )

    def test_hf_vs_megatron(self):
        """HF vs mbridge TP=1 — validates GDN and full model forward+backward.

        Also exercises ``mcore_extra_config`` plumbing: overrides
        ``cp_comm_type`` from the Qwen3.5 default ``'p2p'`` to ``'a2a'``
        and asserts the override actually reaches ``bridge.config``.
        At CP=1 the field is dead config, so numerics are unaffected.
        """
        self._run_hf_vs_mcore(
            cp=1,
            mcore_extra_config={"cp_comm_type": "a2a"},
        )

    def test_hf_vs_megatron_with_cp(self):
        """HF vs mbridge TP=1, CP=2 — validates FLA context-parallel GDN path."""
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < 3:
            self.skipTest(f"need >= 3 GPUs (1 HF + 2 mcore CP=2), only {total_gpus}")
        self._run_hf_vs_mcore(cp=2, master_port=12358)
