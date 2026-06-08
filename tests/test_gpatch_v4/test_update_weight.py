"""End-to-end unit tests for the sampler weight-update path.

Covers the two orthogonal axes that flow through
``SamplerClient.update_weights`` in
``gpatch_v4/client/sampler_client.py``:

============    ==================================    ==================
 placement       sampler backend                       transport
============    ==================================    ==================
 colocate        sglang                                IPC (flattened bucket)
 colocate        vllm                                  IPC (``reduce_tensor``)
 disaggregated   sglang                                NCCL bucket broadcast
 disaggregated   vllm                                  ``NCCLWeightTransferEngine``
============    ==================================    ==================

Rather than re-implementing a bespoke weight-diffing harness, we drive the
production ``debug.debug_engine_update_weight`` pipeline (see
``gpatch_v4/actor/mixin.py::_debug_update_weight_stage1/2``), which already:

1. Saves the trainer's source weights       -> ``src_weights_{dp_rank}/``.
2. Pushes an all-zero overlay to the sampler and snapshots what the sampler
   sees                                     -> ``zero_weights_{dp_rank}/``.
3. Pushes the real weights again and snapshots what the sampler sees
   -> ``real_weights_{dp_rank}/``.

The test then defers the numeric comparison to the pure-function helpers in
``tools/check_sglang_ckpt/check_updated_ckpt.py`` (same script users run by
hand), enforcing:

* ``src_weights == real_weights``  (round-trip fidelity of the real update).
* ``zero_weights`` is all-zero for every non-ignored tensor (the
  ``_k_scale`` / ``_v_scale`` / ``_q_scale`` / ``_prob_scale`` buffers are
  vLLM-side runtime fp8 scales that the trainer does not send and so cannot
  be zeroed out; they are explicitly ignored by the helper).
"""

import importlib.util
import os
import shutil
import unittest
import uuid

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.trainer import GrpoSingleCtrlTrainer, GrpoTrainer
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
    requires_vllm,
)


def _load_check_helpers():
    """Import the pure helpers from ``tools/check_sglang_ckpt/check_updated_ckpt.py``.

    Using ``importlib`` avoids executing the script's ``__main__`` block
    (which would call ``argparse.parse_args()`` against ``sys.argv`` and
    fail under pytest) while still giving us its battle-tested diff and
    zero-check routines without duplicating them.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(
        os.path.join(here, "..", "..", "tools", "check_sglang_ckpt", "check_updated_ckpt.py")
    )
    spec = importlib.util.spec_from_file_location("_check_updated_ckpt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class UpdateWeightTest(unittest.IsolatedAsyncioTestCase):
    """Drive ``debug_update_weight`` and verify the three snapshot dirs."""

    # Parent dir that holds one ``<uuid>`` subdir per test run.  Shared
    # across cases so we can sweep stale leftovers (from SIGKILL'd runs
    # whose tearDown never fired) in one shot.
    _SAVE_ROOT = os.path.abspath(os.path.join("tests", "test_gpatch_v4", "_tmp_update_weight"))

    def setUp(self):
        # Snapshot dirs are multi-GB per run; if a previous invocation
        # was SIGKILL'd (e.g. the CUDA-graph deadlock we hit on vllm),
        # its uuid subdir stays around forever. Sweep the whole parent
        # here so disk doesn't balloon across reruns.
        shutil.rmtree(self._SAVE_ROOT, ignore_errors=True)
        self._save_path = os.path.join(self._SAVE_ROOT, uuid.uuid4().hex)
        os.makedirs(self._save_path, exist_ok=True)
        self._ck = _load_check_helpers()

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()
        shutil.rmtree(self._save_path, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # config factory
    # ------------------------------------------------------------------ #
    def _make_config(self, config_name, backend):
        """Load a reference yaml, flip the debug harness on, and redirect
        the snapshot path to the test-owned tmpdir.

        We also force ``disable_cuda_graph=True`` on every sampler infer
        engine: CUDA graphs are irrelevant for the weight-update pipeline
        and vLLM's CUDA-graph capture (torch.compile + flashinfer
        allreduce fusion) deadlocks intermittently during the initial
        ``torch.cuda.synchronize()`` with TP>1 + Qwen3 on our setup.
        """
        config = load_config(config_name, RlConfig)
        config.sampler.backend = backend
        config.gen_rm.backend = backend
        config.debug.debug_engine_update_weight = True
        config.debug.debug_engine_save_path = self._save_path
        for engine_cfg in config.sampler.infer_engine_configs:
            engine_cfg.disable_cuda_graph = True
        return config

    # ------------------------------------------------------------------ #
    # snapshot verification (pure)
    # ------------------------------------------------------------------ #
    def _discover_grid(self, kind):
        """Return ``(tp_size, [dp_rank, ...])`` by scanning
        ``{save_path}/{kind}_weights_*``.

        Doing this at assert-time (rather than hard-coding tp/dp from the
        yaml) keeps the test resilient to yaml changes and auto-adapts to
        the sampler's configured parallelism.
        """
        dp_ranks = sorted(
            int(d.rsplit("_", 1)[-1]) for d in os.listdir(self._save_path)
            if d.startswith(f"{kind}_weights_") and d.rsplit("_", 1)[-1].isdigit()
        )
        assert dp_ranks, f"no {kind}_weights_* dirs under {self._save_path}"

        first = os.path.join(self._save_path, f"{kind}_weights_{dp_ranks[0]}")
        tp_size = sum(
            1 for f in os.listdir(first)
            if f.startswith("model-rank-") and f.endswith(".safetensors")
        )
        assert tp_size > 0, f"no safetensors shards under {first}"
        return tp_size, dp_ranks

    def _verify_snapshots(self):
        """Run the same checks as ``check_updated_ckpt.py`` programmatically.

        * ``src`` vs ``real``: must be bit-identical (or ``allclose`` for
          floats). Any mismatch means a weight was lost or corrupted on
          its way from the trainer to the sampler.
        * ``zero``: every non-ignored tensor must be all-zero, proving that
          the zero-replacement actually reached the sampler rather than
          silently being dropped.

        We only verify ``dp_rank=0``: every sampler DP rank observes the
        same global weight broadcast, so comparing one is sufficient to
        exercise the update path and keeps the CPU/IO-bound safetensors
        load out of the pytest timeout budget.
        """
        tp_size, dp_ranks = self._discover_grid("real")
        self.assertGreater(tp_size, 0)
        self.assertIn(0, dp_ranks, f"expected dp_rank=0 snapshot, got {dp_ranks}")

        dp = 0
        real_files = self._ck.get_tp_files(
            tp_size, os.path.join(self._save_path, f"real_weights_{dp}")
        )
        src_files = self._ck.get_tp_files(
            tp_size, os.path.join(self._save_path, f"src_weights_{dp}")
        )
        zero_files = self._ck.get_tp_files(
            tp_size, os.path.join(self._save_path, f"zero_weights_{dp}")
        )

        for rank in range(tp_size):
            src_w = self._ck.load_safetensors_file(src_files[rank])
            real_w = self._ck.load_safetensors_file(real_files[rank])
            is_equal, unequal = self._ck.compare_weights_dict(
                src_w, real_w, (src_files[rank], real_files[rank])
            )
            self.assertTrue(
                is_equal,
                f"dp={dp} tp={rank}: src != real after real-weight update; "
                f"first mismatches={unequal[:5]}",
            )

        for rank in range(tp_size):
            zero_w = self._ck.load_safetensors_file(zero_files[rank])
            is_zero, non_zero = self._ck.check_zero_weights(zero_w, zero_files[rank])
            self.assertTrue(
                is_zero,
                f"dp={dp} tp={rank}: zero_weights should be all-zero; "
                f"first non-zero={non_zero[:5]}",
            )

    async def _run_debug(self, config, trainer_cls):
        trainer = trainer_cls()
        await trainer.debug_update_weight(config)

    # ------------------------------------------------------------------ #
    # colocate (IPC handle / reduce_tensor)
    # ------------------------------------------------------------------ #
    @requires_sglang
    async def test_colocate_sglang(self):
        """Colocate + sglang -> flattened IPC bucket path in
        ``UpdateWeightIpcMixin._update_weights_by_ipc_handle_sglang``."""
        config = self._make_config("test_update_weight_colocate", "sglang")
        assert config.placement_type != "disaggregated"
        await self._run_debug(config, GrpoTrainer)
        self._verify_snapshots()

    @requires_vllm
    async def test_colocate_vllm(self):
        """Colocate + vllm -> ``IPCWeightTransferEngine`` /
        ``reduce_tensor`` path in
        ``UpdateWeightIpcMixin._update_weights_by_bucketed_ipc_vllm``."""
        config = self._make_config("test_update_weight_colocate", "vllm")
        assert config.placement_type != "disaggregated"
        await self._run_debug(config, GrpoTrainer)
        self._verify_snapshots()

    # ------------------------------------------------------------------ #
    # disaggregated (NCCL broadcast)
    # ------------------------------------------------------------------ #
    @requires_sglang
    async def test_disaggregated_sglang(self):
        """Disaggregated + sglang -> ``_update_weights_by_distributed_sglang``
        (flattened bucket broadcast over the trainer<->sampler NCCL group)."""
        config = self._make_config("test_math_rl_disaggregated", "sglang")
        assert config.placement_type == "disaggregated"
        await self._run_debug(config, GrpoSingleCtrlTrainer)
        self._verify_snapshots()

    @requires_vllm
    async def test_disaggregated_vllm(self):
        """Disaggregated + vllm -> ``_update_weights_by_distributed_vllm``
        (two-pass bucketed NCCL broadcast + single
        ``gcore_finalize_weights_update`` RPC)."""
        config = self._make_config("test_math_rl_disaggregated", "vllm")
        assert config.placement_type == "disaggregated"
        await self._run_debug(config, GrpoSingleCtrlTrainer)
        self._verify_snapshots()
