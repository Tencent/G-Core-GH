"""Module-level forward smoke test for ``GatedDeltaNet`` (Ray-based).

Mimics the per-rank parallel setup from
``Megatron-LM/tests/unit_tests/ssm/test_gated_delta_net.py::TestGatedDeltaNet``,
but launches each rank as a Ray actor (one GPU each) instead of relying on
``torchrun`` -- same process-control style as
``tests/test_gpatch_v4/test_qwen3_6_moe_text_only_sft.py``.

For each ``(tp, sp, cp)`` combination this test:

1. Spins up ``world_size = tp * cp`` Ray tasks, one GPU per task.
2. Each task initialises ``torch.distributed`` and Megatron model-parallel
   state, builds a single ``GatedDeltaNet`` layer with a small
   ``TransformerConfig``, and runs one forward pass on a constant input.
3. The driver collects per-rank shape/dtype/finite results and asserts
   each rank returned the expected sequence-sliced output.

The matrix mirrors the upstream Megatron unit test:

================  =====  =====  ==========
``tp``            ``sp``  ``cp``  GPUs needed
================  =====  =====  ==========
1                 False  1      1
2                 False  1      2
2                 True   1      2
1                 False  2      2
2                 False  2      4
2                 True   2      4
================  =====  =====  ==========

Manual-run only -- not in CI by default. Requires:

- Ray cluster with at least 4 GPUs
- ``flash-linear-attention`` (``fla``) installed

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=600 \\
        tests/test_gfused/test_gated_delta_net.py
"""

import importlib.util
import os
import socket
import unittest

import ray
import torch

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

_FLA_AVAILABLE = importlib.util.find_spec("fla") is not None
requires_fla = unittest.skipUnless(_FLA_AVAILABLE, "flash-linear-attention (`fla`) not installed")

# Maximum world_size used by any test method below (tp=2, cp=2).
NUM_GPUS_NEEDED = 4

# Dimensions of the smoke-test input. Match the upstream Megatron unit
# test ``TestGatedDeltaNet.test_gpu_forward`` so the GDN config we build
# below stays valid across all (tp, sp, cp) combinations:
#   - SEQ_LENGTH must be divisible by tp*cp*2 worst case.
#   - HIDDEN_SIZE divisible by tp.
SEQ_LENGTH = 64
MICRO_BATCH_SIZE = 2
HIDDEN_SIZE = 256


@ray.remote(num_gpus=1)
def _gdn_smoke_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    tp: int,
    sp: bool,
    cp: int,
    seq_length: int = SEQ_LENGTH,
    micro_batch_size: int = MICRO_BATCH_SIZE,
    hidden_size: int = HIDDEN_SIZE,
):
    """Single-rank worker: build a ``GatedDeltaNet`` and run one forward.

    Parameters mirror the constructor args of
    ``TestGatedDeltaNet.setup_method`` in the upstream Megatron unit test.

    Returns
    -------
    dict
        ``{rank, output_shape, output_dtype, output_finite, expected_seq_len}``
        for the driver to assert on.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    torch.cuda.set_device(0)

    # NB: import inside the worker so the driver process does not need
    # the heavy Megatron / FLA stack (and to defer init_process_group
    # related state until the GPU is actually claimed by Ray).
    import torch.nn.functional as F
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    torch.distributed.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp,
    )
    model_parallel_cuda_manual_seed(123)

    sp_size = tp if sp else 1

    pg_collection = ProcessGroupCollection(
        tp=mpu.get_tensor_model_parallel_group(),
        cp=mpu.get_context_parallel_group(),
    )
    transformer_config = TransformerConfig(
        hidden_size=hidden_size,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=True,
        tensor_model_parallel_size=tp,
        sequence_parallel=sp,
        context_parallel_size=cp,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )
    gdn_submodules = get_experimental_attention_variant_module_spec(
        config=transformer_config,
    ).submodules
    gdn = GatedDeltaNet(
        transformer_config,
        submodules=gdn_submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
    ).cuda().bfloat16()

    local_seq = seq_length // sp_size // cp
    hidden_states = torch.ones(
        (local_seq, micro_batch_size, hidden_size),
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16,
    )
    output, _bias = gdn(hidden_states, attention_mask=None)

    result = {
        "rank": rank,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_finite": bool(torch.isfinite(output).all()),
        "expected_seq_len": local_seq,
    }
    print(
        f"[rank {rank}] tp={tp} sp={sp} cp={cp} "
        f"output.shape={result['output_shape']} "
        f"finite={result['output_finite']}"
    )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return result


@requires_fla
@unittest.skipUnless(
    torch.cuda.device_count() >= NUM_GPUS_NEEDED,
    f"needs {NUM_GPUS_NEEDED} GPUs, have {torch.cuda.device_count()}",
)
class TestGatedDeltaNetSmoke(unittest.TestCase):
    """Module-level (TP/SP/CP) forward smoke test for ``GatedDeltaNet``."""
    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS_NEEDED:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {NUM_GPUS_NEEDED} GPUs in Ray cluster, only {total_gpus}"
            )

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def _run(self, tp: int, sp: bool, cp: int, master_port: int) -> None:
        """Fan out ``tp * cp`` Ray workers and validate per-rank output."""
        world_size = tp * cp
        master_addr = socket.gethostbyname(ray.util.get_node_ip_address())

        futures = [
            _gdn_smoke_worker.remote(
                rank=r,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                tp=tp,
                sp=sp,
                cp=cp,
            ) for r in range(world_size)
        ]
        results = ray.get(futures)

        sp_size = tp if sp else 1
        expected_seq = SEQ_LENGTH // sp_size // cp

        self.assertEqual(
            len(results),
            world_size,
            f"expected {world_size} results, got {len(results)}",
        )
        for r in results:
            tag = f"rank {r['rank']} (tp={tp}, sp={sp}, cp={cp})"
            self.assertEqual(
                r["output_shape"][0],
                expected_seq,
                f"{tag}: seq dim {r['output_shape'][0]} != "
                f"{SEQ_LENGTH} // {sp_size} // {cp} = {expected_seq}",
            )
            self.assertEqual(
                r["output_shape"][1],
                MICRO_BATCH_SIZE,
                f"{tag}: batch dim {r['output_shape'][1]} != {MICRO_BATCH_SIZE}",
            )
            self.assertEqual(
                r["output_shape"][2],
                HIDDEN_SIZE,
                f"{tag}: hidden dim {r['output_shape'][2]} != {HIDDEN_SIZE}",
            )
            self.assertEqual(
                r["output_dtype"],
                "torch.bfloat16",
                f"{tag}: output dtype {r['output_dtype']} != torch.bfloat16",
            )
            self.assertTrue(
                r["output_finite"],
                f"{tag}: output contains NaN/Inf",
            )

    # -- TP-only --
    def test_tp1_cp1(self):
        """Baseline: TP=1, SP=off, CP=1 (1 GPU)."""
        self._run(tp=1, sp=False, cp=1, master_port=12361)

    def test_tp2_cp1(self):
        """TP=2 without SP (2 GPUs)."""
        self._run(tp=2, sp=False, cp=1, master_port=12362)

    def test_tp2_sp_cp1(self):
        """TP=2 with sequence parallel (2 GPUs)."""
        self._run(tp=2, sp=True, cp=1, master_port=12363)

    # -- CP-only --
    def test_tp1_cp2(self):
        """CP=2 (2 GPUs) -- exercises FLA context-parallel GDN path."""
        self._run(tp=1, sp=False, cp=2, master_port=12364)

    # -- TP + CP --
    def test_tp2_cp2(self):
        """TP=2 + CP=2 without SP (4 GPUs)."""
        self._run(tp=2, sp=False, cp=2, master_port=12365)

    def test_tp2_sp_cp2(self):
        """TP=2 + SP + CP=2 (4 GPUs) -- exercises full TP/SP/CP combo."""
        self._run(tp=2, sp=True, cp=2, master_port=12366)


if __name__ == "__main__":
    unittest.main()
