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
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoTokenizer

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None

from mbridge import AutoBridge
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

HF_MODEL_PATH = "hf-hub/Qwen/Qwen3.6-35B-A3B"
HF_MODEL_PATH_SMALL = "hf-hub/Qwen/Qwen3.5-0.8B"
NUM_GPUS = 8
SEQ_LEN = 2048


def _gather_from_cp(tensor, seq_dim, cp_size, cp_group):
    """Gather zigzag-split CP chunks back to full sequence order."""
    assert seq_dim in (0, 1) and tensor.dim() > seq_dim
    tensor = tensor.view(
        *tensor.shape[:seq_dim],
        2,
        tensor.shape[seq_dim] // 2,
        *tensor.shape[seq_dim + 1:],
    )
    gathered = [torch.zeros_like(tensor) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, tensor, group=cp_group)
    reordered = [None] * (2 * cp_size)
    for r in range(cp_size):
        if seq_dim == 1:
            reordered[r] = gathered[r][:, 0]
            reordered[2 * cp_size - r - 1] = gathered[r][:, 1]
        else:
            reordered[r] = gathered[r][0]
            reordered[2 * cp_size - r - 1] = gathered[r][1]
    return torch.cat(reordered, dim=seq_dim)


# ---------------------------------------------------------------------------
# Logits comparison helper
# ---------------------------------------------------------------------------


def _logits_cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-position cosine similarity with exp (softmax-like) normalization.

    Parameters
    ----------
    a, b : torch.Tensor
        Logit tensors of shape ``(batch, seq_len, vocab_size)``.

    Returns
    -------
    torch.Tensor
        Per-position cosine similarity of shape ``(batch, seq_len)``.
    """
    a, b = a.float(), b.float()
    a = torch.exp(a - a.max(dim=-1, keepdim=True)[0])
    b = torch.exp(b - b.max(dim=-1, keepdim=True)[0])
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a * b).sum(dim=-1)


# ---------------------------------------------------------------------------
# HF baseline worker (single GPU, no distributed)
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
def _hf_fwd_bwd_worker(hf_model_path: str, seq_len: int = SEQ_LEN):
    """Single-GPU HF baseline: forward + backward, return logits + grad norms."""
    assert Qwen3_5ForConditionalGeneration is not None, (
        "Qwen3_5ForConditionalGeneration not found; need transformers >= 5.2.0"
    )

    torch.cuda.set_device(0)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    rng = torch.Generator().manual_seed(42)
    input_ids = torch.randint(
        0,
        tokenizer.vocab_size,
        (1, seq_len),
        generator=rng,
    ).cuda()

    print(f"[HF] loading {hf_model_path} ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    model.train()
    print(f"[HF] model loaded, running forward + backward ...")

    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    loss = logits.sum()
    loss.backward()
    logits = logits.detach()

    grad_norms = {}
    total_sq = 0.0
    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            gn = param.grad.float().norm().item()
            grad_norms[name] = gn
            total_sq += gn**2
            if not torch.isfinite(param.grad).all():
                has_nan = True
    total_grad_norm = total_sq**0.5

    print(
        f"[HF] logits.shape={list(logits.shape)} "
        f"logits.mean={logits.float().mean().item():.4f} "
        f"total_grad_norm={total_grad_norm:.4f} num_grads={len(grad_norms)} "
        f"has_nan={has_nan}"
    )

    return {
        "logits": logits.cpu(),
        "grad_norms": grad_norms,
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "logits_mean": logits.float().mean().item(),
        "total_grad_norm": total_grad_norm,
        "num_grads": len(grad_norms),
        "has_nan": has_nan,
    }


# ---------------------------------------------------------------------------
# Megatron/mbridge worker
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
def _mbridge_fwd_bwd_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    hf_model_path: str,
    tp: int,
    ep: int,
    cp: int = 1,
    save_dir: str = None,
    return_logits: bool = False,
    return_grad_norms: bool = False,
    mcore_extra_config: dict = None,
):
    """Single-GPU worker: load model via mbridge, forward + backward."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    torch.cuda.set_device(0)

    torch.distributed.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
    )
    model_parallel_cuda_manual_seed(0)

    print(f"[rank {rank}] loading model (tp={tp}, cp={cp}, ep={ep}) ...")
    bridge = AutoBridge.from_pretrained(
        hf_model_path,
        trust_remote_code=True,
        mcore_extra_config=mcore_extra_config,
    )
    bridge.config.sequence_parallel = tp > 1
    model = bridge.get_model()
    if rank == 0:
        print(f"[rank {rank}] model len={len(model)} config={bridge.config}")
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"[rank {rank}] weights loaded, preparing fwd+bwd ...")
    torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    rng = torch.Generator().manual_seed(42)
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, SEQ_LEN), generator=rng)
    micro_batch_size, real_seq_length = input_ids.size()

    seq_length_factor = tp
    if cp > 1:
        seq_length_factor *= cp * 2
    seq_length = real_seq_length
    if real_seq_length % seq_length_factor != 0:
        seq_length = (
            (real_seq_length + seq_length_factor - 1) // seq_length_factor * seq_length_factor
        )
        input_ids = F.pad(input_ids, (0, seq_length - real_seq_length), value=0)

    sample_list = [{"input_ids": input_ids}]
    logits_container = []

    def fwd_fn(data_iter, model):
        sample = next(data_iter)
        output_tensor = model(
            input_ids=sample["input_ids"].cuda(),
            position_ids=None,
            attention_mask=None,
        )
        if isinstance(output_tensor, tuple):
            output_tensor = output_tensor[0]
        assert isinstance(output_tensor, torch.Tensor)

        def loss_func(output_tensor, non_loss_data=True):
            logits_container.append(output_tensor.detach())
            # Megatron's ``forward_backward_func`` multiplies the returned
            # loss by ``cp_group_size`` before backward (see
            # ``Megatron-LM/megatron/core/pipeline_parallel/schedules.py``,
            # ``compute_forward_loss_and_data``: ``output_tensor *=
            # cp_group_size``) to make per-rank shard losses look like the
            # full-sequence loss for DDP-AVG averaging. This test does its
            # own SUM all-reduce of ``param.grad`` across the CP group
            # below, so the cp_size multiply has to be cancelled here, or
            # the resulting grad-norm comes out as ``cp_size`` times HF.
            cp_size = mpu.get_context_parallel_world_size()
            loss = output_tensor.sum() / cp_size
            return loss, {"loss": loss.detach()}

        return output_tensor, loss_func

    fwd_bwd_function = get_forward_backward_func()
    fwd_bwd_function(
        forward_step_func=fwd_fn,
        data_iterator=iter(sample_list),
        model=model,
        num_microbatches=1,
        forward_only=False,
        seq_length=seq_length,
        decoder_seq_length=seq_length,
        micro_batch_size=micro_batch_size,
    )

    # -- collect logits (on last pipeline stage) --
    fwd_result = {}
    if mpu.is_pipeline_last_stage() and logits_container:
        logits = logits_container[0]
        if mpu.get_context_parallel_world_size() > 1:
            logits = _gather_from_cp(
                logits,
                1,
                mpu.get_context_parallel_world_size(),
                mpu.get_context_parallel_group(),
            )
        if mpu.get_tensor_model_parallel_world_size() > 1:
            logits = gather_from_tensor_model_parallel_region(logits)
        logits = logits[:, :real_seq_length, :]

        fwd_result = {
            "logits_shape": list(logits.shape),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "logits_mean": logits.float().mean().item(),
        }
        print(
            f"[rank {rank}] fwd OK  logits.shape={fwd_result['logits_shape']} "
            f"logits.mean={fwd_result['logits_mean']:.4f}"
        )
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"logits_tp{tp}_cp{cp}_ep{ep}.pt")
            torch.save(logits.cpu(), path)
            print(f"[rank {rank}] saved logits to {path}")
        if return_logits:
            fwd_result["logits"] = logits.cpu()

    # -- reduce gradients across CP (mimics finalize_model_grads) --
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        cp_group = mpu.get_context_parallel_group()
        for param in model[0].parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(
                    param.grad,
                    op=torch.distributed.ReduceOp.SUM,
                    group=cp_group,
                )

    # -- collect per-param grad norms --
    grad_norms = {}
    grad_norm_sq = torch.zeros(1, dtype=torch.float32, device="cuda")
    has_nan = False
    has_zero = True
    for name, param in model[0].named_parameters():
        if param.grad is not None:
            gn = param.grad.float().norm().item()
            grad_norms[name] = gn
            grad_norm_sq += gn**2
            if not torch.isfinite(param.grad).all():
                has_nan = True
            if gn > 0:
                has_zero = False

    # -- reduce norm² across model-parallel group (TP × PP) --
    model_parallel_group = mpu.get_model_parallel_group()
    torch.distributed.all_reduce(
        grad_norm_sq,
        op=torch.distributed.ReduceOp.SUM,
        group=model_parallel_group,
    )
    total_grad_norm = grad_norm_sq.item()**0.5

    bwd_result = {
        "total_grad_norm": total_grad_norm,
        "num_grads": len(grad_norms),
        "has_nan": has_nan,
        "all_zero": has_zero,
    }
    print(
        f"[rank {rank}] bwd OK  total_grad_norm={total_grad_norm:.4f} "
        f"num_grads={len(grad_norms)} has_nan={has_nan} all_zero={has_zero}"
    )
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"grad_norms_tp{tp}_cp{cp}_ep{ep}_rank{rank}.pt")
        torch.save(grad_norms, path)
        print(f"[rank {rank}] saved grad_norms to {path}")
    if return_grad_norms:
        bwd_result["grad_norms"] = grad_norms

    config_snapshot = {
        "cp_comm_type": bridge.config.cp_comm_type,
    }

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return {"rank": rank, **fwd_result, **bwd_result, **config_snapshot}


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
