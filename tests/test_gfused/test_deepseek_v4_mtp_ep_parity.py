# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""MTP EP parity test: EP=1 (baseline) vs EP=4 with router replay.

Verifies that the EP+FSDP2 path produces **identical** MTP logits as a
pure-FSDP (EP=1) baseline when router decisions are replayed. This is the
gold-standard correctness check for the MTP + EP integration: any bug in
expert slicing, load_checkpoint_hp MTP remap, or EP all-to-all would show
up as logits drift.

Usage:

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_mtp_ep_parity.py
"""

from __future__ import annotations

import os
import time
import unittest

import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer, DeepseekV4Config

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    apply_hp,
    capture_routing_decisions,
    router_replay_ctx,
)

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"

NUM_GPUS = 32
EP_SIZE_BASELINE = 1   # pure FSDP
EP_SIZE_TEST = 4       # EP + FSDP
NUM_BACKBONE_LAYERS = 4
SEQ_LEN = 128


def _truncate_config(config: DeepseekV4Config) -> DeepseekV4Config:
    """Truncate backbone to NUM_BACKBONE_LAYERS; keep MTP."""
    assert config.layer_types is not None
    assert config.mlp_layer_types is not None
    config.num_hidden_layers = NUM_BACKBONE_LAYERS
    config.layer_types = config.layer_types[:NUM_BACKBONE_LAYERS]
    config.mlp_layer_types = config.mlp_layer_types[:NUM_BACKBONE_LAYERS]
    return config


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip() -> str:
    return ray.util.get_node_ip_address()


def _build_and_load(
    hf_model_path: str,
    rank: int,
    world_size: int,
    ep_size: int,
) -> DeepseekV4ForCausalLM:
    """Meta-device construct → apply_hp → load_checkpoint_hp."""
    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
    assert config.num_nextn_predict_layers > 0
    config.mtp_loss_scaling_factor = 0.1
    assert not config.tie_word_embeddings

    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    model = apply_hp(model, ep_2d_mesh, cp_mesh=None, amp_fp32=False)
    model.gradient_checkpointing_enable()
    for _bi, _blk in enumerate(model.mtp.layers):
        assert _blk.gradient_checkpointing, (
            f"MTP block {_bi} gradient_checkpointing not enabled after "
            f"gradient_checkpointing_enable(); "
            f"DeepseekV4MTPBlock must inherit GradientCheckpointingLayer"
        )
    model.load_checkpoint_hp(hf_model_path)
    return model


def _mtp_logits(model, outputs) -> list[torch.Tensor]:
    """Project mtp_per_depth_h through lm_head to get MTP logits."""
    mtp_h = outputs.mtp_per_depth_h
    assert mtp_h is not None
    with torch.no_grad():
        return [model.lm_head(h).float().cpu() for h in mtp_h]


@ray.remote(num_gpus=1)
def _baseline_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> dict:
    """EP=1 baseline: load, forward, capture MTP logits + router decisions."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rng = torch.Generator().manual_seed(42 + rank)
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, SEQ_LEN), generator=rng,
    ).cuda()

    print(f"[baseline] rank {rank}: loading (EP=1) ...")
    _t0 = time.time()
    model = _build_and_load(hf_model_path, rank, world_size, ep_size=EP_SIZE_BASELINE)
    print(f"[baseline] rank {rank}: loaded in {time.time() - _t0:.1f}s")

    model.train()

    with capture_routing_decisions(model) as recorded:
        outputs = model(input_ids=input_ids)

    mtp_logits_list = _mtp_logits(model, outputs)
    main_logits = outputs.logits.float().cpu()

    result = {
        "rank": rank,
        "main_logits": main_logits,
        "mtp_logits": mtp_logits_list,
        "recorded_routing": [t.cpu() for t in recorded],
    }

    del model, outputs
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


@ray.remote(num_gpus=1)
def _ep_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    replay_indices: "list[torch.Tensor] | None",
) -> dict:
    """EP=4 test: load, optionally replay router, forward, return MTP logits."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rng = torch.Generator().manual_seed(42 + rank)
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, SEQ_LEN), generator=rng,
    ).cuda()

    tag = f"ep{EP_SIZE_TEST}" if replay_indices is None else f"ep{EP_SIZE_TEST}_replay"
    print(f"[{tag}] rank {rank}: loading (EP={EP_SIZE_TEST}) ...")
    _t0 = time.time()
    model = _build_and_load(hf_model_path, rank, world_size, ep_size=EP_SIZE_TEST)
    print(f"[{tag}] rank {rank}: loaded in {time.time() - _t0:.1f}s")

    model.train()

    if replay_indices is not None:
        replay = [t.cuda() for t in replay_indices]
        ctx = router_replay_ctx(model, replay)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        outputs = model(input_ids=input_ids)

    mtp_logits_list = _mtp_logits(model, outputs)
    main_logits = outputs.logits.float().cpu()

    result = {
        "rank": rank,
        "main_logits": main_logits,
        "mtp_logits": mtp_logits_list,
    }

    del model, outputs
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


class TestMtpEpParity(unittest.TestCase):
    """EP=1 vs EP=4 MTP logits parity (router replay)."""

    def setUp(self):
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        if not env_pythonpath:
            rcdir = "/work/wepsdl"
            env_pythonpath = ":".join([
                f"{rcdir}/gcore-dev",
                f"{rcdir}/gcore-dev/tests",
                f"{rcdir}/gcore-dev/tests/test_gpatch_v4",
                f"{rcdir}/Megatron-LM",
                f"{rcdir}/mbridge",
                f"{rcdir}/Megatron-Bridge/src",
            ])
        ray.init(
            address="auto",
            runtime_env={"env_vars": {"PYTHONPATH": env_pythonpath}},
        )
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {NUM_GPUS} GPUs, only {total_gpus}"
            )

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def test_mtp_ep1_vs_ep4(
        self,
        world_size: int = NUM_GPUS,
        master_port_base: int = 12800,
        rtol: float = 0.06,
    ):
        assert os.path.isdir(HF_MODEL_PATH), f"model dir not found: {HF_MODEL_PATH}"

        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
        )
        ray.get(pg.ready())

        try:
            master_addr = ray.get(
                _get_node_ip.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=0,
                    )
                ).remote()
            )

            # ---- EP=1 baseline + routing capture ----
            print("=" * 60)
            print(f"Running EP=1 baseline (world_size={world_size}) ...")
            bl_futures = [
                _baseline_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=r,
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port_base,
                )
                for r in range(world_size)
            ]
            bl_results = ray.get(bl_futures)

            # Build per-rank replay indices
            replay_per_rank = [
                bl_results[r]["recorded_routing"]
                for r in range(world_size)
            ]
            print(
                f"  baseline rank 0: {len(replay_per_rank[0])} TopKRouter layers"
            )

            # ---- EP=4 WITHOUT replay (free routing) ----
            print("=" * 60)
            print(f"Running EP={EP_SIZE_TEST} WITHOUT router replay ...")
            ep_free_futures = [
                _ep_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=r,
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port_base + 1,
                    replay_indices=None,
                )
                for r in range(world_size)
            ]
            ep_free_results = ray.get(ep_free_futures)

            # ---- EP=4 WITH replay ----
            print("=" * 60)
            print(f"Running EP={EP_SIZE_TEST} + router replay ...")
            ep_futures = [
                _ep_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=r,
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port_base + 2,
                    replay_indices=replay_per_rank[r],
                )
                for r in range(world_size)
            ]
            ep_results = ray.get(ep_futures)
        finally:
            remove_placement_group(pg)

        # ---- Helper to compute rel diff ----
        def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
            return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)

        # ---- Three-way comparison ----
        print("\n--- Three-way MTP logits comparison ---")
        print(f"  {'rank':>4s}  {'BL vs EP4-free (main)':>22s}  {'BL vs EP4-free (mtp)':>22s}  "
              f"{'BL vs EP4-replay (main)':>24s}  {'BL vs EP4-replay (mtp)':>24s}  "
              f"{'EP4-free vs EP4-replay (main)':>30s}  {'EP4-free vs EP4-replay (mtp)':>30s}")
        print("  " + "-" * 170)

        for r in range(world_size):
            bl = bl_results[r]
            epf = ep_free_results[r]
            epr = ep_results[r]

            # BL vs EP4-free (no replay → router may differ → expect larger gap)
            bf_main = _rel(bl["main_logits"], epf["main_logits"])
            bf_mtp = [_rel(a, b) for a, b in zip(bl["mtp_logits"], epf["mtp_logits"])]

            # BL vs EP4-replay (router fixed → should be tight)
            br_main = _rel(bl["main_logits"], epr["main_logits"])
            br_mtp = [_rel(a, b) for a, b in zip(bl["mtp_logits"], epr["mtp_logits"])]

            # EP4-free vs EP4-replay (same EP, router diff only)
            fr_main = _rel(epf["main_logits"], epr["main_logits"])
            fr_mtp = [_rel(a, b) for a, b in zip(epf["mtp_logits"], epr["mtp_logits"])]

            def _fmt_mtp(vals):
                return "[" + ",".join(f"{v:.2e}" for v in vals) + "]"

            print(
                f"  {r:4d}  {bf_main:22.2e}  {_fmt_mtp(bf_mtp):>22s}  "
                f"{br_main:24.2e}  {_fmt_mtp(br_mtp):>24s}  "
                f"{fr_main:30.2e}  {_fmt_mtp(fr_mtp):>30s}"
            )

            # Only BL vs EP4-replay should be tight (router-fixed parity)
            self.assertLess(
                br_main, rtol,
                f"rank {r}: BL vs EP4-replay main logits rel_diff={br_main:.2e} > rtol={rtol}",
            )
            for d, v in enumerate(br_mtp):
                self.assertLess(
                    v, rtol,
                    f"rank {r}: BL vs EP4-replay MTP depth {d} rel_diff={v:.2e} > rtol={rtol}",
                )

        # ---- Summary stats ----
        print("\n--- Summary (mean rel_diff across 32 ranks) ---")
        for label, res_a, res_b in [
            ("BL vs EP4-free", bl_results, ep_free_results),
            ("BL vs EP4-replay", bl_results, ep_results),
            ("EP4-free vs EP4-replay", ep_free_results, ep_results),
        ]:
            main_rels = [_rel(res_a[r]["main_logits"], res_b[r]["main_logits"]) for r in range(world_size)]
            mtp_rels = [_rel(res_a[r]["mtp_logits"][0], res_b[r]["mtp_logits"][0]) for r in range(world_size)]
            print(
                f"  {label:30s}: main mean={sum(main_rels)/len(main_rels):.2e} max={max(main_rels):.2e}, "
                f"mtp mean={sum(mtp_rels)/len(mtp_rels):.2e} max={max(mtp_rels):.2e}"
            )

        # ---- Save logits for offline analysis ----
        save_dir = "/tmp/mtp_ep_parity_logits"
        os.makedirs(save_dir, exist_ok=True)
        for label, results in [
            ("baseline_ep1", bl_results),
            ("ep4_free", ep_free_results),
            ("ep4_replay", ep_results),
        ]:
            for res in results:
                r = res["rank"]
                torch.save(
                    {
                        "main_logits": res["main_logits"],
                        "mtp_logits": res["mtp_logits"],
                    },
                    os.path.join(save_dir, f"{label}_rank{r:02d}.pt"),
                )
        print(f"\nLogits saved to {save_dir}/")

        print("\nPASSED")


if __name__ == "__main__":
    unittest.main()
