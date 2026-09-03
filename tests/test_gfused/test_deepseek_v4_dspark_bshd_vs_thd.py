# coding=utf-8
"""DSpark BSHD vs THD pack-seq：loss / grad 数值对比（Ray worker 精简路径）。

对齐 ``test_deepseek_v4_mtp_smoke.py::test_mtp_bshd_vs_thd``：
  * BSHD：每段独立 pad → forward → 累加 ``dspark_loss`` backward
  * THD：生产路径按 packed segment 采 anchor + router replay → 一次 forward

正式 ``FinetuneTrainer`` 对照见
``test_deepseek_v4_dspark_sft_bshd_vs_thd.py``。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 \\
      tests/test_gfused/test_deepseek_v4_dspark_bshd_vs_thd.py::TestDeepseekV4DSparkBshdVsThd::test_dspark_bshd_vs_thd
"""

from __future__ import annotations

import math
import os
import unittest
import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer

from gpatch_v4.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM, apply_hp
from gpatch_v4.models.deepseek_v4.checkpoint import infer_dspark_num_layers
from gpatch_v4.models.deepseek_v4.dspark import (
    DSparkBatch,
    prepare_dspark_batch,
)
from gpatch_v4.training_backend.fsdp2_backend.dspark_loss import (
    calculate_dspark_loss,
    dspark_loss_denominator,
)
from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

HF_MODEL_PATH = os.environ.get(
    "DSPARK_HF_MODEL_PATH",
    "hf-hub/deepseek-ai/DeepSeek-V4-Flash-0731",
)
NUM_GPUS = 32
EP_SIZE = 4
NUM_LAYERS = 8
NUM_ANCHORS = 2
DSPARK_CE_ALPHA = 0.1
DSPARK_L1_ALPHA = 0.9
DSPARK_CONF_ALPHA = 1.0
DSPARK_LOSS_DECAY_GAMMA = 4.0


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip() -> str:
    return ray.util.get_node_ip_address()


def _prepare_dspark_config(hf_model_path: str) -> DeepseekV4Config:
    config = DeepseekV4Config.from_pretrained(hf_model_path)
    config.dspark_num_layers = infer_dspark_num_layers(hf_model_path)
    config.dspark_num_anchors = NUM_ANCHORS
    assert config.dspark_target_layer_ids
    config.num_hidden_layers = NUM_LAYERS
    if config.layer_types is not None:
        config.layer_types = config.layer_types[:NUM_LAYERS]
    if config.mlp_layer_types is not None:
        config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
    num_target_layers = len(config.dspark_target_layer_ids)
    assert NUM_LAYERS >= num_target_layers
    config.dspark_target_layer_ids = list(
        range(NUM_LAYERS - num_target_layers, NUM_LAYERS)
    )
    assert not config.tie_word_embeddings, (
        "meta-device init may hang with tie_word_embeddings=True"
    )
    return config


def _make_fake_segments(fake_seq_lens, tokenizer, device, seed=42):
    pad_id = tokenizer.pad_token_id
    vocab = tokenizer.vocab_size
    ids_list, labels_list = [], []
    for i, s in enumerate(fake_seq_lens):
        rng = torch.Generator().manual_seed(seed + i)
        ids = torch.randint(0, vocab, (s,), generator=rng).to(device)
        ids = torch.where(ids == pad_id, (ids + 1) % vocab, ids)
        lab = torch.roll(ids, shifts=-1)
        lab[-1] = -100
        ids_list.append(ids)
        labels_list.append(lab)
    return ids_list, labels_list


def _build_model_for_dspark(hf_model_path, ep_2d_mesh, cp_mesh=None):
    config = _prepare_dspark_config(hf_model_path)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)
    assert model.dspark_enabled
    assert model.mtp is not None
    model = apply_hp(
        model,
        ep_2d_mesh,
        cp_mesh=cp_mesh,
        amp_fp32=True,
        attn_backend="fused",
        indexer_backend="fused",
    )
    model.gradient_checkpointing_enable()
    model.load_checkpoint_hp(hf_model_path)
    model.train()
    return model, config


def _slice_replay_for_contiguous_cp(
    replay_indices: list[torch.Tensor],
    *,
    global_seq_len: int,
    cp_rank: int,
    cp_size: int,
) -> list[torch.Tensor]:
    """Slice backbone routing to the local CP chunk; keep DSpark draft routing intact."""
    s_local = global_seq_len // cp_size
    start = cp_rank * s_local
    end = start + s_local
    sliced = []
    for idx in replay_indices:
        if idx.shape[0] == global_seq_len:
            sliced.append(idx[start:end].contiguous())
        else:
            sliced.append(idx)
    return sliced


def _pad_segment(ids, lab, pad_token_id, pad_mul, device):
    s = ids.shape[0]
    s_padded = ((s + pad_mul - 1) // pad_mul) * pad_mul
    seg_ids = torch.full((1, s_padded), pad_token_id, dtype=torch.long, device=device)
    seg_labels = torch.full((1, s_padded), -100, dtype=torch.long, device=device)
    seg_ids[0, :s] = ids
    seg_labels[0, :s] = lab
    seg_labels[0, s - 1] = -100
    loss_mask = (seg_labels != -100).float()
    position_ids = torch.arange(s_padded, device=device).unsqueeze(0)
    return seg_ids, seg_labels, loss_mask, position_ids


def _prepare_dspark_batch(
    input_ids,
    labels,
    loss_mask,
    *,
    block_size: int,
    seed: int,
    packed_seq_params=None,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=NUM_ANCHORS,
        block_size=block_size,
        packed_seq_params=packed_seq_params,
    )


def _split_packed_dspark_batch(
    batch: DSparkBatch,
    cu_seqlens_q_padded: torch.Tensor,
    block_size: int,
) -> list[DSparkBatch]:
    boundaries = cu_seqlens_q_padded.tolist()
    num_segments = len(boundaries) - 1
    assert batch.anchor_positions.shape[1] == num_segments * NUM_ANCHORS
    block_offsets = torch.arange(
        block_size,
        device=batch.anchor_positions.device,
    ).view(1, 1, -1)
    batches = []
    for seg_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        anchor_slice = slice(seg_idx * NUM_ANCHORS, (seg_idx + 1) * NUM_ANCHORS)
        block_keep_mask = batch.block_keep_mask[:, anchor_slice]
        anchor_positions = batch.anchor_positions[:, anchor_slice] - start
        anchor_positions = torch.where(
            block_keep_mask,
            anchor_positions,
            torch.zeros_like(anchor_positions),
        )
        target_hidden_indices = (
            anchor_positions.unsqueeze(-1) + block_offsets
        ).clamp(max=end - start - 1)
        batches.append(
            DSparkBatch(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                target_ids=batch.target_ids[:, anchor_slice],
                eval_mask=batch.eval_mask[:, anchor_slice],
                prev_token_ids=batch.prev_token_ids[:, anchor_slice],
                target_hidden_indices=target_hidden_indices,
            )
        )
    return batches


def _calculate_dspark_step_loss(dspark_output, global_denominator):
    return calculate_dspark_loss(
        outputs=dspark_output,
        global_denominator=global_denominator,
        dp_size=1,
        ce_loss_alpha=DSPARK_CE_ALPHA,
        l1_loss_alpha=DSPARK_L1_ALPHA,
        confidence_loss_alpha=DSPARK_CONF_ALPHA,
        loss_decay_gamma=DSPARK_LOSS_DECAY_GAMMA,
    )


@ray.remote(num_gpus=1)
def _dspark_bshd_perseg_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    fake_seq_lens: list[int],
    seed: int = 42,
) -> dict:
    """BSHD：每段独立 forward，累加 dspark_loss；捕获 routing 供 THD replay。"""
    from gpatch_v4.models.deepseek_v4.router_replay import capture_routing_decisions
    from gpatch_v4.models.deepseek_v4.thd import pack_sequences

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ids_list, labels_list = _make_fake_segments(
        fake_seq_lens, tokenizer, torch.device("cuda"), seed,
    )
    model, config = _build_model_for_dspark(hf_model_path, ep_2d_mesh)
    pad_mul = max(config.compress_rates.values())
    block_size = int(config.dspark_block_size)
    device = torch.device("cuda")

    padded = [
        _pad_segment(ids, lab, tokenizer.pad_token_id, pad_mul, device)
        for ids, lab in zip(ids_list, labels_list)
    ]
    packed_ids, _, packed_labels, psp = pack_sequences(
        ids_list,
        labels_list,
        config=config,
        pad_to_multiple_of=pad_mul,
        cp_size=1,
        pad_token_id=tokenizer.pad_token_id,
        label_ignore_index=-100,
    )
    assert packed_labels is not None
    packed_batch = _prepare_dspark_batch(
        packed_ids,
        packed_labels,
        (packed_labels != -100).float(),
        block_size=block_size,
        seed=seed + 1000,
        packed_seq_params=psp,
    )
    seg_batches = _split_packed_dspark_batch(
        packed_batch,
        psp.cu_seqlens_q_padded,
        block_size,
    )

    global_denom = torch.zeros((), device=device)
    for batch in seg_batches:
        global_denom += dspark_loss_denominator(
            batch,
            block_size=block_size,
            loss_decay_gamma=DSPARK_LOSS_DECAY_GAMMA,
        )
    global_denom = global_denom.clamp_min(1.0)

    reported_loss = 0.0
    per_seg_routing: list[list[torch.Tensor]] = []
    for seg_i, ((seg_ids, _, _, position_ids), batch) in enumerate(
        zip(padded, seg_batches)
    ):
        with capture_routing_decisions(model) as recorded:
            outputs = model(
                input_ids=seg_ids,
                position_ids=position_ids,
                dspark_batch=batch,
            )
        assert outputs.dspark_output is not None
        per_seg_routing.append([t.detach().cpu() for t in recorded if t is not None])
        result = _calculate_dspark_step_loss(outputs.dspark_output, global_denom)
        result.loss.backward()
        reported_loss += result.loss.item()

    total_grad_norm = float(model.clip_grad_norm_(2.0))

    n_router_layers = len(per_seg_routing[0])
    recorded_routing = [
        torch.cat([per_seg_routing[s][layer_i] for s in range(len(per_seg_routing))], dim=0)
        for layer_i in range(n_router_layers)
    ]

    out = {
        "rank": rank,
        "dspark_loss": reported_loss,
        "total_grad_norm": total_grad_norm,
        "recorded_routing": recorded_routing,
        "global_denom": global_denom.item(),
    }
    print(
        f"[dspark_bshd_perseg] rank {rank}: dspark_loss={reported_loss:.6f}, "
        f"total_grad_norm={total_grad_norm:.4f}, "
        f"global_denom={global_denom.item():.4f}"
    )
    del model, outputs
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return out


@ray.remote(num_gpus=1)
def _dspark_thd_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    fake_seq_lens: list[int],
    seed: int = 42,
    replay_indices: list[torch.Tensor] | None = None,
    cp_size: int = 1,
    pack_cp_size: int | None = None,
    capture_routing: bool = False,
) -> dict:
    """THD pack：生产路径按 segment 采 anchor；``cp_size>1`` 时走 contiguous CP。"""
    from contextlib import ExitStack

    from gpatch_v4.models.deepseek_v4.cp import cp_chunk_data
    from gpatch_v4.models.deepseek_v4.dspark import shard_dspark_batch_for_contiguous_cp
    from gpatch_v4.models.deepseek_v4.router_replay import (
        capture_routing_decisions,
        router_replay_ctx,
    )
    from gpatch_v4.models.deepseek_v4.thd import pack_sequences

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if pack_cp_size is None:
        pack_cp_size = cp_size
    assert world_size % ep_size == 0
    assert world_size % cp_size == 0
    assert pack_cp_size >= cp_size

    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    if cp_size > 1:
        cp_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // cp_size, cp_size),
            mesh_dim_names=("dp", "cp"),
        )["cp"]
    else:
        cp_mesh = None
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ids_list, labels_list = _make_fake_segments(
        fake_seq_lens, tokenizer, torch.device("cuda"), seed,
    )
    model, config = _build_model_for_dspark(hf_model_path, ep_2d_mesh, cp_mesh=cp_mesh)
    pad_mul = max(config.compress_rates.values())
    block_size = int(config.dspark_block_size)

    packed_ids, packed_pos, packed_labels, psp = pack_sequences(
        ids_list,
        labels_list,
        config=config,
        pad_to_multiple_of=pad_mul,
        cp_size=pack_cp_size,
        pad_token_id=tokenizer.pad_token_id,
        label_ignore_index=-100,
    )
    assert packed_labels is not None
    packed_batch = _prepare_dspark_batch(
        packed_ids,
        packed_labels,
        (packed_labels != -100).float(),
        block_size=block_size,
        seed=seed + 1000,
        packed_seq_params=psp,
    )

    global_denom = dspark_loss_denominator(
        packed_batch,
        block_size=block_size,
        loss_decay_gamma=DSPARK_LOSS_DECAY_GAMMA,
    ).clamp_min(1.0)

    local_ids = packed_ids
    local_pos = packed_pos
    local_psp = psp
    local_batch = packed_batch
    cp_group = None
    cp_rank = 0
    if cp_size > 1:
        cp_group = model._cp_group
        cp_rank = model._cp_rank
        s_local = packed_ids.shape[1] // cp_size
        assert s_local >= config.sliding_window, (
            f"s_local={s_local} < sliding_window={config.sliding_window}"
        )
        assert s_local >= block_size - 1, (
            f"s_local={s_local} < block_size-1={block_size - 1}"
        )
        local_batch = shard_dspark_batch_for_contiguous_cp(
            packed_batch,
            sequence_length=packed_ids.shape[1],
            cp_rank=cp_rank,
            cp_size=cp_size,
        )
        local_ids, _, _, local_pos, local_psp = cp_chunk_data(
            cp_rank,
            cp_size,
            tokens=packed_ids,
            labels=packed_labels,
            position_ids=packed_pos,
            packed_seq_params=psp,
        )

    stack = ExitStack()
    recorded = None
    if replay_indices is not None and len(replay_indices) > 0:
        replay = [t.cuda() for t in replay_indices]
        if cp_size > 1:
            replay = _slice_replay_for_contiguous_cp(
                replay,
                global_seq_len=packed_ids.shape[1],
                cp_rank=cp_rank,
                cp_size=cp_size,
            )
        stack.enter_context(router_replay_ctx(model, replay))
    if capture_routing:
        recorded = stack.enter_context(capture_routing_decisions(model))

    with stack:
        outputs = model(
            input_ids=local_ids,
            position_ids=local_pos,
            packed_seq_params=local_psp,
            dspark_batch=local_batch,
        )
        assert outputs.dspark_output is not None
        result = _calculate_dspark_step_loss(outputs.dspark_output, global_denom)
        reported = torch.stack(
            [
                result.loss.detach(),
                result.ce_loss.detach(),
                result.l1_loss.detach(),
                result.confidence_loss.detach(),
            ]
        )
        if cp_group is not None:
            dist.all_reduce(reported, group=cp_group)
        result.loss.backward()
    total_grad_norm = float(model.clip_grad_norm_(2.0))

    recorded_routing = None
    if capture_routing:
        assert recorded is not None
        recorded_routing = []
        for i, t in enumerate(recorded):
            assert t is not None, f"layer {i} routing not captured"
            recorded_routing.append(t.detach().cpu())

    out = {
        "rank": rank,
        "dspark_loss": reported[0].item(),
        "ce_loss": reported[1].item(),
        "l1_loss": reported[2].item(),
        "confidence_loss": reported[3].item(),
        "total_grad_norm": total_grad_norm,
        "global_denom": global_denom.item(),
        "packed_labels_valid": int((packed_labels != -100).sum().item()),
        "packed_seq_len": int(packed_ids.shape[1]),
        "recorded_routing": recorded_routing,
    }
    print(
        f"[dspark_thd cp{cp_size}] rank {rank}: dspark_loss={out['dspark_loss']:.6f}, "
        f"ce={out['ce_loss']:.6f}, l1={out['l1_loss']:.6f}, "
        f"conf={out['confidence_loss']:.6f}, "
        f"total_grad_norm={total_grad_norm:.4f}, "
        f"global_denom={global_denom.item():.4f}"
    )
    del model, outputs
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return out


class TestDeepseekV4DSparkBshdVsThd(unittest.TestCase):
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
                f"need >= {NUM_GPUS} GPUs in Ray cluster, only {total_gpus}"
            )

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def test_dspark_bshd_vs_thd(
        self,
        world_size: int = NUM_GPUS,
        ep_size: int = EP_SIZE,
        master_port: int = 13700,
    ):
        """BSHD per-seg vs THD pack：dspark_loss / grad_norm 应接近。"""
        hf_model_path = os.path.abspath(HF_MODEL_PATH)
        if not os.path.isdir(hf_model_path):
            raise unittest.SkipTest(
                f"DSpark checkpoint not found: {hf_model_path}; "
                "set DSPARK_HF_MODEL_PATH"
            )
        assert world_size % ep_size == 0

        fake_seq_lens = [100, 80, 120]

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
            bshd_futures = [
                _dspark_bshd_perseg_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=r,
                    )
                ).remote(
                    hf_model_path,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    ep_size=ep_size,
                    fake_seq_lens=fake_seq_lens,
                )
                for r in range(world_size)
            ]
            bshd_results = ray.get(bshd_futures)
        finally:
            remove_placement_group(pg)

        self.tearDown()
        self.setUp()

        pg2 = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
        )
        ray.get(pg2.ready())
        try:
            master_addr2 = ray.get(
                _get_node_ip.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg2, placement_group_bundle_index=0,
                    )
                ).remote()
            )
            replay = bshd_results[0].get("recorded_routing")
            thd_futures = [
                _dspark_thd_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg2, placement_group_bundle_index=r,
                    )
                ).remote(
                    hf_model_path,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr2,
                    master_port=master_port + 1,
                    ep_size=ep_size,
                    fake_seq_lens=fake_seq_lens,
                    replay_indices=replay,
                )
                for r in range(world_size)
            ]
            thd_results = ray.get(thd_futures)
        finally:
            remove_placement_group(pg2)

        for res in bshd_results + thd_results:
            self.assertTrue(math.isfinite(res["dspark_loss"]))
            self.assertTrue(math.isfinite(res["total_grad_norm"]))

        bshd_r0 = bshd_results[0]
        thd_r0 = thd_results[0]
        print("\n" + "=" * 60)
        print("DSpark BSHD vs THD")
        print(
            f"  BSHD: dspark_loss={bshd_r0['dspark_loss']:.6f}  "
            f"grad_norm={bshd_r0['total_grad_norm']:.4f}  "
            f"denom={bshd_r0['global_denom']:.4f}"
        )
        print(
            f"  THD:  dspark_loss={thd_r0['dspark_loss']:.6f}  "
            f"grad_norm={thd_r0['total_grad_norm']:.4f}  "
            f"denom={thd_r0['global_denom']:.4f}"
        )
        loss_rel = abs(bshd_r0["dspark_loss"] - thd_r0["dspark_loss"]) / (
            abs(bshd_r0["dspark_loss"]) + 1e-8
        )
        gn_rel = abs(bshd_r0["total_grad_norm"] - thd_r0["total_grad_norm"]) / (
            abs(bshd_r0["total_grad_norm"]) + 1e-8
        )
        print(f"  loss rel_diff={loss_rel:.6f}  grad_norm rel_diff={gn_rel:.6f}")
        print("=" * 60 + "\n")

        self.assertLess(loss_rel, 0.005, f"DSpark loss rel_diff {loss_rel:.6f} > 0.5%")
        self.assertLess(gn_rel, 0.01, f"DSpark grad_norm rel_diff {gn_rel:.6f} > 1%")


if __name__ == "__main__":
    unittest.main()
