# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""FSDP2 e2e: DeepSeek-V4 CSA indexer freeze vs weight-decay on a real ckpt.

Truncates Flash to 4 backbone layers, loads real weights, mirrors the training
engine path ``DeepseekV4PostInitModel`` → ``setup_optimizer``, then one
fwd / bwd / step.

- freeze ON (default): indexer excluded from AdamW; tensors unchanged after step
  (even with elevated ``weight_decay``).
- freeze OFF: indexer stays in AdamW; after a real bwd, zero-fill any still-None
  grads (FSDP-like) and assert WD shrinks them.

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_freeze_csa_indexer_e2e.py
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, DeepseekV4Config

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.extended_model.deepseek_v4 import DeepseekV4PostInitModel
from gpatch_v4.models.deepseek_v4 import DeepseekV4ForCausalLM, apply_hp
from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer
from gpatch_v4.training_backend.fsdp2_backend.optimizer import setup_optimizer

HF_MODEL_PATH = "hf-hub/sgl-project/DeepSeek-V4-Flash-FP8/"
NUM_GPUS = 8
EP_SIZE = 4
NUM_BACKBONE_LAYERS = 4
SEQ_LEN = 256
# High WD so a single zero-grad step is easy to see.
WD = 0.5
LR = 0.1


def _truncate_config(config: DeepseekV4Config) -> DeepseekV4Config:
    assert config.layer_types is not None
    assert config.mlp_layer_types is not None
    config.num_hidden_layers = NUM_BACKBONE_LAYERS
    config.layer_types = config.layer_types[:NUM_BACKBONE_LAYERS]
    config.mlp_layer_types = config.mlp_layer_types[:NUM_BACKBONE_LAYERS]
    # LM-loss only; MTP not needed for indexer WD coverage.
    config.num_nextn_predict_layers = 0
    return config


def _training_ns(*, freeze_csa_indexer: bool) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(
            freeze_csa_indexer=freeze_csa_indexer,
            freeze_router_weight=False,
            freeze_router_correction_bias=True,
        )
    )


def _setup_optimizer_cfg(*, weight_decay: float = WD, lr: float = LR):
    return SimpleNamespace(
        optimizer=SimpleNamespace(
            optimizer_type="adamw",
            lr=lr,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=weight_decay,
            adam_epsilon=1e-8,
        ),
        checkpoint=SimpleNamespace(no_load_optim=True),
    )


def _as_local_tensor(t: torch.Tensor) -> torch.Tensor:
    x = t.detach()
    if isinstance(x, DTensor):
        x = x.full_tensor()
    return x


def _indexer_named_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    out: list[tuple[str, nn.Parameter]] = []
    for name, module in model.named_modules():
        if isinstance(module, DeepseekV4Indexer):
            for pname, p in module.named_parameters():
                out.append((f"{name}.{pname}", p))
    return out


def _snapshot_indexer(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: _as_local_tensor(p).cpu().clone() for n, p in _indexer_named_params(model)}


def _opt_param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(p) for g in optimizer.param_groups for p in g["params"]}


def _fill_zero_grads(params, *, force: bool = False) -> int:
    """Fill ``grad=zeros`` where missing; return how many tensors were filled."""
    n = 0
    for p in params:
        if not (force or p.requires_grad):
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
            n += 1
    return n


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip() -> str:
    return ray.util.get_node_ip_address()


@ray.remote(num_gpus=1)
def _freeze_indexer_e2e_worker(
    hf_model_path: str,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    ep_size: int,
    freeze_csa_indexer: bool,
    seq_len: int = SEQ_LEN,
) -> dict:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    assert world_size % ep_size == 0
    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rng = torch.Generator(device="cpu").manual_seed(7 + rank)
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, seq_len), generator=rng,
    ).cuda()
    # Avoid pad id in labels path.
    input_ids = torch.where(
        input_ids == tokenizer.pad_token_id,
        (input_ids + 1) % tokenizer.vocab_size,
        input_ids,
    )

    config = DeepseekV4Config.from_pretrained(hf_model_path)
    _truncate_config(config)
    assert not config.tie_word_embeddings

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    # fused indexer matches production GRPO recipes; CPU unit tests use eager.
    model = apply_hp(
        model,
        ep_2d_mesh,
        cp_mesh=None,
        amp_fp32=False,
        indexer_backend="fused",
        attn_backend="fused",
    )
    model.load_checkpoint_hp(hf_model_path)
    model.train()

    indexer_params = _indexer_named_params(model)
    assert indexer_params, (
        f"[rank {rank}] no DeepseekV4Indexer after truncate to "
        f"{NUM_BACKBONE_LAYERS} layers; layer_types={list(config.layer_types)}"
    )

    # Mirror fsdp2_engine_lm.setup_model_and_get_optimizer ordering.
    DeepseekV4PostInitModel(_training_ns(freeze_csa_indexer=freeze_csa_indexer))(model)
    opt = setup_optimizer(_setup_optimizer_cfg(), model)
    opt_ids = _opt_param_ids(opt)

    in_opt = [n for n, p in indexer_params if id(p) in opt_ids]
    req = {n: p.requires_grad for n, p in indexer_params}

    labels = input_ids.clone()
    labels = torch.roll(labels, shifts=-1, dims=-1)
    labels[:, -1] = -100

    before = _snapshot_indexer(model)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits.float()
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss), f"[rank {rank}] non-finite loss={loss}"
    loss.backward()

    none_grads = sum(1 for _, p in indexer_params if p.grad is None)
    filled = 0
    if not freeze_csa_indexer:
        # Real CSA path often leaves indexer.grad as None; FSDP-like zero fill
        # is what makes AdamW WD fire — same failure mode freeze exists for.
        filled = _fill_zero_grads([p for _, p in indexer_params], force=False)
        if filled == 0 and none_grads == len(indexer_params):
            filled = _fill_zero_grads([p for _, p in indexer_params], force=True)

    model.clip_grad_norm_(1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    after = _snapshot_indexer(model)
    changed = [n for n in before if not torch.equal(before[n], after[n])]
    abs_before = {n: before[n].float().abs().sum().item() for n in before}
    abs_after = {n: after[n].float().abs().sum().item() for n in after}

    if freeze_csa_indexer:
        assert all(v is False for v in req.values()), req
        assert not in_opt, f"frozen indexer still in optimizer: {in_opt}"
        assert not changed, f"frozen indexer moved under WD: {changed}"
    else:
        assert all(v is True for v in req.values()), req
        assert len(in_opt) == len(indexer_params), (
            f"unfrozen indexer missing from optimizer: in={in_opt} "
            f"all={[n for n, _ in indexer_params]}"
        )
        assert changed, (
            f"unfrozen indexer unchanged after WD step "
            f"(none_grads={none_grads}, filled={filled})"
        )
        for n in changed:
            assert abs_after[n] < abs_before[n], (
                f"{n}: expected WD shrink, before_abs={abs_before[n]}, "
                f"after_abs={abs_after[n]}"
            )

    result = {
        "rank": rank,
        "freeze_csa_indexer": freeze_csa_indexer,
        "n_indexer_params": len(indexer_params),
        "indexer_names": [n for n, _ in indexer_params],
        "in_optimizer": in_opt,
        "requires_grad": req,
        "loss": float(loss.detach()),
        "none_grads_after_bwd": none_grads,
        "zero_grads_filled": filled,
        "changed": changed,
        "n_changed": len(changed),
    }
    print(
        f"[freeze_indexer_e2e] rank={rank} freeze={freeze_csa_indexer} "
        f"n_idx={len(indexer_params)} in_opt={len(in_opt)} "
        f"none_grads={none_grads} filled={filled} changed={len(changed)} "
        f"loss={result['loss']:.4f}"
    )

    del model, opt, outputs
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


class TestDeepseekV4FreezeCsaIndexerE2E(unittest.TestCase):
    def setUp(self):
        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def _run(self, *, freeze_csa_indexer: bool, master_port: int) -> list[dict]:
        assert os.path.isdir(HF_MODEL_PATH), HF_MODEL_PATH
        world_size = NUM_GPUS
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
            futures = [
                _freeze_indexer_e2e_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=r,
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    ep_size=EP_SIZE,
                    freeze_csa_indexer=freeze_csa_indexer,
                )
                for r in range(world_size)
            ]
            return ray.get(futures)
        finally:
            remove_placement_group(pg)

    def test_freeze_on_and_off_against_real_ckpt(self):
        """freeze ON: no WD; freeze OFF: in-opt + WD shrink (FSDP-like zero grads)."""
        on_results = self._run(freeze_csa_indexer=True, master_port=29711)
        self.assertTrue(all(r["n_changed"] == 0 for r in on_results))
        self.assertTrue(all(not r["in_optimizer"] for r in on_results))
        self.assertTrue(all(r["n_indexer_params"] > 0 for r in on_results))

        off_results = self._run(freeze_csa_indexer=False, master_port=29712)
        self.assertTrue(all(r["n_changed"] > 0 for r in off_results))
        self.assertTrue(
            all(len(r["in_optimizer"]) == r["n_indexer_params"] for r in off_results)
        )

        print("\n" + "=" * 60)
        print("  CSA indexer freeze e2e (4-layer Flash ckpt)")
        print("=" * 60)
        r0_on, r0_off = on_results[0], off_results[0]
        print(
            f"  freeze ON : n_idx={r0_on['n_indexer_params']} "
            f"in_opt={len(r0_on['in_optimizer'])} changed={r0_on['n_changed']} "
            f"loss={r0_on['loss']:.4f}"
        )
        print(
            f"  freeze OFF: n_idx={r0_off['n_indexer_params']} "
            f"in_opt={len(r0_off['in_optimizer'])} changed={r0_off['n_changed']} "
            f"none_grads={r0_off['none_grads_after_bwd']} "
            f"filled={r0_off['zero_grads_filled']} loss={r0_off['loss']:.4f}"
        )
        print("=" * 60 + "\n")


if __name__ == "__main__":
    unittest.main()
