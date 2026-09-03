# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""离线重排前后的 full model forward 等价性校验。

对同一份 fake THD 输入，分别 load 原始 checkpoint 与离线重排后的 checkpoint，
各跑一次前向，比较输出是否**逐比特相等**：主 LM-head ``logits``；``enable_mtp``
时追加每个 MTP depth 的 prediction hidden ``mtp.{d}``。
"""
from __future__ import annotations

import os

import hydra
import ray
import torch
from hydra.utils import get_original_cwd
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer, DeepseekV4Config

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.models.deepseek_v4 import DeepseekV4ForCausalLM, apply_hp
from gpatch_v4.models.deepseek_v4.cp import cp_chunk_data
from gpatch_v4.models.deepseek_v4.thd import pack_sequences
from gpatch_v4.orches.placement_group import _create_placement_group

NUM_GPUS = 32
EP_SIZE = 8
CP_SIZE = 4
FAKE_SEQ_LENS = [100, 200, 100]
PAD_TO_MULTIPLE_OF = 128
FAKE_QA_SEED = 20260530
_MASTER_PORT = 12900


# ---------------------------------------------------------------------------
# helpers（结构对齐 test_deepseek_v4_ep_cp_thd.py，仅保留 forward 所需部分）
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip():
    return ray.util.get_node_ip_address()


def _make_fake_qa(tokenizer, device, fake_seq_lens=None):
    """3 段 ids + per-seg roll-shift labels（next-token 预测，末位 -100）。

    所有 rank 用同一 ``FAKE_QA_SEED`` → vanilla baseline 全 rank loss/grad
    一致（DP 复制），仅借多卡均摊 FSDP unshard 显存；pack 全 rank 同 input
    → 共享同一份 baseline routing。
    """
    if fake_seq_lens is None:
        fake_seq_lens = FAKE_SEQ_LENS
    pad_id = tokenizer.pad_token_id
    vocab = tokenizer.vocab_size
    assert pad_id < vocab, (
        f"pad_id ({pad_id}) >= vocab_size ({vocab}); the (ids+1)%vocab "
        f"collision-fix below assumes pad_id is in randint range"
    )
    ids_list, labels_list = [], []
    for i, s in enumerate(fake_seq_lens):
        rng = torch.Generator().manual_seed(FAKE_QA_SEED + i)
        ids = torch.randint(0, vocab, (s,), generator=rng).to(device)
        # DSV4 pad_id 大，randint(pad_id+1, vocab) 会压扁低位 vocab；
        # 改成把撞 pad_id 的位置移一格
        ids = torch.where(ids == pad_id, (ids + 1) % vocab, ids)
        labels = torch.roll(ids, shifts=-1)
        labels[-1] = -100
        ids_list.append(ids)
        labels_list.append(labels)
    return ids_list, labels_list


def _setup_dist(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _setup_ep_cp_meshes(world_size):
    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // EP_SIZE, EP_SIZE),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    cp_full_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // CP_SIZE, CP_SIZE),
        mesh_dim_names=("dp", "cp"),
    )
    return ep_2d_mesh, cp_full_mesh["cp"]


def _build_fork_model(ckpt_dir, hf_config):
    assert all(PAD_TO_MULTIPLE_OF % r == 0 for r in hf_config.compress_rates.values()), (
        f"PAD_TO_MULTIPLE_OF={PAD_TO_MULTIPLE_OF} must be divisible by every "
        f"compress_rate; got {dict(hf_config.compress_rates)}"
    )
    assert not hf_config.tie_word_embeddings, (
        "meta-device init may hang with tie_word_embeddings=True"
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(hf_config)
    finally:
        torch.set_default_dtype(prev_dtype)
    return model


def _gather_cp(tensor, cp_group):
    """CP>1 时按 cp_group 沿 seq 维 all-gather 回全 T."""
    cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    if cp_size > 1:
        gathered = [torch.empty_like(tensor) for _ in range(cp_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=cp_group)
        return torch.cat(gathered, dim=1)
    else:
        return tensor.clone()


def _forward_logits(ckpt_dir, hf_config, enable_mtp, ep_2d_mesh, cp_mesh,
                    local_ids, local_position_ids, local_psp):
    """Build+load 一个 ckpt 的 fork 模型跑一次 THD 前向，返回待比对张量 dict。

    key ``logits`` 为 LM-head 输出；enable_mtp 时追加 ``mtp.{d}`` 各 depth 的
    prediction hidden。返回前 all-gather 回全 T 并释放模型显存，供顺序对比第二个
    ckpt 复用。
    """
    model = _build_fork_model(ckpt_dir, hf_config)
    model = apply_hp(
        model,
        ep_2d_mesh,
        cp_mesh=cp_mesh,
        amp_fp32=False,
        attn_backend="eager",
        indexer_backend="eager",
        ep_backend="eager",
    )
    model.load_checkpoint_hp(ckpt_dir)
    # use_mtp = (mtp is not None and self.training)，且 MTP 输出不进入主 logits：
    # 要验 MTP 重排必须 train() 触发并单独比对 mtp_per_depth_h。
    model.train()

    cp_group = model._cp_group if CP_SIZE > 1 else None
    # 同 seed 起跑，屏蔽潜在 dropout/随机算子在两次顺序前向间的 RNG 漂移。
    torch.manual_seed(FAKE_QA_SEED)
    torch.cuda.manual_seed_all(FAKE_QA_SEED)
    with torch.no_grad():
        outputs = model(
            input_ids=local_ids,
            position_ids=local_position_ids,
            packed_seq_params=local_psp,
        )

    out = {"logits": _gather_cp(outputs.logits.detach(), cp_group)}
    if enable_mtp:
        per_depth = outputs.mtp_per_depth_h
        assert per_depth is not None, (
            "enable_mtp but mtp_per_depth_h is None (train() must be on to run MTP)"
        )
        for d, h in enumerate(per_depth):
            out[f"mtp.{d}"] = _gather_cp(h.detach(), cp_group)

    del model
    torch.cuda.empty_cache()
    return out


@ray.remote(num_gpus=1)
def _forward_compare_worker(
    src_ckpt: str, dst_ckpt: str, rank: int, world_size: int,
    master_addr: str, master_port: int, hf_config: DeepseekV4Config, enable_mtp: bool,
):
    """在同一 rank 上先后前向 src / dst 两个 ckpt，本地逐张量比对 bit-equal。

    比对项：主 ``logits``；enable_mtp 时追加每个 MTP depth 的 ``mtp.{d}``。
    """
    _setup_dist(rank, world_size, master_addr, master_port)
    ep_2d_mesh, cp_mesh = _setup_ep_cp_meshes(world_size)

    tokenizer = AutoTokenizer.from_pretrained(src_ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ids_list, labels_list = _make_fake_qa(
        tokenizer, device=torch.device("cuda"), fake_seq_lens=FAKE_SEQ_LENS
    )

    packed_ids, packed_position_ids, packed_labels, psp = pack_sequences(
        ids_list, labels_list,
        config=hf_config,
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
        cp_size=CP_SIZE,
        pad_token_id=tokenizer.pad_token_id,
        label_ignore_index=-100,
    )
    assert packed_labels is not None

    cp_size = cp_mesh.size()
    cp_rank = cp_mesh.get_local_rank()
    if cp_size > 1:
        local_ids, _, _, local_position_ids, local_psp = cp_chunk_data(
            cp_rank, cp_size,
            tokens=packed_ids, labels=packed_labels,
            position_ids=packed_position_ids, packed_seq_params=psp,
        )
    else:
        local_ids, local_position_ids, local_psp = packed_ids, packed_position_ids, psp

    # 同一份输入喂给两个 ckpt；重排等价 → 每个输出张量应逐比特一致。
    out_src = _forward_logits(
        src_ckpt, hf_config, enable_mtp, ep_2d_mesh, cp_mesh,
        local_ids, local_position_ids, local_psp,
    )
    out_dst = _forward_logits(
        dst_ckpt, hf_config, enable_mtp, ep_2d_mesh, cp_mesh,
        local_ids, local_position_ids, local_psp,
    )

    assert out_src.keys() == out_dst.keys(), (
        f"output keys mismatch: src={sorted(out_src)} dst={sorted(out_dst)}"
    )
    per_key = {}
    bitequal = True
    finite = True
    max_abs = 0.0
    for key in out_src:
        a, b = out_src[key], out_dst[key]
        assert a.shape == b.shape, (
            f"{key} shape mismatch: src={list(a.shape)} dst={list(b.shape)}"
        )
        eq = bool(torch.equal(a, b))
        mad = (a.float() - b.float()).abs().max().item()
        fin = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        per_key[key] = {"bitequal": eq, "max_abs_diff": mad, "finite": fin, "shape": list(a.shape)}
        bitequal = bitequal and eq
        finite = finite and fin
        max_abs = max(max_abs, mad)
    print(
        f"[repermute_fwd] rank {rank}: bitequal={bitequal} max_abs_diff={max_abs:.3e} "
        f"finite={finite} keys={list(per_key)}",
        flush=True,
    )

    del out_src, out_dst
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return {"rank": rank, "bitequal": bitequal, "max_abs_diff": max_abs, "finite": finite, "per_key": per_key}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _resolve(path: str, base: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base, path)


def run_forward_equivalence(
    src_ckpt: str, dst_ckpt: str, hf_config: DeepseekV4Config, enable_mtp: bool
) -> None:
    """32 卡 EP8/CP4 THD 前向，比对 src / dst 两个 ckpt 输出是否 bit-equal。"""
    assert os.path.isdir(src_ckpt), f"src ckpt dir not found: {src_ckpt}"
    assert os.path.isdir(dst_ckpt), f"dst ckpt dir not found: {dst_ckpt}"
    assert NUM_GPUS % EP_SIZE == 0, f"NUM_GPUS={NUM_GPUS} not divisible by EP_SIZE={EP_SIZE}"
    assert NUM_GPUS % CP_SIZE == 0, f"NUM_GPUS={NUM_GPUS} not divisible by CP_SIZE={CP_SIZE}"

    ray.init(address="auto")
    try:
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        assert total_gpus >= NUM_GPUS, (
            f"need >= {NUM_GPUS} GPUs in Ray cluster, only {total_gpus}"
        )
        pg_obj, bundle_indices = _create_placement_group(NUM_GPUS)
        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg_obj, placement_group_bundle_index=bundle_indices[0],
                )
            ).remote()
        )
        print("=" * 60)
        print(
            f"[repermute_fwd] EP={EP_SIZE} CP={CP_SIZE} world={NUM_GPUS} enable_mtp={enable_mtp} "
            f"num_hidden_layers={hf_config.num_hidden_layers} fake_seq_lens={FAKE_SEQ_LENS}\n"
            f"  src={src_ckpt}\n  dst={dst_ckpt}"
        )
        futures = [
            _forward_compare_worker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg_obj, placement_group_bundle_index=bundle_indices[r],
                )
            ).remote(
                src_ckpt, dst_ckpt,
                rank=r, world_size=NUM_GPUS,
                master_addr=master_addr, master_port=_MASTER_PORT,
                hf_config=hf_config, enable_mtp=enable_mtp,
            )
            for r in range(NUM_GPUS)
        ]
        results = ray.get(futures)
        remove_placement_group(pg_obj)
    finally:
        kill_all_actors_and_shutdown_ray()

    max_abs_overall = max(res["max_abs_diff"] for res in results)
    failures = [res for res in results if not res["bitequal"] or not res["finite"]]
    print(
        f"\n[repermute_fwd] {NUM_GPUS} ranks done, max_abs_diff over all ranks={max_abs_overall:.3e}"
    )
    if failures:
        for res in failures:
            bad_keys = {
                k: v for k, v in res["per_key"].items() if not v["bitequal"] or not v["finite"]
            }
            print(
                f"[repermute_fwd] FAIL rank {res['rank']}: bitequal={res['bitequal']} "
                f"finite={res['finite']} max_abs_diff={res['max_abs_diff']:.3e} bad_keys={bad_keys}"
            )
        raise SystemExit(1)
    print("[repermute_fwd] PASSED: repermuted checkpoint is bit-equal to the original.")
