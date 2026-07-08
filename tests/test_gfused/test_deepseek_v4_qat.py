# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""FP8 QAT smoke test: forward with/without fp8_qat produces close but
different log-probs (quantization noise is present but bounded).

Router replay with balanced expert indices ensures both runs go through
the same expert assignment — isolating QAT noise from routing variance.

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_qat.py
"""

import os
import unittest

import ray
import torch
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer, DeepseekV4Config

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray
from gpatch_v4.models.deepseek_v4 import DeepseekV4ForCausalLM, apply_hp
from gpatch_v4.models.deepseek_v4.router_replay import (
    enable_router_replay,
    router_replay_ctx,
)
from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter
from gpatch_v4.orches.placement_group import _create_placement_group

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
NUM_GPUS = 32
NUM_LAYERS = 4
EP_SIZE = 8
SEQ_LEN = 128


def _setup_dist(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _make_balanced_replay_indices(model, seq_len, device):
    """Per-TopKRouter balanced expert indices: token t picks experts
    [(t*top_k + k) % num_experts for k in range(top_k)].
    """
    indices_per_layer = []
    for m in model.modules():
        if isinstance(m, DeepseekV4TopKRouter):
            n_experts = m.num_experts
            top_k = m.top_k
            t = torch.arange(seq_len, device=device)
            idx = torch.stack([(t * top_k + k) % n_experts for k in range(top_k)], dim=-1)
            indices_per_layer.append(idx)
    return indices_per_layer


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip():
    return ray.util.get_node_ip_address()


@ray.remote(num_gpus=1)
def _qat_worker(
    hf_model_path: str, rank: int, world_size: int,
    master_addr: str, master_port: int,
):
    _setup_dist(rank, world_size, master_addr, master_port)

    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // EP_SIZE, EP_SIZE),
        mesh_dim_names=("ep_fsdp", "ep"),
    )

    config = DeepseekV4Config.from_pretrained(hf_model_path)
    config.num_hidden_layers = NUM_LAYERS
    config.layer_types = config.layer_types[:NUM_LAYERS]
    config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
    config.num_nextn_predict_layers = 0

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    model = apply_hp(model, ep_2d_mesh, fp8_qat=True)
    model.load_checkpoint_hp(hf_model_path)
    model.eval()
    dist.barrier()

    # 1. fake input
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    rng = torch.Generator().manual_seed(42)
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, SEQ_LEN), generator=rng).cuda()
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)

    # 2. 造均衡的 replay indices
    replay_indices = _make_balanced_replay_indices(model, SEQ_LEN, device=torch.device("cuda"))
    n_routers = len(replay_indices)
    if rank == 0:
        print(f"  {n_routers} TopKRouter layers, replay shape {list(replay_indices[0].shape)}")

    # 3. forward WITH qat + replay
    with torch.no_grad(), router_replay_ctx(model, replay_indices):
        out_on = model(input_ids=input_ids, position_ids=position_ids)
        logps_on = torch.log_softmax(out_on.logits.float(), dim=-1)

    # 4. toggle fp8_qat off, forward WITHOUT qat + same replay
    model.config.fp8_qat = False
    with torch.no_grad(), router_replay_ctx(model, replay_indices):
        out_off = model(input_ids=input_ids, position_ids=position_ids)
        logps_off = torch.log_softmax(out_off.logits.float(), dim=-1)

    # 5. 对比
    diff = (logps_on - logps_off).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    are_identical = torch.equal(logps_off, logps_on)

    result = {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "are_identical": are_identical,
    }
    if rank == 0:
        print(f"  max_abs_diff  = {max_abs_diff:.6f}")
        print(f"  mean_abs_diff = {mean_abs_diff:.6f}")
        print(f"  are_identical = {are_identical}")

    del model
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


class TestFP8QAT(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {NUM_GPUS} GPUs, only {total_gpus}"
            )

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def test_qat_changes_output(self):
        assert os.path.isdir(HF_MODEL_PATH), (
            f"model dir not found: {HF_MODEL_PATH}"
        )
        pg = _create_placement_group(NUM_GPUS)
        pg_obj, bundle_indices = pg

        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg_obj, placement_group_bundle_index=bundle_indices[0],
                )
            ).remote()
        )

        futures = []
        for r in range(NUM_GPUS):
            futures.append(
                _qat_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg_obj,
                        placement_group_bundle_index=bundle_indices[r],
                    )
                ).remote(
                    HF_MODEL_PATH,
                    rank=r, world_size=NUM_GPUS,
                    master_addr=master_addr, master_port=23456,
                )
            )
        results = ray.get(futures)

        r0 = results[0]
        self.assertFalse(r0["are_identical"], "fp8_qat should change log-probs")
        self.assertLess(r0["mean_abs_diff"], 0.10, "mean log-prob diff too large")
        self.assertGreater(r0["max_abs_diff"], 0, "expected nonzero diff from quantization noise")
