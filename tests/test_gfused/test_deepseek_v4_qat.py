# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""FP8/FP4 QAT smoke test with routed-MoE FP4 priority.

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
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray
from gpatch_v4.orches.placement_group import _create_placement_group

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
NUM_GPUS = 32
NUM_LAYERS = 4
EP_SIZE = 8
SEQ_LEN = 128


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip():
    return ray.util.get_node_ip_address()


@ray.remote(num_gpus=1)
def _qat_worker(
    hf_model_path: str, rank: int, world_size: int,
    master_addr: str, master_port: int,
):
    import os

    import torch
    from torch import distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoTokenizer, DeepseekV4Config

    from gpatch_v4.models.deepseek_v4 import DeepseekV4ForCausalLM, apply_hp
    from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter
    from gpatch_v4.models.deepseek_v4.router_replay import router_replay_ctx

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // EP_SIZE, EP_SIZE),
        mesh_dim_names=("ep_fsdp", "ep"),
    )

    config = DeepseekV4Config.from_pretrained(hf_model_path)
    config.num_hidden_layers = NUM_LAYERS
    config.layer_types = config.layer_types[:NUM_LAYERS]
    config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    model = apply_hp(model, ep_2d_mesh, fp8_qat=True, fp4_qat=True)
    model.load_checkpoint_hp(hf_model_path)
    model.train()
    dist.barrier()
    assert model.config.fp4_qat
    assert all(layer.config.fp4_qat for layer in model.mtp.layers)

    # 1. fake input
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    rng = torch.Generator().manual_seed(42)
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, SEQ_LEN), generator=rng).cuda()
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)

    # 2. 造均衡的 replay indices
    replay_indices = []
    for module in model.modules():
        if isinstance(module, DeepseekV4TopKRouter):
            token_idx = torch.arange(SEQ_LEN, device="cuda")
            replay_indices.append(
                torch.stack(
                    [
                        (token_idx * module.top_k + k) % module.num_experts
                        for k in range(module.top_k)
                    ],
                    dim=-1,
                )
            )
    n_routers = len(replay_indices)
    if rank == 0:
        print(f"  {n_routers} TopKRouter layers, replay shape {list(replay_indices[0].shape)}")

    def set_qat_flags(*, fp4_qat, fp8_qat):
        model.config.fp4_qat = fp4_qat
        model.config.fp8_qat = fp8_qat
        for layer in model.mtp.layers:
            layer.config.fp4_qat = fp4_qat
            layer.config.fp8_qat = fp8_qat

    def run_forward():
        torch.manual_seed(1234)
        with torch.no_grad(), router_replay_ctx(model, replay_indices):
            outputs = model(input_ids=input_ids, position_ids=position_ids)
        assert outputs.mtp_per_depth_h is not None
        assert len(outputs.mtp_per_depth_h) == config.num_nextn_predict_layers
        assert all(torch.isfinite(output).all() for output in outputs.mtp_per_depth_h)
        return torch.log_softmax(outputs.logits.float(), dim=-1)

    logps_fp4_and_fp8 = run_forward()
    set_qat_flags(fp4_qat=True, fp8_qat=False)
    logps_fp4 = run_forward()
    set_qat_flags(fp4_qat=False, fp8_qat=True)
    logps_fp8 = run_forward()
    set_qat_flags(fp4_qat=False, fp8_qat=False)
    logps_off = run_forward()

    fp4_diff = (logps_fp4_and_fp8 - logps_fp8).abs()
    fp8_with_fp4_diff = (logps_fp4_and_fp8 - logps_fp4).abs()
    fp8_diff = (logps_fp8 - logps_off).abs()

    result = {
        "fp4_max_abs_diff": fp4_diff.max().item(),
        "fp4_mean_abs_diff": fp4_diff.mean().item(),
        "fp8_with_fp4_max_abs_diff": fp8_with_fp4_diff.max().item(),
        "fp8_with_fp4_are_identical": torch.equal(logps_fp4_and_fp8, logps_fp4),
        "fp8_max_abs_diff": fp8_diff.max().item(),
        "fp8_mean_abs_diff": fp8_diff.mean().item(),
        "fp8_are_identical": torch.equal(logps_fp8, logps_off),
    }
    if rank == 0:
        print(f"  fp4_max_abs_diff  = {result['fp4_max_abs_diff']:.6f}")
        print(f"  fp4_mean_abs_diff = {result['fp4_mean_abs_diff']:.6f}")
        print(f"  fp8_with_fp4_max_abs_diff = {result['fp8_with_fp4_max_abs_diff']:.6f}")
        print(f"  fp8_max_abs_diff  = {result['fp8_max_abs_diff']:.6f}")
        print(f"  fp8_mean_abs_diff = {result['fp8_mean_abs_diff']:.6f}")

    del model
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return result


class TestQAT(unittest.TestCase):

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

    def test_fp4_qat_and_fp8_qat_change_output(self):
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