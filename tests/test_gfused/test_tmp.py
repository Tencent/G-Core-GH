import os
import gc
import re
import shutil
import time
import unittest

import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM, apply_hp

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
SAVE_DIR = "/work/wepsdl/_test_tmp_save"
NUM_GPUS = 32
EP_SIZE = 4
CP_SIZE = 2
SEQ_LEN = 256
NUM_LAYERS: "int | None" = None


def _truncate_config(config):
    """Truncate to NUM_LAYERS decoder layers for Stage-1 fast iteration."""
    if NUM_LAYERS is not None:
        config.num_hidden_layers = NUM_LAYERS
        config.layer_types = config.layer_types[:NUM_LAYERS]
        config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
    return config


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip():
    """Return the externally-reachable IP of the node this task works on."""
    return ray.util.get_node_ip_address()


@ray.remote(num_gpus=1)
def _tmp_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
):
    """One Ray task per GPU. Runs load → save → reload → compare."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    ep_size = EP_SIZE
    cp_size = CP_SIZE
    assert world_size % ep_size == 0
    assert world_size % cp_size == 0
    ep_2d_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // ep_size, ep_size),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    cp_full_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size // cp_size, cp_size),
        mesh_dim_names=("dp", "cp"),
    )
    cp_mesh = cp_full_mesh["cp"]

    config = DeepseekV4Config.from_pretrained(HF_MODEL_PATH)
    _truncate_config(config)
    assert not config.tie_word_embeddings, (
        "meta-device init may hang with tie_word_embeddings=True"
    )

    print(f"[rank {rank}] phase 1: building meta model + load_checkpoint_hp ...")
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model1 = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    model1 = apply_hp(model1, ep_2d_mesh, cp_mesh=cp_mesh, amp_fp32=False)
    model1.load_checkpoint_hp(HF_MODEL_PATH, dtype=torch.bfloat16)

    if rank == 0 and os.path.isdir(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    dist.barrier()

    print(f"[rank {rank}] phase 2: save_checkpoint_hp ...")
    t0 = time.time()
    model1.save_checkpoint_hp(SAVE_DIR, orig_ckpt_dir=HF_MODEL_PATH)
    dist.barrier()
    t1 = time.time()

    if rank == 0:
        try:
            total_bytes = sum(
                os.path.getsize(os.path.join(SAVE_DIR, f))
                for f in os.listdir(SAVE_DIR)
                if f.endswith(".safetensors")
            )
            print(
                f"[save BENCH] NUM_LAYERS={NUM_LAYERS} world_size={world_size} "
                f"ep={EP_SIZE} took {t1 - t0:.2f}s, "
                f"written {total_bytes / (1024**3):.2f} GiB → "
                f"{total_bytes / max(t1 - t0, 1e-9) / (1024**3):.2f} GiB/s aggregate"
            )
        finally:
            shutil.rmtree(SAVE_DIR, ignore_errors=True)

    dist.barrier()
    dist.destroy_process_group()
    return {"rank": rank, "ok": True, "save_seconds": t1 - t0}



class TestTmp1(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto")

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    @unittest.skip("OOM when cluster GPUs are dirty from prior tests; needs clean GPU")
    def test_tmp_1(self):
        world_size = NUM_GPUS
        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK",
        )
        ray.get(pg.ready())

        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0,
                )
            ).remote()
        )

        futures = [
            _tmp_worker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=r,
                )
            ).remote(
                rank=r,
                world_size=world_size,
                master_addr=master_addr,
                master_port=11000,
            )
            for r in range(world_size)
        ]
        results = ray.get(futures)
        remove_placement_group(pg)
