import os
import time
import asyncio
import unittest
import shutil

import numpy as np
import pytest
import ray
import torch
import pynvml
from PIL import Image

import torch
import torch.distributed.autograd as autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed.rpc as rpc

from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.orches.placement_group import (
    create_gen_rm_group,
    create_placement_groups,
    create_train_group,
)
from gpatch_v4.trainer import T2iGrpoTrainer
from gpatch_v4_test_helper import load_config


class MyModule(nn.Module):
    def __init__(self, device, comm_mode):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(1000, 1000).to(device)
        self.comm_mode = comm_mode

    def forward(self, x):
        # x.to() is a no-op if x is already on self.device
        y = self.linear(x.to(self.device))
        return y.cpu() if self.comm_mode == "cpu" else y

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


def measure(comm_mode):
    # local module on "worker0/cuda:0"
    lm = MyModule("cuda:0", comm_mode)
    # remote module on "worker1/cuda:1"
    rm = rpc.remote("worker1", MyModule, args=("cuda:1", comm_mode))
    # prepare random inputs
    x = torch.randn(1000, 1000).cuda(0)

    tik = time.time()
    for _ in range(10):
        with autograd.context() as ctx:
            y = rm.rpc_sync().forward(lm(x))
            autograd.backward(ctx, [y.sum()])
    # synchronize on "cuda:0" to make sure that all pending CUDA ops are
    # included in the measurements
    torch.cuda.current_stream("cuda:0").synchronize()
    tok = time.time()
    print(f"{comm_mode} RPC total execution time: {tok - tik}")


def run_worker(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    if rank == 0:
        options.set_device_map("worker1", {0: 1})
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=2, rpc_backend_options=options)
        measure(comm_mode="cpu")
        measure(comm_mode="cuda")
    else:
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=2, rpc_backend_options=options)

    # block until all rpcs finish
    rpc.shutdown()


def run_worker_2(rank):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '11000'

    if rank == 0:
        rpc.init_rpc("worker0", rank=0, world_size=2)
        ret = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(2), 3))
    else:
        rpc.init_rpc("worker1", rank=1, world_size=2)
    rpc.shutdown()


class T2iGrpoOteam4_4Test(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_placeholder(self):
        """Placeholder test — CUDA RPC tests are commented out (see test_1, test_2)."""
        pass

    '''
    def test_1(self):
        # https://h-huang.github.io/tutorials/recipes/cuda_rpc.html
        world_size = 2
        mp.spawn(run_worker, nprocs=world_size, join=True)

    def test_2(self):
        # https://docs.pytorch.org/docs/stable/rpc.html
        world_size = 2
        mp.spawn(run_worker_2, nprocs=world_size, join=True)
    '''
