# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import asyncio
import logging
import os
import pickle
import sys
import time
import uuid
from typing import Any, Dict

import torch
import torch.distributed
import torch.distributed as dist

from megatron.core import mpu

from gpatch.rpc import once_rpc
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    get_model_parallel_group_gloo,
    get_model_parallel_src_rank_gloo,
    init_pg,
    initlize_parallel_state,
    is_mp_head,
)
from gpatch_v4.default_provider.default_infer_engine_provider import default_sampler_model_provider
from gpatch_v4.trainer.base_server import BaseServer
from gpatch_v4.trainer.helper import RlTokeinizerMixin
from gpatch_v4.trainer.sampler_helper import (
    SamplerCallbackHook,
    SamplerGenerateMixin,
    SamplerServerTestMixin,
)
from gpatch_v4.utils import log, logging_memory_tracking, perf_time


class SamplerServer(BaseServer, RlTokeinizerMixin, SamplerGenerateMixin, SamplerServerTestMixin):
    def __init__(
        self, endpoint_ip: str, endpoint_port: int, config: RlConfig, legacy_sampler_hook: SamplerCallbackHook = None
    ):
        super().__init__(endpoint_ip, endpoint_port, config, legacy_hook=legacy_sampler_hook)
        if legacy_sampler_hook is None or legacy_sampler_hook.engine_provider is None:
            self.engine_provider = default_sampler_model_provider
        else:
            self.engine_provider = legacy_sampler_hook.engine_provider
        self.build_tokenizer()
        self.tokenizer = self.get_sampler_tokenizer()

    def bcast_between_mp_group(self, cmd):
        # sometimes call bcast prevent  worker timeout
        cmd_obj = [{'cmd': cmd}]
        torch.distributed.broadcast_object_list(
            cmd_obj, src=get_model_parallel_src_rank_gloo(), group=get_model_parallel_group_gloo()
        )

    def _register_routes(self, app):
        super()._register_routes(app)

        @app.post("/setup")
        @once_rpc(**self.monitor_kwargs)
        async def setup(req_dict):
            assert is_mp_head()
            log(f"Sampler setup {torch.distributed.get_rank()}")
            async with self.lock:
                assert self.infer_engine is None
                self.bcast_between_mp_group(cmd="setup")
                self.infer_engine = self.engine_provider(self.config.sampler_config)
            assert self.infer_engine is not None
            logging_memory_tracking("memory tracking after sampler setup", rank=0)
            return {"ret": "ok"}

        @app.post("/update_weights")
        @once_rpc(**self.monitor_kwargs)
        async def update_weights(req_dict):
            with perf_time("update_weight", rank=0):
                ret = await self.infer_engine.update_weights(**req_dict)
                log(f"trace update weight result {ret}")
                resp_dict = {"update_success": ret[0]}
            return resp_dict

        @app.post("/flush_cache")
        @once_rpc(**self.monitor_kwargs)
        async def flush_cache(req_dict):
            await self.infer_engine.flush_cache()
            return {"ret": True}

        @app.post("/wake_up")
        @once_rpc(**self.monitor_kwargs)
        async def wake_up(req_dict):
            tag_names = req_dict["tag_names"]
            ret = await self.infer_engine.wake_up(tags=tag_names)
            return {"ret": ret}

        @app.post("/sleep")
        @once_rpc(**self.monitor_kwargs)
        async def sleep(req_dict):
            ret = await self.infer_engine.sleep()
            return {"ret": ret}

        @app.post("/generate")
        @once_rpc(**self.monitor_kwargs)
        async def generate(req_dict):
            sampler_dp_rank = req_dict['sampler_dp_rank']
            sampling_repeat_n = req_dict['sampling_repeat']
            batch_data = req_dict['batch']
            assert sampler_dp_rank == mpu.get_data_parallel_rank(
            ), f'{sampler_dp_rank=} != {mpu.get_data_parallel_rank()=}'

            resp: dict = await self.generate_rollouts(batch_data, sampling_repeat_n)
            assert 'ready' not in resp, '`ready` is a reserved name'

            #TODO: add ppo_wecube_report?
            return resp

        @app.post("/mark_ppo_step_begin")
        @once_rpc(**self.monitor_kwargs)
        async def mark_ppo_step_begin(req_dict):
            self.bcast_between_mp_group(cmd="mark_ppo_step_begin")
            tag_names = req_dict["tag_names"]
            async with self.lock:
                ret = await self.infer_engine.wake_up(tags=tag_names)
            return {"ret": ret}

        @app.post("/mark_ppo_step_end")
        @once_rpc(**self.monitor_kwargs)
        async def mark_ppo_step_end(req_dict):
            self.bcast_between_mp_group(cmd="mark_ppo_step_end")
            async with self.lock:
                ret = await self.infer_engine.sleep()
            return {"ret": ret}

        # register test route
        self.register_test_routes(app)


def run_grpo_sampler_server(config, legacy_sampler_hook: SamplerCallbackHook = None):
    #TODO: pick ep ip and port
    ep_ip = config.sampler_config.sampler_ips[mpu.get_data_parallel_rank()]
    ep_port = config.sampler_config.sampler_ports[mpu.get_data_parallel_rank()]

    server = SamplerServer(
        endpoint_ip=ep_ip,
        endpoint_port=ep_port,
        config=config,
        legacy_sampler_hook=legacy_sampler_hook,
    )
    server.serve_forever(config.sampler_config.server_timeout_keep_alive)


def run_grpo_sampler_worker(config, legacy_sampler_hook: SamplerCallbackHook = None):
    infer_engine = None
    while True:
        try:
            cmd_obj = [None]
            torch.distributed.broadcast_object_list(
                cmd_obj, src=get_model_parallel_src_rank_gloo(), group=get_model_parallel_group_gloo()
            )
            cmd_obj = cmd_obj[0]
            if cmd_obj['cmd'] == 'setup':
                if config.sampler_config.infer_engine_impl == 'sglang' \
                    and mpu.get_tensor_model_parallel_rank() % config.dist_config.num_gpus_per_node == 0:
                    assert infer_engine is None, "infer_engine should be None"

                    if legacy_sampler_hook is not None and legacy_sampler_hook.engine_provider is not None:
                        engine_provider = legacy_sampler_hook.engine_provider
                    else:
                        engine_provider = default_sampler_model_provider
                    infer_engine = engine_provider(config.sampler_config)
                    assert infer_engine is not None, "infer_engine should not be None"
            else:
                pass
        except Exception as e:
            logging.error(f"Sampler worker {torch.distributed.get_rank()} error: {e}")

        time.sleep(2)


def test_sampler(config, test_hook_func):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    tensor = torch.tensor([42.0]).to(torch.cuda.current_device()
                                    ) if rank == 0 else torch.tensor([0.0]).to(torch.cuda.current_device())
    dist.broadcast(tensor, src=0)
    logging.info(f"Sampler {torch.distributed.get_rank()} received: {tensor}")

    if torch.distributed.get_rank() == 2:
        test_hook_func(config)


def run_grpo_sampler(config, test_hook_func):
    # setup parallel state group
    initlize_parallel_state(config)
    init_pg(config.dist_config)

    message = f"role {config.train_config.role} tp_rank {mpu.get_tensor_model_parallel_rank()} pp_rank {mpu.get_pipeline_model_parallel_rank()}" \
              f" tp_size {mpu.get_tensor_model_parallel_world_size()} pp_size {mpu.get_pipeline_model_parallel_world_size()}" \
              f" dp_rank {mpu.get_data_parallel_rank()} dp_size {mpu.get_data_parallel_world_size()}"

    log(message)
    test_sampler(config, test_hook_func)
    cpu_barrier()

    #TODO: 兼容之前 hook func 传入的方法，保留外传 provider 的接口
    if is_mp_head():
        run_grpo_sampler_server(config, legacy_sampler_hook=None)
    else:
        run_grpo_sampler_worker(config, legacy_sampler_hook=None)
