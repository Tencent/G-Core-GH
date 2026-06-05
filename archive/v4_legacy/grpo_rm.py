# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import logging
import os
import sys
import time
from typing import Any, Dict, List, Union

import torch
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
from gpatch_v4.reward import RewardFactory
from gpatch_v4.trainer.base_server import BaseServer
from gpatch_v4.trainer.helper import RlTokeinizerMixin
from gpatch_v4.utils import clear_memory, log, logging_memory_tracking, logging_rank0


class RewardServer(BaseServer, RlTokeinizerMixin):
    def __init__(self, endpoint_ip: str, endpoint_port: int, config: RlConfig, legacy_reward_hook=None):
        super().__init__(endpoint_ip, endpoint_port, config, legacy_hook=legacy_reward_hook)
        self.build_tokenizer()
        self.tokenizer = self.get_rm_tokenizer()
        self.reward_engine = None
        self.computed = False
        self.batching_reqs: List[Dict[str, Union[int, List[Any]]]] = []
        self.infer_rm_critic_results: Dict[int, Dict[int, Dict]] = {}
        """
        self.infer_rm_critic_results 格式如： [dp_rank: ppo_step: Dict]
        """

    def bcast_between_mp_group(self, cmd):
        # sometimes call bcast prevent  worker timeout
        cmd_obj = [{'cmd': cmd}]
        torch.distributed.broadcast_object_list(
            cmd_obj, src=get_model_parallel_src_rank_gloo(), group=get_model_parallel_group_gloo()
        )

    def _register_routes(self, app):
        super()._register_routes(app)

        #TODO:
        @app.post("/setup")
        @once_rpc(**self.monitor_kwargs)
        async def setup(req_dict):
            assert is_mp_head()
            log(f"Reward setup {torch.distributed.get_rank()}")
            async with self.lock:
                assert self.reward_engine is None
                self.bcast_between_mp_group("setup")
                self.reward_engine = RewardFactory.get_reward_engine(self.config, self.tokenizer)
            assert self.reward_engine is not None
            logging_memory_tracking("memory tracking after reward setup", rank=0)
            return {"ret": "ok"}

        @app.post("/wake_up")
        @once_rpc(**self.monitor_kwargs)
        async def wake_up(req_dict):
            async with self.lock:
                self.bcast_between_mp_group("wake_up")
                assert self.reward_engine is not None
                self.reward_engine.wake_up()
            return {"ret": "ok"}

        @app.post("/sleep")
        @once_rpc(**self.monitor_kwargs)
        async def sleep(req_dict):
            async with self.lock:
                self.bcast_between_mp_group("sleep")
                assert self.reward_engine is not None
                self.reward_engine.sleep()
            clear_memory()
            return {"ret": "ok"}

        @app.post("/mark_ppo_step_begin")
        @once_rpc(**self.monitor_kwargs)
        async def mark_ppo_step_begin(req_dict):
            async with self.lock:
                self.computed = False
                self.bcast_between_mp_group("mark_ppo_step_begin")
                assert self.reward_engine is not None
                self.reward_engine.wake_up()

            return {"ret": "ok"}

        @app.post("/mark_ppo_step_end")
        @once_rpc(**self.monitor_kwargs)
        async def mark_ppo_step_begin(req_dict):
            async with self.lock:
                self.bcast_between_mp_group("mark_ppo_step_end")
                assert self.reward_engine is not None
                self.reward_engine.sleep()
            return {"ret": "ok"}

        @app.post("/issue_infer_rm")
        @once_rpc(**self.monitor_kwargs)
        async def issue_infer_rm(req_dict):
            assert self.computed == False

            self.batching_reqs.append(req_dict)
            #TODO: report_ppo_metrics
            return {"ret": "ok"}

        @app.post("/get_infer_rm_result")
        @once_rpc(**self.monitor_kwargs)
        async def get_infer_rm_result(req_dict):
            actor_dp_rank = req_dict['actor_dp_rank']
            ppo_step = req_dict['ppo_step']
            sample_idx = req_dict['sample_idx']
            sampling_repeat_n = req_dict['sampling_repeat_n']

            async with self.lock:
                if not self.computed:
                    self.computed = True
                    for batch in self.batching_reqs:
                        assert batch["ppo_step"] == ppo_step

                    self.bcast_between_mp_group("get_infer_rm_critic_result")
                    resp_dicts = self.reward_engine.compute_rewards(self.batching_reqs, sampling_repeat_n)

                    for batch, b_resp_dict in zip(self.batching_reqs, resp_dicts, strict=True):
                        b_actor_dp_rank = batch['actor_dp_rank']
                        b_sample_idx = batch['sample_idx']

                        tmpd = self.infer_rm_critic_results.setdefault(b_actor_dp_rank, {})
                        tmpdd = tmpd.setdefault(ppo_step, {})
                        tmpdd[b_sample_idx] = b_resp_dict
                    self.batching_reqs = []

            tmpd = self.infer_rm_critic_results[actor_dp_rank][ppo_step]
            resp_dict = tmpd[sample_idx]
            return resp_dict


class RewardWorker(RlTokeinizerMixin):
    def __init__(self, config: RlConfig, legacy_reward_hook=None):
        self.config = config
        self.build_tokenizer()
        self.tokenizer = self.get_rm_tokenizer()
        self.reward_engine = None
        self.legacy_reward_hook = legacy_reward_hook

    def run_grpo_rm_worker_forever(self):
        while True:
            cmd_obj = [None]
            torch.distributed.broadcast_object_list(
                cmd_obj, src=get_model_parallel_src_rank_gloo(), group=get_model_parallel_group_gloo()
            )
            cmd = cmd_obj[0]['cmd']

            if cmd == 'setup':
                assert self.reward_engine is None
                self.reward_engine = RewardFactory.get_reward_engine(self.config, self.tokenizer)
                assert self.reward_engine is not None
            elif cmd in ['wake_up', 'mark_ppo_step_begin']:
                assert self.reward_engine is not None
                self.reward_engine.wake_up()
            elif cmd in ['sleep', 'mark_ppo_step_end']:
                assert self.reward_engine is not None
                self.reward_engine.sleep()
            elif cmd == 'get_infer_rm_critic_result':
                self.reward_engine.compute_rewards(None, None)
            else:
                raise ValueError(f"Unknown cmd: {cmd}")
            time.sleep(2)


def run_grpo_rm_server(config: RlConfig, legacy_reward_hook=None):
    ep_ip = config.bt_reward_config.reward_ips[mpu.get_data_parallel_rank()]
    ep_port = config.bt_reward_config.reward_ports[mpu.get_data_parallel_rank()]

    server = RewardServer(
        endpoint_ip=ep_ip, endpoint_port=ep_port, config=config, legacy_reward_hook=legacy_reward_hook
    )
    server.serve_forever(config.bt_reward_config.server_timeout_keep_alive)


def run_grpo_rm_worker(config: RlConfig, legacy_reward_hook=None):
    worker = RewardWorker(
        config=config,
        legacy_reward_hook=legacy_reward_hook,
    )
    worker.run_grpo_rm_worker_forever()


def test_rm(config, test_hook_func):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    tensor = torch.tensor([float(rank)]).to(torch.cuda.current_device())
    gathered = [torch.zeros(1).to(tensor.device) for _ in range(world_size)] if rank == 0 else None
    dist.gather(tensor, gathered if rank == 0 else None, dst=0)
    if rank == 0:
        logging.info(f"RM rank 0 gathered: {[t.item() for t in gathered]}")

    if torch.distributed.get_rank() == 1:
        test_hook_func(config)


#TODO: bt_reward 的 reward，直接走 fsdp backend？ 先不管 moe 的情况
def run_grpo_rm(config: RlConfig, test_hook_func):
    # setup parallel state group
    initlize_parallel_state(config)
    init_pg(config.dist_config)

    message = f"role {config.train_config.role} tp_rank {mpu.get_tensor_model_parallel_rank()} pp_rank {mpu.get_pipeline_model_parallel_rank()}" \
              f" tp_size {mpu.get_tensor_model_parallel_world_size()} pp_size {mpu.get_pipeline_model_parallel_world_size()}" \
              f" dp_rank {mpu.get_data_parallel_rank()} dp_size {mpu.get_data_parallel_world_size()}"

    log(message)
    test_rm(config, test_hook_func)
    cpu_barrier()

    #TODO: init_wecube_reporter

    if is_mp_head():
        run_grpo_rm_server(config, legacy_reward_hook=None)
    else:
        run_grpo_rm_worker(config, legacy_reward_hook=None)
