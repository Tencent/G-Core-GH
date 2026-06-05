# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import copy
import json
import logging
import os
import sys
from typing import Callable

import setproctitle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.parallel_state import cpu_barrier, init_distributed
from gpatch_v4.trainer.grpo_actor import train_grpo_actor
from gpatch_v4.trainer.grpo_rm import run_grpo_rm
from gpatch_v4.trainer.grpo_sampler import run_grpo_sampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#TODO:
# 将 auto_place.py 生成的逻辑放到这个进程上面来做？
def load_place_config(path):
    config_path = os.path.join(path, 'config.json')
    with open(config_path, 'r') as inf:
        place_config = json.load(inf)
    return place_config


def init_from_place_config(config):
    place_config = load_place_config("place-config")

    # sampler 相关
    sampler_ips = [x['ip'] for x in place_config['sampler']['rpc_servers']]
    sampler_ports = [str(x['port']) for x in place_config['sampler']['rpc_servers']]
    sampler_dist_init_addrs = [str(x['dist_init_addr']) for x in place_config['sampler']['rpc_servers']]
    sampler_tp_size = place_config['sampler']['tp_size']
    sampler_pp_size = place_config['sampler']['pp_size']
    sampler_ep_size = place_config['sampler']['ep_size']

    config.sampler_config.dist_init_addrs = sampler_dist_init_addrs
    config.sampler_config.sampler_ips = sampler_ips
    config.sampler_config.sampler_ports = sampler_ports
    config.sampler_config.infer_engine.tp_size = sampler_tp_size
    config.sampler_config.infer_engine.pp_size = sampler_pp_size
    config.sampler_config.infer_engine.ep_size = sampler_ep_size
    config.sampler_config.sampler_mp_size = sampler_tp_size * sampler_pp_size
    config.sampler_config.sampler_world_size = len(sampler_ports) * sampler_tp_size * sampler_pp_size

    # rm server
    rm_ips = [x['ip'] for x in place_config['critic']['rpc_servers']]
    rm_ports = [str(x['port']) for x in place_config['critic']['rpc_servers']]
    rm_tp_size = place_config['critic']['tp_size']
    rm_pp_size = place_config['critic']['pp_size']
    config.bt_reward_config.reward_ips = rm_ips
    config.bt_reward_config.reward_ports = rm_ports
    config.bt_reward_config.tp_size = rm_tp_size
    config.bt_reward_config.pp_size = rm_pp_size

    actor_ips = [x for x in place_config['actor']['ips']]
    config.policy_config.sampler_client.endpoint_ips = sampler_ips
    config.policy_config.sampler_client.endpoint_ports = sampler_ports
    config.policy_config.bt_reward_client.endpoint_ips = rm_ips
    config.policy_config.bt_reward_client.endpoint_ports = rm_ports
    config.monitor_config.monitor_server_ip = actor_ips[0]


def validate_and_prepare_config(config):
    #TODO: 根据 len(dataset) 来推算，测试阶段 hard code
    config.train_config.ppo_step_per_epoch = 100
    config.train_config.total_ppo_step = config.train_config.ppo_step_per_epoch * \
        config.train_config.num_train_epoches

    assert config.train_config.rollout_mbs == 1, f"rollout_mbs {config.train_config.rollout_mbs} must be 1"

    if config.train_config.role == "actor":
        pass
    elif config.train_config.role == "sampler":
        # sampler 使用 tp_size 等值覆盖一把
        # gen_rm 也一样
        config.dist_config.tensor_model_parallel_size = \
            config.sampler_config.infer_engine.tp_size
        config.dist_config.pipeline_model_parallel_size = \
            config.sampler_config.infer_engine.pp_size
        config.dist_config.expert_model_parallel_size = 1
        config.dist_config.expert_tensor_parallel_size = 1
        config.dist_config.context_parallel_size = 1
        config.dist_config.use_tp_pp_dp_mapping = True
        #TODO: more
    elif config.train_config.role == "rm":
        config.dist_config.tensor_model_parallel_size = \
            config.bt_reward_config.tp_size
        config.dist_config.pipeline_model_parallel_size = \
            config.bt_reward_config.pp_size
        config.dist_config.expert_model_parallel_size = 1
        config.dist_config.expert_tensor_parallel_size = 1
        config.dist_config.context_parallel_size = 1
    elif config.train_config.role == "gen_rm":
        pass
    else:
        raise ValueError(f"Invalid role: {config.train_config.role}")


def launcher_worker(meta_info, config, test_hook_func):
    role = meta_info["role"]
    rank = meta_info["rank"]
    config.train_config.role = role
    setproctitle.setproctitle(f"python grpo_{role} {rank=}")

    init_distributed(meta_info)
    cpu_barrier()

    validate_and_prepare_config(config)
    cpu_barrier()

    if role == "actor":
        train_grpo_actor(config, test_hook_func)
    elif role == "sampler":
        run_grpo_sampler(config, test_hook_func)
    elif role == "rm":
        run_grpo_rm(config, test_hook_func)


def clear_env():
    for var in ['TORCHELASTIC_USE_AGENT_STORE']:
        os.environ.pop(var, None)
    os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"


def launch(config: RlConfig, test_hook_func: Callable):
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])

    if rank == 0:
        logging.info(f"Starting main process {local_rank=} {rank=} {world_size=} {master_addr=} {master_port=}")

    # default master_port 29500
    ports = [29501, 29502, 29503]
    roles = ["actor", "sampler", "rm"]
    # 动态分配：ports = [get_free_port() for _ in range(3)]

    clear_env()

    init_from_place_config(config)

    mp.set_start_method('spawn', force=True)

    meta_info = {
        "role_master_addr": master_addr,
        "role_master_port": master_port,
        "rank": rank,
        "world_size": world_size,
        "role": None,
        "torch_dist_timeout_minutes": config.dist_config.torch_dist_timeout_minutes,
    }

    processes = []
    for i in range(len(ports)):
        meta_info_ = copy.deepcopy(meta_info)
        meta_info_["role_master_port"] = ports[i]
        meta_info_["role"] = roles[i]
        process = mp.Process(name=roles[i], target=launcher_worker, args=(meta_info_, config, test_hook_func))
        processes.append(process)

    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    launch()
