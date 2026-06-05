import argparse
import json
import os
import pprint
import subprocess
import sys
import math
import shutil

import socket
import psutil

try:
    import ray
except:
    ray = None

USE_SGLANG = False
try:
    import sglang as sgl
    USE_SGLANG = True
except:
    USE_SGLANG = False


def get_args():
    parser = argparse.ArgumentParser(description='placement calculator')
    parser.add_argument(
        '--fn',
        type=str,
        required=True,
        choices=['gen', 'get', 'init_ray', 'gen_and_init_ray'],
    )
    parser.add_argument(
        '--get-fn',
        type=str,
        choices=[
            'sampler-nnodes',
            'sampler-master-addr',
            'sampler-svr-ips',
            'sampler-svr-ports',
            'sampler-node-ips',
            'sampler-dist-init-addrs',
            'gen-rm-nnodes',
            'gen-rm-master-addr',
            'gen-rm-svr-ips',
            'gen-rm-svr-ports',
            'gen-rm-node-ips',
            'gen-rm-dist-init-addrs',
            'critic-nnodes',
            'critic-master-addr',
            'critic-svr-ips',
            'critic-svr-ports',
            'critic-node-ips',
            'actor-nnodes',
            'actor-master-addr',
            'actor-svr-ips',
            'actor-svr-ports',
            'actor-node-ips',
            'actor-tp-size',
            'actor-pp-size',
            'actor-cp-size',
            'actor-ep-size',
            'actor-etp-size',
            'sampler-tp-size',
            'sampler-pp-size',
            'sampler-ep-size',
            'critic-tp-size',
            'critic-pp-size',
            'critic-ep-size',
            'critic-cp-size',
            'critic-etp-size',
            'gen-rm-tp-size',
            'gen-rm-pp-size',
            'gen-rm-ep-size',
            'hetero-gen-rm-num',
        ],
    )
    parser.add_argument(
        '--schema',
        type=str,
        default='colocate-all',
        choices=['colocate-all', 'disjoint', 'ppo-schema-1', 'ppo-gen-rms-schema-1'],
        help='''摆放方案：
        1. colocate-all 全部一起摆放
        2. disjoint 全部独立
        3. ...''',
    )
    parser.add_argument(
        '--interface-name',
        type=str,
        default='bond1',
        help='network adapters interface',
    )
    parser.add_argument('--nnodes', type=int, default=1, help='nnodes')
    parser.add_argument('--num-gpus-per-node', type=int, default=8, help='num gpus per node')
    parser.add_argument('--actor-nnodes', type=int, default=1, help='actor nnodes')
    parser.add_argument('--critic-nnodes', type=int, default=1, help='critic nnodes')
    parser.add_argument('--sampler-nnodes', type=int, default=1, help='sampler nnodes')
    parser.add_argument('--gen-rm-nnodes', type=int, default=1, help='gen rm nnodes')
    parser.add_argument(
        '--config-folder', type=str, default='place-config', help='folder of config files'
    )
    parser.add_argument('--sampler-tp-size', type=int, default=1)
    parser.add_argument('--sampler-pp-size', type=int, default=1)
    parser.add_argument('--sampler-ep-size', type=int, default=1)
    parser.add_argument('--gen-rm-tp-size', type=int, default=1)
    parser.add_argument('--gen-rm-pp-size', type=int, default=1)
    parser.add_argument('--gen-rm-ep-size', type=int, default=1)
    parser.add_argument('--hetero-gen-rm-num', type=int, default=1)
    parser.add_argument('--hetero-gen-rm-tp-sizes', type=int, nargs='*', default=[])
    parser.add_argument('--hetero-gen-rm-pp-sizes', type=int, nargs='*', default=[])
    parser.add_argument('--hetero-gen-rm-ep-sizes', type=int, nargs='*', default=[])
    parser.add_argument('--get-fn-hetero-gen-rm-idx', type=int, default=None)
    parser.add_argument('--critic-tp-size', type=int, default=1)
    parser.add_argument('--critic-pp-size', type=int, default=1)
    parser.add_argument('--critic-cp-size', type=int, default=1)
    parser.add_argument('--critic-ep-size', type=int, default=1)
    parser.add_argument('--critic-etp-size', type=int, default=1)
    parser.add_argument('--actor-tp-size', type=int, default=1)
    parser.add_argument('--actor-pp-size', type=int, default=1)
    parser.add_argument('--actor-cp-size', type=int, default=1)
    parser.add_argument('--actor-ep-size', type=int, default=1)
    parser.add_argument('--actor-etp-size', type=int, default=1)
    parser.add_argument(
        '--sampler-begin-port',
        type=int,
        default=61000,
        help='task worker i will occupy port `x + i`'
    )
    parser.add_argument(
        '--sampler-dist-init-begin-port',
        type=int,
        default=61100,
        help='task worker i will occupy port `x + i`'
    )
    parser.add_argument(
        '--gen-rm-begin-port',
        type=int,
        default=61200,
        help='task worker i will occupy port `x + i`'
    )
    parser.add_argument(
        '--gen-rm-dist-init-begin-port',
        type=int,
        default=61300,
        help='task worker i will occupy port `x + i`'
    )
    parser.add_argument(
        '--critic-begin-port',
        type=int,
        default=61400,
        help='task worker i will occupy port `x + i`'
    )
    parser.add_argument(
        '--actor-begin-port',
        type=int,
        default=61500,
        help='task worker i will occupy port `x + i`'
    )
    parser.add_argument('--ray-port', type=int, default=6379, help='ray start port')
    parser.add_argument(
        '--no-patch-sglang', action='store_false', dest='patch_sglang', help='disable sglang patch'
    )

    args = parser.parse_args()
    return args


def validate_args(args):
    if args.fn == 'gen':
        if args.hetero_gen_rm_num <= 0:
            raise ValueError('hetero_gen_rm_num must be greater than 0')
        elif args.hetero_gen_rm_num == 1:
            # 兼容以前的参数逻辑
            args.hetero_gen_rm_tp_sizes = [args.gen_rm_tp_size]
            args.hetero_gen_rm_pp_sizes = [args.gen_rm_pp_size]
            args.hetero_gen_rm_ep_sizes = [args.gen_rm_ep_size]
        if args.hetero_gen_rm_num != len(
            args.hetero_gen_rm_tp_sizes
        ) or args.hetero_gen_rm_num != len(
            args.hetero_gen_rm_pp_sizes
        ) or args.hetero_gen_rm_num != len(args.hetero_gen_rm_ep_sizes):
            raise ValueError(
                'hetero_gen_rm_num must be equal to len(hetero_gen_rm_tp_sizes) and len(hetero_gen_rm_pp_sizes) and len(hetero_gen_rm_ep_sizes)'
            )


def get_ip_from_interface(interface_name):
    addrs = psutil.net_if_addrs()
    if interface_name in addrs:
        for addr in addrs[interface_name]:
            if addr.family == 2:  # AF_INET
                return addr.address
    return None


def get_hostnames():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # 这个 env 只能在 wxg gemini work
    hostname = os.environ.get('POD_NAME', None)
    if hostname is None:
        hostname = socket.gethostname()
    hostnames = comm.allgather(hostname)
    return hostnames


def get_node_ips(args):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # 这个 env 只能在 wxg gemini work
    ip = os.environ.get('__HOST_IP__', None)
    if ip is None:
        ip = get_ip_from_interface(args.interface_name)
    ips = comm.allgather(ip)
    return ips


def get_hostname_to_ip_map(args):
    hostnames = get_hostnames()
    ips = get_node_ips(args)
    return dict((x, y) for x, y in zip(hostnames, ips))


def get_vllm_config(
    args,
    ips,
    hostnames,
    tp_size,
    pp_size,
    server_begin_port,
    dist_init_begin_port,
    cp_size=1,
    ep_size=1
):
    # 对于 vllm 实例，暂时强制 cp_size 和 ep_size 为 1
    assert cp_size == 1
    n_gpus = len(ips) * args.num_gpus_per_node
    mp_size = tp_size * pp_size
    assert n_gpus % mp_size == 0
    dp_size = n_gpus // mp_size

    gpu_ips = [ip for ip in ips for _ in range(args.num_gpus_per_node)]
    svr_ports = [server_begin_port + i % args.num_gpus_per_node for i in range(dp_size)]
    dist_init_ports = [dist_init_begin_port + i % args.num_gpus_per_node for i in range(dp_size)]

    rpc_servers = []
    for dp_rank in range(dp_size):
        server = dict(
            dp_rank=dp_rank,
            ip=gpu_ips[dp_rank * mp_size],
            port=svr_ports[dp_rank],
            dist_init_addr=f'{gpu_ips[dp_rank * mp_size]}:{dist_init_ports[dp_rank]}'
        )
        rpc_servers.append(server)
    topo_config = dict(
        ips=ips,
        hostnames=hostnames,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        rpc_servers=rpc_servers,
    )
    return topo_config


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


# vllm 的 topo layout 和 megatron 不一样
def get_mlm_config(
    args,
    ips,
    hostnames,
    tp_size,
    pp_size,
    cp_size,
    server_begin_port,
    ep_size=1,
    etp_size=1,
    is_critic=False
):
    n_gpus = len(ips) * args.num_gpus_per_node
    mp_and_cp_size = tp_size * pp_size * cp_size
    assert n_gpus % mp_and_cp_size == 0
    dp_size = n_gpus // mp_and_cp_size

    gpu_ips = [ip for ip in ips for _ in range(args.num_gpus_per_node)]
    svr_ports = [server_begin_port + i % args.num_gpus_per_node for i in range(dp_size)]

    rpc_servers = []
    for dp_rank in range(dp_size):
        # TODO 暂时假定 tp-cp-dp-pp 的顺序
        server = dict(
            dp_rank=dp_rank, ip=gpu_ips[dp_rank * (tp_size * cp_size)], port=svr_ports[dp_rank]
        )
        rpc_servers.append(server)

    if False and is_critic and ep_size > 1:
        # 测试中，暂时不进到这个分支
        assert ep_size % 2 == 0 and dp_size % 2 == 0, f"please use a number of nodes that is a power of 2"
        # ep_size > 1, 要求 tp0 cp0 pp0 ep0 才启 server
        tp_cp_size = tp_size * cp_size
        max_server_size = lcm(tp_cp_size, ep_size)
        server_num = dp_size * tp_size * cp_size / max_server_size
        svr_ports = [server_begin_port + i % args.num_gpus_per_node for i in range(server_num)]

        # 从数学来讲，server_num 只能是 dp_size 的约数
        shard_off = dp_size // server_num

        rpc_servers = []
        for si in range(server_num):
            # TODO 暂时假定 tp-cp-dp-pp 的顺序
            server = dict(
                dp_rank=si * shard_off, ip=gpu_ips[si * max_server_size], port=svr_ports[si]
            )
            rpc_servers.append(server)

    topo_config = dict(
        ips=ips,
        hostnames=hostnames,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        etp_size=etp_size,
        rpc_servers=rpc_servers
    )
    return topo_config


def gen(args):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    hostnames = get_hostnames()
    ips = get_node_ips(args)

    if args.schema == 'colocate-all':
        s_ips = ips
        s_hostnames = hostnames
    elif args.schema == 'disjoint':
        off = 0
        n = args.sampler_nnodes
        s_ips = ips[off:off + n]
        s_hostnames = hostnames[off:off + n]
    else:
        raise NotImplementedError('')
    sampler_config = get_vllm_config(
        args,
        s_ips,
        s_hostnames,
        args.sampler_tp_size,
        args.sampler_pp_size,
        args.sampler_begin_port,
        args.sampler_dist_init_begin_port,
        ep_size=args.sampler_ep_size,
    )

    if args.schema == 'colocate-all':
        gr_ips = ips
        gr_hostnames = hostnames
    elif args.schema == 'disjoint':
        off += n
        n = args.gen_rm_nnodes
        gr_ips = ips[off:off + n]
        gr_hostnames = hostnames[off:off + n]
    else:
        raise NotImplementedError('')

    gen_rm_config_list = []
    for i in range(args.hetero_gen_rm_num):
        # 100 / 8 = 12 应该用不到 12 个异构 gen-rm
        gen_rm_config_list.append(
            get_vllm_config(
                args,
                gr_ips,
                gr_hostnames,
                args.hetero_gen_rm_tp_sizes[i],
                args.hetero_gen_rm_pp_sizes[i],
                args.gen_rm_begin_port + i * args.num_gpus_per_node,
                args.gen_rm_dist_init_begin_port + i * args.num_gpus_per_node,
                ep_size=args.hetero_gen_rm_ep_sizes[i]
            )
        )

    gen_rm_config = {
        "hetero-gen-rm-num": args.hetero_gen_rm_num,
        "gen-rm-config": gen_rm_config_list
    }

    if args.schema == 'colocate-all':
        c_ips = ips
        c_hostnames = hostnames
    elif args.schema == 'disjoint':
        off += n
        n = args.critic_nnodes
        c_ips = ips[off:off + n]
        c_hostnames = hostnames[off:off + n]
    else:
        raise NotImplementedError('')
    assert args.critic_cp_size == 1
    assert args.critic_etp_size == 1
    critic_config = get_mlm_config(
        args,
        c_ips,
        c_hostnames,
        args.critic_tp_size,
        args.critic_pp_size,
        args.critic_cp_size,
        args.critic_begin_port,
        ep_size=args.critic_ep_size,
        etp_size=args.critic_etp_size,
        is_critic=True
    )

    if args.schema == 'colocate-all':
        a_ips = ips
        a_hostnames = hostnames
    elif args.schema == 'disjoint':
        off += n
        n = args.actor_nnodes
        a_ips = ips[off:off + n]
        a_hostnames = hostnames[off:off + n]
    else:
        raise NotImplementedError('')
    actor_config = get_mlm_config(
        args,
        a_ips,
        a_hostnames,
        args.actor_tp_size,
        args.actor_pp_size,
        args.actor_cp_size,
        args.actor_begin_port,
        ep_size=args.actor_ep_size,
        etp_size=args.actor_etp_size
    )
    ray_config = dict(port=args.ray_port)

    config = {
        'args': args.__dict__,
        'sampler': sampler_config,
        'gen-rm': gen_rm_config,
        'critic': critic_config,
        'actor': actor_config,
        'ray': ray_config,
    }
    if rank != 0:
        return

    os.makedirs(args.config_folder, exist_ok=True)
    config_path = os.path.join(args.config_folder, 'config.json')
    with open(config_path, 'w') as outf:
        json.dump(config, outf, indent=4)
    with open(os.path.join(args.config_folder, 'sampler.hostfile'), 'w') as outf:
        for hostname in sampler_config['hostnames']:
            outf.write(f'{hostname} slots=1\n')
    with open(os.path.join(args.config_folder, 'gen-rm.hostfile'), 'w') as outf:
        for hostname in sampler_config['hostnames']:
            outf.write(f'{hostname} slots=1\n')
    with open(os.path.join(args.config_folder, 'critic.hostfile'), 'w') as outf:
        for hostname in critic_config['hostnames']:
            outf.write(f'{hostname} slots=1\n')
    with open(os.path.join(args.config_folder, 'actor.hostfile'), 'w') as outf:
        for hostname in actor_config['hostnames']:
            outf.write(f'{hostname} slots=1\n')


def get(args):
    config_path = os.path.join(args.config_folder, 'config.json')
    with open(config_path, 'r') as inf:
        config = json.load(inf)

    if args.get_fn in [
        'gen-rm-nnodes', 'gen-rm-master-addr', 'gen-rm-svr-ips', 'gen-rm-svr-ports',
        'gen-rm-node-ip', 'gen-rm-dist-init-addrs', 'gen-rm-tp-size', 'gen-rm-pp-size',
        'gen-rm-ep-size'
    ]:
        if config['gen-rm']['hetero-gen-rm-num'] == 1:
            args.get_fn_hetero_gen_rm_idx = 0
        elif args.get_fn_hetero_gen_rm_idx is None:
            raise ValueError(
                f"hetero-gen-rm-num = {config['gen-rm']['hetero-gen-rm-num']}, but hetero-gen-rm-idx is not specified"
            )

    if args.get_fn == 'sampler-nnodes':
        print(len(config['sampler']['ips']))
    elif args.get_fn == 'sampler-master-addr':
        print(config['sampler']['ips'][0])
    elif args.get_fn == 'sampler-svr-ips':
        ips = [x['ip'] for x in config['sampler']['rpc_servers']]
        print(' '.join(ips))
    elif args.get_fn == 'sampler-svr-ports':
        ports = [str(x['port']) for x in config['sampler']['rpc_servers']]
        print(' '.join(ports))
    elif args.get_fn == 'sampler-node-ips':
        ips = [x for x in config['sampler']['ips']]
        print(' '.join(ips))
    elif args.get_fn == 'sampler-dist-init-addrs':
        addrs = [str(x['dist_init_addr']) for x in config['sampler']['rpc_servers']]
        print(' '.join(addrs))

    elif args.get_fn == 'gen-rm-nnodes':
        print(len(config['gen-rm']['gen-rm-config'][args.get_fn_hetero_gen_rm_idx]['ips']))
    elif args.get_fn == 'gen-rm-master-addr':
        print(config['gen-rm']['gen-rm-config'][args.get_fn_hetero_gen_rm_idx]['ips'][0])
    elif args.get_fn == 'gen-rm-svr-ips':
        ips = [
            x['ip']
            for x in config['gen-rm']['gen-rm-config'][args.get_fn_hetero_gen_rm_idx]['rpc_servers']
        ]
        print(' '.join(ips))
    elif args.get_fn == 'gen-rm-svr-ports':
        ports = [
            str(x['port'])
            for x in config['gen-rm']['gen-rm-config'][args.get_fn_hetero_gen_rm_idx]['rpc_servers']
        ]
        print(' '.join(ports))
    elif args.get_fn == 'gen-rm-node-ips':
        ips = [x for x in config['gen-rm']['gen-rm-config'][args.get_fn_hetero_gen_rm_idx]['ips']]
        print(' '.join(ips))
    elif args.get_fn == 'gen-rm-dist-init-addrs':
        addrs = [
            str(x['dist_init_addr'])
            for x in config['gen-rm']['gen-rm-config'][args.get_fn_hetero_gen_rm_idx]['rpc_servers']
        ]
        print(' '.join(addrs))

    elif args.get_fn == 'critic-nnodes':
        print(len(config['critic']['ips']))
    elif args.get_fn == 'critic-master-addr':
        print(config['critic']['ips'][0])
    elif args.get_fn == 'critic-svr-ips':
        ips = [x['ip'] for x in config['critic']['rpc_servers']]
        print(' '.join(ips))
    elif args.get_fn == 'critic-svr-ports':
        ports = [str(x['port']) for x in config['critic']['rpc_servers']]
        print(' '.join(ports))
    elif args.get_fn == 'critic-node-ips':
        ips = [x for x in config['critic']['ips']]
        print(' '.join(ports))

    elif args.get_fn == 'actor-nnodes':
        print(len(config['actor']['ips']))
    elif args.get_fn == 'actor-master-addr':
        print(config['actor']['ips'][0])
    elif args.get_fn == 'actor-svr-ips':
        ips = [x['ip'] for x in config['actor']['rpc_servers']]
        print(' '.join(ips))
    elif args.get_fn == 'actor-svr-ports':
        ports = [str(x['port']) for x in config['actor']['rpc_servers']]
        print(' '.join(ports))
    elif args.get_fn == 'actor-node-ips':
        ips = [x for x in config['actor']['ips']]
        print(' '.join(ips))

    elif args.get_fn == 'actor-tp-size':
        print(config['actor']['tp_size'])
    elif args.get_fn == 'actor-pp-size':
        print(config['actor']['pp_size'])
    elif args.get_fn == 'actor-cp-size':
        print(config['actor']['cp_size'])
    elif args.get_fn == 'actor-ep-size':
        print(config['actor']['ep_size'])
    elif args.get_fn == 'actor-etp-size':
        print(config['actor']['etp_size'])
    elif args.get_fn == 'critic-tp-size':
        print(config['critic']['tp_size'])
    elif args.get_fn == 'critic-pp-size':
        print(config['critic']['pp_size'])
    elif args.get_fn == 'critic-cp-size':
        print(config['critic']['cp_size'])
    elif args.get_fn == 'critic-ep-size':
        print(config['critic']['ep_size'])
    elif args.get_fn == 'critic-etp-size':
        print(config['critic']['etp_size'])
    elif args.get_fn == 'sampler-tp-size':
        print(config['sampler']['tp_size'])
    elif args.get_fn == 'sampler-pp-size':
        print(config['sampler']['pp_size'])
    elif args.get_fn == 'sampler-ep-size':
        print(config['sampler']['ep_size'])
    elif args.get_fn == 'gen-rm-tp-size':
        print(config['gen-rm']['gen-rm-config'][args.get_fn_hetero_gen_rm_idx]['tp_size'])
    elif args.get_fn == 'gen-rm-pp-size':
        print(config['gen-rm']['gen-rm-config'][args.get_fn_hetero_gen_rm_idx]['pp_size'])
    elif args.get_fn == 'gen-rm-ep-size':
        print(config['gen-rm']['gen-rm-config'][args.get_fn_hetero_gen_rm_idx]['ep_size'])
    elif args.get_fn == 'hetero-gen-rm-num':
        print(config['gen-rm']['hetero-gen-rm-num'])

    else:
        raise ValueError(f'unknown get_fn {args.get_fn}')


# ray up is not usable in wxg gemini cluster due to permission related issues
# assuming tp-pp-dp order for vllm / sglang
def init_ray(args):
    if ray is None:
        return
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    node_rank = comm.Get_rank()

    config_path = os.path.join(args.config_folder, 'config.json')
    with open(config_path, 'r') as inf:
        config = json.load(inf)

    if config['args']['schema'] == 'disjoint':
        if node_rank >= config['args']['sampler_nnodes']:
            return

    cmd = f'ray stop --force'
    subprocess.run(cmd, shell=True, check=True)

    ips = [x['ip'] for x in config['sampler']['rpc_servers']]

    sampler_mp_size = config['args']['sampler_tp_size'] * config['args']['sampler_pp_size']
    if sampler_mp_size <= config['args']['num_gpus_per_node']:
        is_head = True
    else:
        assert sampler_mp_size % config['args']['num_gpus_per_node'] == 0, 'invalid mp size'
        is_head = node_rank * config['args']['num_gpus_per_node'] % sampler_mp_size == 0
    head_ip = ips[node_rank * config['args']['num_gpus_per_node'] // sampler_mp_size]

    ray_port = config['ray']['port']
    if is_head:
        cmd = f'ray start --head --node-ip-address {head_ip} --port {ray_port} --disable-usage-stats'
    else:
        cmd = f'ray start --address="{head_ip}:{ray_port}"'

    print(f'init_ray {node_rank=} {cmd}')
    subprocess.run(cmd, shell=True, check=True)


def temporary_patch_sglang():
    # schedule 修改源于 sglang 0.4.10.post2 权重更新需要，修改了 sglang 的两个函数
    # 以及修复 DeepSeek-V3 update_weights_from_disk error
    print(f"patch sglang", flush=True)

    import sglang
    import sglang.srt.managers.tp_worker as tp_worker
    import sglang.srt.model_executor.model_runner as model_runner
    import sglang.srt.sampling.penaltylib.orchestrator as orchestrator
    import sglang.srt.entrypoints.engine as engine
    try:
        import sglang.srt.layers.quantization.fp8 as fp8
    except ImportError:
        fp8 = None
    try:
        import sglang.srt.models.gpt_oss as gpt_oss
    except ImportError:
        gpt_oss = None

    try:
        import sglang.srt.mem_cache.memory_pool as memory_pool
    except ImportError:
        memory_pool = None

    def patch_func(lib_obj, modified_file_path):
        path_org = lib_obj.__file__
        path_bak = path_org + ".bak"

        if not os.path.exists(path_bak):
            shutil.copyfile(path_org, path_bak)

        assert os.path.exists(
            modified_file_path
        ), f"Sglang patch file not find: {modified_file_path}"
        shutil.copyfile(modified_file_path, path_org)

    gcore_path = os.path.dirname(os.path.join(os.path.dirname(__file__), "../"))
    sglang_version = sglang.__version__
    assert sglang_version in [
        '0.4.6.post5', '0.4.10.post2', '0.5.1.post2', '0.5.2', '0.5.3', '0.5.4.post3', '0.5.7',
        '0.5.9'
    ], f"{sglang_version}"

    patch_func(
        tp_worker,
        modified_file_path=f"{gcore_path}/mpatch/sglang-{sglang_version}/srt/managers/tp_worker.py"
    )
    patch_func(
        model_runner,
        modified_file_path=
        f"{gcore_path}/mpatch/sglang-{sglang_version}/srt/model_executor/model_runner.py"
    )
    if sglang_version not in ['0.5.7', '0.5.9']:
        patch_func(
            orchestrator,
            modified_file_path=
            f"{gcore_path}/mpatch/sglang-{sglang_version}/srt/sampling/penaltylib/orchestrator.py"
        )
    if sglang_version == '0.5.3':
        # for https://github.com/sgl-project/sglang/blob/v0.5.3/python/sglang/srt/models/deepseek_v2.py#L2843
        from sglang.srt.models import deepseek_v2
        patch_func(
            deepseek_v2,
            modified_file_path=
            f"{gcore_path}/mpatch/sglang-{sglang_version}/srt/models/deepseek_v2.py"
        )

    if sglang_version == '0.4.10.post2':
        assert fp8 is not None, "mistakes in sglang version"
        patch_func(
            fp8,
            modified_file_path=
            f"{gcore_path}/mpatch/sglang-{sglang_version}/srt/layers/quantization/fp8.py"
        )
    elif sglang_version == '0.5.1.post2':
        assert gpt_oss is not None, "mistakes in sglang version"
        patch_func(
            gpt_oss,
            modified_file_path=f"{gcore_path}/mpatch/sglang-{sglang_version}/srt/models/gpt_oss.py"
        )
    elif sglang_version == '0.5.2':
        assert fp8 is not None, "mistakes in sglang version"
        patch_func(
            fp8,
            modified_file_path=
            f"{gcore_path}/mpatch/sglang-{sglang_version}/srt/layers/quantization/fp8.py"
        )
    elif sglang_version == '0.5.4.post3':
        assert memory_pool is not None, "mistakes in sglang version"
        patch_func(
            memory_pool,
            modified_file_path=
            f"{gcore_path}/mpatch/sglang-{sglang_version}/srt/mem_cache/memory_pool.py"
        )
    elif sglang_version == '0.5.7':
        patch_func(
            engine,
            modified_file_path=
            f"{gcore_path}/mpatch/sglang-{sglang_version}/srt/entrypoints/engine.py"
        )


if __name__ == '__main__':
    args = get_args()
    validate_args(args)
    if args.fn == 'gen':
        gen(args)
        if USE_SGLANG and args.patch_sglang:
            temporary_patch_sglang()
    elif args.fn == 'get':
        get(args)
    elif args.fn == 'init_ray':
        init_ray(args)
    elif args.fn == 'gen_and_init_ray':
        gen(args)
        init_ray(args)
    else:
        raise ValueError(f'unknown fn {args.fn}')
