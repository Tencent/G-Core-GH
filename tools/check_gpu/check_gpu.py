import fcntl
import os
import socket
import struct
from datetime import timedelta

import torch
import torch.distributed as dist


def get_ip_by_ifname(ifname='bond1'):
    """获取指定网卡的 IP 地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(
            fcntl.ioctl(
                s.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack('256s', ifname[:15].encode('utf-8'))
            )[20:24]
        )
        return ip
    except Exception:
        return socket.gethostname()


def check_gpu(local_rank):
    """检测指定 GPU 是否正常"""
    try:
        torch.cuda.set_device(local_rank)
        # 简单的 GPU 运算测试
        x = torch.randn(100, 100, device=f'cuda:{local_rank}')
        y = torch.matmul(x, x)
        # 确保运算完成
        torch.cuda.synchronize(local_rank)
        return True, f"GPU {local_rank} - normal"
    except Exception as e:
        return False, f"GPU {local_rank} - abnormal: {str(e)}"


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    hostname = get_ip_by_ifname('bond1')

    # 初始化分布式环境
    dist.init_process_group(backend='gloo', timeout=timedelta(seconds=300))

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 检测当前进程对应的 GPU
    success, msg = check_gpu(local_rank)
    result_msg = f"[{hostname}] [device rank {rank//8}] [gpu-rank {rank}] {msg}"
    print(result_msg)

    # 等待所有进程完成检测
    dist.barrier()

    # 收集所有结果到 rank 0
    all_results = [None for _ in range(world_size)] if rank == 0 else None
    pod_name = os.environ.get("POD_NAME", "unknown")
    result_obj = {
        "hostname": hostname,
        "pod_name": pod_name,
        "device_rank": rank // 8,
        "local_rank": local_rank,
        "success": success,
        "msg": msg
    }
    dist.gather_object(result_obj, all_results, dst=0)

    if rank == 0:
        print("\n" + "=" * 60)
        print("GPU 检测汇总报告")
        print("=" * 60)

        # 按机器分组
        host_results = {}
        for r in all_results:
            h = r["device_rank"]
            if h not in host_results:
                host_results[h] = []
            host_results[h].append(r)

        total_gpus = 0
        failed_gpus = 0
        for h in sorted(host_results.keys()):
            results = sorted(host_results[h], key=lambda x: x["local_rank"])
            print(f"\n机器-{h} IP: {results[0]['hostname']} (Pod: {results[0]['pod_name']})")
            for r in results:
                total_gpus += 1
                status = "✓" if r["success"] else "✗"
                if not r["success"]:
                    failed_gpus += 1
                print(f"  [{status}] {r['msg']}")

        print(f"\n{'=' * 60}")
        print(f"总计: {total_gpus} 个 GPU, {total_gpus - failed_gpus} 个正常, {failed_gpus} 个异常")
        if failed_gpus == 0:
            print("所有 GPU 检测通过! ✓")
        else:
            print(f"警告: 有 {failed_gpus} 个 GPU 异常! ✗")
        print("=" * 60)

    dist.barrier()
    dist.destroy_process_group()
