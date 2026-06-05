# 惠州集群测试记录

记录惠州 h20 集群（可能网络有问题?） lib 的测试吞吐，可能并不合理，仅供参考。

关闭透明大页，并清空 page cache（tlinux 特殊姿势）。
```bash
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
sync
echo 3 >/proc/sys/vm/drop_caches
echo 1 >/proc/sys/vm/compact_memory
```

## sglang

sglang 0.4.6.post5，`mem_fraction_static`=0.75`, very short prompt, min and max new tokens 2048.

|     Model      | TP size | EP size | GBS  | Throughput Max | Throughput Min |
| :------------: | :-----: | :-----: | :--: | :------------: | :------------: |
| Qwen 2.5 1.5B  |    1    |    1    | 1024 |     17894      |      9190      |
| Qwen 2.5 1.5B  |    1    |    1    | 2048 |     19077      |   9335 (OOM)   |
| Qwen 2.5 1.5B  |    2    |    1    | 1024 |     23243      |     14191      |
| Qwen 2.5 1.5B  |    2    |    1    | 2048 |     25005      |                |
|  Qwen 2.5 32B  |    1    |    1    | 1024 |      ~439      |                |
|  Qwen 2.5 32B  |    1    |    1    | 128  |      ~439      |      241       |
|  Qwen 2.5 32B  |    4    |    1    | 128  |      4121      |      2590      |
|  Qwen 2.5 32B  |    4    |    1    | 256  |      4346      |      2793      |
| Qwen 3 30B A3B |    4    |    1    | 128  |     10662      |      6806      |
| Qwen 3 30B A3B |    4    |    1    | 256  |     14141      |      7942      |
| Qwen 3 30B A3B |    4    |    1    | 512  |      9524      |      8938      |
| Qwen 3 30B A3B |    1    |    4    | 512  |                |                |
|                |         |         |      |                |                |


