# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指引。

[TOC]


## 仓库结构

本仓库是 **WeChat-YATT / G-Core** 的 monorepo，一个用于 LLM、VLM 和 DiT 的分布式训练库。主要组件：

- **`gcore-dev/`** — G-Core 主库（主要工作目录）
- **`Megatron-LM/`** — 我们的 Megatron-LM fork（分支：`dev`），用于分布式训练后端
- **`mbridge/`** — HF ↔ Megatron 桥接
- **`Megatron-Bridge/`** — Nvidia HuggingFace ↔ Megatron checkpoint 转换
- **`sglang/`** — 用于 rollout 生成的 SGLang 推理引擎
- **`vllm/`** — 用于 rollout 生成的 vLLM 推理引擎


## GPU 服务器（测试资源）

本地机器没有 GPU。如需在 GPU 上运行脚本，请向用户索取连接命令（格式如 `./gemini-go <container> <token>`），然后解析出 container 和 token，调用 gemini-remote MCP `wx-gemini-remote` 进行连接。

### 工作目录

有两个代码目录：
- 本地开发目录：`/xxx/work/wepsdl`（也就是当前的项目路径）
- 远程测试目录：分布式文件系统路径 `/work/wepsdl`

1. 始终在本地编辑代码
2. 远程测试目录是分布式文件系统的目录，同时挂载在本地开发机和测试机上，在本地编辑完成后，一般 1 秒之内会**自动同步到远程测试目录**，不需要额外同步。
3. **注意**：工作目录是 `/work/wepsdl/gcore-dev`，而非远程代码目录 `/work/wepsdl`；一定要 `cd /work/wepsdl/gcore-dev` 确保工作目录正确。

### 远程操作

1. 通过 `wx-gemini-remote` MCP 在容器内执行命令，运行训练任务或者测试。

2. 提供的 GPU 节点只是头节点，不是整个集群。集群节点信息记录在 `/etc/mpi/hostfile`。
代码通过 MPI run 在集群上集体执行（例如初始化 Ray，或直接运行）。示例：
```bash
mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  hostname
```


## 运行测试

执行 `tests/test.sh` 即可（尽量通过这个脚本执行测试，因为他们包括了清理逻辑，避免意外的资源占用）：
```bash
cd /work/wepsdl/gcore-dev
bash tests/test.sh
```

如果要执行单个测试，务必先做好准备和清理：
```bash
RCDIR="/work/wepsdl"
export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
source tests/test_gpatch_v4/mpirun-stop-ray.sh
source tests/test_gpatch_v4/mpirun-init-ray.sh
pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_xxx.py
```

测试比较缓慢，所以每 5 分钟和用户汇报下测试进展。
为了方便开发者观察，测试的 log 要重定向到 `/tmp` 目录下。


## 编程指导

1. 不要用 web2.0 时代的业务逻辑来容错，对于不符合预期的情况，不允许默默出错，使用 assert 更好（除非情况确实存在，而不是 BUG）。
坏逻辑：
```python
tokens = batch.get('tokens', torch.zeros())
```
好逻辑：
```python
tokens = batch['tokens'] # raise if `tokens` is not provided
```

2. `gpatch`, `mpatch`, `tools/auto_place.py` 属于 trainer v3 的老代码，`gpatch_v4` 属于 trainer v4 的新代码，写代码的时候要分清楚边界。

3. 使用 Facebook Python 代码风格，docstring 使用 NumPy 风格。

4. 注意可读性：
 - 继承的函数注意标注 `override` decorator；
 - 函数签名尽可能标注 type hints（参数与返回值）；
 - 如果函数实在是太长了，注意按照一定的逻辑拆分函数；
 - 不要为了一个只用在一个地方使用的两行代码创建一个包装函数，很难看，除非可能需要 override（interface）。