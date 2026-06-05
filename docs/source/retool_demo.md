# ReTool: Multi-Turn Tool-Calling Agent RL

本文档介绍如何使用 G-Core V4 进行多轮工具调用 Agent 的强化学习训练（ReTool 框架）。

This document introduces how to use G-Core V4 for multi-turn tool-calling agent RL training (ReTool framework).

## Overview / 概览

ReTool 实现了基于轨迹的 DAPO (Direct Alignment from Preference Optimization) Agent 训练：

1. Agent 接收数学问题，通过多轮对话调用 `code_interpreter` 工具
2. 沙箱执行代码，返回结果
3. Agent 根据执行结果决定继续调用工具或给出最终答案
4. 通过规则奖励（数学正确性）进行 GRPO 训练

```
Problem → Agent (Qwen2.5-7B)
              │
              ├─ Turn 1: call code_interpreter(code)
              │           └─ Sandbox → execution result
              ├─ Turn 2: call code_interpreter(code)
              │           └─ Sandbox → execution result
              ├─ ...
              └─ Turn N: final answer \boxed{...}
                          └─ Reward: correct? 1.0 / 0.0
```

### 关键特性 / Key Features

- **Multi-turn trajectory**: 最多 8 轮对话，每轮最多 2048 tokens
- **Code sandbox**: 本地沙箱执行代码，20s 超时，1GB 内存限制
- **TrajEnvManager**: 基于轨迹的环境管理器，支持并行环境
- **GRPO + reward normalization**: 按轨迹组归一化奖励

## Quick Start / 快速开始

```bash
cd gcore-dev
bash tasks/retool/agentic_rl/run_retool_dapo_agentic.sh
```

启动脚本会自动完成：

1. 初始化 Ray 集群
2. 启动本地代码执行沙箱
3. 等待沙箱就绪
4. 启动 Agent RL 训练

核心启动命令：

```bash
export PYTHONPATH="$PWD:/root/Megatron-LM:/root/mbridge:/root/Megatron-Bridge/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

# 启动沙箱
export SANDBOX_LOCAL_PORT=8008
nohup python3 tasks/retool/local_sandbox_server.py &

# 启动训练
python3 tasks/retool/agentic_rl/retool_dapo_agentic.py \
    --config-path="." --config-name="retool_dapo_agentic.yaml"
```

## Configuration / 配置

```yaml
training:
  seq_length: 18432              # 多轮对话需要更长序列
  rollout_gbs: 32
  sampling_repeat_n: 8           # 每个问题生成 8 条轨迹
  sampling_keep_n: 8
  train_gbs: 128

  agentic:
    adv_estimator: grpo
    reward_normalization:
      grouping: traj_group_id    # 按轨迹组归一化
      method: mean_std

    train_env_manager:
      max_traj_per_env: 8        # 每个环境最多 8 条轨迹
      group_per_worker: 4
      group_replicate: 8

    env_cfg_template:
      env_type: retool_dapo
      custom_env_cls: tasks.retool.agentic_rl.env:RetoolDapoEnv
      max_steps: 8               # 最多 8 轮对话
      max_tokens_per_step: 2048  # 每轮最多 2048 tokens

      env_tool_config:
        use_tools: true
        tools_json_path: tasks/retool/agentic_rl/tools_code_interpreter.json
        tool_call_parser: sglang/qwen25

# Policy: Qwen2.5-7B-Instruct
policy:
  model_arch: "qwen2"
  hf_model_path: "hf-hub/Qwen/Qwen2.5-7B-Instruct"
  dist_config:
    tensor_model_parallel_size: 4
```

## Environment / 环境

### RetoolDapoEnv

`tasks/retool/agentic_rl/env.py` 中的 `RetoolDapoEnv` 实现了 Gym-style 环境接口：

- `reset()` — 加载数学问题，设置初始 observation
- `step(action, tool_call_parse_result)` — 处理 Agent 的响应
  - 如果有 tool call → 沙箱执行代码，返回结果，继续对话
  - 如果无 tool call → 提取 `\boxed{}` 答案，计算奖励，结束

### Code Sandbox

`tasks/retool/local_sandbox_server.py` 提供 HTTP API 执行代码：

- 端口: `SANDBOX_LOCAL_PORT` (默认 8008)
- 超时: 20s
- 内存限制: 1GB
- 输出截断: 训练 50 行 / 推理 200 行

## Customization / 自定义扩展

1. **Environment** — 继承 `Env` 基类，实现 `reset()` 和 `step()`
2. **Tools** — 修改 `tools_code_interpreter.json`，添加自定义工具定义
3. **Reward** — 修改 `tasks/retool/agentic_rl/reward_utils.py` 中的 `compute_retool_reward()`
4. **Dataset** — 修改 `tasks/retool/agentic_rl/dataset_utils.py`

## File Structure / 文件结构

```
tasks/retool/
├── agentic_rl/                         # Trajectory-based DAPO
│   ├── run_retool_dapo_agentic.sh      # 启动脚本
│   ├── retool_dapo_agentic.yaml        # 训练配置
│   ├── retool_dapo_agentic.py          # Hydra 入口
│   ├── env.py                          # RetoolDapoEnv 环境
│   ├── dataset_utils.py                # 数据加载
│   ├── reward_utils.py                 # 奖励计算
│   ├── sandbox_sync.py                 # 沙箱通信
│   └── tools_code_interpreter.json     # 工具定义
├── local_sandbox_server.py             # 代码执行沙箱
├── internal_agent/                     # 内部 Agent（async_generate）
└── external_agent/                     # 外部 Agent（OpenAI API）
```
