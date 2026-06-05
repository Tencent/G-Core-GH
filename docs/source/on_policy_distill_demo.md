# On-Policy Distillation Demo

本文档介绍如何使用 G-Core V4 进行文本模型的在线策略蒸馏 (On-Policy Knowledge Distillation)。

This document introduces how to use G-Core V4 for on-policy knowledge distillation on text-only models.

## Overview / 概览

On-Policy Distillation 的核心思想：

1. **学生模型**在线生成 rollout（采样响应）
2. **教师模型**对学生的输出进行评估，提供 KL 散度信号
3. 结合规则奖励（如数学正确性）和教师 KL 惩罚进行 GRPO 训练

The core idea of on-policy distillation:

1. The **student model** generates rollouts (sampled responses) online
2. The **teacher model** evaluates the student's outputs, providing KL divergence signal
3. GRPO training combines rule-based rewards (e.g., math correctness) with teacher KL penalty

```
Student (Qwen3-30B-A3B)         Teacher (Qwen3-32B)
       │                              │
       ├─ generate rollouts ──────────┤
       │                              ├─ compute KL penalty
       │                              │
       ├─ rule reward (math accuracy) │
       │                              │
       └─ GRPO update ◄──────────────┘
```

## Quick Start / 快速开始

```bash
cd gcore-dev
bash tasks/on_policy_distill/scripts/on_policy_distill.sh
```

核心启动命令：

```bash
export PYTHONPATH="$PWD:/root/Megatron-LM:/root/mbridge:/root/Megatron-Bridge/src:$PYTHONPATH"

# 初始化 Ray 集群
source tasks/on_policy_distill/scripts/mpirun-init-ray.sh

# 启动蒸馏训练
python3 -u gpatch_v4/entry/train_on_policy_distill.py \
    --config-path="../../tasks/on_policy_distill/yamls" \
    --config-name="distill_demo"
```

## Configuration / 配置

```yaml
training:
  rollout_gbs: 32              # Rollout 全局 batch size
  train_gbs: 512               # 训练全局 batch size
  seq_length: 16384            # 序列长度
  sampling_repeat_n: 8         # 每个 prompt 采样 8 次
  sampling_keep_n: 8           # 保留全部样本

# 学生模型: Qwen3-30B-A3B (MoE)
policy:
  model_arch: "qwen3"
  hf_model_path: "hf-hub/Qwen/Qwen3-30B-A3B"

# 教师模型: Qwen3-32B
teacher:
  model_arch: "qwen3"
  hf_model_path: "hf-hub/Qwen/Qwen3-32B"

# 数据集: GSM8K
data:
  data_pathes:
    - "hf-hub/openai/gsm8k-jsonl/train"
  py_path: "tasks/on_policy_distill/simple_dataset.py"
  fn_name: "get_dataset_and_dataloader"
```

## Customization / 自定义扩展

1. **YAML 配置** — 替换学生/教师模型路径、调整并行度
2. **Dataset** — 参考 `tasks/on_policy_distill/simple_dataset.py`，实现 `get_dataset_and_dataloader()` 接口
3. **Reward** — 参考 `tasks/on_policy_distill/bt_reward.py`，实现规则奖励逻辑

## File Structure / 文件结构

```
tasks/on_policy_distill/
├── scripts/
│   ├── on_policy_distill.sh       # 启动脚本
│   └── mpirun-init-ray.sh        # Ray 初始化
├── yamls/
│   └── distill_demo.yaml         # 训练配置
├── simple_dataset.py              # 数据集加载
├── bt_reward.py                   # 规则奖励（数学正确性）
└── train_on_policy_distill.py     # 入口
```
