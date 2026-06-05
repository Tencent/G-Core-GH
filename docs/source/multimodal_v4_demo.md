# Multimodal V4 Demo

本文档介绍如何使用 G-Core V4 进行多模态视觉语言模型 (VLM) 的训练，包括 SFT 微调和 On-Policy 蒸馏。

This document introduces how to use G-Core V4 for multimodal vision-language model (VLM) training, including SFT fine-tuning and on-policy distillation.

## Overview / 概览

`tasks/multimodal_v4/` 提供了以下多模态训练流水线：

| Pipeline | Description | Entry Point |
|----------|-------------|-------------|
| **SFT Fine-tuning** | 视觉语言模型监督微调 | `gpatch_v4/entry/train_lm_sft.py` |
| **On-Policy Distill** | 在线蒸馏（学生生成 + 教师评估） | `gpatch_v4/entry/train_on_policy_distill.py` |
| **Off-Policy Distill** | 离线蒸馏（预生成数据 + 教师 KL 惩罚） | `gpatch_v4/entry/train_off_policy_distill.py` |

支持的模型架构：Qwen3-VL 系列（2B / 4B / 32B）。

Supported model architectures: Qwen3-VL series (2B / 4B / 32B).

## Quick Start / 快速开始

### 1. SFT Fine-tuning

```bash
cd gcore-dev
export PYTHONPATH="$PWD:/root/Megatron-LM:/root/mbridge:/root/Megatron-Bridge/src:$PYTHONPATH"

# 初始化 Ray 集群
source tasks/multimodal_v4/finetune/scripts/run_once.sh

# 启动 SFT 训练
python3 -u gpatch_v4/entry/train_lm_sft.py \
    --config-path="../../tasks/multimodal_v4/finetune/yaml" \
    --config-name="qwen3vl_sft"
```

SFT 配置示例：

```yaml
training:
  training_backend: mcore
  train_gbs: 256
  train_mbs: 1
  seq_length: 4096
  num_train_epoches: 1

policy:
  model_arch: "qwen3_vl"
  hf_model_path: "hf-hub/Qwen/Qwen3-VL-4B-Instruct"
  dist_config:
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1

data:
  py_path: "tasks/multimodal_v4/finetune/qwen3vl_sft_v4.py"
  fn_name: "get_dataset_and_dataloader"
```

### 2. On-Policy Distillation

学生模型在线生成 → 教师模型评估 → 计算 KL 惩罚 → GRPO 训练。

Student generates online → Teacher evaluates → KL penalty → GRPO training.

```bash
cd gcore-dev
bash tasks/multimodal_v4/on_policy_distill/script/on_policy_distill.sh
```

核心启动命令：

```bash
source tasks/multimodal_v4/on_policy_distill/script/mpirun_run_once.sh \
    tasks/multimodal_v4/on_policy_distill/script/run_once.sh

python3 -u gpatch_v4/entry/train_on_policy_distill.py \
    --config-path="../../tasks/multimodal_v4/on_policy_distill/yaml" \
    --config-name="qwen3vl_distill"
```

配置示例：

```yaml
training:
  rollout_gbs: 128
  train_gbs: 640
  sampling_repeat_n: 5        # 每个 prompt 采样 5 次
  sampling_keep_n: 5          # 保留全部 5 个样本

# 学生模型: Qwen3-VL-2B
policy:
  model_arch: "qwen3_vl"
  hf_model_path: "hf-hub/Qwen/Qwen3-VL-2B-Instruct"
  rollout_gen_type: "on_policy_distill"

# 教师模型: Qwen3-VL-32B
teacher:
  model_arch: "qwen3_vl"
  hf_model_path: "hf-hub/Qwen/Qwen3-VL-32B-Instruct"

data:
  data_pathes:
    - "hf-hub/hiyouga/geometry3k"
  py_path: "tasks/multimodal_v4/on_policy_distill/qwen3vl_simple_dataset.py"
  fn_name: "get_dataset_and_dataloader"
```

## Customization / 自定义扩展

使用 multimodal V4 训练自定义任务，通常只需修改：

1. **YAML 配置文件** — 模型路径、并行度、数据路径
2. **Dataset 加载函数** — 参考 `qwen3vl_sft_v4.py` 或 `qwen3vl_simple_dataset.py`，需处理图片输入
3. **Reward 函数**（On-Policy Distill） — 参考 `bt_reward.py`

## File Structure / 文件结构

```
tasks/multimodal_v4/
├── finetune/
│   ├── scripts/run_once.sh            # Ray 初始化
│   ├── yaml/qwen3vl_sft.yaml         # SFT 配置
│   └── qwen3vl_sft_v4.py             # 数据集加载
├── on_policy_distill/
│   ├── script/on_policy_distill.sh    # 启动脚本
│   ├── yaml/qwen3vl_distill.yaml     # 蒸馏配置
│   ├── qwen3vl_simple_dataset.py      # 数据集加载
│   ├── bt_reward.py                   # 规则奖励
│   └── train_on_policy_distill.py     # 入口
└── off_policy_distill/
    ├── script/off_policy_distill.sh   # 启动脚本
    ├── yaml/off_policy_distill_demo.yaml
    └── ...
```
