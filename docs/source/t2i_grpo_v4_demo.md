# Text-to-Image GRPO V4 Demo

本文档介绍如何使用 G-Core V4 进行文生图扩散模型的 GRPO 训练，基于 FSDP2 训练后端。

This document introduces how to use G-Core V4 for text-to-image diffusion model GRPO training, powered by the FSDP2 training backend.

## Overview / 概览

T2I GRPO 训练的核心流程：

1. 从 prompt 数据集采样文本描述
2. 扩散模型生成图片
3. **BT Reward Model** (HPS v2) 评估美学质量
4. **Generative Reward Model** (Qwen2.5-VL) 评估语义一致性
5. GRPO 更新扩散模型参数

```
Prompts → Diffusion Model → Generated Images
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              BT RM (HPS v2)  Gen RM (VLM)    Rule Rewards
              美学质量评分      语义一致性        格式奖励
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                              GRPO Update
```

### 关键特性 / Key Features

- **FSDP2 Backend**: 使用 PyTorch Fully Sharded Data Parallel v2，适合大型扩散模型
- **Dual Reward**: HPS v2 美学评分 + VLM 语义评估
- **支持 Flux / OTeam4-4 等扩散模型架构**

## Quick Start / 快速开始

### Flux 模型训练

```bash
cd gcore-dev
bash tasks/t2i_grpo_tv4/scripts/grpo_flux.sh
```

核心启动命令：

```bash
export PYTHONPATH="$PWD:/root/Megatron-LM:/root/mbridge:/root/Megatron-Bridge/src:$PYTHONPATH"

# 初始化 Ray 集群
source tasks/t2i_grpo_tv4/scripts/mpirun-init-ray.sh

# 启动训练
python3 -u gpatch_v4/entry/train_t2i_grpo.py \
    --config-path="../../tasks/t2i_grpo_tv4/yaml" \
    --config-name="flux_rl_config"
```

## Configuration / 配置

```yaml
training:
  training_backend: fsdp2          # FSDP2 训练后端
  rollout_gbs: 16                  # Rollout batch size
  train_gbs: 64                    # 训练 batch size
  seq_length: 300

# 图片生成参数
image_generation:
  width: 720
  height: 720
  num_steps: 16                    # 扩散步数
  guidance_scale: 3.5
  eta: 0.3
  shift: 3

# Generative Reward Model（语义质量）
gen_rm:
  model_arch: "qwen2_5_vl"
  hf_model_path: "hf-hub/Qwen/Qwen2.5-VL-7B-Instruct"

# BT Reward Model（美学评分）
bt_rm:
  name: "hpsv2"
  clip_model: "CLIP-ViT-H-14"

# PPO 参数
ppo:
  grpo_advantage_epsilon: 1e-8
  ppo_ratio_eps: 1e-4              # T2I 用极小的 clip range
  grpo_kl_loss_beta: 0.01
```

## Reward Models / 奖励模型

### BT Reward — HPS v2

基于 CLIP-ViT-H-14 的美学评分模型，评估生成图片的光照、构图、细节等视觉质量。

参考 `tasks/t2i_grpo_tv4/bt_rewards.py`。

### Generative Reward — VLM

使用 Qwen2.5-VL-7B 视觉语言模型，评估生成图片与文本描述之间的语义一致性。

参考 `tasks/t2i_grpo_tv4/gen_rewards.py`。

## Customization / 自定义扩展

1. **YAML 配置** — 调整图片生成参数、模型路径
2. **Dataset** — 参考 `tasks/t2i_grpo_tv4/simple_dataset.py`，加载 `.txt` 或 `.jsonl` 格式的 prompt 数据
3. **Reward** — 替换或组合 BT/Gen reward model

## File Structure / 文件结构

```
tasks/t2i_grpo_tv4/
├── scripts/
│   ├── grpo_flux.sh               # Flux 模型启动脚本
│   └── grpo_oteam4_4.sh           # OTeam4-4 启动脚本
├── yaml/
│   └── flux_rl_config.yaml        # 训练配置
├── simple_dataset.py              # Prompt 数据集
├── prompts.txt                    # 训练 prompt 集（~5000 条）
├── bt_rewards.py                  # HPS v2 美学奖励
└── gen_rewards.py                 # VLM 语义奖励
```
