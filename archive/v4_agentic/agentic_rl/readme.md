# Agentic RL - Sokoban 强化学习训练

## 项目简介

基于 Qwen2.5-VL 模型的 Sokoban 推箱子游戏强化学习训练项目，使用 GRPO 算法进行多步交互式训练。

## 快速开始

### 环境要求

```bash
# 安装依赖
pip install TensorDict codetiming transformers==4.55 gym_sokoban
pip install git+https://github.com/axon-rl/gem.git
pip install antlr4-python3-runtime==4.9.3 msgspec==0.20.0
```

### 运行训练

```bash
# 直接运行测试脚本
./test_rollout.sh

# 或手动运行
python3 tasks/agentic_rl/test_agentic.py --config-path="." --config-name="rl_config.yaml"
```

## 配置说明

### 主要配置参数

- **模型配置**: Qwen2.5-VL-7B 模型，支持视觉语言任务
- **环境配置**: Sokoban 推箱子游戏，6x6 网格，1个箱子
- **训练参数**: 
  - Rollout 批次大小: 128
  - 训练批次大小: 512
  - 学习率: 5.0e-6
  - 序列长度: 1024

### 环境指令

```
"You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets. When you are right next to a box, you can push it by moving in the same direction. You cannot push a box through a wall, and you cannot pull a box. The answer must be one of action in a turn, format is <answer>Right</answer>"
```

## 项目结构

```
tasks/agentic_rl/
├── rl_config.yaml          # 主配置文件
├── test_agentic.py         # 训练入口脚本
├── test_rollout.sh         # 运行脚本
├── requirements.txt        # 依赖列表
└── readme.md              # 本文档
```

## 核心组件

### GrpoTrainer
主要训练器，位于 `gpatch_v4.trainer.GrpoTrainer`

### StepVLTrajEnvManager
Sokoban 环境管理器，支持多步轨迹管理

### 分布式训练
- 使用 Ray 进行分布式采样
- 支持模型并行 (tensor_model_parallel_size: 2)
- 集成 WandB 进行实验追踪

## 监控与调试

- **WandB 集成**: 自动记录训练指标
- **调试模式**: 支持权重更新调试
- **数据保存**: 可保存首次 rollout 数据用于分析

## 注意事项

1. 确保有足够的 GPU 内存运行 7B 模型
2. 需要正确设置 PYTHONPATH 包含 Megatron-LM 和 mbridge 路径
3. 支持模型格式转换 (mcore ↔ hf)

## 故障排除

- 内存不足: 减小批次大小或使用梯度累积
- 依赖问题: 检查 Python 路径和依赖版本
- 模型加载失败: 确认模型路径和格式正确