# Math GRPO V4 Demo

本文档基于 `tasks/math_rl_v4/scripts/grpo.sh` 脚本，介绍 G-Core V4 版本 GRPO (Group Relative Policy Optimization) 训练的使用方法与代码设计。

This document is based on `tasks/math_rl_v4/scripts/grpo.sh` and introduces the usage and code design of the G-Core V4 GRPO training pipeline.

## Overview / 概览

V4 版本相比 V3 进行了重要的架构升级：

- **Ray-based orchestration**: 使用 Ray 统一管理所有组件（Sampler、Reward Model、Train Group），取代了 V3 中基于 mpirun 的多进程方案。
- **Hydra configuration**: 使用 Hydra + OmegaConf 进行配置管理，支持 YAML 配置文件的灵活组合与覆盖。
- **Async pipeline**: 训练流程完全异步化，通过 `asyncio` 编排各组件的初始化和训练循环。
- **Automatic retry**: 内置 actor 卡死检测与自动重启机制。

Compared to V3, V4 features several important architectural upgrades:

- **Ray-based orchestration**: All components (Sampler, Reward Model, Train Group) are unified under Ray, replacing V3's mpirun-based multi-process approach.
- **Hydra configuration**: Uses Hydra + OmegaConf for configuration management, supporting flexible YAML composition and overrides.
- **Async pipeline**: The entire training pipeline is fully asynchronous, orchestrated via `asyncio`.
- **Automatic retry**: Built-in actor stuck detection and automatic restart mechanism.

## Quick Start / 快速开始

### 1. Launch Script / 启动脚本

入口脚本 `tasks/math_rl_v4/scripts/grpo.sh`：

```bash
MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config.yaml"
```

脚本分为三个步骤：

The script consists of three steps:

1. **设置 PYTHONPATH**: 将 Megatron-LM、mbridge、Megatron-Bridge 等依赖加入 Python 路径。
2. **初始化 Ray 集群**: 通过 `mpirun-init-ray.sh` 在所有节点上启动 Ray，自动组建分布式集群。
3. **启动训练**: 调用 `train_lm_grpo.py` 入口，通过 Hydra 加载 YAML 配置。

### 2. Ray Cluster Initialization / Ray 集群初始化

`mpirun-init-ray.sh` 使用 mpirun 在所有节点上执行 `init-ray.sh`：

```bash
# mpirun-init-ray.sh: use mpirun to launch Ray on all nodes
mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH ... \
  bash tasks/math_rl_v4/scripts/init-ray.sh $__HOST_IP__
```

```bash
# init-ray.sh: start Ray head or worker node
node_rank=$OMPI_COMM_WORLD_RANK
if [[ "$node_rank" == '0' ]]; then
  ray start --head --node-ip-address $MIP --port=6379
else
  ray start --address="$MIP:6379"
fi
```

Rank 0 节点启动 Ray head，其他节点作为 worker 加入集群。之后所有组件的资源分配和通信都通过 Ray 统一管理。

Rank 0 starts the Ray head node, and other nodes join as workers. After this, all component resource allocation and communication are managed through Ray.

## Code Design / 代码设计

### Architecture / 整体架构

```
grpo.sh
  └─ train_lm_grpo.py          # Hydra entry point
       └─ GrpoTrainer           # Trainer orchestrator
            ├─ orches.init(config)              # Initialize Ray
            ├─ create_placement_groups()  # Allocate GPU resources
            ├─ SamplerGroup               # Inference / rollout generation (sglang)
            ├─ BtRmGroup                  # Rule-based reward model
            ├─ TrainGroup                 # Policy training (Megatron-Core)
            └─ train_loop()               # Main training loop
```

### Entry Point / 入口

`gpatch_v4/entry/train_lm_grpo.py` 是 Hydra 入口：

```python
@hydra.main(config_path="../configs/yaml", config_name="test_config", version_base=None)
def main(cfg: RlConfig):
    default_config = RlConfig()
    merged_config = OmegaConf.merge(default_config, cfg)
    merged_obj = OmegaConf.to_object(merged_config)

    trainer = GrpoTrainer()
    asyncio.run(trainer.launch_with_retry(merged_obj))
```

它会将 YAML 配置与 `RlConfig` dataclass 默认值合并，然后创建 `GrpoTrainer` 并启动训练。通过 `--config-path` 和 `--config-name` 指定任务配置文件。

It merges the YAML config with the `RlConfig` dataclass defaults, then creates a `GrpoTrainer` and launches training. Use `--config-path` and `--config-name` to specify the task configuration file.

### GrpoTrainer / 训练编排

`gpatch_v4/trainer/grpo_trainer.py` 中的 `GrpoTrainer` 继承了 `TrainerRetryMixin`，其 `launch` 方法按顺序初始化各组件：

```python
class GrpoTrainer(TrainerRetryMixin):
    async def launch(self, config: RlConfig):
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        # 1. Initialize sampler (inference engine)
        self.sampler_group = create_sampler_group(config, pgs)
        await self.sampler_group.init()

        # 2. Initialize reward model (if enabled)
        if config.training.use_bt_rm_reward:
            self.bt_rm_group = create_bt_rm_group(config, pgs)
            await self.bt_rm_group.init()

        # 3. Initialize training group (policy model)
        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()

        await self.train_group.setup_client()
        await self.train_group.setup_rollout_generator()
        await self.train_group.setup_model_and_optimizer()
```

`TrainerRetryMixin.launch_with_retry` 在 `launch` 之后启动训练循环，并监控 actor 是否卡死。如果检测到卡死，则自动关闭 Ray 并重启整个流程：

```python
class TrainerRetryMixin:
    async def launch_with_retry(self, config):
        while restart_count <= max_restart_attempts:
            await self.launch(config)
            task_train_loop = asyncio.create_task(self.train_group.train_loop())
            task_check_actor_stuck = asyncio.create_task(self.train_group.check_actor_stuck())
            # If actor is stuck, restart; otherwise, return normally.
```

### Placement Groups / GPU 资源分配

`create_placement_groups` 根据配置创建 Ray placement groups，支持两种模式：

- **colocate**（默认）: 所有组件共享同一组 GPU，通过时分复用。
- **disaggregated**: 各组件使用独立的 GPU。

### Configuration / 配置文件

`tasks/math_rl_v4/yaml/rl_config.yaml` 是核心配置文件，包含以下主要部分：

| Section | Description |
|---------|-------------|
| `training` | Training hyperparams: batch sizes, save interval, reward type, etc. |
| `checkpoint` | Checkpoint load/save paths, HF conversion options. |
| `optimizer` | Learning rate, warmup, decay, Adam parameters. |
| `data` | Dataset paths, custom dataset loader (`simple_dataset.py`). |
| `policy` | Model architecture, parallelism config (TP/PP), HF model paths. |
| `sampler` | Inference engine backend (sglang), generation params (temperature, top_p, max_tokens). |
| `bt_rm` | Reward model type (`rule_only`), reward function path. |
| `ppo` | GRPO-specific params: advantage epsilon, PPO ratio clip. |
| `report` | W&B reporting configuration. |

示例配置关键字段：

```yaml
training:
  use_bt_rm_reward: True       # Use rule-based reward
  rollout_gbs: 256             # Rollout global batch size
  sampling_repeat_n: 32        # Number of samples per prompt

policy:
  model_arch: "qwen2"
  dist_config:
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 2

sampler:
  backend: sglang
  infer_engine_configs:
    - temperature: 1.0
      top_p: 0.9
      generate_max_tokens: 1024

bt_rm:
  reward_type: "rule_only"
```

### Dataset / 数据集

`tasks/math_rl_v4/simple_dataset.py` 提供了自定义数据集加载函数 `get_dataset_and_dataloader`，在 YAML 中通过 `data.py_path` 和 `data.fn_name` 引用：

```yaml
data:
  data_pathes:
    - "hf-hub/openai/gsm8k-jsonl/train"
  py_path: "tasks/math_rl_v4/simple_dataset.py"
  fn_name: "get_dataset_and_dataloader"
```

数据集处理流程：

1. 从 JSONL 文件加载数据（`question` + `answer` 字段）。
2. 使用 tokenizer 的 `apply_chat_template` 构建 prompt。
3. Tokenize 后返回 `input_ids`、`prompt_len` 和 `gt_label`（ground truth answer）。

Dataset processing flow:

1. Load data from JSONL files (`question` + `answer` fields).
2. Build prompts using the tokenizer's `apply_chat_template`.
3. After tokenization, return `input_ids`, `prompt_len`, and `gt_label` (ground truth answer).

### Reward Function / 奖励函数

`tasks/math_rl_v4/bt_reward.py` 中的 `math_rl_rule_reward` 是基于规则的奖励函数，包含两种奖励信号：

- **Accuracy Reward**: 检查模型输出中 `\boxed{}` 的内容是否与 ground truth 匹配。
- **Format Reward**: 检查模型输出是否包含正确格式的 `\boxed{number}`。

最终奖励 = accuracy_reward + format_reward。

在 YAML 中通过 `bt_rm.reward_model_info.reward_py_path` 和 `parse_reward_fn_name` 引用：

```yaml
bt_rm:
  reward_type: "rule_only"
  reward_model_info:
    - reward_py_path: tasks/math_rl_v4/bt_reward.py
      parse_reward_fn_name: math_rl_rule_reward
      reward_weight: 1.0
```

## Customization / 自定义扩展

如果你需要在自己的任务上使用 GRPO V4 训练，通常只需修改以下三部分：

To use GRPO V4 training on your own task, you typically only need to modify the following three parts:

1. **YAML 配置文件**: 修改模型路径、并行度、数据路径等。
2. **Dataset 加载函数**: 参考 `simple_dataset.py`，实现 `get_dataset_and_dataloader` 接口，返回包含 `train_dataset`、`train_sampler`、`train_dataloader` 的 dict。
3. **Reward 函数**: 参考 `bt_reward.py`，实现自定义的奖励计算逻辑，函数签名为 `fn(batched_data, tokenizer, actor_tokenizer) -> (rule_reward, per_token_reward, metrics)`。
