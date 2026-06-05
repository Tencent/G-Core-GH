# G-Core 开发手册

G-Core 是一个 scalable、简单、高效且可用于生产环境的 RL 训练库，构建于 Megatron、vLLM 和 SGLang 之上。

---

```{contents}
:depth: 2
:local:
```

---

## 项目架构

```
gcore-dev/
├── gpatch_v4/              # V4 Trainer（推荐，Ray + Hydra + asyncio）
│   ├── orches/             # Ray 编排层
│   ├── client/             # Ray 客户端封装
│   ├── configs/            # Hydra 配置 dataclass
│   ├── entry/              # 训练入口（train_lm_grpo.py 等）
│   ├── trainer/            # 训练编排（GrpoTrainer 等）
│   ├── training_backend/   # 训练后端（Megatron / FSDP2）
│   ├── generation_backend/ # 推理后端（SGLang / vLLM）
│   ├── rollout_generator/  # Rollout 生成（含 async rollout）
│   ├── reward/             # 奖励计算
│   ├── actor/              # Actor 模型
│   ├── models/             # 模型定义（Qwen3-VL、Bagel、Flux 等）
│   ├── agentic/            # Agentic RL（环境、工具）
│   └── utils/              # 工具函数
├── gpatch/                 # V3 Trainer（Legacy，mpirun 多进程）
│   ├── core/               # 核心功能实现
│   │   ├── models/         # 模型定义 (Actor, Critic, RM)
│   │   ├── transformer/    # Transformer 配置
│   │   ├── ppo_helper.py   # PPO/GRPO 辅助函数
│   │   └── aligner_helper.py # 对齐辅助函数
│   ├── training/           # 训练相关
│   │   └── arguments.py    # 命令行参数定义
│   ├── rpc/                # RPC 通信
│   └── model_yamls/        # 模型配置文件
├── mpatch/                 # Megatron-Core 补丁
├── gdataset/               # 数据集处理
├── megatron_datasets/      # Megatron 数据集适配
├── tasks/                  # 具体任务实现
│   ├── math_rl_v4/         # Math RL V4 任务
│   ├── math_rl_v3/         # Math RL V3 任务
│   ├── qwen3vl/            # Qwen3 VL 任务
│   ├── dpo/                # DPO 任务
│   └── my_custom_loss/     # 自定义 Loss 示例（V3）
├── docs/                   # 文档
└── tools/                  # 工具脚本
```

---

## 核心模块

| 模块 | 作用 | 关键文件 |
|------|------|----------|
| `gpatch_v4/` | V4 训练器：Ray 编排、Hydra 配置、async pipeline | `entry/`, `trainer/`, `configs/` |
| `gpatch/` | V3 训练器：mpirun 编排、CLI 参数 | `core/models/`, `training/arguments.py` |
| `mpatch/` | Megatron-Core 补丁 | 分布式训练、检查点、并行状态 |
| `gdataset/` | 数据集处理 | 惰性加载、多存储后端、智能填充 |
| `megatron_datasets/` | Megatron 数据集适配 | 各模型的数据集实现 |
| `tasks/` | 具体任务实现 | Math RL、VQA、DPO 等任务 |

---

## V4 GRPO 训练架构（推荐）

> 详细使用指南请参考 [Math GRPO V4 Demo](math_grpo_v4_demo.md)。

### V4 vs V3 对比

| 特性 | V3 | V4 |
|------|----|----|
| **编排方式** | mpirun 多进程，手动分别启动 | Ray 统一管理 |
| **配置系统** | 命令行参数 (argparse) | Hydra + OmegaConf (YAML) |
| **训练流水线** | 同步 | asyncio 异步 |
| **容错** | 无 | 内置卡死检测 + 自动重启 |
| **启动方式** | 3 条 mpirun 命令 | 1 条 python 命令 |
| **GPU 布局** | `tools/auto_place.py` 预生成 | `create_placement_groups()` 动态分配 |

### 整体架构

```
grpo.sh
  └─ train_lm_grpo.py            # Hydra 入口
       └─ GrpoTrainer             # 训练编排
            ├─ orches.init(config)              # 初始化 Ray 集群
            ├─ create_placement_groups()  # GPU 资源分配
            ├─ SamplerGroup               # 推理 / rollout 生成 (SGLang)
            ├─ BtRmGroup                  # 奖励模型
            ├─ TrainGroup                 # 策略训练 (Megatron-Core)
            └─ train_loop()               # 异步训练循环
```

### 启动流程

```bash
# 1. 设置环境变量
export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

# 2. 初始化 Ray 集群（mpirun 在所有节点上启动 Ray）
source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

# 3. 一条命令启动训练
python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config.yaml"
```

### 配置示例

```yaml
training:
  use_bt_rm_reward: True       # 使用规则奖励
  rollout_gbs: 256             # Rollout 全局 batch size
  sampling_repeat_n: 32        # 每个 prompt 采样次数

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

### 自定义扩展

使用 V4 进行自定义任务，通常只需修改三部分：

1. **YAML 配置文件** — 模型路径、并行度、数据路径等
2. **Dataset 加载函数** — 参考 `tasks/math_rl_v4/simple_dataset.py`
3. **Reward 函数** — 参考 `tasks/math_rl_v4/bt_reward.py`

---

## V3 GRPO 训练架构（Legacy）

> V3 文档的详细内容请参考 [Trainer V3 (Legacy)](v3_legacy.md)。

### 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GRPO 训练循环                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
    │      Actor       │      │     Sampler      │      │    Critic/RM     │
    │    (Megatron)    │      │  (vLLM/SGLang)   │      │    (Megatron)    │
    └────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
             │                         │                         │
             │  ① 发送 prompt          │                         │
             │ ───────────────────────>│                         │
             │                         │                         │
             │  ② 返回 rollout         │                         │
             │ <───────────────────────│                         │
             │                         │                         │
             │  ③ 发送 rollout                                   │
             │ ─────────────────────────────────────────────────>│
             │                                                   │
             │  ④ 返回 rewards                                   │
             │ <─────────────────────────────────────────────────│
             │                         │                         │
    ┌────────▼─────────┐               │                         │
    │  ⑤ 计算优势值     │               │                         │
    │  (advantages)    │               │                         │
    └────────┬─────────┘               │                         │
             │                         │                         │
    ┌────────▼─────────┐               │                         │
    │  ⑥ 前向传播       │               │                         │
    │  计算 log_probs  │               │                         │
    └────────┬─────────┘               │                         │
             │                         │                         │
    ┌────────▼─────────┐               │                         │
    │  ⑦ 计算 Loss     │               │                         │
    │  - actor_loss    │               │                         │
    │  - kl_loss       │               │                         │
    │  - entropy_bonus │               │                         │
    └────────┬─────────┘               │                         │
             │                         │                         │
    ┌────────▼─────────┐               │                         │
    │  ⑧ 反向传播       │               │                         │
    │  计算梯度         │               │                         │
    └────────┬─────────┘               │                         │
             │                         │                         │
    ┌────────▼─────────┐               │                         │
    │  ⑨ 参数更新       │               │                         │
    │  optimizer.step  │               │                         │
    └────────┬─────────┘               │                         │
             │                         │                         │
    ┌────────▼─────────┐               │                         │
    │  ⑩ 同步权重到     │               │                         │
    │  Sampler (可选)  │ ─────────────>│                         │
    └──────────────────┘               │                         │
             │                         │                         │
             └─────────────── 下一轮迭代 ──────────────────────────┘
```

### 组件说明

| 组件 | 作用 | 启动方式 |
|------|------|----------|
| **Actor** | 策略模型训练，计算 GRPO loss 并更新参数 | `train_ppo_actor.py` |
| **Sampler** | 独立推理进程，使用 vLLM/SGLang 生成 rollout | `train_ppo_sampler.py` |
| **Critic/RM** | 计算奖励，支持规则奖励和模型奖励 | `train_ppo_critic.py` |

### 训练流程详解

| 步骤 | 说明 | 关键代码/函数 |
|------|------|---------------|
| ① 发送 prompt | Actor 从数据集获取 prompt，发送给 Sampler | `rollout_get_batch()` |
| ② 返回 rollout | Sampler 生成多个采样结果返回 | `gen_rollouts()` |
| ③ 发送 rollout | Actor 将生成结果发送给 Critic/RM | RPC 通信 |
| ④ 返回 rewards | Critic/RM 计算奖励值返回 | 规则奖励 / 模型奖励 |
| ⑤ 计算优势值 | 根据 rewards 计算 GRPO advantages | `calculate_grpo_advantages()` |
| ⑥ 前向传播 | 计算当前策略的 log_probs | `from_parallel_logits_to_logprobs()` |
| ⑦ 计算 Loss | 计算 actor_loss + kl_loss - entropy_bonus | `loss_func()` |
| ⑧ 反向传播 | 计算梯度 | `loss.backward()` |
| ⑨ 参数更新 | 优化器更新参数 | `optimizer.step()` |
| ⑩ 同步权重 | 将更新后的权重同步到 Sampler（可选） | 权重同步机制 |

---

## 快速开始

### 推荐入门路径

1. **V4 GRPO 训练（推荐）** - 参考 [Math GRPO V4 Demo](math_grpo_v4_demo.md)
2. **V3 例子** - 参考 [Trainer V3 (Legacy)](v3_legacy.md)（含 SFT、GRPO、VQA）

---

## 自定义 Loss 开发指南

通过继承 `GptPpoActorModel` 并重写 `get_actor_grpo_forward_output_and_loss_func` 方法来实现自定义的 loss 计算逻辑，**无需修改原有代码**。

### 文件结构

```
tasks/my_custom_loss/
├── __init__.py
├── custom_actor_model.py    # 自定义 Actor 模型 (DeepSeek 3.2 无偏 KL 估计)
├── train_actor.py           # 训练入口
└── README.md                # 详细文档
```

### 使用方式

运行训练时，将原来的 `train_ppo_actor.py` 替换为 `train_actor.py`：

```bash
python tasks/my_custom_loss/train_actor.py [原有参数...]
```

---

### 示例：DeepSeek 3.2 无偏 KL 估计

DeepSeek 3.2 论文中提出了无偏的 KL 散度估计方法，通过乘上重要性采样系数来修正 KL 估计的偏差。

#### 背景

在 GRPO 训练中，我们用旧策略 π_old 采样，但想估计当前策略 π_θ 对参考策略 π_ref 的 KL 散度：

```
原始 KL 估计 (有偏): E_{π_old}[KL(π_θ || π_ref)]
无偏 KL 估计:        E_{π_old}[(π_θ/π_old) * KL(π_θ || π_ref)]
```

通过乘上重要性采样系数 `ratio = π_θ/π_old`，将基于旧策略采样的 KL 估计修正为对当前策略的无偏估计。

#### 核心代码改动

```python
# 1. 计算重要性采样系数 (importance sampling ratio)
#    ratio = π_θ(a|s) / π_θ_old(a|s) = exp(log π_θ - log π_θ_old)
log_ratio = curr_log_probs - prev_log_probs
ratios = log_ratio.exp()

# 2. 计算原始 KL loss (有偏估计)
raw_kl = calculate_kl_loss(
    cur_log_probs=curr_log_probs,
    ref_log_probs=ref_log_probs,
    use_absolute_kl=False,
    use_low_var_kl=True,
)

# 3. 【核心改动】DeepSeek 3.2 无偏 KL 估计
#    无偏估计: E_{π_old}[(π_θ/π_old) * KL(π_θ || π_ref)]
#    注意: 对 ratio 做 detach，避免 KL loss 的梯度通过 ratio 传播
unbiased_kl = raw_kl * ratios.detach()

# 4. 计算最终的 KL loss
kl_loss = masked_mean(unbiased_kl, mask)
```

### 需要修改的位置

用 `################################################################` 注释标记，搜索 `【开始】` 和 `【结束】` 即可定位：

| 文件 | 标记 | 说明 |
|------|------|------|
| `custom_actor_model.py` | `【开始】DeepSeek 3.2 无偏 KL 估计实现` | 无偏 KL 估计核心逻辑 |
| `custom_actor_model.py` | `【新增 metrics】对比有偏和无偏 KL` | 新增的 metrics |
| `train_actor.py` | `【开始】导入自定义 Actor 模型` | 导入语句 |
| `train_actor.py` | `【开始】自定义 Actor Provider` | Actor 创建逻辑 |

---

### 可用的变量

在 `loss_func` 内部可以使用以下变量：

| 变量 | 类型 | 说明 |
|------|------|------|
| `mask` | `torch.Tensor` | 响应部分的 mask |
| `advantages` | `torch.Tensor` | 优势值 |
| `prev_log_probs` | `torch.Tensor` | 上一轮策略的 log 概率 (π_old) |
| `ref_log_probs` | `torch.Tensor` | 参考模型的 log 概率 (π_ref) |
| `curr_log_probs` | `torch.Tensor` | 当前策略的 log 概率 (π_θ) |
| `target` | `torch.Tensor` | 目标 token ids |
| `scaled_entropy` | `torch.Tensor` | 缩放后的熵 |
| `self.config` | `GpatchTransformerConfig` | 配置对象 |
| `self.ratio_eps` | `float` | clip ratio 的 epsilon |
| `self.entropy_bonus` | `float` | 熵 bonus 系数 |

---

## 可用的辅助函数

```python
from gpatch.core.aligner_helper import (
    from_parallel_logits_to_logprobs,  # logits -> log probs
    masked_mean,                        # mask 平均
    average_losses_across_data_parallel_group,  # DP 组平均
)
from gpatch.core.ppo_helper import (
    vocab_parallel_entropy,     # 熵计算
    calculate_kl_loss,          # KL loss
    calculate_grpo_advantages,  # GRPO 优势
    create_mask,                # 创建 mask
)
```

---

## 配置系统

### V4 配置（Hydra + OmegaConf）

V4 使用 Hydra YAML 配置，所有配置 dataclass 定义在 `gpatch_v4/configs/`。
详细 API 参考请查看 [V4 Config API](api/configs_v4.rst)。

### V3 配置（命令行参数）

模型配置文件位于 `gpatch/model_yamls/`，支持的模型包括：

- Qwen 系列: `qwen2.5-*`, `qwen3-*`, `qwen3vl-*`
- LLaMA 系列: `llama2-*`, `llama3.1-*`, `llama3.2-*`
- DeepSeek 系列: `deepseek-v2-*`, `deepseek-v3`
- 其他: `gemma3-*`, `internvl3-*`, `glm4*`

训练参数定义在 `gpatch/training/arguments.py`，完整参数列表参考 [V3 参数说明](trainer_v3_args.md)。

---

## 参考文档

- [Math GRPO V4 Demo](math_grpo_v4_demo.md)（推荐入口）
- [Math SFT](math_sft.md)
- [Math GRPO](math_grpo.md)
- [VQA](vqa.md)
- [GDataset](gdataset.md)
- [Trainer V3 (Legacy)](v3_legacy.md)

---

## Citation

```
@misc{wu2025wechatyattscalablesimpleefficient,
      title={WeChat-YATT: A Scalable, Simple, Efficient, and Production Ready Training Library}, 
      author={Junyu Wu and Weiming Chang and Xiaotao Liu and Guanyou He and Tingfeng Xian and Haoqiang Hong and Boqi Chen and Hongtao Tian and Tao Yang and Yunsheng Shi and Feng Lin and Ting Yao and Jiatao Xu},
      year={2025},
      eprint={2508.07970},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.07970}, 
}
```
