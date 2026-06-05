# Dump Metrics V4

## 功能概述

V4 训练框架支持在 **RL 训练（GRPO/GSPO）** 和 **SFT 训练** 期间，按配置间隔采集每个 ppo_step（或 train_step）的训练指标并落盘保存。采集内容包括每条样本的 token ids、reward、各阶段 logprobs（rollout / ref / curr）、ratio、是否被 clip、per-token entropy、advantage、top-k logprobs 及对应 token ids，以及 MoE 路由 top-k 信息。


## 配置参数

在 yaml 配置文件的 `training` 节下设置：

| 参数 | dtype | default | 说明 |
|------|------|------------|------|
| `ppo_dump_metrics_interval` | int | -1 | 每隔 N 个 ppo_step 落盘一次，`-1` 表示关闭 |
| `ppo_dump_metrics_dir` | str | `""` | 落盘目录路径，开启落盘时必须设置 |
| `ppo_dump_per_token_entropy` | bool | False | 是否落盘 per-token entropy（注意⚠️此开关仅用于控制sft, 是由于sft在拿到per_token_entropy时会有重复计算，影响训练效率，所以加了个开关，默认关闭，需要才打开；⚠️grpo不会产生重复计算，默认会对per_token_entropy落盘，不用加这个开关） |
| `dump_metrics_logprobs_topk` | int | 0 | 算loss前得到的输出层 logprobs 取 top-k，[b,s,vocab_size]->[b,s,topk] |
| `ppo_dump_moe_topk` | int | 0 | MoE routing 取 ppo_dump_moe_topk个experts（可以>训练的topk），`0` 表示关闭 |

### 配置示例

```yaml
training:
  ppo_dump_metrics_interval: 1
  ppo_dump_metrics_dir: "dump_moe_v4_grpo_0326"
  ppo_dump_per_token_entropy: True
  ppo_dump_moe_topk: 32
  dump_metrics_logprobs_topk: 10
```


## Metrics字段说明

### GRPO/GSPO RL 训练

> **关于 shape 中的长度标记**：由于 dump 数据来自不同的处理阶段，各字段的长度并不统一：
>
> | 标记 | 含义 | 来源 |
> |------|------|------|
> | `T` | 实际 token 序列长度 | rollout 生成的原始序列长度 |
> | `P` | precompute 阶段的 padded_seqlen | `compute_log_probs()` 时 smart_pad 分组的 batch max seqlen |
> | `S` | 训练阶段的 padded_seqlen | 训练 forward 时 batch max seqlen |
>
> 一般关系：`T ≤ P ≤ S`。不使用 smart_pad 且 batch 组成相同时 `P = S`；使用 smart_pad 时三者通常不同。
>
> shifted 类字段（logprobs、mask、entropy 等）长度为 `S-1` 或 `P-1`，因为自回归模型 N 个 token 只产生 N-1 个 next-token 预测。

#### 1.Rollout 相关metrics

| 字段 | shape | dtype | 说明 |
|------|-------|-------|------|
| `ppo_step` | scalar | int | 当前 ppo step |
| `tokens` | `[T]` | int32 | 完整 token ids（prompt + response） |
| `gt_label` | scalar 或 dict | - | ground truth 标签，取决于数据集 |
| `rewards` | `[1]` | float32 | 该样本的 reward |
| `rewards_details` | - | - | reward 明细（如有） |
| `rollout_logprobs` | `[T]` | bfloat16 | rollout 阶段推理引擎生成的 shifted logprobs，`rollout[i]` = log P(token[i+1] \| tokens[0:i+1])，prompt 位置为哨兵值 1.0 |
| `pre_logprobs` | `[P-1]` | bfloat16 | 当前 policy model（训练前）的 shifted logprobs，即 PPO ratio 的分母 π_old |
| `advantages` | `[P-1]` | bfloat16 | 每个 token 的 advantage 值 |


（实现：在 rollout 完成后，epoch2 循环外一次性采集，跨 `ppo_max_epochs_2` 共享。`pre_logprobs` 和 `advantages` 在 `compute_log_probs()` 阶段产生，其长度取决于该阶段 smart_pad 分组的 padded seqlen。）


#### 2.Loss 相关metrics

| 字段 | shape | dtype | 说明 |
|------|-------|-------|------|
| `curr_logprobs` | `[S-1]` | bfloat16 | 当前 policy model（训练中）的 shifted logprobs |
| `per_token_entropy` | `[S-1]` | bfloat16 | 每个 token 的 entropy（未 mask） |
| `topk_logprobs` | `[S, topk]` | bfloat16 | 输出层 top-k logprobs，需要配置 `dump_metrics_logprobs_topk` |
| `topk_token_ids` | `[S, topk]` | int32 | topk_logprobs 对应的 token-ids |
| `ppo_ratio_unclamped` | `[S-1]` | bfloat16 | clip 前的 ratio |
| `is_ppo_ratio_clamped` | `[S-1]` | bool | 该 token 的 ratio 是否被 clip |
| `mask` | `[S-1]` | bool | response mask |

（实现：在训练的每个 forward 中采集，由 pipeline last stage 广播到所有 PP ranks。）


#### 3.MoE Routing相关Metrics
moe_topk_info 由 `ppo_dump_moe_topk > 0` 控制，可以自定义需要dump的topk（支持ppo_dump_moe_topk>训练topk）

| 字段 | dtype | 说明 |
|------|-------|------|
| `moe_topk_info` | dict | 每层 MoE routing 的 top-k 信息 |

```python
"moe_topk_info": {
    "layer1": {
        "topk_scores": Tensor[S, ppo_dump_moe_topk],    # bfloat16
        "topk_indices": Tensor[S, ppo_dump_moe_topk],   # int16
    },
    "layer2": { ... },
    ...
}
```

（实现：在所有PP—ranks上采集其对应layers的数据，然后gather到pp0，pp0最后会拿到所有layers的moe topk数据,然后存盘。）

### SFT 微调

SFT 场景落盘字段较少（SFT 只有训练阶段，`S` 为该 batch 的 padded_seqlen）：

| 字段 | shape | dtype | 说明 |
|------|-------|-------|------|
| `train_step` | scalar | int | 当前 train step 编号 |
| `tokens` | `[T]` | int32 | 实际 token 序列（未 pad，长度为原始 token 数） |
| `mask` | `[S-1]` | bool | loss mask |
| `topk_logprobs` | `[S, topk]` | bfloat16 | 可选，需设置 `dump_metrics_logprobs_topk` |
| `topk_token_ids` | `[S, topk]` | int32 | 可选，与 topk_logprobs 同步 |
| `per_token_entropy` | `[S-1]` | bfloat16 | 可选，需设置 `ppo_dump_per_token_entropy` |
| `moe_topk_info` | dict | - | 可选，需设置 `ppo_dump_moe_topk > 0` |



## 目录结构

```
ppo_dump_metrics_dir/
├── tmp/
│   ├── PpoStep0_SubEpoch0_20260326_2135/
│   │   ├── dp0_rank0.pt
│   │   ├── dp1_rank4.pt
│   │   └── ...
│   ├── PpoStep0_SubEpoch1_20260326_2135/
│   │   └── ...
│   ├── PpoStep1_SubEpoch0_20260326_2137/
│   │   └── ...
│   └── ...
├── train_info_20260326_2200.tar    # 每 100 个目录自动打包
└── train_info_20260326_2300.tar
```

- **目录命名**：`PpoStep{step}_SubEpoch{epoch}_{YYYYMMDD_HHMM}`（RL 场景）或 `TrainStep{step}_{YYYYMMDD_HHMM}`（SFT 场景）
- **文件命名**：`dp{dp_rank}_rank{global_rank}.pt` 每个dp存一份.pt文件
- **自动打包**：`tmp/` 下累积满 100 个目录后，由 rank 0 在后台线程自动打包为 `.tar` 并删除原目录

> **注意**：当 `ppo_max_epochs_2 > 1` 时，同一个 ppo_step 会产生多个 SubEpoch 目录，每个 SubEpoch 的 loss 相关字段（如 `curr_logprobs`、`ppo_ratio_unclamped` 等）不同，但rollout相关字段（`tokens`、`rewards` 等）相同。


## pt 文件格式

每个 `.pt` 文件是一个 list，每个元素对应一条样本。注意各字段长度来自不同处理阶段（详见上方 shape 说明）：

```python
# dp{dp_rank}_rank{global_rank}.pt
[
    {
        "ppo_step": 0,
        "tokens": Tensor[T],                    # int32, 实际 token 序列
        "gt_label": ...,
        "rewards": Tensor[1],                    # float32
        "rewards_details": ...,
        "rollout_logprobs": Tensor[T],           # bfloat16
        "pre_logprobs": Tensor[P-1],             # bfloat16
        "advantages": Tensor[P-1],               # bfloat16
        "curr_logprobs": Tensor[S-1],            # bfloat16
        "per_token_entropy": Tensor[S-1],        # bfloat16
        "topk_logprobs": Tensor[S, topk],        # bfloat16
        "topk_token_ids": Tensor[S, topk],       # int32
        "ppo_ratio_unclamped": Tensor[S-1],      # bfloat16
        "is_ppo_ratio_clamped": Tensor[S-1],     # bool
        "mask": Tensor[S-1],                     # bool
        "moe_topk_info": {
            "layer1": {"topk_scores": Tensor[S, moe_topk], "topk_indices": Tensor[S, moe_topk]},
            "layer2": { ... },
            ...
        }
    },
    { ... },  # sample 2
    ...
]
```


## 读取示例

```python
import torch

data = torch.load("rank_0.pt", map_location="cpu", weights_only=False)
print(f"样本数: {len(data)}")

sample = data[0]
print(f"ppo_step: {sample['ppo_step']}")
print(f"tokens shape: {sample['tokens'].shape}")
print(f"reward: {sample['rewards']}")
print(f"mask 中 True 的数量: {sample['mask'].sum().item()}")

# 查看 top-k logprobs
if "topk_logprobs" in sample:
    print(f"topk_logprobs shape: {sample['topk_logprobs'].shape}")

# 查看 MoE routing 信息
if "moe_topk_info" in sample:
    print(f"MoE 层数: {len(sample['moe_topk_info'])}")
    layer1 = sample["moe_topk_info"]["layer1"]
    print(f"layer1 topk_probs shape: {layer1['topk_probs'].shape}")
```

另外，可以通过脚本 tools/align_dump_sample/align_sample.py 读取对齐后的字段