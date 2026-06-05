# Dump Metrics

## 功能概述

训练期间采集各个iteration相关信息（支持配置间隔x个ppo_step执行一次）并落盘，采集信息包含每条样本每个token的tokenid、reward、gt_label、ratio、是否被clip、entropy、top-k logps及其对应vocabulary的indices。


## 新增参数

- `ppo-dump-metrics`: 常规训练metrics落盘开关
- `ppo-dump-moe-metrics`: MoE相关metrics落盘开关
- `ppo-dump-logprobs-topk`: 输出层映射到词表时，topk的logprobs的落盘开关
- `ppo-save-train-data-interval`: 每间隔X个ppo-step保存一次training metrics, 必须>0才会保存
- `ppo-save-train-data-dir metrics`: 保存的位置, 需要存盘时必须设置，默认为None保存不了
- `ppo-save-train-data-logits-topk`: logprobs取topk


## 可落盘metrics

### 基础metrics ###
只要设置了 ppo-save-train-data-interval 和 ppo-save-train-data-dir 就会落盘
0. iteration
1. tokens
2. gt_label
3. rewards
4. rewards_details (if this key exists)

### loss相关metrics ###
由 开关 ppo-dump-metrics 控制
5. topk_logprobs 需要指定参数ppo-save-train-data-logits-topk，不然返回None。
6. topk_token_ids 与topk_logprobs同步
7. topk_logits 与topk_logprobs同步
8. ppo_ratio_unclamped 截断前的ratio
9. is_ppo_ratio_clamped 是否被截断
10. per_token_entropy 求mean之前，每个token单独的entropy（未mask）
11. mask

### MoE Routing Topk Metrics ###
由 开关 ppo-dump-moe-metrics 控制
12. moe_topk_info 训练阶段每一个MoELayer Routing选择的topk experts的scores和indices


## 保存方式

### 目录结构组织

每个ppo-step创建一个独立目录，格式：`iteration{iteration}_ppo_step{step}_{timestamp}`

```
ppo-save-train-data-dir
├── tmp/
├──── iteration8_ppo_step0_20251216_200038/
│     ├── dp0.pt
│     ├── dp1.pt
│     ├── dp2.pt
│     ├── ...
│     └── dp7.pt
├──── iteration8_ppo_step1_20251216_200139/
│     ├── dp0.pt
│     ├── dp1.pt
│     └── ...
│     ...
├── train_info_20251216_200056.tar  # 每100个ppo-step打包一次
└── train_info_20251216_201200.tar
```

### 文件命名规则

- **目录命名**：`iteration{iteration}_ppo_step{step}_{YYYYMMDD_HHMMSS}`(每个ppo-step存一个文件夹)
- **文件命名**：`dp{dp_rank}.pt`（ppo-step内每个dp-rank对应一个文件）
- **打包文件**：`train_info_{YYYYMMDD_HHMMSS}.tar`（目前每100个ppo-step打包一次，防止文件过多）

### pt文件格式

每个pt文件保存对应dp-rank上该ppo-step训练的所有samples。
**数组长度**：对应dp-rank上该ppo-step训练的总samples数量

```python
# dp{dp_rank}.pt 文件内容格式
[
    # sample1
    {
        "iteration": ...,
        "tokens": ...,
        "rewards": ...,
        "rewards_detals": ...,
        "gt_label": {
            "test_case": ...,
            "data_id": ...,
            "image_uin": ...,
            "image_key": ...
        }
        "topk_logprobs": ...,
        "topk_token_ids": ...,
        "topk_logits": ...,
        "ppo_ratio_unclamped": ...,
        "is_ppo_ratio_clamped": ...,
        "per_token_entropy": ...,
        "mask": ...,
        "moe_topk_info": {
            "layer1": {
                "topk_scores": ..., 
                "topk_indices": ...
            }, 
            "layer2": {
                ...
            }, 
            ...
        }
    },
    # sample2
    {
        ...
    },
    # ... 更多samples
]
```
