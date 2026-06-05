# Dynamic Context Parallel (Dynamic CP)

## 背景 / Background

在 RL 训练中，同一个 batch 内的序列长度差异往往很大（短 prompt 可能只有几十 token，长推理链可能上千 token）。使用固定的 Context Parallel (CP) 大小时，所有序列都按相同粒度切分到多个 GPU，导致短序列浪费大量 GPU 算力、长序列分配不均。

In RL training, sequence lengths within a batch can vary drastically. Fixed Context Parallel assigns the same CP size to all sequences, wasting GPU compute on short sequences and creating imbalance on long ones.

**Dynamic CP** 根据每条序列的长度动态调整 CP 组大小：长序列分配更多 GPU 做 CP 切分，短序列用更少 GPU 甚至打包到同一 GPU 上。这样可以在不改变总 GPU 数的前提下显著提高训练吞吐。

**Dynamic CP** dynamically adjusts the CP group size per sequence: long sequences get more GPUs for CP splitting, short sequences get fewer GPUs or are packed together. This significantly improves training throughput without changing the total GPU count.

## 已支持的训练类型 / Supported Training Types

| 训练类型 | 状态 | 说明 |
|---|---|---|
| GRPO / PPO (RL) | ✅ 已支持 | RL 特有字段 (advantages, logprobs 等) 自动处理 |
| SFT (Supervised Fine-Tuning) | ✅ 已支持 | 标准 CE loss，shift 在 converter 中完成 |
| DPO | 🔜 待适配 | 需要新增 converter + forward step |
| On/Off-Policy Distill | 🔜 待适配 | 需要新增 converter + forward step |

## 配置方法 / Configuration

在 `rl_config.yaml` 或 `math_sft.yaml` 的 `policy.dist_config` 下添加三个字段：

Add three fields under `policy.dist_config` in your config yaml:

```yaml
policy:
  dist_config:
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 2
    context_parallel_size: 2          # 最大 CP 组大小 / max CP group size
    dynamic_context_parallel: True     # 开启 Dynamic CP / enable Dynamic CP
    max_seqlen_per_dp_cp_rank: 4096   # 每个 rank 最大 token 数 / max tokens per rank
    min_dynamic_context_parallel_size: 1  # 最小 CP 组大小 / min CP group size
```

### 参数说明 / Parameter Reference

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `dynamic_context_parallel` | bool | `False` | 是否开启 Dynamic CP。开启后训练时自动按序列长度动态调整 CP 分组 |
| `max_seqlen_per_dp_cp_rank` | int | `None` | 每个 DPxCP rank 能容纳的最大 token 数。建议从模型在单 GPU 上能跑的最大序列长度开始调。**开启 Dynamic CP 时必填** |
| `min_dynamic_context_parallel_size` | int | `1` | 最小 CP 组大小。`1` 表示最短的序列可以不做 CP 切分（独占一个 GPU）。设为 `2` 则至少 2 GPU 做 CP |
| `context_parallel_size` | int | `1` | 此字段定义了 DPxCP 资源池的大小上限。Dynamic CP 会在 `[min, context_parallel_size × dp_size]` 范围内动态选择 |

## 使用前提 / Prerequisites

- 需要 Megatron-LM 版本包含 Dynamic CP 支持（PR #2924, #3405, #2000）
- `context_parallel_size` 须 ≥ 2（否则 Dynamic CP 没有意义）
- 当前仅支持 Megatron 后端 (`training_backend: mcore`)
- 不支持与 CUDA Graphs 同时使用

## 当前限制 / Known Limitations

下面这些能力暂未接入 Dynamic CP，开启时会直接 assert 失败，请保持关闭：

| 不兼容能力 | 行为 |
|---|---|
| `virtual_pipeline_model_parallel_size > 1` (VPP) | `DistConfig.__post_init__` 阶段 assert |
| Multi-Token Prediction (`model_arch == deepseek_v3`) | 运行时进入 `_run_dyn_cp_training` 时 assert |
| `ppo_dump_metrics_interval > 0` | 运行时进入 `_run_dyn_cp_training` 时 assert |

## 示例 / Example

基于 `tasks/math_rl_v4/scripts/grpo.sh` 的 GRPO demo 已默认开启 Dynamic CP，直接运行即可：

The GRPO demo at `tasks/math_rl_v4/scripts/grpo.sh` has Dynamic CP enabled by default:

```bash
bash tasks/math_rl_v4/scripts/grpo.sh
```

SFT 训练同样支持 Dynamic CP，在 config 中配置即可。

SFT training also supports Dynamic CP — just add the config fields above to your SFT yaml.

关闭 Dynamic CP 只需设置 `dynamic_context_parallel: False`，系统会回退到原有的固定 CP 模式。

To disable Dynamic CP, set `dynamic_context_parallel: False` and the system falls back to fixed CP.

## 调参建议 / Tuning Tips

1. **`max_seqlen_per_dp_cp_rank`** 是最关键的参数。设得太小会导致 microbatch 数量增多、调度开销变大；设得太大可能 OOM。建议从单 GPU 能跑的最大序列长度的 80% 开始试。

2. **`min_dynamic_context_parallel_size`** 通常保持 `1` 即可。如果你的 attention 实现要求 CP ≥ 2，则设为 `2`。

3. 观察日志中的 `[TRAIN-DYN-CP]` 或 `[SFT-DYN-CP]` 标记判断调度效果，`scheduled num_micro_batches=...` 越少，GPU 利用率越高。

## 接入新的训练类型 / Adding Support for a New Training Type

如果要为新的训练类型（如 DPO、Distill）接入 Dynamic CP，按下面三步：

To add Dynamic CP support for a new training type, follow these three steps:

### Step 1: 编写 converter / Write a converter

在 `gpatch_v4/utils/dynamic_cp_utils.py` 中添加 `convert_xxx_samples_to_dyn_cp_format(samples)`，把每条原始样本转换成 Dynamic CP 标准格式。padding 对齐由内部的 `_get_total_pad_divisor()` 处理。

Add `convert_xxx_samples_to_dyn_cp_format(samples)` in `gpatch_v4/utils/dynamic_cp_utils.py`. Padding alignment is handled internally by `_get_total_pad_divisor()`.

每条样本必须包含的字段（长度 = padded_len）：
- `tokens` (int64)，`labels` (int64)，`loss_mask` (float32)，`position_ids` (int64)
- `original_seq_len`、`padded_seq_len`：`torch.tensor([len], dtype=torch.int32)`

额外的 per-token 字段写进 dict 即可自动参与调度（all-to-all、packing、CP slicing），长度需与 `tokens` 一致。

参考 `convert_rl_samples_to_dyn_cp_format` 与 `convert_sft_samples_to_dyn_cp_format` 实现。

### Step 2: 编写 forward step / Write a forward step

在 `gpatch_v4/training_backend/megatron_backend/mixin.py` 添加 `xxx_forward_step_dyn_cp()`，返回一个 `fwd_output_and_loss_func` 闭包。内部用 `get_batch_for_dyn_cp(data_iterator, dynamic_cp=True, extra_token_keys=(...))` 取一条 packed batch（SFT 不用额外字段时传 `()`）。

```python
def xxx_forward_step_dyn_cp(self):
    def fwd_output_and_loss_func(data_iterator, model):
        batch, packed_seq_params = get_batch_for_dyn_cp(
            data_iterator, dynamic_cp=True,
            extra_token_keys=("my_extra_field",),
        )
        # ... model forward with packed_seq_params ...
        # ... loss computation (不要做 static CP all-reduce) ...
    return fwd_output_and_loss_func
```

### Step 3: 在 training step 入口分流 / Dispatch from training step

在对应的 step 方法（如 `_finetune_step` / `_update_policy`）开头加 Dynamic CP 分支，复用共享 helper：

```python
if self.dist_config.dynamic_context_parallel:
    return self._run_dyn_cp_training(
        batch,
        num_microbatches,
        converter=convert_xxx_samples_to_dyn_cp_format,
        forward_step_func=self.xxx_forward_step_dyn_cp(),
        metric_prefix="xxx",
        log_tag="XXX-DYN-CP",
        forward_only=False,        # SFT eval 路径可传 True
    )
```

`_run_dyn_cp_training` 负责：converter → `run_dyn_cp_schedule` → 前后向 → metrics 聚合并广播。

### 注意事项 / Notes

- packed 后每个 microbatch 在 token 维度上是 **THD 格式**（所有 sample 拼接成一条），所以 `micro_batch_size=1`。
- Loss 函数内 **不要** 用 `mpu.get_context_parallel_group()` 做 static CP all-reduce，CP 通信由 attention 层内部处理。
- 长度为 `seq_len - 1` 的额外字段（如 RL 的 logprobs）需在 converter 中先对齐到 `seq_len`、再 pad 到 `padded_len`，loss 函数中再 `[:, :-1]` 截回；converter 已把位置 `seq_len-1`（THD 子序列边界）的 `loss_mask` 置 0，跨子序列预测不会污染 loss。


# SFT 对齐
## math(qwen2)
1. 原版：
脚本：gcore-dev/tasks/math_rl_v4/scripts/sft.sh
wandb: http://wandb.testsite.woa.com:8080/plt2/dynamic_cp_riona/runs/0byzv68x?nw=nwuserplt2
acc: 0.575
fmt: 0.695

2. dynamic-cp:
脚本：gcore-dev/tasks/math_rl_v4/scripts/sft_dyn_cp.sh
wandb: http://wandb.testsite.woa.com:8080/plt2/dynamic_cp_riona/runs/s8qmm7rk?nw=nwuserplt2
acc: 0.578
fmt: 0.697

## qwen3vl
1. 原版：
脚本：gcore-dev/tasks/multimodal_v4/finetune/scripts/finetune_qwen3vl_pmc_vqa.sh
wandb: http://wandb.testsite.woa.com:8080/plt2/qwen3_vl_v4/runs/ptn8fb2m?nw=nwuserplt2
acc: 0.65

2. dynamic-cp:
脚本：gcore-dev/tasks/multimodal_v4/finetune/scripts/finetune_qwen3vl_dyn_cp.sh
wandb：http://wandb.testsite.woa.com:8080/plt2/qwen3_vl_v4/runs/i9kml6ew?nw=nwuserplt2
acc: 0.65
