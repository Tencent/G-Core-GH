# Dynamic Sampling V4

每个 PPO step 按 group 过滤低学习价值的 rollout，不够就 refill，直到凑满 `rollout_gbs` 个 group 再训练。一个 group = 一条 prompt + `sampling_repeat_n` 条 response。

默认过滤规则来自 DAPO：组内 reward 全相同（全对 / 全错）的 group 不进训练。


## 当前支持

打开方式：`policy.rollout_gen_type: dynamic_sampling`。

### Epoch 口径

| `epoch_mode` | 停训条件 | 和 baseline 比什么 |
|---|---|---|
| `fixed_ppo_steps`（默认） | `num_train_epoches * ppo_step_per_epoch` 个 PPO step | 同样更新次数 |
| `consumed_data_epochs` | 累计消费满 `num_train_epoches` 个 epoch 的 prompt | 同样数据量 |

DS 每步会超采，所以这两种口径的 PPO step 数不一样。细节见下面「算法」。

### Filter

- **默认**：组内 `rewards` 不全相同才保留。只对 `advantage_type` 为 `grpo` / `gdpo` / `gdpo_sample_bn` / `group_gdpo` / `group_gdpo_sample_bn` 生效；其它 advantage 不过滤。
- **自定义**：同时设 `filter_py_path` 和 `filter_fn_name`。

### 可以一起用

- placement：colocate、disaggregated（按 wave 调度，方便 colocate 分时占卡）
- reward：BT-RM、Gen-RM、external reward（每波 generation 之后、filter 之前算完）
- eval：走普通 `BaseRolloutGenerator`，不做 DS filter
- 断点续训：保存 / 恢复 DataSource 游标（`current_epoch`、`batches_consumed`）、自包含的 `prompt_buffer`、以及 `ema_expansion_ratio`。需要 `ResumableDistributedSampler`；换 DP 数续训不支持。

### 暂不支持

- `async_rollout`、`single_controller`
- `stream_external_reward`（DS 下关掉；external reward 改在 generator 内部按 wave 算）
- prompt 级 early stop + abort（filter 要等整波 reward；colocate 只能按 batch onload/offload）
- 换 DP 数之后从 DS checkpoint 续训

### 数据约束

- dataloader 每个 batch 的 prompt 数必须等于 `rollout_mbs`（一般是 1）
- prompt batch 必须是 list-valued


## 算法

每个 PPO step 的目标是收集 `target = rollout_gbs` 个 **valid** group（按 DP 切分后，本 rank 目标是 `num_microbatches * rollout_mbs`）。

```
ema = init_expansion_ratio
for ppo_step:
    selected = []
    for wave in 1 .. (1 + max_refill_times):
        gap = target - len(selected)
        if gap <= 0:
            break
        n = ceil(gap * ema * oversampling_ratio)
        n = 上取整到 rollout_mbs 的倍数
        从 prompt_buffer 或 dataloader 取 n 条 prompt（buffer 只在够一整波时用；存的是未 strip 的完整 prompt）
        sampler → external reward → gen-rm → bt-rm
        按 group filter:
            valid 且 selected 未满 → 放入 selected
            valid 但 selected 已满 → 把原始 prompt 放回 prompt_buffer（生成结果丢弃，下次再采）
            invalid → 记下，必要时用来 pad
    if selected 仍不足:
        用 invalid group 补齐，并打 sample_mask=False
        # 第一波 n >= target（ema、oversampling_ratio 均 >= 1），issued >= target，
        # invalid = issued - selected >= need，一定够 pad
    用本 step 全局 valid/invalid 数更新 ema
    训练 selected
```

EMA（先在 DP 上 all-reduce 再算）：

```
current_ratio = max_expansion_ratio          # num_valid == 0
              = min(total / num_valid, max_expansion_ratio)  # otherwise
ema = ema_decay * ema + (1 - ema_decay) * current_ratio
```

`current_ratio` 的含义是「要凑 1 个 valid group，平均要采多少个 group」。下一波按这个估计超采。

### epoch_mode

DS 每步会多吃 prompt，所以「一个 epoch」有两种口径：

| `epoch_mode` | 停训条件 | 适用对比 |
|---|---|---|
| `fixed_ppo_steps`（默认） | 和普通 GRPO 一样：`num_train_epoches * ppo_step_per_epoch` 个 PPO step | 同样更新次数下 DS 有没有用 |
| `consumed_data_epochs` | 某个 PPO step 里累计消费满 `num_train_epoches` 个 epoch 的 prompt 后停。最后一步为了 refill 可能跨进下一个 epoch | 同样数据量下 DS 有没有用 |

`fixed_ppo_steps` 下 DS 的 PPO step 数和 baseline 一样，但 generation 量和吃掉的 prompt 都更多。


## 参数

在 yaml 的 `training.dynamic_sampling` 下：

| 参数 | dtype | default | 含义 |
|---|---|---|---|
| `oversampling_ratio` | float `>= 1` | 1.2 | 在 EMA 估计之上再多采一点，减少 refill 次数 |
| `ema_decay` | float `(0, 1)` | 0.9 | 越接近 1，expansion ratio 更新越慢 |
| `init_expansion_ratio` | float `>= 1` | 1.0 | 第一步的 `ema` |
| `max_expansion_ratio` | float `>= init` | 10.0 | `current_ratio` 和 EMA 的上限，防止单波 batch 爆炸 |
| `max_refill_times` | int `>= 0` | 3 | 第一波之后最多再 refill 几波。`0` 表示只采一波；valid 不够时用 invalid pad 补齐（第一波已 `>= target`，一定够） |
| `filter_py_path` / `filter_fn_name` | str or null | null | 自定义 filter，必须成对出现或都省略 |
| `epoch_mode` | str | `fixed_ppo_steps` | `fixed_ppo_steps` 或 `consumed_data_epochs` |

自定义 filter 签名：

```python
def my_filter(config, group) -> tuple[bool, str]:
    # group: 一条 prompt 的 sampling_repeat_n 条 sample
    # 返回 (是否保留, 原因字符串)，原因会打到 dynamic_sampling/filter_reason/<reason>
    ...
```

默认 filter：组内 `rewards` 不全相等则 `(True, "valid")`，否则 `(False, "equal_rewards")`。


## 使用方式

打开 DS 只需要两处：

```yaml
policy:
  rollout_gen_type: dynamic_sampling

training:
  dynamic_sampling:
    oversampling_ratio: 1.2
    ema_decay: 0.9
    init_expansion_ratio: 1.0
    max_expansion_ratio: 10.0
    max_refill_times: 3
    epoch_mode: fixed_ppo_steps
```

参考配置和启动脚本：

- `tasks/math_rl_v4/yaml/rl_dynamic_sampling.yaml`
- `tasks/math_rl_v4/scripts/grpo_dynamic_sampling.sh`

```bash
cd /path/to/gcore
bash tasks/math_rl_v4/scripts/grpo_dynamic_sampling.sh
```

脚本内部是：

```bash
python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_dynamic_sampling.yaml"
```

和 baseline 对比时，两边的 `rollout_gbs` / `sampling_repeat_n` / 模型 / 数据应对齐，只改 `rollout_gen_type` 和 `training.dynamic_sampling`。`epoch_mode` 决定「对齐 step 数」还是「对齐数据量」。


## Metrics

每步会多报一组 `dynamic_sampling/*`（过滤后的 train `rollout-rewards/*` 仍会报，但是 **post-filter** 口径）：

| key | 含义 |
|---|---|
| `dynamic_sampling/ema_expansion_ratio` | 更新后的 EMA，下一 step 用 |
| `dynamic_sampling/num_waves` | 本 step 实际采了几波 |
| `dynamic_sampling/issued_groups` | 本 step 发出去生成的 group 数 |
| `dynamic_sampling/selected_groups` | 送进训练的 group 数（valid + 必要时的 invalid pad） |
| `dynamic_sampling/num_valid_groups` | 本 step 判定 valid 的 group 数（含没装进 selected、退回 buffer 的） |
| `dynamic_sampling/num_invalid_groups` | 被 filter 掉的 group 数 |
| `dynamic_sampling/padded_invalid_groups` | 用 invalid 补齐训练 batch 的数量 |
| `dynamic_sampling/buffer_size` | 当前 `prompt_buffer` 长度 |
| `dynamic_sampling/filter_reason/<reason>` | 各 filter 原因的计数 |

判断「DS 有没有训得更好」应看 **eval**（eval 不做 filter，和 baseline 协议一致）。train 上的 `rollout-rewards/global_acc_reward` 只统计送进训练的 group，全对/全错被滤掉后会偏，不能直接和 baseline 叠曲线。
