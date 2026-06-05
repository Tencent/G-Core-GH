Sampling Strategy
=================

Gcore 当前支持以下 rollout sampling 策略：

- dynamic sampling
- partial rollout
- oversampling

## dynamic sampling

coming soon ...

## partial rollout

coming soon ...

## oversampling

### why oversampling

rollout 阶段可能存在学习价值不大的 sample，
例如 grpo 算法中 group 内所有 rewards 都相同的 samples，这些 samples 会拖慢 RL 训练的学习速度。
针对这些 samples，可以通过过滤掉学习价值不大的 samples 加速 RL 训练。

Oversampling 策略在 rollout 阶段，选取数量大于 rollout global batch size （之后简写为 rollout_gbs）
数目的 prompt 进行 rollout， 然后从中选取 rollout_gbs 的 samples 进行后续的 RL 训练。

相比于 dynamic sampling 策略，Oversampling 可以避免多次 backfill （减少 rollout 次数），提高 rollout 速度。

### 实现原理

具体实现原理参考[GLM-4.5V](https://arxiv.org/abs/2507.01006v5) RLCS。
训练中每次 rollout 从数据中拿出 `expansion_ratio * rollout_gbs` 条 prompt做rollout，
之后根据`score_func`选取 `rollout_gbs` 条 samples / groups，进行后续训练。

当前实现的版本采用 ema 方式更新 `expansion_ratio`（以 GRPO 为例）：

- 在训练开始设置 `expansion_ratio=init_expansion_ratio`
- 每一个 PPO step
  1. 采样`expansion_ratio * rollout_gbs` 条 prompts 做 rollout，得到`expansion_ratio * rollout_gbs`个group 
  2. 根据`score_func`计算每个 group 的 score，以及该 group 是否 valid，计算所有 group 中 invalid group 的占比`invalid_ratio`
  3. 当前 step 的 `step_expansion_ratio` = `min(1 - (1 / invalid_ratio), max_expansion_ratio)`
  4. 更新 `expansion_ratio = expansion_ratio * ema_decay + (1 - ema_decay) * step_expansion_ratio`
  5. 后续训练 ...

### 使用

#### 运行
通过显示设置`--ppo-oversampling-init-expansion-ratio`方式 enable oversampling。

具体运行方式可参考`tests/test_gpatch_v3/test_oversampling/grpo_oversampling.sh`
以下参数与 oversampling 有关

- `--ppo-oversampling-ema-decay`
  - 对应上述 ema_decay，设置 ema decay 的指数
  - float，0 <= ema_decay <= 1
- `--ppo-oversampling-init-expansion-ratio`
  - 对应上述 init_expansion_ratio，设置 expansion_ratio 初始值
  - float，init_expansion_ratio >= 1.0
- `--ppo-oversampling-max-expansion-ratio`
  - 对应上述 max_expansion_ratio，对 step_expansion_ratio 做截断，防止batch size 爆炸
  - ppo_step 的采样 prompt 数目不会大于这个值
  - float，max_expansion_ratio >= max_expansion_ratio
- `--ppo-oversampling-curriculum-score-func-path`
  - 对应上述 score_func，设置用户自定义 score_func
  - List of str，`list[0]`为 score_func 所在 py 文件 path，`list[1]` 为 score_func 名字 
  - score_func return scalar，<= 表示该 group 无效，值越大表示该 group 学习价值越大
  - 默认 score_func 见 `gpatch/training/v3/oversampling.py`的`default_score_group_func`
- `--ppo-oversampling-delay-filter`
  - 将 oversampling 选择从超采样 groups 中 filter 出 rollout_gbs 个 groups 的过程推迟到 logps 计算之后
  - store_true，默认 False
  - 原因：计算 score 依赖 rewards
    - 默认情况下 logps 计算在 rewards 之后串行之行，filter 过程在 logps 之前
    执行，可以节省部分 logps 计算量
    - 在 logps 与 rewards 计算并行场景下，需要打开该选项等待 rewards 计算完毕后才可以 filter

#### metrics

**NOTE**:开启 oversampling 会使用 **Dual Logging Strategy**，会多 log 一次超采样的所有 sample 的 metrics。
具体原因可以参考 https://github.com/volcengine/verl/pull/2988（简单来讲就是对于数据集，log 过滤后的 samples 是有偏的，而超采的 samples 是无偏的）。
`rollout-metrics` section 下的所有 metrics 都会对超采的 samples 计算一次放在`oversampling-rollout-metrics`作为训练时的参考。

另外一些跟 oversampling 单独有关的 metircs 如下：
- `oversampling-meta/expansion_ratio`: 每个step的expansion_ratio
- `oversampling-meta/selected_score`: 选择的 group 数目的平均学习价值
- `oversampling-meta/selected_num_batches'` 选择的 group 数目
- `oversampling-meta/oversampling_score'` 超采 group 的平均学习价值
- `oversampling-meta/oversampling_num_batches'` 超采的总 group 数目

### 实验结果

math_rl 结果待补充