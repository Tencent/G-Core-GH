# V4 自定义 Policy Loss

本文档描述 **Megatron / mcore** 路径下自定义 policy loss 的注册方式与输入输出约定。
按 `ppo.use_legacy_loss` 分为两套接口，**请勿混用**。

## `use_legacy_loss=True`（legacy，后续计划弃用）

本文档暂不提供 legacy custom loss 的接口细节。已有 legacy 使用方可参考
`gpatch_v4/training_backend/loss_factory.py`；新增 custom loss 请使用下文
`use_legacy_loss=False` 的新版接口。

## `use_legacy_loss=False`

新版 policy loss 使用按 backend 隔离的统一注册表。内置实现见
`gpatch_v4/training_backend/loss/ppo_loss.py`，注册表实现在
`gpatch_v4/training_backend/loss/registry.py`。

### 注册

在配置中给 loss 指定一个未注册的名字，并提供 Python 文件的绝对路径和函数名：

```yaml
ppo:
  use_legacy_loss: false
  loss_func: my_loss
  loss_func_py_path: /abs/path/to/my_loss.py
  loss_func_py_name: my_loss_fn
```

`GrpoTrainActor` 初始化时按下面的顺序处理：

1. 查询 `("mcore", ppo.loss_func)` 是否已经注册。
2. 已注册时直接使用注册表中的函数，忽略 `loss_func_py_path` 和
  `loss_func_py_name`。
3. 未注册时从 `loss_func_py_path` 导入 `loss_func_py_name`，再以
  `backend="mcore"`、`loss_name=ppo.loss_func` 注册。

因此，自定义名字不要与当前 MCore 内置 loss 重名。通过配置加载的自定义
Python 文件只需导出普通函数，**不要再给该函数添加** `@register_loss` **装饰器**，
否则导入模块时会提前注册，随后自动注册将因重名失败。

通常不需要手工注册。确需在代码中注册时，接口为：

```python
from gpatch_v4.training_backend.loss import register_custom_loss_fn

register_custom_loss_fn(
    backend="mcore",
    loss_name="my_loss",
    py_path="/abs/path/to/my_loss.py",
    fn_name="my_loss_fn",
)
```

仓库内随模块导入而注册的实现可使用：

```python
from gpatch_v4.training_backend.loss import register_loss

@register_loss(backends=("mcore",), loss_name="my_loss")
def my_loss_fn(config, loss_input):
    ...
```

这种方式与上述配置自动注册方式二选一。

### 函数签名

```python
import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.training_backend.loss import PolicyLossInput


def my_loss_fn(
    config: RlConfig,
    loss_input: PolicyLossInput,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    ...
    return bwd_loss, bwd_count, metrics
```

- `config` 是完整训练配置，policy loss 的参数通常从 `config.ppo` 读取。
- `loss_input` 是新版
`gpatch_v4.training_backend.loss.PolicyLossInput`，不要从
`loss_factory` 导入 legacy 同名类型。
- 返回值必须是 `(bwd_loss, bwd_count, metrics)` 三元组。



### 输入：`PolicyLossInput`

进入新版 loss 前，MCore engine 会将当前策略输出和 rollout 数据整理为以
response 为单位的 padded tensor。静态 CP 的结果已 all-gather；dyn-CP/THD
数据也会先重建并转换成 `[B, S]`，因此自定义 loss 不处理 CP shard 或 THD
pack。


| 字段                                               | 类型              | 当前 MCore policy 路径中的含义                                                                             |
| ------------------------------------------------ | --------------- | -------------------------------------------------------------------------------------------------- |
| `advantages`                                     | `Tensor`        | 优势，通常为 `[B, S]`。自定义 top-k 算法可以使用额外维度，但传给 `agg` 前必须归约成 `[B, S]`                                     |
| `prev_log_probs`                                 | `Tensor | None` | 旧策略 logprob `[B, S]`；`ppo.skip_prev_logps=True` 时可能为 `None`                                        |
| `ref_log_probs`                                  | `Tensor | None` | reference logprob `[B, S]`；未启用 reference 时为 `None`                                                 |
| `curr_log_probs`                                 | `Tensor`        | 当前策略 logprob `[B, S]`，保留梯度                                                                         |
| `response_mask`                                  | `Tensor`        | 有效 response token mask `[B, S]`                                                                    |
| `scaled_entropy`                                 | `Tensor`        | 已按 mask 聚合的 entropy 标量                                                                             |
| `rollout_log_probs`                              | `Tensor | None` | sampler 侧 logprob `[B, S]`，用于 off-policy correction 等算法                                            |
| `per_token_entropy`                              | `Tensor | None` | 当前策略逐 token entropy `[B, S]`；当前 MCore actor 会填充                                                    |
| `prev_per_token_entropy`                         | `Tensor | None` | 旧策略逐 token entropy `[B, S]`，仅部分算法提供                                                                |
| `parallel_logits`                                | `Tensor | None` | 本 TP rank 的 vocab-parallel logits；linear CE 或 response-padded dyn-CP 下为 `None`，不保证与 `[B, S]` 输入同布局 |
| `sample_mask`                                    | `Tensor | None` | 样本 mask `[B]`；0 表示该样本不参与 seq-mean 聚合                                                               |
| `token_weights`                                  | `Tensor | None` | `[B, 1]` 或 `[B, S]`；只重加权聚合分子，不改变计数分母                                                               |
| `global_retention_ratio`                         | `Tensor | None` | rollout/filter 提供的全局保留比例辅助量                                                                        |
| `entropy_aux_figures`                            | `Tensor | None` | entropy 相关辅助统计                                                                                     |
| `teacher_log_probs`                              | `Tensor | None` | teacher logprob，供 distillation 类算法使用                                                               |
| `prev_topk_logprobs` / `curr_topk_logprobs`      | `Tensor | None` | student top-k token 上的新旧 logprob，通常为 `[B, S-1, K]`                                                 |
| `dumped_topk_logprobs` / `dumped_topk_token_ids` | `Tensor | None` | engine 已生成的 dump 数据，可能已位于 CPU                                                                      |
| `should_dump_metrics`                            | `bool`          | 是否允许返回本 step 的 dump metrics                                                                        |
| `calculate_per_token_loss`                       | `bool`          | `True` 使用 token-mean；`False` 使用 seq-mean，见下文                                                       |
| `cu_seqlens_padded`                              | `None`          | 新版 loss 的固定输入；THD 信息已在 engine 中消费                                                                  |
| `local_cp_size`                                  | `int`           | 新版 loss 中固定为 `1`                                                                                   |


自定义函数应根据自身算法显式 `assert` 必需的 optional 字段。例如依赖旧策略
logprob 时，应断言 `loss_input.prev_log_probs is not None`。

### 输出与归一化

统一返回：

```python
bwd_loss, bwd_count, metrics
```

- `bwd_loss`：需要反向传播的 0-D `Tensor`，必须是**未做 GBS 或全局
token/sample 归一化的局部 sum**。
- `bwd_count`：0-D `Tensor`，表示与 `bwd_loss` 对应的局部 token 数或有效
sample 数。当前 MCore caller 会解包该字段，但梯度归一化使用 engine
自己维护的 token count / alive GBS；不要依赖 caller 用
`bwd_count` 缩放 `bwd_loss`。
- `metrics`：`dict`。参与日志聚合的 tensor 应 `detach()`，避免保留计算图。

建议调用 `gpatch_v4.training_backend.loss.utils.agg` 生成
`(bwd_sum, bwd_count)`。该 helper 只接受 `[B, S]` 的逐 token value 和
mask，并断言 `cu_seqlens_padded is None`、`local_cp_size == 1`。

#### `calculate_per_token_loss=True`

按 token 聚合：


| 返回值         | 语义               | 无 `token_weights` 时的计算                   |
| ----------- | ---------------- | ---------------------------------------- |
| `bwd_loss`  | 有效 token loss 之和 | `(per_token_loss * response_mask).sum()` |
| `bwd_count` | 有效 token 数       | `response_mask.sum()`                    |


Engine 将 `bwd_loss` 作为 token sum 交给 Megatron，并根据自己统计的全局
token 数归一化梯度。自定义函数不要再除以 token 数或 GBS。

#### `calculate_per_token_loss=False`

先计算每个样本的 response token mean，再对有效样本求和：


| 返回值         | 语义                | 无额外 mask/weight 时的计算                                |
| ----------- | ----------------- | --------------------------------------------------- |
| `bwd_loss`  | 各样本 token mean 之和 | `sum_i((L_i * M_i).sum() / M_i.sum().clamp(min=1))` |
| `bwd_count` | 有效 sample 数       | `B`，或 `sample_mask.sum()`                           |


Engine 会按 alive GBS 对该 sum 做全局归一化。`sample_mask=0` 的样本不计入
分子和分母；`token_weights` 只重加权分子，分母仍由
`response_mask`/`sample_mask` 决定。

#### `metrics`

推荐使用以下两种值：

- 加权均值：`torch.stack([numerator, denominator])`。最终会跨 DP rank
先分别求和，再计算 `sum(numerator) / sum(denominator)`。
- 普通标量：0-D tensor 或 Python `int`/`float`。默认跨 DP rank 求均值；
key 以 `_min`、`_max`、`_sum` 结尾时分别取 min、max、sum。

常见 key 为 `loss`、`policy_loss`、`scaled_entropy`、`grpo_kl_loss` 和
`ppo_ratio`。`[sum, count]` metric 不需要手工 all-reduce；若函数还返回
需要在 loss 内立即跨 DP 规约的 0-D CUDA tensor，可调用
`reduce_metrics_across_data_parallel_group(metrics)`。

dump key 使用 `dump/` 前缀。`should_dump_metrics=True` 时可以返回 CPU
tensor，例如 `dump/curr_logprobs`、`dump/mask`；这些字段不会走普通 metric
聚合。

### 最小示例

自定义函数会直接替代整个 policy loss；框架不会在函数外自动补充
off-policy correction、reference KL 或 entropy bonus。下面的示例只实现
unclipped policy gradient 和 entropy bonus，实际算法需要自行加入其他项。

```python
# /abs/path/to/my_loss.py
import torch

from gpatch_v4.training_backend.loss import PolicyLossInput
from gpatch_v4.training_backend.loss.utils import agg


def my_loss_fn(
    config,
    loss_input: PolicyLossInput,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    assert loss_input.prev_log_probs is not None
    assert loss_input.per_token_entropy is not None

    ratios = (loss_input.curr_log_probs - loss_input.prev_log_probs).exp()
    per_token_policy_loss = -loss_input.advantages * ratios
    agg_kwargs = {
        "calculate_per_token_loss": loss_input.calculate_per_token_loss,
        "sample_mask": loss_input.sample_mask,
        "token_weights": loss_input.token_weights,
        "cu_seqlens_padded": loss_input.cu_seqlens_padded,
        "local_cp_size": loss_input.local_cp_size,
    }
    policy_sum, policy_count = agg(
        per_token_policy_loss,
        loss_input.response_mask,
        **agg_kwargs,
    )
    entropy_sum, entropy_count = agg(
        loss_input.per_token_entropy,
        loss_input.response_mask,
        **agg_kwargs,
    )
    bwd_loss = policy_sum - config.ppo.ppo_entropy_bonus * entropy_sum
    bwd_count = policy_count

    metrics = {
        "loss": torch.stack([bwd_loss.detach(), bwd_count.detach()]),
        "policy_loss": torch.stack([policy_sum.detach(), policy_count.detach()]),
        "scaled_entropy": torch.stack([entropy_sum.detach(), entropy_count.detach()]),
    }
    return bwd_loss, bwd_count, metrics
```



### 与 legacy 路径的主要差异

- 注册入口从
`gpatch_v4.training_backend.loss_factory.register_custom_loss_fn(name, ...)`
改为按 backend 注册的
`gpatch_v4.training_backend.loss.register_custom_loss_fn(backend, loss_name, ...)`。
- `PolicyLossInput` 必须从 `gpatch_v4.training_backend.loss` 导入。
- legacy 返回 `(bwd_loss, metrics)`；新版返回
`(bwd_loss, bwd_count, metrics)`。
- 新版 loss 只接收 response-padded `[B, S]` 数据；dyn-CP/THD 的重建由
engine 在调用 loss 前完成。
- 新路径当前没有注册 legacy 中的 `opd`、`fipo` 实现；需要这些算法时应继续
使用 legacy，或按本文接口自行实现。
- 新路径的内置 GRPO 支持 `ppo.ppo_entropy_regularization_type`；
以 GRPO 为基底的 STEER 也会应用相同的 entropy regularization，GSPO 暂不支持。
新路径实现在 `gpatch_v4/training_backend/loss/ppo_loss.py`，legacy 路径保留
`loss_factory.py` 中原有实现。
- 当前自定义 policy loss 注册仅覆盖 MCore actor 训练路径；FSDP2 RL 仍走
legacy `loss_factory.py`。

