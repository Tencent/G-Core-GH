# 自定义 Loss 开发指南 - DeepSeek 3.2 无偏 KL 估计

通过继承 `GptPpoActorModel` 并重写 `get_actor_grpo_forward_output_and_loss_func` 方法来实现自定义的 loss 计算逻辑，无需修改原有代码。

本示例实现了 **DeepSeek 3.2 论文中的无偏 KL 散度估计方法**。

## 背景

在 GRPO 训练中，我们用旧策略 π_old 采样，但想估计当前策略 π_θ 对参考策略 π_ref 的 KL 散度：

```
原始 KL 估计 (有偏): E_{π_old}[KL(π_θ || π_ref)]
无偏 KL 估计:        E_{π_old}[(π_θ/π_old) * KL(π_θ || π_ref)]
```

通过乘上重要性采样系数 `ratio = π_θ/π_old`，将基于旧策略采样的 KL 估计修正为对当前策略的无偏估计。

**参考论文**: https://arxiv.org/abs/2501.12948

## 文件结构

```
tasks/my_custom_loss/
├── __init__.py
├── custom_actor_model.py    # 自定义 Actor 模型 (DeepSeek 3.2 无偏 KL 估计)
├── train_actor.py           # 训练入口
└── README.md                # 本文档
```

## 使用方式

运行训练时，将原来的 `train_ppo_actor.py` 替换为 `train_actor.py`：

```bash
python tasks/my_custom_loss/train_actor.py [原有参数...]
```

---

## 核心代码改动

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

---

## 需要修改的位置

用 `################################################################` 注释标记，搜索 `【开始】` 和 `【结束】` 即可定位：

| 文件 | 标记 | 说明 |
|------|------|------|
| `custom_actor_model.py` | `【开始】DeepSeek 3.2 无偏 KL 估计实现` | 无偏 KL 估计核心逻辑 |
| `custom_actor_model.py` | `【新增 metrics】对比有偏和无偏 KL` | 新增的 metrics |
| `train_actor.py` | `【开始】导入自定义 Actor 模型` | 导入语句 |
| `train_actor.py` | `【开始】自定义 Actor Provider` | Actor 创建逻辑 |

---

## 新增的 Metrics

训练过程中会记录以下 metrics，方便对比有偏和无偏 KL 估计的差异：

| Metric | 说明 |
|--------|------|
| `kl_loss_unbiased` | DeepSeek 3.2 无偏 KL 估计 |
| `kl_loss_biased` | 原始有偏 KL 估计 (用于对比) |
| `ppo_ratio` | 重要性采样系数 π_θ/π_old |

---

## 可用的变量

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
