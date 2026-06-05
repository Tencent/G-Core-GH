# reward — 奖励模型

奖励计算工厂 + 多种奖励实现。RL 训练中用于对采样结果打分。

## RewardFactory

根据 `config.bt_rm.reward_type` 返回对应的奖励引擎：

| reward_type | 实现类 | 说明 |
|-------------|--------|------|
| `rule_only` | `RuleReward` | 纯规则奖励（正则匹配、格式检查等） |
| `rm_only` | `RmReward` | 纯 RM 模型打分 |
| `rm_and_rule` | `MixRmAndRuleReward` | RM + 规则混合奖励 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `base_reward.py` | 奖励基类 `RewardAbc` |
| `mixin.py` | 奖励计算公共 mixin |
| `rule_reward.py` | 基于规则的奖励实现 |
| `rm_reward.py` | 基于 RM 模型的奖励实现 |
| `mix_rm_and_rule_reward.py` | 混合奖励实现 |
| `base_reward_model_t2i.py` | T2I 专用奖励模型基类 `BaseT2iRewardModel` |
| `base_external_reward.py` | `BaseExternalReward` — 外部异步奖励抽象基类，支持与 GPU 计算重叠的异步 I/O 调用 |

## 扩展方式

1. 继承 `RewardAbc` 实现新的奖励类
2. 在 `RewardFactory.get_reward_engine()` 中添加新的 `reward_type` 分支
3. 外部奖励：继承 `BaseExternalReward`，实现 `calc_external_reward` 异步方法
