# rollout_generator — Rollout 生成器

定义 rollout 生成策略。`RolloutGeneratorFactory` 根据 `config.policy.rollout_gen_type` 选择具体实现。

## 已注册生成器

| rollout_gen_type | 生成器类 | 说明 |
|------------------|----------|------|
| `base` | `BaseRolloutGenerator` | 标准 rollout：采样 → RM 打分 → 返回 |
| `replay` | `ReplayRolloutGenerator` | 从历史数据回放 rollout |
| `dynamic_sampling` | `DynamicSamplingRolloutGenerator` | 动态采样策略 |
| `on_policy_distill` | `OnPolicyDistillRolloutGenerator` | On-policy 蒸馏 rollout（含 teacher logits） |
| `off_policy_distill` | `OffPolicyDistillRolloutGenerator` | Off-policy 蒸馏 rollout |
| `agentic` | `AgenticRolloutGenerator` | Agentic RL rollout（来自 `gpatch_v4.agentic`） |
| `external` | 动态加载 | 通过 `rollout_gen_py_path` / `rollout_gen_cls_name` 动态导入用户自定义生成器 |
| `partial_rollout` | — | 预留，尚未实现（`NotImplementedError`） |

## 文件说明

| 文件 | 说明 |
|------|------|
| `__init__.py` | `RolloutGeneratorFactory` — 根据 `rollout_gen_type` 分发 |
| `generator_abc.py` | 生成器抽象基类 |
| `base_generator.py` | `BaseRolloutGenerator` — 标准实现 |
| `ds_generator.py` | `DynamicSamplingRolloutGenerator` — 动态采样 |
| `replay_generator.py` | `ReplayRolloutGenerator` — 历史回放 |
| `on_policy_distill_generator.py` | On-policy 蒸馏生成器 |
| `off_policy_distill_generator.py` | Off-policy 蒸馏生成器 |
| `mixin.py` | 生成器公共 mixin |

### `async_rollout/` — 异步 Rollout 子模块

| 文件 | 说明 |
|------|------|
| `rollout_controller.py` | `RolloutController` / `GenerateResult` — 异步 rollout 控制器 |
| `agent_loop_actor.py` | `AgentLoopActor` — agentic 循环 actor |
| `two_turn_reflect_agent_loop_actor.py` | `TwoTurnReflectAgentLoopActor` — 两轮反思 agent 循环 |

## 扩展方式

1. 继承 `generator_abc.py` 中的基类
2. 在 `RolloutGeneratorFactory.get_rollout_generator()` 中添加新的 `rollout_gen_type` 分支；或使用 `external` 类型动态加载
