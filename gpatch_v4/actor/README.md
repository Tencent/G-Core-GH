# actor — Ray Worker Actors

Ray remote actor 定义，每种训练 / 推理角色对应一个 Actor 类。Trainer 通过 `orches/` 将这些 Actor 部署到 Ray 集群上。

## 核心概念

- 每个 Actor 是一个 `@ray.remote` 类，被 `orches/*_group.py` 创建并管理
- Actor 内部持有 training engine 或 inference engine，执行实际的计算逻辑
- Mixin 模式：`mixin.py` 提供 tokenizer 构建、指标统计、checkpoint 转换、权重 offload/onload、PPO 数据生成等可复用能力

## 文件说明

| 文件 | 说明 |
|------|------|
| `mixin.py` | 公共 Mixin 集合：`TokenizerMixin`、`MetricsMixin`、`RlTrainerMixin`、`OffloadManager`/`OnloadManager`（CPU offload）、`ProfileMixin`、`RetryActorMixin`、`T2iTokenizerMixin`、`CheckpointConverterMixin`、`TestActorMixin`、`TrainingPltMixin`、`FlopsCounterMixin` 等 |
| `grpo_train_actor.py` | GRPO 训练 Actor（文本 RL） |
| `grpo_async_train_actor.py` | GRPO 异步训练 Actor（异步 rollout） |
| `grpo_agentic_train_actor.py` | GRPO Agentic 训练 Actor（agentic RL） |
| `grpo_sampler_actor.py` | GRPO 采样 Actor，运行推理引擎生成 rollout |
| `grpo_bt_rm_actor.py` | Batch reward model Actor（文本） |
| `grpo_gen_rm_actor.py` | Generative reward model Actor（文本） |
| `finetune_actor.py` | SFT 微调 Actor |
| `dpo_actor.py` | DPO 训练 Actor |
| `distill_student_actor.py` | On-policy 蒸馏学生端 Actor |
| `distill_teacher_actor.py` | On-policy 蒸馏教师端 Actor |
| `off_policy_distill_student_actor.py` | Off-policy 蒸馏学生端 Actor |
| `off_policy_distill_sampler_actor.py` | Off-policy 蒸馏采样 Actor |
| `t2i_grpo_train_actor.py` | T2I GRPO 训练 Actor |
| `t2i_edit_sft_actor.py` | T2I 编辑 SFT Actor |
| `t2i_grpo_bt_rm_actor.py` | T2I Batch RM Actor |
| `t2i_grpo_gen_rm_actor.py` | T2I Generative RM Actor |
| `evaluate_actor.py` | 评估 Actor |
| `kv_store_actor.py` | 分布式 KV 存储 Actor |
| `training_plt_actor.py` | 训练平台上报 Actor |
| `inference_actor.py` | 纯推理 Worker Actor |

## 扩展方式

新增训练范式时，继承 `orches/base_actor.py` 的 `BaseActor` 或直接使用 mixin 组合，注册到 `orches/custom_actor_registry.py` 即可被 trainer 动态加载。
