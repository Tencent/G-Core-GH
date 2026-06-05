# configs — 配置体系

基于 Python `dataclass` + Hydra 的分层配置系统。每种训练范式对应一个顶层 Config 类，各子模块的配置拆分为独立 `*_config.py`。

## 顶层 Config（定义在 `config.py`）

| Config 类 | 用途 |
|-----------|------|
| `FinetuneConfig` | SFT 微调 |
| `RlConfig` | 文本 RL（GRPO） |
| `AgenticRlConfig` | Agentic RL（继承 `RlConfig`） |
| `T2iRlConfig` | T2I RL（GRPO） |
| `DpoConfig` | DPO |
| `OnPolicyDistillConfig` | On-policy 蒸馏（支持多 teacher 路由） |
| `OffPolicyDistillConfig` | Off-policy 蒸馏 |
| `T2iEditSftConfig` | T2I 编辑 SFT |
| `EvaluateConfig` | 评估 |
| `InferenceConfig` | 纯推理 |

### 其他文件中的顶层 Config

| Config 类 | 文件 | 用途 |
|-----------|------|------|
| `T2iDpoConfig` | `t2i_dpo_config.py` | T2I DPO |
| `BagelConfig` / `BagelT2iRlConfig` | `bagel_configs.py` | BAGEL 训练 |

## 子模块 Config

| 文件 | 说明 |
|------|------|
| `policy_config.py` | 模型路径、并行度、model_arch 等 |
| `training_config.py` | 训练超参：lr、batch_size、backend 选择（fsdp2 / mcore） |
| `sampler_config.py` | 推理引擎参数：engine 类型、TP/PP、生成参数 |
| `ppo_config.py` | PPO/GRPO 算法参数、advantage 类型、KL 惩罚 |
| `reward_config.py` | RM 权重、reward_type（rule_only / rm_only / rm_and_rule） |
| `data_config.py` | 数据集路径、tokenizer 配置 |
| `checkpoint_config.py` | Checkpoint 保存 / 加载策略 |
| `optimizer_config.py` | 优化器、LR scheduler |
| `report_config.py` | 实验追踪（WandB 等）、profiling |
| `debug_config.py` | 调试开关 |
| `infer_engine_config.py` | 推理引擎详细配置 |
| `ema_config.py` | EMA 权重配置（T2I） |
| `kv_config.py` | KV Store 配置 |
| `dist_config.py` | 分布式并行度配置（TP/PP/CP/EP/DP） |
| `client_config.py` | RPC 客户端配置 |
| `profile_config.py` | 性能 profiling 配置 |
| `agentic_config.py` | Agentic RL 配置 |
| `transformer_config.py` | Transformer 模型参数覆盖 |
| `evaluate_config.py` | 评估结果输出配置 |
| `inference_config.py` | 推理结果输出配置 |
| `t2i_dpo_config.py` | T2I DPO 配置 |
| `bagel_configs.py` | BAGEL 系列配置 |
| `utils.py` | 配置工具：`MappingProtocol`、`merge_hydra_config` 等 |

## YAML 默认值

`yaml/` 目录下的 YAML 文件作为 Hydra 默认配置，运行时与 dataclass 默认值合并。

## 使用方式

```python
@hydra.main(config_path="../configs/yaml", config_name="test_config")
def main(cfg: RlConfig):
    default_config = RlConfig()
    merged = OmegaConf.merge(default_config, cfg)
    ...
```
