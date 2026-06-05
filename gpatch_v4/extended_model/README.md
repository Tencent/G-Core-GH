# extended_model — 模型架构适配层

根据 `MODEL_ARCH` 提供不同模型架构的定制化钩子，通过 Factory + Registry 模式分发。解耦了训练 / 采样逻辑与具体模型实现。

## 扩展点

| 扩展点 | Factory | 说明 |
|--------|---------|------|
| Rollout Attribute | `ApplySamplingRolloutAttrFactory` | 采样前对 rollout 属性做架构特定处理 |
| Sampler Generate | `SamplerGenerateFuncFactory` | 采样生成函数（LLM 文本 vs 多模态） |
| Prepare Data Forward | `PrepareDataForwardFactory` | 训练 forward 前的数据预处理（RL / SFT / DPO / Distill / Agentic 各有不同） |

## 注册表

`__init__.py` 中定义了按 `MODEL_ARCH` 分派的注册表（dict）：

- **默认**走 LLM 路径
- **VLM**（Qwen3-VL、Qwen3.5、Qwen3-Omni 等）走 MultiModal 路径
- **WeLM-V4** 走 `welm_v4.py` 中的专用路径（可选，`ImportError` 时跳过）
- **Agentic** 走 `PrepareDataForwardAgentic`（来自 `gpatch_v4.agentic`）

## 文件说明

| 文件 | 说明 |
|------|------|
| `base.py` | 扩展点抽象基类 |
| `llm.py` | LLM 实现：`PrepareDataForwardLLM`、`SamplerGenerateFuncLLM`、`DpoPrepareDataForwardLLM`、`OffPoilicyDistillPrepareDataForwardLLM` 等 |
| `multi_modal.py` | 多模态实现：`ApplySamplingRolloutAttrMultiModal`、`SamplerGenerateFuncMultiModal`、`ApplySamplingRolloutAttrQwen3_5` / `_MOE` |
| `qwen3_vl.py` | Qwen3-VL 特化：多模态 forward 数据准备（RL / SFT / DPO / Distill） |
| `welm_v4.py` | WeLM-V4 特化：`WelmV4PrepareDataForwardLLM`（可选，打包时可能不存在） |
| `rollout_attr_hook.py` | `ApplySamplingRolloutAttrLLM` — LLM rollout 属性处理 |

## 扩展方式

1. 在对应文件中新增实现类
2. 在 `__init__.py` 的 `REGISTER_*` 字典中注册新的 `MODEL_ARCH` 映射
