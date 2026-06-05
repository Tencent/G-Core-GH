# extended_pipeline — T2I / 多模态推理管线

为不同的 T2I 模型架构提供端到端推理管线（采样 + 去噪 + 解码），通过 `ExtendPipelineFactory` 按 `MODEL_ARCH` 分发。

## 已注册管线

| MODEL_ARCH | Pipeline 类 | 说明 |
|------------|------------|------|
| `flux` | `FluxPipeline` | Flux 扩散模型 |
| `oteam4_3` | `Oteam43Pipeline` | 内部 4.3 扩散模型 |
| `oteam4_4` | `Oteam44Pipeline` | 内部 4.4 扩散模型 |
| `bagel` | `FSDP2EngineBagel` | BAGEL 多模态 |
| `qwen_image_edit` | `QwenImageEditPipeline` | Qwen 图像编辑 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `mixin.py` | 管线公共 mixin |
| `pipeline_base.py` | 基础管线抽象 |
| `pipeline_flux.py` | Flux 管线实现 |
| `pipeline_oteam4_3.py` | Oteam 4.3 管线 |
| `pipeline_oteam4_4.py` | Oteam 4.4 管线 |
| `pipeline_bagel.py` | BAGEL 管线 |
| `pipeline_qwen_image_edit.py` | Qwen 图像编辑管线 |
| `pipeline_fsdp2_omni_base.py` | FSDP2 Omni 基础管线 |

## 扩展方式

1. 新建 `pipeline_xxx.py` 继承 `pipeline_base.py` 中的基类
2. 在 `__init__.py` 的 `REGISTER_PIPELINE` 中注册 `MODEL_ARCH` → Pipeline 映射
