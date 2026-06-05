# models — 模型实现

自定义模型架构实现，包括扩散模型 transformer、CLIP、HPS 等。`__init__.py` 通过 `REGISTER_MODEL_CLS` 将 `MODEL_ARCH` 映射到 transformer 类。

## 已注册模型类

| MODEL_ARCH | Transformer 类 |
|------------|----------------|
| `flux` | `FluxTransformer2DModel`（diffusers） |
| `oteam4_3` | `FluxTransformer2DModel`（diffusers） |
| `oteam4_4` | `Oteam4_4Transformer2DModel`（自定义） |
| `qwen_image_edit` | `QwenImageTransformer2DModel`（diffusers） |

## 子目录说明

| 子目录 | 说明 |
|--------|------|
| `bagel/` | BAGEL 多模态模型：data 工具、推理器、建模（autoencoder、SigLIP、Qwen2、TaylorSeer cache） |
| `omni_common/` | Omni 模型公共工具：forward_utils、inference_mixin、modeling_utils |
| `oteam4_4/` | 自定义 Flux-style transformer、attention、Qwen 适配 |
| `oteam4_5_moe/` | MoE Flux transformer、DeepEP、autoencoder、并行化工具 |
| `weclip_v2/` / `weclip_v3/` | WeCLIP 视觉编码器变体（NaViT、ConvNeXt 等） |
| `wegen_hpsv3/` | HPS v3 风格的图像质量评分模型 |
