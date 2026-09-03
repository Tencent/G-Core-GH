# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.

# Model arch 这个字段从 transformers 内获取，避免不规范的写法，例如 "qwen3moe" "qwen2p5vl" "qwen25vl"
# https://github.com/huggingface/transformers/tree/main/src/transformers/models
#
# 由于实际上开源社区有很多个人项目的操作并不规范，或者是内部项目有时候操之过急，有些逻辑只能用
# model arch 特判，所以只能吃大便了。但名字上 follow 下 transformers 的 naming 规范，避免 'qwen2p5'
# 这种。
#
# 例如：
# ```
# model_arch = 'oteam4_5_moe'
# class Oteam4_5_MoeXxxYyy:
# ```
# ```
# model_arch = 'oteam8_8_vl_moe'
# class Oteam8_8_VLMoeAaaaBbbbCccc:
# ```
#
# 如果社区有人命名特殊，那么尽量 follow。


class MODEL_ARCH:
    """Registry of known model architecture identifiers.

    Values follow the naming convention of HuggingFace Transformers.
    """
    # LM and VLM
    LLAMA = 'llama'
    LLAMA4 = 'llama4'
    QWEN2 = 'qwen2'  # transformers 里 qwen25 和 qwen2 是同一个
    QWEN2_5_VL = 'qwen2_5_vl'
    QWEN3 = 'qwen3'
    QWEN3_MOE = 'qwen3_moe'
    QWEN3_VL = 'qwen3_vl'
    QWEN3_VL_MOE = 'qwen3_vl_moe'
    QWEN3_5 = 'qwen3_5'
    QWEN3_5_MOE = 'qwen3_5_moe'
    QWEN3_OMNI_MOE = 'qwen3_omni_moe'
    WELM_MOE = 'welm_moe'
    WELMV4_MOE = 'welmv4_moe'
    WELM_OMNI_V4_5 = 'welm_omni_v4_5'
    QWEN2_MOE = 'qwen2_moe'
    QWEN2_VL = 'qwen2_vl'
    DEEPSEEK_V3 = 'deepseek_v3'
    DEEPSEEK_V4 = 'deepseek_v4'
    QWEN4_EXP = 'qwen4_exp'  # Qwen3.8-Flash-Next
    MISTRAL = 'mistral'
    GEMMA3_TEXT = 'gemma3_text'
    GEMMA4 = 'gemma4'
    SEED_OSS = 'seed_oss'
    APERTUS = 'apertus'
    GLM4V = 'glm4v'
    GPT_OSS = 'gpt_oss'

    # WEMM
    QWEN3_5_WEMM = 'qwen3_5_wemm'
    QWEN3_5_MOE_WEMM = 'qwen3_5_moe_wemm'
    QWEN3_VL_WEMM = 'qwen3_vl_wemm'
    WEMM3_EMBEDDING = 'wemm3_embedding'
    WEMM3_5_EMBEDDING = 'wemm3_5_embedding'
    WEMM3_5_MOE_EMBEDDING = 'wemm3_5_moe_embedding'

    # diffusion
    STABLE_DIFFUSION_V1_5 = 'stable_diffusion_v1_5'  # https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
    FLUX = 'flux'
    FLUX_KONTEXT = 'flux_kontext'
    OTEAM4_3 = 'oteam4_3'
    OTEAM4_4 = 'oteam4_4'
    BAGEL = 'bagel'
    QWEN_IMAGE_EDIT = 'qwen_image_edit'
    # vae
    # 暂无特殊规则
