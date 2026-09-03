from gpatch_v4.core.constants import MODEL_ARCH

try:
    from diffusers import QwenImageTransformer2DModel
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

    try:
        from gpatch_v4.models.oteam4_4.transformer_flux import (
            FluxTransformer2DModel as Oteam4_4Transformer2DModel,
        )
        _OTEAM_MODEL_CLS = {
            MODEL_ARCH.OTEAM4_3: FluxTransformer2DModel,
            MODEL_ARCH.OTEAM4_4: Oteam4_4Transformer2DModel,
        }
    except ImportError:
        _OTEAM_MODEL_CLS = {}

    REGISTER_MODEL_CLS = {
        MODEL_ARCH.FLUX: FluxTransformer2DModel,
        MODEL_ARCH.QWEN_IMAGE_EDIT: QwenImageTransformer2DModel,
        **_OTEAM_MODEL_CLS,
    }
except (ImportError, RuntimeError):
    # LLM-only paths import gpatch_v4.models.* and must not require diffusers/torchvision.
    REGISTER_MODEL_CLS = {}
