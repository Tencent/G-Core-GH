# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com
"""Patch vLLM MoE model parameters so fused ``w13_weight`` / ``w2_weight``
receive the correct ``weight_loader``.

Background
----------
Since vLLM 0.8.2, fused MoE models (e.g. Qwen2/Qwen3 MoE, DeepSeek-V2/V3,
Mixtral) keep their expert weights as raw ``Parameter`` s under
``mlp.experts.w13_weight`` / ``mlp.experts.w2_weight`` without attaching
``weight_loader`` to the parameters themselves. Only the *module*
``mlp.experts`` exposes ``weight_loader``. vLLM's generic
``model.load_weights`` looks up ``param.weight_loader`` and silently skips
those parameters when it is missing, which is exactly the MoE corruption
we have been chasing.

Ported from ``verl/verl/utils/vllm/patch.py::patch_vllm_moe_model_weight_loader``.
"""

SUPPORTED_MOE_MODELS: list[type] = []


def _collect_supported_moe_models() -> None:
    global SUPPORTED_MOE_MODELS
    if SUPPORTED_MOE_MODELS:
        return

    def _try_append(module_path: str, name: str) -> None:
        try:
            mod = __import__(module_path, fromlist=[name])
            SUPPORTED_MOE_MODELS.append(getattr(mod, name))
        except Exception:
            # vLLM version may not include this arch; skip silently.
            pass

    _try_append("vllm.model_executor.models.deepseek_v2", "DeepseekV2ForCausalLM")
    _try_append("vllm.model_executor.models.deepseek_v2", "DeepseekV3ForCausalLM")
    _try_append("vllm.model_executor.models.mixtral", "MixtralForCausalLM")
    _try_append("vllm.model_executor.models.qwen2_moe", "Qwen2MoeForCausalLM")
    _try_append("vllm.model_executor.models.qwen3_moe", "Qwen3MoeForCausalLM")
    _try_append("vllm.model_executor.models.qwen3_vl_moe", "Qwen3MoeLLMForCausalLM")
    _try_append("vllm.model_executor.models.qwen3_next", "Qwen3NextForCausalLM")
    _try_append("vllm.model_executor.models.kimi_vl", "KimiVLForConditionalGeneration")
    _try_append("vllm.model_executor.models.qwen3_5", "Qwen3_5MoeForCausalLM")
    _try_append("vllm.model_executor.models.deepseek_v4", "DeepseekV4ForCausalLM")


def patch_vllm_moe_model_weight_loader(model) -> None:
    """Attach ``experts.weight_loader`` to fused MoE tensor parameters.

    Safe to call multiple times; no-op for non-MoE models.
    """
    _collect_supported_moe_models()
    if not SUPPORTED_MOE_MODELS:
        return

    original_model_type = type(model)
    # Unwrap ACL graph wrapper if present (matches verl's handling).
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(original_model_type):
        model = model.runnable
        original_model_type = type(model)

    mlp_attr_mapping: dict[type, str] = {}
    try:
        from vllm.model_executor.models.mixtral import MixtralForCausalLM
        mlp_attr_mapping[MixtralForCausalLM] = "block_sparse_moe"
    except Exception:
        pass
    try:
        from vllm.model_executor.models.deepseek_v4 import DeepseekV4ForCausalLM
        mlp_attr_mapping[DeepseekV4ForCausalLM] = "ffn"
    except Exception:
        pass
    default_mlp_attr = "mlp"

    inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner_model is None:
        raise ValueError(
            "patch_vllm_moe_model_weight_loader: model has no 'model' or "
            "'language_model' attribute"
        )

    moe_types = tuple(SUPPORTED_MOE_MODELS)
    if not isinstance(model, moe_types) and not isinstance(inner_model, moe_types):
        return

    # Qwen3-VL / Qwen3.5 nest an extra level.
    if type(inner_model).__name__ in ("Qwen3MoeLLMForCausalLM", "Qwen3_5MoeForCausalLM"):
        inner_model = inner_model.model

    for layer in inner_model.layers:
        mlp_attr = mlp_attr_mapping.get(original_model_type, default_mlp_attr)
        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue

        experts = getattr(mlp, "experts", None)
        if not experts or not hasattr(experts, "weight_loader"):
            continue

        for name, param in mlp.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = experts.weight_loader
            if "weight_scale" in name:
                param.weight_loader = experts.weight_loader
                if getattr(param, "quant_method", None) is None:
                    param.quant_method = "block"
