# copying form verl


def get_peft_cls(config, bridge, provider, dtype=None):
    """Get PEFT class from model config.

    Args:
        config: Model configuration object.
        bridge: Megatron-Bridge AutoBridge instance.
        provider:

    Returns:
        LoRAConfig / CanonicalLoRAConfig / DoRAConfig, or None.
    """
    from gpatch_v4.training_backend.megatron_backend.megatron_bridge import (
        CanonicalLoRA,
        DoRA,
        LoRA,
        VLMLoRA,
    )

    peft_cls = None
    if not hasattr(config, "lora"):
        return peft_cls

    lora_cfg = config.lora
    # Only enable if rank > 0
    if lora_cfg.get("rank", 0) <= 0:
        return peft_cls

    assert bridge is not None and provider is not None, "LoRA/PEFT only supported via Megatron-Bridge"

    lora_dtype = lora_cfg.get("dtype", dtype)
    # if lora_dtype is not None:
    #     lora_dtype = PrecisionType.to_dtype(lora_dtype)

    lora_type = lora_cfg.get("type", "lora")
    if lora_type == "lora":
        peft_cls = LoRA(
            target_modules=lora_cfg.get(
                "target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
            ),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.0),
            dropout_position=lora_cfg.get("dropout_position", "pre"),
            lora_A_init_method=lora_cfg.get("lora_A_init_method", "xavier"),
            lora_B_init_method=lora_cfg.get("lora_B_init_method", "zero"),
            a2a_experimental=lora_cfg.get("a2a_experimental", False),
            lora_dtype=lora_dtype,
            exclude_modules=lora_cfg.get("exclude_modules", []),
        )
    if lora_type == "vlm_lora":
        peft_cls = VLMLoRA(
            target_modules=lora_cfg.get(
                "target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
            ),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.0),
            dropout_position=lora_cfg.get("dropout_position", "pre"),
            lora_A_init_method=lora_cfg.get("lora_A_init_method", "xavier"),
            lora_B_init_method=lora_cfg.get("lora_B_init_method", "zero"),
            a2a_experimental=lora_cfg.get("a2a_experimental", False),
            lora_dtype=lora_dtype,
            freeze_vision_model=lora_cfg.get("freeze_vision_model", True),
            freeze_vision_projection=lora_cfg.get("freeze_vision_projection", True),
            freeze_language_model=lora_cfg.get("freeze_language_model", True),
            exclude_modules=lora_cfg.get("exclude_modules", []),
        )
    elif lora_type == "canonical_lora":
        peft_cls = CanonicalLoRA(
            target_modules=lora_cfg.get(
                "target_modules",
                [
                    "linear_q",
                    "linear_k",
                    "linear_v",
                    "linear_proj",
                    "linear_fc1_up",
                    "linear_fc1_gate",
                    "linear_fc2",
                ],
            ),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.0),
            dropout_position=lora_cfg.get("dropout_position", "pre"),
            lora_A_init_method=lora_cfg.get("lora_A_init_method", "xavier"),
            lora_B_init_method=lora_cfg.get("lora_B_init_method", "zero"),
            exclude_modules=lora_cfg.get("exclude_modules", []),
        )
    elif lora_type == "dora":
        peft_cls = DoRA(
            target_modules=lora_cfg.get(
                "target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
            ),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.0),
            dropout_position=lora_cfg.get("dropout_position", "pre"),
            lora_A_init_method=lora_cfg.get("lora_A_init_method", "xavier"),
            lora_B_init_method=lora_cfg.get("lora_B_init_method", "zero"),
            exclude_modules=lora_cfg.get("exclude_modules", []),
        )

    print(
        f"Enabling {lora_type.upper()} with rank={lora_cfg.get('rank')}, "
        f"alpha={lora_cfg.get('alpha')}, dropout={lora_cfg.get('dropout')}"
    )
    return peft_cls


__all__ = [
    "get_peft_cls",
]
