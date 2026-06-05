"""Offline block-wise FP8 quantization for HuggingFace checkpoints.

Uses transformers FineGrainedFP8Config to quantize BF16/FP32 models to
block-wise FP8 (E4M3) format, compatible with SGLang / vLLM / Transformers.

Designed for Qwen3.5 / Qwen3.6 MoE-VL but works for any HuggingFace model.

Usage
-----
    python quantize_hf_to_fp8.py \
        --input-hf-path /path/to/bf16_model \
        --output-fp8-path /path/to/fp8_model

    # Qwen3.5/3.6 hub-aligned skips (default):
    python quantize_hf_to_fp8.py \
        --input-hf-path /path/to/bf16_model \
        --output-fp8-path /path/to/fp8_model \
        --skip-policy official

    # Import exact list from an official FP8 checkpoint config:
    python quantize_hf_to_fp8.py \
        --input-hf-path /path/to/bf16_model \
        --output-fp8-path /path/to/fp8_model \
        --reference-fp8-config /path/to/Qwen3.6-35B-A3B-FP8

The output checkpoint can be loaded by:
    # SGLang
    python -m sglang.launch_server --model /path/to/fp8_model --quantization fp8

    # vLLM
    vllm serve /path/to/fp8_model --quantization fp8

    # Transformers (auto-detected from config.json)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("/path/to/fp8_model", device_map="auto")
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from argparse import ArgumentParser
from glob import glob

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    FineGrainedFP8Config,
    Qwen3_5MoeForConditionalGeneration,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from qwen3_mtp_fp8_postprocess import append_mtp_weights_to_fp8_checkpoint
from qwen3_official_fp8_skips import (
    build_qwen3_official_modules_to_not_convert_from_path,
    load_modules_to_not_convert_from_reference,
)

# Legacy broad regex skips (pre-official alignment).
_LEGACY_MODULES_TO_NOT_CONVERT = [
    ".*lm_head",
    ".*linear_attn",
    ".*shared_expert_gate",
    ".*visual",
]


def resolve_auto_model_class(input_hf_path: str):
    """Pick HF Auto class from config architectures (e.g. Qwen3.6 VL vs text-only)."""
    config_path = os.path.join(input_hf_path, "config.json")
    with open(config_path) as f:
        architectures = json.load(f).get("architectures", [])

    arch = architectures[0] if architectures else ""
    if "Qwen3_5MoeForConditionalGeneration" in arch:
        return Qwen3_5MoeForConditionalGeneration
    if "ConditionalGeneration" in arch or "ImageTextToText" in arch:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def resolve_modules_to_not_convert(
    input_hf_path: str,
    skip_policy: str,
    reference_fp8_config: str | None,
    modules_to_not_convert: list[str] | None,
) -> list[str]:
    """Resolve BF16 skip list from CLI flags."""
    if modules_to_not_convert is not None:
        return list(modules_to_not_convert)

    if reference_fp8_config is not None:
        return load_modules_to_not_convert_from_reference(reference_fp8_config)

    if skip_policy == "official":
        return build_qwen3_official_modules_to_not_convert_from_path(input_hf_path)

    if skip_policy == "legacy":
        return list(_LEGACY_MODULES_TO_NOT_CONVERT)

    raise ValueError(
        f"Unknown skip_policy={skip_policy!r}; use 'official', 'legacy', or pass "
        "--modules-to-not-convert / --reference-fp8-config"
    )


def patch_saved_quantization_config(
    output_fp8_path: str,
    modules_to_not_convert: list[str],
) -> None:
    """Align saved ``config.json`` quantization metadata with HF hub FP8 checkpoints."""
    config_path = os.path.join(output_fp8_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    qcfg = cfg.setdefault("quantization_config", {})
    qcfg["quant_method"] = "fp8"
    qcfg["activation_scheme"] = qcfg.get("activation_scheme", "dynamic")
    qcfg["fmt"] = "e4m3"
    qcfg["modules_to_not_convert"] = modules_to_not_convert
    if "weight_block_size" not in qcfg:
        qcfg["weight_block_size"] = [128, 128]

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def copy_processor_configs(input_hf_path: str, output_fp8_path: str) -> list[str]:
    """Copy VL processor config files needed by SGLang / AutoProcessor."""
    copied_files = []
    for src_path in sorted(glob(os.path.join(input_hf_path, "*processor*.json"))):
        dst_path = os.path.join(output_fp8_path, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        copied_files.append(os.path.basename(src_path))
    return copied_files


def main(
    input_hf_path: str,
    output_fp8_path: str,
    block_size: tuple[int, int] = (128, 128),
    activation_scheme: str = "dynamic",
    skip_policy: str = "official",
    reference_fp8_config: str | None = None,
    modules_to_not_convert: list[str] | None = None,
    include_mtp: bool = True,
):
    skip_modules = resolve_modules_to_not_convert(
        input_hf_path,
        skip_policy=skip_policy,
        reference_fp8_config=reference_fp8_config,
        modules_to_not_convert=modules_to_not_convert,
    )

    model_cls = resolve_auto_model_class(input_hf_path)

    print(f"Input:                 {input_hf_path}")
    print(f"Output:                {output_fp8_path}")
    print(f"AutoModel class:       {model_cls.__name__}")
    print(f"Skip policy:           {skip_policy}")
    if reference_fp8_config:
        print(f"Reference FP8 config:  {reference_fp8_config}")
    print(f"Block size:            {list(block_size)}")
    print(f"Activation scheme:     {activation_scheme}")
    print(f"modules_to_not_convert: {len(skip_modules)} entries")
    if len(skip_modules) <= 12:
        print(f"  {skip_modules}")
    else:
        print(f"  {skip_modules[:6]} ... {skip_modules[-3:]}")
    print()

    quant_config = FineGrainedFP8Config(
        weight_block_size=block_size,
        activation_scheme=activation_scheme,
        modules_to_not_convert=skip_modules,
    )

    print("Loading and quantizing model...")
    model = model_cls.from_pretrained(
        input_hf_path,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Saving FP8 checkpoint to {output_fp8_path} ...")
    model.save_pretrained(output_fp8_path)
    patch_saved_quantization_config(output_fp8_path, skip_modules)

    if include_mtp:
        print("Appending MTP weights from the input checkpoint...")
        mtp_tensor_count = append_mtp_weights_to_fp8_checkpoint(
            input_hf_path,
            output_fp8_path,
            block_size=block_size,
            modules_to_not_convert=skip_modules,
        )
        print(f"MTP tensors written: {mtp_tensor_count}")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(input_hf_path)
        tokenizer.save_pretrained(output_fp8_path)
        print("Tokenizer saved.")
    except Exception:
        print("No tokenizer found or failed to save, skipping.")

    processor_files = copy_processor_configs(input_hf_path, output_fp8_path)
    if processor_files:
        print(f"Processor configs copied: {processor_files}")
    else:
        print("No processor configs found, skipping.")

    print(f"\nDone. FP8 checkpoint saved to: {output_fp8_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Offline block-wise FP8 quantization via transformers FineGrainedFP8Config"
    )
    parser.add_argument(
        "--input-hf-path", type=str, required=True, help="Path to BF16/FP32 HuggingFace checkpoint"
    )
    parser.add_argument(
        "--output-fp8-path",
        type=str,
        required=True,
        help="Output path for FP8 quantized checkpoint",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs=2,
        default=[128, 128],
        help="Block size for quantization, default: 128 128",
    )
    parser.add_argument(
        "--activation-scheme",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Activation quantization scheme, default: dynamic",
    )
    parser.add_argument(
        "--skip-policy",
        type=str,
        default="official",
        choices=["official", "legacy"],
        help=(
            "How to choose modules_to_not_convert: 'official' (Qwen3.5/3.6 hub-aligned, "
            "default) or 'legacy' (broad regex, skips entire linear_attn)."
        ),
    )
    parser.add_argument(
        "--reference-fp8-config",
        type=str,
        default=None,
        help=(
            "Path to an official FP8 checkpoint (or its config.json); "
            "imports modules_to_not_convert verbatim. Overrides --skip-policy."
        ),
    )
    parser.add_argument(
        "--modules-to-not-convert",
        type=str,
        nargs="*",
        default=None,
        help="Explicit submodule names to keep in BF16 (overrides --skip-policy).",
    )
    parser.add_argument(
        "--no-include-mtp",
        action="store_true",
        help="Do not append top-level mtp.* weights after HF save_pretrained.",
    )
    args = parser.parse_args()

    main(
        input_hf_path=args.input_hf_path,
        output_fp8_path=args.output_fp8_path,
        block_size=tuple(args.block_size),
        activation_scheme=args.activation_scheme,
        skip_policy=args.skip_policy,
        reference_fp8_config=args.reference_fp8_config,
        modules_to_not_convert=args.modules_to_not_convert,
        include_mtp=not args.no_include_mtp,
    )
