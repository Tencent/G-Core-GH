"""Test FP8 quantized model with HuggingFace generate.

Loads the FP8 checkpoint and runs generation to verify correctness.
Optionally compares output with the original BF16 model.

Usage
-----
    # Test FP8 model only
    python tools/convert_fp8/qwen3_5/tests/test_fp8_hf_generate.py --fp8-path /path/to/fp8_model

    # Compare FP8 vs BF16
    python tools/convert_fp8/qwen3_5/tests/test_fp8_hf_generate.py \
        --fp8-path /path/to/fp8_model \
        --bf16-path /path/to/bf16_model

    # Custom prompt
    python tools/convert_fp8/qwen3_5/tests/test_fp8_hf_generate.py \
        --fp8-path /path/to/fp8_model \
        --prompt "Explain quantum computing in simple terms."
"""

import json
import os
from argparse import ArgumentParser
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    Qwen3_5MoeForConditionalGeneration,
)


def resolve_auto_model_class(model_path: str):
    with open(os.path.join(model_path, "config.json")) as f:
        architectures = json.load(f).get("architectures", [])
    arch = architectures[0] if architectures else ""

    if "Qwen3_5MoeForConditionalGeneration" in arch:
        return Qwen3_5MoeForConditionalGeneration
    if "ConditionalGeneration" in arch or "ImageTextToText" in arch:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Write a short Python function that computes fibonacci numbers.",
    "Explain the theory of relativity in one paragraph.",
]


def load_and_generate(
    model_path: str,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    enable_thinking: bool = False,
    label: str = "Model",
) -> list[str]:
    print(f"\n{'='*60}")
    print(f"Loading {label}: {model_path}")
    print(f"{'='*60}")

    model_cls = resolve_auto_model_class(model_path)
    print(f"Model class: {model_cls}")
    model = model_cls.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    outputs = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        outputs.append(response)

        print(f"\n--- Prompt {i+1} ---")
        print(f"Q: {prompt}")
        print(f"A: {response}")

    del model
    torch.cuda.empty_cache()
    return outputs


def main():
    parser = ArgumentParser(description="Test FP8 quantized model generation")
    parser.add_argument(
        "--fp8-path", type=str, required=True, help="Path to FP8 quantized checkpoint"
    )
    parser.add_argument(
        "--bf16-path",
        type=str,
        default=None,
        help="Path to original BF16 checkpoint (for comparison)"
    )
    parser.add_argument(
        "--prompt", type=str, nargs="*", default=None, help="Custom prompt(s) to test"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Use Qwen thinking chat template (default: off, direct answer)",
    )
    args = parser.parse_args()

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS

    tokenizer = AutoTokenizer.from_pretrained(args.fp8_path)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": args.enable_thinking,
    }

    fp8_outputs = load_and_generate(
        args.fp8_path, tokenizer, prompts, label="FP8 model", **gen_kwargs
    )

    if args.bf16_path:
        bf16_outputs = load_and_generate(
            args.bf16_path, tokenizer, prompts, label="BF16 model", **gen_kwargs
        )

        print(f"\n{'='*60}")
        print("Comparison: FP8 vs BF16")
        print(f"{'='*60}")
        for i, (fp8_out, bf16_out) in enumerate(zip(fp8_outputs, bf16_outputs)):
            match = fp8_out.strip() == bf16_out.strip()
            print(f"\nPrompt {i + 1}: {'EXACT MATCH' if match else 'DIFFERENT'}")
            print(f"  FP8:  {fp8_out[:500]}...")
            print(f"  BF16: {bf16_out[:500]}...")

    print("\nAll generation tests passed.")


if __name__ == "__main__":
    main()
