# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""Single-process HF smoke test for truncated DeepSeek-V4-Flash.

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/transformers/src:$PYTHONPATH"
    pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_hf.py
"""

# 参考：https://github.com/huggingface/transformers/commit/b6fb4ec0f24cfad232de46a02441b9cc6362cd22

import math
import os
import unittest

import torch
import torch.nn.functional as F

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
NUM_LAYERS = 1
SEQ_LEN = 128


def _truncate_config(config):
    """Truncate config to a few decoder layers for a local smoke test."""
    assert config.layer_types is not None
    assert config.mlp_layer_types is not None
    config.num_hidden_layers = NUM_LAYERS
    config.layer_types = config.layer_types[:NUM_LAYERS]
    config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
    return config


class TestDeepseekV4Hf(unittest.TestCase):
    """Load HF DeepSeek-V4, then run one forward/backward step."""

    def setUp(self):
        if not os.path.isdir(HF_MODEL_PATH):
            raise unittest.SkipTest(
                f"model dir not found: {HF_MODEL_PATH}; "
                f"download or convert DeepSeek-V4-Flash into hf-hub/ first"
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda is required for this smoke test")
        free_bytes, _ = torch.cuda.mem_get_info(0)
        free_gib = free_bytes / 1024**3

    def test_hf_fwd_bwd(self):
        """Verify truncated HF model produces finite logits and gradients."""
        from transformers import AutoTokenizer, DeepseekV4ForCausalLM, FineGrainedFP8Config
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

        torch.manual_seed(42)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        rng = torch.Generator().manual_seed(42)
        input_ids = torch.randint(
            0,
            tokenizer.vocab_size,
            (1, SEQ_LEN),
            generator=rng,
        ).to(device)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        print(f"loading {HF_MODEL_PATH} with {NUM_LAYERS} decoder layers ...")
        config = DeepseekV4Config.from_pretrained(HF_MODEL_PATH)
        _truncate_config(config)
        assert config.layer_types is not None
        assert config.mlp_layer_types is not None
        self.assertEqual(config.num_hidden_layers, NUM_LAYERS)
        self.assertEqual(len(config.layer_types), NUM_LAYERS)
        self.assertEqual(len(config.mlp_layer_types), NUM_LAYERS)
        quantization_config = FineGrainedFP8Config(dequantize=True)
        model = DeepseekV4ForCausalLM.from_pretrained(
            HF_MODEL_PATH,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        self.assertEqual(len(model.model.layers), NUM_LAYERS)
        model.train()
        model.gradient_checkpointing_enable()

        for name, param in model.named_parameters():
            print(f'{name}: {param.dtype} {param.shape}')

        torch.cuda.reset_peak_memory_stats()
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        self.assertTrue(torch.isfinite(logits).all(), "logits contain NaN/Inf")

        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        self.assertTrue(math.isfinite(loss.item()), "loss is NaN/Inf")
        loss.backward()
        print(f'loss={loss.item():.4f}')

        grad_norms = []
        has_nonfinite_grad = False
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            if not torch.isfinite(grad).all():
                has_nonfinite_grad = True
            grad_norms.append(grad.norm().item())

        self.assertFalse(has_nonfinite_grad, "grads contain NaN/Inf")
        self.assertGreater(len(grad_norms), 0, "model produced no gradients")
        self.assertGreater(
            max(grad_norms),
            0.0,
            "all gradients are zero",
        )

        mem_after_bwd = torch.cuda.memory_allocated(device) / 1024**3
        mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(
            f"logits.shape={list(logits.shape)} "
            f"logits.mean={logits.float().mean().item():.4f} "
            f"loss={loss.item():.4f} "
            f"num_grads={len(grad_norms)} "
            f"max_grad_norm={max(grad_norms):.4f}\n"
            f"  mem_after_bwd={mem_after_bwd:.2f} GiB, mem_peak={mem_peak:.2f} GiB"
        )
