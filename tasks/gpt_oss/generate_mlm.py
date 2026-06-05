from types import SimpleNamespace
import argparse
import json
import os
import pprint
import re
import shutil
import sys
import gc
import shutil
import importlib
import time
from typing import Dict

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch
import transformers

from gpatch.core.utils import print_with_rank_and_datetime
from megatron.core import dist_checkpointing
from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.core.enums import ModelType
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from megatron.legacy import fused_kernels
from megatron.training import get_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args, set_global_variables
from megatron.training.initialize import _set_random_seed, _initialize_distributed
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.training.utils import unwrap_model
from megatron.training import get_tokenizer
from megatron.inference.text_generation.generation import ForwardStep
from megatron.inference.text_generation.generation import generate_tokens_probs_and_return_on_first_stage
from megatron.inference.text_generation.generation import score_and_return_on_first_stage
from megatron.core.transformer.module import Float16Module
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)

import argparse

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

from megatron.training import print_rank_0

from megatron.core.inference.contexts import StaticInferenceContext

from gpatch.training.arguments import gpatch_extra_args
from gpatch.core.transformer.transformer_config import GpatchTransformerConfig
from gpatch.patch_mcore import init_gpatch_for_mcore


class SingleBatchIterator:
    """Iterator that yields a single batch of data for text generation.
    Required by the forward_backward_func function.

    This class creates an iterator that yields exactly one batch containing
    input tokens, position IDs, and attention mask, then raises StopIteration.
    Used for single-step inference in the forward pass.
    """
    def __init__(self, input_ids, position_ids, attention_mask):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def text_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for text generation.
    Required by the forward_backward_func function.

    Extracts a batch from the data iterator and runs the model forward pass
    with the provided input tokens, position IDs, and attention mask.

    Args:
        data_iterator: Iterator providing batches of input data
        model: The Megatron model to run forward pass on
        **kwargs: Additional keyword arguments (unused)

    Returns:
        Tuple of (model_output, loss_function)
    """
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def gen_args(parser):
    parser = gpatch_extra_args(parser)

    group = parser.add_argument_group(title='Gen')
    group.add_argument('--prompts', type=str, nargs='*', required=True)
    group.add_argument('--max_new_tokens', type=int, default=32)
    return parser


def load_model():
    args = get_args()
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args, GpatchTransformerConfig)

    mod = importlib.import_module(args.load_model_provider)
    model = mod.model_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    iteration, _ = load_checkpoint([model], None, None)

    model.eval()
    model = Float16Module(config, model).cuda()
    return model


@torch.no_grad()
def gen(model, prompt_idx):
    # Tokenize the input prompt
    args = get_args()
    tokenizer = get_tokenizer()
    prompt = args.prompts[prompt_idx]

    chat = [
        {
            'role': 'user',
            'content': prompt,
        },
    ]
    input_ids_list = tokenizer._tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device="cuda")
    input_ids = input_ids.unsqueeze(0)
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long,
                     device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    generated_ids = input_ids.clone()

    stop_tokens = [tokenizer._tokenizer.eos_token_id]

    # Greedy generation loop
    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")

            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(input_ids, position_ids, attention_mask)

            output = fwd_bwd_function(
                forward_step_func=text_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                # All-gather operation
                dist.all_gather(
                    gathered_tensors,
                    output,
                    group=parallel_state.get_tensor_model_parallel_group()
                )
                # Concatenate along last dimension (dim=2)
                output = torch.cat(gathered_tensors, dim=2)
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                # Debug: print token information
                if step < 5:  # Only for first few iterations
                    print_with_rank_and_datetime(
                        f"Step {step}: output shape={output.shape}, var={output.var():.4f}"
                    )
                    logits = output[0, -1, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [tokenizer._tokenizer.decode([idx]) for idx in top5_ids]
                    print_with_rank_and_datetime(
                        f"Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}"
                    )
                    print_with_rank_and_datetime(
                        f"Selected: '{tokenizer._tokenizer.decode([next_token_ids.item()])}' (id={next_token_ids.item()})"
                    )
            else:
                next_token_ids = torch.ones(
                    (1, 1), device=generated_ids.device, dtype=generated_ids.dtype
                )

            torch.distributed.broadcast(next_token_ids, torch.distributed.get_world_size() - 1)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long,
                             device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() in stop_tokens:
                break

    # Decode the generated sequence
    generated_text = tokenizer._tokenizer.decode(list(generated_ids[0]))
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    print_rank_0(f"Prompt: {prompt}")
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("=======================================")


def main():
    init_gpatch_for_mcore()
    args = parse_args(gen_args)
    args = validate_args(args)
    set_global_variables(args, build_tokenizer=True)
    args = get_args()
    _initialize_distributed(get_embedding_ranks=None, get_position_embedding_ranks=None, store=None)
    _set_random_seed(args.seed, args.data_parallel_random_init)
    fused_kernels.load(args)
    torch.distributed.barrier()

    model = load_model()
    torch.distributed.barrier()
    for i in range(len(args.prompts)):
        gen(model, i)


if __name__ == "__main__":
    main()
