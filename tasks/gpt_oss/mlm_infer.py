# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# astrachang@tencent.com, yeazhao@tencent.com
# reference https://github.com/NVIDIA/Megatron-LM/blob/main/examples/inference/README.md
import argparse
import importlib

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import AutoTokenizer

from megatron.core import parallel_state as mpu
from megatron.core.transformer.module import Float16Module
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.text_generation_controllers.text_generation_controller import TextGenerationController
from megatron.legacy import fused_kernels
from megatron.training import get_args, get_tokenizer
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.global_vars import set_args, set_global_variables
from megatron.training.initialize import _set_random_seed, _initialize_distributed
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from gpatch.training.arguments import gpatch_extra_args
from gpatch.core.transformer.transformer_config import GpatchTransformerConfig
from gpatch.patch_mcore import init_gpatch_for_mcore


def gen_args(parser):
    parser = gpatch_extra_args(parser)

    group = parser.add_argument_group(title='Gen')
    group.add_argument('--prompts', type=str, nargs='*', required=True)
    group.add_argument('--max-new-tokens', type=int, default=32)
    group.add_argument('--max-batch-size', type=int, default=10)
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


def main():
    init_gpatch_for_mcore()
    args = parse_args(gen_args)
    args = validate_args(args)
    setattr(args, 'inference_max_requests', 8)  # can not pass with cli, set manually
    set_global_variables(args, build_tokenizer=True)
    args = get_args()
    _initialize_distributed(get_embedding_ranks=None, get_position_embedding_ranks=None, store=None)
    _set_random_seed(args.seed, args.data_parallel_random_init)
    fused_kernels.load(args)
    torch.distributed.barrier()

    model = load_model()
    torch.distributed.barrier()
    tokenizer = get_tokenizer()
    inference_wrapped_model = GPTInferenceWrapper(model, args)

    # Define a sampling loop.
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )

    # Create a static or dynamic inference engine.
    inference_engine = StaticInferenceEngine(
        text_generation_controller=text_generation_controller,
        max_batch_size=args.max_batch_size,
    )

    # customize your sampling params here
    c = SamplingParams(temperature=0.5, top_k=10)
    results: List[InferenceRequest] = inference_engine.generate(
        prompts=args.prompts, sampling_params=c
    )

    if torch.distributed.get_rank() == 0:
        for idx, result in enumerate(results):
            print(f' ------------- RESULT FOR PROMPT {idx} --------------- ')
            result = {
                'id': result.request_id,
                'input_prompt': result.prompt,
                'generated_text': result.generated_text,
                'generated_tokens': result.generated_tokens
            }
            print(result)


if __name__ == "__main__":
    main()
"""
test output:
 ------------- RESULT FOR PROMPT 0 --------------- 
{'id': '0', 'input_prompt': "what's the result of 1+1", 'generated_text': '?\n\nThe result of 1 + 1 is 2.\n\nSure! If you have any more questions or need further assistance, feel free', 'generated_tokens': tensor([   30,   279,   976,  1534,   328,   220,    16,   659,   220,    16,
          382,   220,    17,    13,   279, 62915,     0,  1843,   481,   679,
         1062,   945,  5359,   503,  1309,  6544, 14647,    11,  3195,  2240],
       device='cuda:0')}
 ------------- RESULT FOR PROMPT 1 --------------- 
{'id': '1', 'input_prompt': 'write a poem about openai', 'generated_text': "'s\n\nHere is a short poem about OpenAI:\n\nOpenAI, a shining star\nIn the world of AI, you are a shining bar\n", 'generated_tokens': tensor([  885,   279, 12253,   382,   261,  4022, 41339,  1078,  7788, 17527,
         1402,  6447, 17527,    11,   261, 77082,  8253,   198,   637,   290,
         2375,   328, 20837,    11,   481,   553,   261, 77082,  3608,   198],
       device='cuda:0')}
 ------------- RESULT FOR PROMPT 2 --------------- 
{'id': '2', 'input_prompt': 'do you know tomorin?', 'generated_text': '\n\nYes, Tomori is a Japanese name that can be written in different ways using different kanji characters. It can also be a surname. However', 'generated_tokens': tensor([  279, 13022,    11, 11838,  6510,   382,   261, 18938,  1308,   484,
          665,   413,  7582,   306,  2647,  6984,  2360,  2647,  3163,  4133,
         9862,    13,  1225,   665,  1217,   413,   261, 68009,    13,  5551],
       device='cuda:0')}
 ------------- RESULT FOR PROMPT 3 --------------- 
{'id': '3', 'input_prompt': "what's your name?", 'generated_text': '\n\nI am an AI language model developed by OpenAI and I don\'t have a personal name. You can call me "Assistant" or "AI".', 'generated_tokens': tensor([  279,    40,   939,   448, 20837,  6439,  2359,  9742,   656,  7788,
        17527,   326,   357,  4128,   679,   261,  3832,  1308,    13,  1608,
          665,  2421,   668,   392, 91655,     1,   503,   392, 17527,  4050],
       device='cuda:0')}

"""
