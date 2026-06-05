import sys
import os
import json
import random
import time
import re

import torch


def get_ppo_prompt_format(args, tokenizer):
    if args.px_apply_chat_template:
        return None, tokenizer._tokenizer.eos_token

    # 其实应该用 tokenizer.apply_chat_template 的，不过有些 model 也没有。
    if args.model_arch == "qwen2-72b":
        prompt_format = "<|im_start|>system\nyou are a helpful assistant<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = tokenizer._tokenizer.eos_token
    elif args.model_arch == "yi_9b":
        prompt_format = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = None
    elif args.model_arch in ["qwen2.5-math-rm-72b", "qwen2.5-math-1.5b"]:
        prompt_format = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = tokenizer._tokenizer.eos_token
    elif args.model_arch in ["qwen3", "qwen3-moe", "qwen3-next-moe"]:
        if torch.distributed.get_rank() == 0:
            print(f"define your own prompt_format according to your task")
        prompt_format = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = tokenizer._tokenizer.eos_token
    elif args.model_arch == "llama":
        prompt_format = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nPlease reason step by step, and put your final answer within \\boxed{}.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{problem}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
        eos_token = tokenizer._tokenizer.eos_token
    else:
        # NOTE: 注意！这里根据业务实际情况来的！有时候业务的数据会修改字段名！注意不要踩坑！
        prompt_format = "###{problem}\n### Response:\n"
        eos_token = tokenizer._tokenizer.eos_token
    return prompt_format, eos_token


def get_gen_rm_prompt_format(args):
    # 这只是个 demo，简单起见，就不一个个 model 写了。
    # 其实应该用 tokenizer.apply_chat_template 的，不过有些 model 也没有。
    prompt_format = """<|im_start|>You are a math teacher. Grade the Solution, verifying correctness step by step. Use Expected Answer to find any erroneous step in the Solution. At the end of the Solution verification, when you give your final grade, write it in the form "Verification: Is the answer correct (Yes/No)? X",  where X is either Yes or No.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"""
    return prompt_format
