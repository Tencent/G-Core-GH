import asyncio
import json
import argparse
from datetime import timedelta

import torch
import numpy as np
from transformers import AutoProcessor

from megatron_datasets.tools.lmdb_read_cli import fetch_images_from_lmdb
from megatron_datasets.mm_dataset import convert_conversations, remove_bos
from megatron_datasets.qwen2vl_dataset import resize_image

from gpatch.rpc import call_once_rpc
from gpatch.core.utils import load_and_call
from tasks.multimodal_comm.extra_args import ppo_mm_extra_args
from tasks.multimodal_grpo_critic_utils import (
    match_results_captcha,
    match_results_nq_hotpotq,
    match_results_geo3k,
    match_results_geo3k_v2,
    match_results_pmc_vqa,
    cal_rewards,
)


def parse_jsonl(
    sammple,
    rm_bos,
    lmdb_port,
    min_pixels_num,
    max_pixels_num,
    processor,
    dataset_type: str = "sft",
):
    assert dataset_type in ["sft", "grpo"]

    json_data = json.loads(sammple.strip())
    if dataset_type == "sft":
        assert json_data['conversations'][-1]['role'] == "assistant"
        answer = json_data['conversations'][-1]['content']
        conversations = convert_conversations(json_data['conversations'][:-1])
    elif dataset_type == "grpo":
        assert json_data['conversations'][-1]['role'] == "user"
        answer = json_data['label']
        conversations = convert_conversations(json_data['conversations'])
    else:
        assert False

    prompt_texts = processor.apply_chat_template(
        conversations,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
    )
    if rm_bos:
        prompt_texts = remove_bos(prompt_texts)
    prompt_ids = processor.tokenizer([prompt_texts])["input_ids"][0]

    imgs = []
    if "images" in json_data and len(json_data["images"]) > 0:
        imgs = fetch_images_from_lmdb(json_data["images"], lmdb_port)
        imgs = [
            np.array(resize_image(ele, img, min_pixels_num, max_pixels_num))
            for ele, img in zip(json_data['images'], imgs)
        ]

    return prompt_texts, prompt_ids, answer, imgs


async def run_eval_client(args, rank, world_size):
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    ip_port_map = [(x['ip'], x['port']) for x in config['sampler']['rpc_servers']]
    num_workers = len(ip_port_map)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # parse the data info
    with open(args.jsonl_path) as f:
        samples = f.readlines()

    samples_cnt = len(samples)
    samples_cnt_each_rank = (samples_cnt + world_size - 1) // world_size
    begin_idx = rank * samples_cnt_each_rank
    end_idx = min((rank + 1) * samples_cnt_each_rank, samples_cnt)
    samples = samples[begin_idx:end_idx]
    print(f"{rank=} send the sample cnt:{len(samples)}")

    gather_len_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gather_len_list, len(samples))
    min_sampler = min(gather_len_list)
    gather_once = None
    if args.max_cnt_sync_once is not None:
        gather_once = args.max_cnt_sync_once // world_size
        sync_times = min_sampler // args.max_cnt_sync_once

    cos = []
    answers = []
    batch_input_ids = []
    resp_cos = []
    for i, sample in enumerate(samples):
        ep_ip, ep_port = ip_port_map[i % num_workers]
        url = f'http://{ep_ip}:{ep_port}/generate'
        prompt_texts, prompt_ids, answer, imgs = parse_jsonl(
            sample,
            False,
            args.lmdb_port,
            args.min_pixels_num,
            args.max_pixels_num,
            processor,
            args.dataset_type,
        )
        req_dict = dict(
            prompt=prompt_texts,
            prompt_token_ids=prompt_ids,
            image=imgs,
            sampling_params=dict(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
                max_tokens=args.max_tokens,
            ),
        )
        batch_input_ids.append(prompt_ids)
        cos.append(call_once_rpc(url, req_dict, timeout=10 * 60))
        answers.append(answer)
        if gather_once is not None and len(cos) >= gather_once:
            resp_cos.extend(await asyncio.gather(*cos))
            cos = []
            print(f"{rank=} finish req:{len(resp_cos)=}")
            if sync_times >= 0:
                torch.distributed.barrier()
                sync_times -= 1

    metric_dict = {}
    if len(cos) > 0:
        resp_cos.extend(await asyncio.gather(*cos))
    batch_output_ids = [req_resp["output_token_ids"] for req_resp in resp_cos]

    resp_strs = processor.tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
    prompt_strs = processor.tokenizer.batch_decode(batch_input_ids, skip_special_tokens=False)
    if args.dataset_type in ["sft"]:
        acc_val = []
        for answer, resp_str in zip(answers, resp_strs):
            if answer.strip() == resp_str.strip():
                acc_val.append(1.0)
            else:
                acc_val.append(0)
        acc_tensor = torch.tensor(acc_val, dtype=torch.float32, device="cpu").view(-1, 1)
        metric_dict["acc"] = acc_tensor
    elif args.dataset_type in ["grpo"]:
        # 与 tasks/multimodal_grpo_critic_utils.py rule_based_rm 函数保持一致
        rule_type = args.ppo_mm_rule_type
        if rule_type in ['captcha']:
            acc_reward_tensor, fmt_reward_tensor = match_results_captcha(resp_strs, answers)
        elif rule_type in ['nq_hotpotq']:
            acc_reward_tensor, fmt_reward_tensor = match_results_nq_hotpotq(resp_strs, answers)
        elif rule_type in ['geometry3k']:
            acc_reward_tensor, fmt_reward_tensor = match_results_geo3k(resp_strs, answers)
        elif rule_type in ['geometry3k-v2']:
            acc_reward_tensor, fmt_reward_tensor = match_results_geo3k_v2(resp_strs, answers)
        elif rule_type in ['pmc_vqa']:
            acc_reward_tensor, fmt_reward_tensor = match_results_pmc_vqa(resp_strs, answers)
        elif rule_type in ['agent']:
            acc_reward_tensor, fmt_reward_tensor = cal_rewards(resp_strs, answers)
        elif rule_type == 'import_file':
            _, metric_dict = load_and_call(
                args.ppo_custom_rule_file,
                "compute_score",
                resp_strs,
                answers,
                prompt_strs=prompt_strs,
            )
            return metric_dict
        else:
            print(f"not support to this type: {rule_type}")
            raise NotImplementedError(f"not support to this type: {rule_type}")
        metric_dict["acc_rewards"] = acc_reward_tensor
        metric_dict["fmt_rewards"] = fmt_reward_tensor
    else:
        assert False

    return metric_dict


def client_args(parser):
    group = parser.add_argument_group(description="Evaluation client for image and text queries")
    group.add_argument("--jsonl-path", type=str, help="Path to the JSONL file")
    group.add_argument("--config-path", type=str, help="Path to the config file")
    group.add_argument("--model-path", type=str, help="Path to the model")
    group.add_argument("--lmdb-port", type=int, help="Port for LMDB")
    # sample param
    group.add_argument("--temperature", type=float, default=0.)
    group.add_argument("--top-k", type=int, default=1)
    group.add_argument("--top-p", type=float, default=1.)
    group.add_argument("--max-tokens", type=int, default=2048)
    group.add_argument("--seed", type=int, default=42)
    # dataset and rule param
    group.add_argument(
        "--dataset-type",
        type=str,
        default="sft",
        choices=["sft", "grpo"],
        help="jsonl dataset type"
    )
    group.add_argument("--min-pixels-num", type=int, default=None, help="min image width * height")
    group.add_argument("--max-pixels-num", type=int, default=None, help="max image width * height")
    group.add_argument('--ppo-custom-rule-file', type=str, default="", help='ppo custom rule file')

    # 多少个样本同步一次
    group.add_argument("--max-cnt-sync-once", type=int, default=None)

    return parser


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation client for image and text queries")
    parser = ppo_mm_extra_args(client_args, parser)
    return parser.parse_args()


if __name__ == '__main__':
    # 8h
    torch.distributed.init_process_group(backend='gloo', timeout=timedelta(seconds=8 * 60 * 60))
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank == 0:
        print("init distruibed finish")

    args = parse_args()
    metric_dict = asyncio.run(run_eval_client(args, rank, world_size))
    print(f"{rank=} infer finish")
    torch.distributed.barrier()
    gather_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gather_list, metric_dict)
    if rank == 0:
        stat_dict = {}
        for ele in gather_list:
            for k, v in ele.items():
                if k not in stat_dict:
                    stat_dict[k] = v.view(-1).clone()
                else:
                    stat_dict[k] = torch.cat([stat_dict[k], v.view(-1)], dim=-1)

        for k, v in stat_dict.items():
            print(f"{k=} {v.mean().item()}")
    torch.distributed.barrier()
