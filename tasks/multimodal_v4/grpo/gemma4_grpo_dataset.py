"""Gemma4 GRPO dataset for geometry3k (text-only).

Loads geometry3k from HuggingFace Hub, converts each sample into a
prompt+label pair compatible with the GRPO pipeline, and tokenises
using the Gemma4 processor / tokenizer.

Tokenization uses ``add_special_tokens=False`` to match vLLM's
``Gemma4ProcessingInfo``, which avoids a double-BOS because the
chat template already embeds a literal ``<bos>`` in the rendered text.
"""

import json
from functools import partial
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.data.distributed import DistributedSampler

from gpatch_v4.configs.config import RlConfig

instruction_following = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <reason> </reason> tags. The final answer MUST BE put in \boxed{}."
)

g_uniq_id = 0


def gen_unique_id(dp_rank):
    global g_uniq_id
    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0
    g_uniq_id += 1
    return f"dp_rank_{dp_rank}_worker_id_{worker_id}_{g_uniq_id}"


def convert_sample(sample):
    answer = sample["answer"]
    problem = sample["problem"]
    label = json.dumps(dict(answer=answer, problem=problem))
    conversation = [
        dict(role="user", content=problem + " " + instruction_following),
    ]
    return dict(conversations=conversation, label=label)


def collate_func(config, tokenizer, dp_rank, instances):
    prompt_token_ids = []
    prompt_lens = []
    labels = []

    seqlen = config.training.seq_length
    max_gen = config.sampler.infer_engine_configs[0].generate_max_tokens

    for instance in instances:
        messages = convert_sample(instance)

        prompt_text = tokenizer.apply_chat_template(
            messages["conversations"],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        plen = len(ids)
        if plen > (seqlen - max_gen):
            plen = seqlen - max_gen
            ids = ids[:plen]

        prompt_token_ids.append({"prompt_token_ids": ids})
        prompt_lens.append(torch.tensor(plen, dtype=torch.long))
        labels.append(messages["label"])

    return {
        "prompt_token_ids": prompt_token_ids,
        "prompt_lens": prompt_lens,
        "gt_label": labels,
    }


def get_dataset_and_dataloader(config: RlConfig = None, tokenizer=None, dp_rank=0, dp_size=1):
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(
        config.policy.hf_tokenizer_path, trust_remote_code=True
    )

    dataset = load_dataset(config.data.data_pathes[0])
    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"].shuffle(seed=42)

    train_sampler = DistributedSampler(
        train_dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed,
    )
    eval_sampler = DistributedSampler(
        eval_dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=partial(collate_func, config, _tokenizer, dp_rank),
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.rollout_mbs,
        num_workers=config.data.dataloader_num_workers,
        prefetch_factor=config.data.dataloader_prefetch_factor,
        drop_last=True,
    )

    eval_rollout_mbs = (
        config.training.eval_rollout_mbs
        if config.training.eval_rollout_mbs else config.training.rollout_mbs
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        collate_fn=partial(collate_func, config, _tokenizer, dp_rank),
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=eval_rollout_mbs,
        num_workers=config.data.dataloader_num_workers,
        prefetch_factor=config.data.dataloader_prefetch_factor,
        drop_last=True,
    )

    return {
        "train_dataset": train_dataset,
        "train_sampler": train_sampler,
        "train_dataloader": train_dataloader,
        "eval_dataset": eval_dataset,
        "eval_sampler": eval_sampler,
        "eval_dataloader": eval_dataloader,
    }


def verify_dataloader_func(train_dataset, train_sampler, train_dataloader):
    dl_iter = iter(train_dataloader)
    data = next(dl_iter)
    print(f"{data.keys()=}")
    print(f"{data=}")
