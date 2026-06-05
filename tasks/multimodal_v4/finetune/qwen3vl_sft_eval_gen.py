import json
import os
import re
import uuid
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from megatron.core import mpu
from megatron_datasets.mm_dataset import convert_conversations
from megatron_datasets.qwenvl_dataset_map import resize_image

from gpatch_v4.utils import log


class EvalMapFunc:
    """Extracts prompt token IDs, images, and ground truth label from each sample."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, example):
        imgs = example.pop("__images_feat__", [])
        json_data = {k: v for k, v in example.items()}
        conversations = json_data.get("conversations", [])

        gt_label = ""
        if conversations and conversations[-1]["role"] == "assistant":
            gt_label = conversations[-1]["content"]

        prompt_convs = convert_conversations(conversations[:-1])

        all_text = self.processor.apply_chat_template(
            prompt_convs,
            tools=None,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_token_ids = self.processor.tokenizer([all_text])["input_ids"][0]

        imgs_np = None
        if imgs and "images" in json_data and len(json_data["images"]) > 0:
            imgs_np = [
                np.array(resize_image(ele, img, None, None))
                for ele, img in zip(json_data["images"], imgs)
            ]

        return {
            "prompt_token_ids": prompt_token_ids,
            "gt_label": gt_label,
            "imgs_np": imgs_np,
        }


def eval_collate_fn(instances):
    prompt_token_ids = []
    prompt_lens = []
    gt_labels = []

    for inst in instances:
        inp = {"prompt_token_ids": inst["prompt_token_ids"]}
        if inst["imgs_np"] is not None:
            pil_imgs = [Image.fromarray(arr) for arr in inst["imgs_np"]]
            inp["multi_modal_data"] = {"image": pil_imgs}
        prompt_token_ids.append(inp)
        prompt_lens.append(torch.tensor(len(inst["prompt_token_ids"]), dtype=torch.long))
        gt_labels.append(inst["gt_label"])

    return {
        "prompt_token_ids": prompt_token_ids,
        "prompt_lens": prompt_lens,
        "gt_label": gt_labels,
    }


def get_eval_dataset_and_dataloader(config=None, tokenizer=None, dp_rank=0, dp_size=1):
    from gdataset import GDatasetV4
    from gdataset.feat import PilImageListFeat
    from transformers import AutoProcessor

    eval_path = config.data.data_pathes[-1]

    feats = {
        "images": PilImageListFeat(
            lmdb=True,
            return_src_data=True,
            convert_to_rgb=True,
            new_name="__images_feat__",
        ),
    }

    processor = AutoProcessor.from_pretrained(config.policy.hf_tokenizer_path)

    num_workers = config.data.dataloader_num_workers
    prefetch = config.data.dataloader_prefetch_factor

    dataset = GDatasetV4(
        eval_path,
        dp_rank=dp_rank,
        dp_size=dp_size,
        gbs=max(dp_size * max(num_workers, 1) * config.training.train_mbs, dp_size),
        shuffling_buffer_size=0,
        consumed=0,
        feats=feats,
        seed=42,
        smart_padding_compare_func=None,
        smart_padding_buffer_size=0,
        mbs=config.training.train_mbs,
    )
    dataset.map(EvalMapFunc(processor))
    dataset.set_epoch(0)

    dl_kwargs = {}
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch

    dataloader = DataLoader(
        dataset,
        collate_fn=eval_collate_fn,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.train_mbs,
        num_workers=num_workers,
        drop_last=True,
        **dl_kwargs,
    )

    return {
        "train_dataset": dataset,
        "train_dataloader": dataloader,
    }


async def offline_generate_func(
    config, infer_engine, idx, tokenizer, student_tokenizer, batched_data, sampling_repeat_n
):
    prompt_token_ids = batched_data["prompt_token_ids"]
    prompt_lens = batched_data["prompt_lens"]
    gt_labels = batched_data["gt_label"]

    sampling_params = infer_engine.get_sampling_params_from_config(
        config.sampler.infer_engine_configs[idx], tokenizer.eos_token_id
    )
    async_gens = []
    for i in range(len(prompt_token_ids)):
        for j in range(sampling_repeat_n):
            tmp_sampling_params = infer_engine.copy_sampling_params_with_seed_offset(
                sampling_params, i * sampling_repeat_n + j
            )
            gen = infer_engine.async_generate(
                prompt_token_ids[i], tmp_sampling_params, str(uuid.uuid4().hex)
            )
            async_gens.append(gen)

    gen_outputs = await infer_engine.wait_and_get_async_generate_output(async_gens)

    tokens_lst = []
    seq_length_lst = []
    prompt_len_lst = []
    gt_label_lst = []

    for gi, gen_out in enumerate(gen_outputs):
        i = gi // sampling_repeat_n
        assert len(gen_out.outputs) == 1
        resp_tokens = list(gen_out.outputs[0].token_ids)
        prompt_ids = prompt_token_ids[i]["prompt_token_ids"]
        token = list(prompt_ids) + resp_tokens
        if len(token) > config.training.seq_length:
            token = token[: config.training.seq_length]
        tokens_lst.append(torch.tensor(token, dtype=torch.long))
        seq_length_lst.append(torch.tensor(len(token), dtype=torch.long))
        prompt_len_lst.append(prompt_lens[i])
        gt_label_lst.append(gt_labels[i])

    return {
        "tokens": tokens_lst,
        "sequence_lengths": seq_length_lst,
        "prompt_lengths": prompt_len_lst,
        "gt_label": gt_label_lst,
    }


def eval_rollout_samples(config=None, tokenizer=None, rbs: List[Dict[str, Any]] = None):
    correct = 0
    total = 0
    results = []

    for rb in rbs:
        seq_len = rb["sequence_lengths"]
        pmt_len = rb["prompt_lengths"]
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.item()
        if isinstance(pmt_len, torch.Tensor):
            pmt_len = pmt_len.item()

        gen_tokens = rb["tokens"][pmt_len:seq_len]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        gt_label = rb["gt_label"]

        is_correct = gen_text.strip() == gt_label.strip()
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "gt_label": gt_label,
            "generated": gen_text.strip(),
            "correct": is_correct,
        })

    rank = mpu.get_data_parallel_rank()
    acc = correct / total if total > 0 else 0
    log(f"[Rank {rank}] Eval: total={total} correct={correct} acc={acc:.4f}")
    # gather all results
    gathered = [None] * mpu.get_data_parallel_world_size()
    torch.distributed.all_gather_object(gathered, results, group=mpu.get_data_parallel_group())
    all_results = [r for sub in gathered if sub is not None for r in sub]
    all_correct = sum(1 for r in all_results if r["correct"])
    all_total = len(all_results)
    all_acc = all_correct / all_total if all_total > 0 else 0
    log(f"Eval: total={all_total} correct={all_correct} acc={all_acc:.4f}")
    
    # save all results to a jsonl file
    if mpu.get_data_parallel_rank() == 0:
        out_dir = config.evaluate_result.output_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{config.evaluate_result.output_prefix}.jsonl")
        with open(out_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            # save summary
            summary = {
                "summary": True,
                "total": all_total,
                "correct": all_correct,
                "acc": all_acc,
            }
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        log(f"Saved eval results to {out_path}")
