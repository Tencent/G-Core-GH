import copy
import json
import os
import re
import uuid
from typing import Any, Dict, List

import torch

from megatron.core import mpu

from gpatch_v4.utils import log


async def offline_generate_func(
    config, infer_engine, idx, tokenizer, student_tokenizer, batched_data, sampling_repeat_n
):
    prompt_token_ids = batched_data["prompt_token_ids"]
    prompt_lens = batched_data["prompt_lens"]
    gt_label = batched_data["gt_label"]

    is_same_tokenizer = tokenizer.get_vocab() == student_tokenizer.get_vocab()
    assert is_same_tokenizer, f"tokenizer is not the same"

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
        j = gi % sampling_repeat_n
        assert len(gen_out.outputs) == 1

        resp_tokens = list(gen_out.outputs[0].token_ids)
        token = prompt_token_ids[i]['prompt_token_ids'] + resp_tokens
        assert len(token) <= config.training.seq_length
        tokens_lst.append(torch.tensor(token, dtype=torch.long))
        seq_length_lst.append(torch.tensor(len(token), dtype=torch.long))
        prompt_len_lst.append(prompt_lens[i])
        gt_label_lst.append(gt_label[i])

    assert len(tokens_lst) == len(gt_label_lst)

    rollout_batch = {
        'tokens': tokens_lst,
        'sequence_lengths': seq_length_lst,
        'prompt_lengths': prompt_len_lst,
        'gt_label': gt_label_lst,
    }
    return rollout_batch


def parse_question(text):
    pattern = r'<|im_start|>user\n(.*?)<|im_end|>'
    # re.findall 找到所有匹配项，并返回捕获组 (.*?) 中的内容
    matches = re.findall(pattern, text)
    if matches:
        first_res = None
        for match in matches:
            if match != "":
                first_res = match
                break
        return first_res
    else:
        return None


def save_jsonl(config, question_texts, label_texts, seq_len_list, prompt_len_list, rank):
    os.makedirs(config.evaluate_result.output_dir, exist_ok=True)
    assert config.evaluate_result.output_prefix is not None
    output_dir = os.path.join(
        config.evaluate_result.output_dir, config.evaluate_result.output_prefix
    )
    os.makedirs(output_dir, exist_ok=True)
    log(f"save rollout samples to {output_dir}")
    none_cnt = 0
    with open(
        os.path.join(output_dir, f"{config.evaluate_result.output_prefix}_rank_{rank}.jsonl"), 'w'
    ) as f:
        for i in range(len(question_texts)):
            if question_texts[i] is None:
                log(
                    f"Rank {rank} question parse as None and its detail is question={question_texts[i]} label={label_texts[i]}"
                )
                none_cnt += 1
                continue
            item = {
                "problem": question_texts[i],
                "solution": label_texts[i],
                "seq_len": seq_len_list[i],
                "prompt_len": prompt_len_list[i],
            }
            f.write(json.dumps(item) + '\n')

    log(f"Rank {rank} finished none_cnt={none_cnt}")


def save_rollout_samples(config=None, tokenizer=None, rbs: List[Dict[str, Any]] = None):
    labels_list = []
    question_list = []
    seq_len_list = []
    prompt_len_list = []

    for rb in rbs:
        seq_len = rb["sequence_lengths"]
        pmt_len = rb["prompt_lengths"]
        seq_len_list.append(seq_len if isinstance(seq_len, int) else seq_len.item())
        prompt_len_list.append(pmt_len if isinstance(pmt_len, int) else pmt_len.item())
        question_list.append(rb["tokens"][:pmt_len])
        labels_list.append(rb["tokens"][pmt_len:seq_len])

    question_texts = tokenizer.batch_decode(question_list, skip_special_tokens=False)
    question_texts = [parse_question(text) for text in question_texts]
    label_texts = tokenizer.batch_decode(labels_list, skip_special_tokens=False)
    assert len(question_texts) == len(label_texts), f"{len(question_texts)} != {len(label_texts)}"
    save_jsonl(
        config, question_texts, label_texts, seq_len_list, prompt_len_list,
        mpu.get_data_parallel_rank()
    )
