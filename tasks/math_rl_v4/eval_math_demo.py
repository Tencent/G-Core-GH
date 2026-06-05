import copy
import json
import os
import uuid
from typing import Any, Dict, List

import torch
import torch.distributed

from tasks.math_rl_v4.bt_reward import cal_accuracy_reward, cal_format_reward

from megatron.core import mpu

from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.utils import log


async def eval_generate_func(
    config, infer_engine, idx, tokenizer, student_tokenizer, batched_data, sampling_repeat_n
):
    prompt_token_ids = batched_data["prompt_token_ids"]
    prompt_lens = batched_data["prompt_lens"]
    gt_label = batched_data["gt_label"]

    # 先简单粗暴判断
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


def save_jsonl(config, total_conversation, gt_answer, acc_reward, fmt_reward, rank):
    os.makedirs(config.evaluate_result.output_dir, exist_ok=True)
    assert config.evaluate_result.output_prefix is not None
    output_dir = os.path.join(
        config.evaluate_result.output_dir, config.evaluate_result.output_prefix
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, f"{config.evaluate_result.output_prefix}_rank_{rank}.jsonl"), 'w'
    ) as f:
        for i in range(len(total_conversation)):
            item = {
                "gt_answer": gt_answer[i],
                "acc_reward": acc_reward[i],
                "fmt_reward": fmt_reward[i],
                "conversation": total_conversation[i],
            }
            f.write(json.dumps(item) + '\n')


def eval_acc_and_fmt_score(config=None, tokenizer=None, rbs: List[Dict[str, Any]] = None):
    eval_dir = "eval-tmp"
    os.makedirs(eval_dir, exist_ok=True)
    torch.save(rbs, f"{eval_dir}/eval_results_{torch.distributed.get_rank()}.pt")
    tokens_list = []
    gt_label_list = []
    for rb in rbs:
        tokens_list.append(rb["tokens"])
        gt_label_list.append(rb["gt_label"].item())

    total_conversation = tokenizer.batch_decode(tokens_list, skip_special_tokens=False)
    assert len(total_conversation
              ) == len(gt_label_list), f"{len(total_conversation)} != {len(gt_label_list)}"
    acc_reward, _, _ = cal_accuracy_reward(total_conversation, gt_label_list)
    fmt_reward = cal_format_reward(total_conversation)

    cpu_barrier()
    save_jsonl(
        config, total_conversation, gt_label_list, acc_reward, fmt_reward,
        mpu.get_data_parallel_rank()
    )

    acc_reward_tensor = torch.tensor(acc_reward, dtype=torch.float32).cuda()
    fmt_reward_tensor = torch.tensor(fmt_reward, dtype=torch.float32).cuda()
    local_sample_nums = torch.tensor(acc_reward_tensor.shape[0]).view(1).cuda()
    acc_reward_sum = acc_reward_tensor.sum().view(1)
    fmt_reward_sum = fmt_reward_tensor.sum().view(1)
    accuracy_local = acc_reward_sum / local_sample_nums
    format_local = fmt_reward_sum / local_sample_nums
    data = torch.cat([acc_reward_sum, fmt_reward_sum, local_sample_nums])

    torch.distributed.all_reduce(
        data, group=mpu.get_data_parallel_group(), op=torch.distributed.ReduceOp.SUM
    )
    accuracy = data[0] / data[2]
    format = data[1] / data[2]

    log(
        f"Accuracy: {accuracy}, Format: {format} samples {data[2]}| local accuracy: {accuracy_local}, local format: {format_local} local num {local_sample_nums}"
    )
