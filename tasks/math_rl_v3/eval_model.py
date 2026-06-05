import os
import argparse
import json
import re
import sys
from mpi4py import MPI

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from math_rule_rm import cal_accuracy_reward, cal_format_reward

device = "cuda"


def extract_gt_answer(text):
    match = re.search(r'####\s*(-?\d+(\.\d+)?)', text)
    if match:
        return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    else:
        return None


def create_model(args):
    print("model_path", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(tokenizer.eos_token, tokenizer.eos_token_id, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    return model, tokenizer


def save_jsonl(args, total_conversation, gt_answer, acc_reward, fmt_reward, rank):
    with open(args.output_path + f"_rank_{rank}.jsonl", 'w') as f:
        for i in range(len(total_conversation)):
            item = {
                "gt_answer": gt_answer[i],
                "acc_reward": acc_reward[i],
                "fmt_reward": fmt_reward[i],
                "conversation": total_conversation[i],
            }
            f.write(json.dumps(item) + '\n')


def do_eval(args, comm, world_size, rank, model, tokenizer):
    total_conversation = []
    gt_answer = []
    total_lines = 0
    with open(args.data_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    lines_per_rank = (total_lines + world_size - 1) // world_size
    start_line = lines_per_rank * rank
    end_line = min(start_line + lines_per_rank, total_lines)
    print(f"Rank {rank} total line {total_lines} {start_line} {end_line}")

    with open(args.data_path, 'r') as f:
        # seek
        print_flag = True
        for current_line, line in enumerate(tqdm(f)):
            if current_line < start_line:
                continue
            elif current_line >= end_line:
                break
            else:
                if print_flag or current_line % 10 == 0:
                    print(f"Rank {rank} process data index {current_line}")
                    print_flag = False

                try:
                    data = json.loads(line)
                except Exception as e:
                    print(e, line)
                    sys.exit()
                question = data["question"]
                answer = extract_gt_answer(data["answer"])
                gt_answer.append(answer)

                messages = [
                    {
                        "role":
                            "system",
                        "content":
                            "Please reason step by step, and put your final answer within \\boxed{}."
                    }, {
                        "role": "user",
                        "content": question
                    }
                ]

                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = tokenizer([input_text], return_tensors="pt").to(device)

                generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                total_conversation.append(input_text + response)

    acc_reward, _, _ = cal_accuracy_reward(total_conversation, gt_answer)
    fmt_reward = cal_format_reward(total_conversation)

    acc_reward_tensor = torch.tensor(acc_reward, dtype=torch.float32, device=model.device)
    fmt_reward_tensor = torch.tensor(fmt_reward, dtype=torch.float32, device=model.device)

    assert fmt_reward_tensor.shape[0] == acc_reward_tensor.shape[0]
    len_shape = acc_reward_tensor.shape[0]
    acc_reward_sum = acc_reward_tensor.sum()
    fmt_reward_sum = fmt_reward_tensor.sum()

    save_jsonl(args, total_conversation, gt_answer, acc_reward, fmt_reward, rank)

    comm.Barrier()
    total_acc_sum = comm.allreduce(acc_reward_sum, op=MPI.SUM)
    total_fmt_sum = comm.allreduce(fmt_reward_sum, op=MPI.SUM)
    total_len = comm.allreduce(len_shape, op=MPI.SUM)
    assert total_len == total_lines, f"total_len {total_len}, total_lines {total_lines}"

    accuracy = total_acc_sum / total_len
    fmt_matching_degress = total_fmt_sum / total_len
    if rank == 0:
        print(f"eval accuracy {accuracy.item():.5f}")
        print(f"format matching degree {fmt_matching_degress.item():.5f}")


def get_args():
    parser = argparse.ArgumentParser(
        description="test eval", allow_abbrev=False, conflict_handler='resolve'
    )
    parser.add_argument('--model_path', type=str, required=True, help="model path")
    parser.add_argument('--data_path', type=str, required=True, help='data_path')
    parser.add_argument("--output_path", type=str, required=True, help="output file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    comm = MPI.COMM_WORLD  # 假定使用 mpi，其实用 torchrun 也可以，但是 mpi 更加方便。
    rank = comm.Get_rank()
    size = comm.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)
    print(f"begin preprocess {rank} {size}", flush=True)
    print(f"inspect args {args}", flush=True)

    model, tokenizer = create_model(args)
    do_eval(args, comm, size, rank, model, tokenizer)


if __name__ == "__main__":
    main()
