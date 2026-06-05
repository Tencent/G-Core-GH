import argparse
import sys
import time
import pprint

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def get_run_args():
    parser = argparse.ArgumentParser(description='Hf generate demo')
    parser.add_argument('--hf_ckpt', type=str, required=True)
    parser.add_argument('--prompts', type=str, nargs='*', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--use_fa', action='store_true')
    args = parser.parse_args()
    return args


def demo():
    args = get_run_args()
    torch.set_printoptions(precision=4, sci_mode=False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_ckpt,
        trust_remote_code=True,
        padding_side='left',
    )
    tokenizer.pad_token = tokenizer.eos_token

    # welm 和 baichuan 都没有 bos，忽略
    # for pi, prompt in enumerate(args.prompts):
    #     args.prompts[pi] = f'{tokenizer.bos_token}{prompt}'
    tokenized = tokenizer(
        args.prompts, add_special_tokens=False, padding='longest', return_tensors='pt'
    )
    input_ids = tokenized.input_ids.cuda()
    attention_mask = tokenized.attention_mask.cuda()
    print(f'input_ids {input_ids}')
    # print(f'attention_mask {pprint.pformat(attention_mask, indent=2)}')

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2' if args.use_fa else 'eager',
        trust_remote_code=True
    ).cuda()
    model.eval()
    for tryi in range(1):
        t0 = time.time()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            return_dict_in_generate=True,
            output_scores=True,
        )
        t1 = time.time()
        time_elpased = t1 - t0
        sequences = outputs['sequences']
        # print(f'generate outputs {pprint.pformat(outputs, indent=2)}')

        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        print(f'time_elpased {time_elpased}')
        print(
            f'inputs shape {input_ids.shape} sequences shape {sequences.shape} gen len {sequences.shape[1] - input_ids.shape[1]}'
        )
        print(f'texts {texts}')
        print(f'sequences {sequences}')
        # print(
        #     f'scores shape {scores.shape} {scores} sum {torch.sum(scores)} logit of "，" {scores[0, 101]} "太阳" {scores[0, 8893]} "他" {scores[0, 60061]}'
        # )

        # https://huggingface.co/docs/transformers/zh/internal/generation_utils
        # for tok_i, score in enumerate(outputs['scores']):
        #     print(f'{tok_i=} {score.shape=}')
        #     print(f'{torch.mean(score)=}')
        # print(f'{torch.min(score)=}')
        # print(f'{torch.max(score)=}')
        # print(f'{torch.sum(score)=}')
        # torch.save(score, f'hf-logits-{tok_i}.pt')


if __name__ == '__main__':
    demo()
